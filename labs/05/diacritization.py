#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import List, Dict, Tuple, Optional
import urllib.request

import numpy as np
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--dictionary", default=False, action="store_true", help="Use the dictionary")
parser.add_argument("--dev", default=None, type=float, help="Use given fraction as dev")
parser.add_argument("--hidden_layers", nargs="+", default=[100], type=int, help="Hidden layer sizes")
parser.add_argument("--max_iter", default=1000, type=int, help="Max iters")
parser.add_argument("--model", default="lr", type=str, help="Model to use")
parser.add_argument("--model_kind", default="single", type=str, help="Model kind (single/per_letter)")
parser.add_argument("--model_path", default="diacritization1.model", type=str, help="Model path")
parser.add_argument("--prune", default=0, type=int, help="Prune features with <= given counts")
parser.add_argument("--solver", default="saga", type=str, help="LR solver")
parser.add_argument("--target_mode", default="marks", type=str, help="Target mode (letters/marks)")
parser.add_argument("--window_chars", default=1, type=int, help="Window characters to use")
parser.add_argument("--window_ngrams", default=4, type=int, help="Window ngrams to use")


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

    TARGET_LETTERS = sorted(set(LETTERS_NODIA + LETTERS_DIA))

    @staticmethod
    def letter_to_target(letter, target_mode):
        if target_mode == "letters":
            return Dataset.TARGET_LETTERS.index(letter)
        elif target_mode == "marks":
            if letter in "áéíóúý":
                return 1
            if letter in "čďěňřšťůž":
                return 2
            return 0

    @staticmethod
    def target_to_letter(target, letter, target_mode):
        if target_mode == "letters":
            return Dataset.TARGET_LETTERS[target]
        elif target_mode == "marks":
            if target == 1:
                index = "aeiouy".find(letter)
                return "áéíóúý"[index] if index >= 0 else letter
            if target == 2:
                index = "cdenrstuz".find(letter)
                return "čďěňřšťůž"[index] if index >= 0 else letter
            return letter

    def get_features(self, args):
        processed = self.data.lower()
        features, targets, indices = [], [], []
        for i in range(len(processed)):
            if processed[i] not in Dataset.LETTERS_NODIA:
                continue
            features.append([processed[i]])
            for o in range(1, args.window_chars):
                features[-1].append(processed[i - o:i - o + 1])
                features[-1].append(processed[i + o:i + o + 1])
            for s in range(1, args.window_ngrams):
                for o in range(-s, 0 + 1):
                    features[-1].append(processed[max(i + o, 0):i + o + s + 1])
            targets.append(self.letter_to_target(self.target[i].lower(), args.target_mode))
            indices.append(i)

        return features, targets, indices


def prune_features(features: List[List[str]], prune_threshold: int) -> List[List[str]]:
    if prune_threshold:
        feature_counts: Dict[str, int] = {}
        for feature_list in features:
            for feature in feature_list:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        # Replace feature with '<unk>' if its count is <= prune_threshold
        features = [[feature if feature_counts.get(feature, 0) > prune_threshold else "<unk>"
                     for feature in feature_list]
                    for feature_list in features]
    return features


def create_model(args: argparse.Namespace):
    pipeline_steps = [
        ("one-hot", sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")),
        ("estimator",
         sklearn.neural_network.MLPClassifier(hidden_layer_sizes=args.hidden_layers, max_iter=args.max_iter, verbose=1)
         if args.model == "mlp" else
         sklearn.linear_model.LogisticRegression(solver=args.solver, multi_class="multinomial", max_iter=args.max_iter,
                                                 verbose=1)
         if args.model == "lr" else
         sklearn.linear_model.LogisticRegressionCV(solver=args.solver, Cs=np.geomspace(1, 1e4, 10),
                                                   max_iter=args.max_iter, cv=10, n_jobs=-1))
    ]
    return sklearn.pipeline.Pipeline(steps=pipeline_steps)


def postprocess_model(model: sklearn.pipeline.Pipeline, args: argparse.Namespace):
    if args.model == "mlp":
        mlp = model.named_steps["estimator"]
        mlp._optimizer = None  # Remove the optimizer before saving the model
        # Convert coefficients and intercepts to float16 to save space
        for i in range(len(mlp.coefs_)):
            mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        for i in range(len(mlp.intercepts_)):
            mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)
    if args.model == "lrr":
        lrr = model.named_steps["estimator"]
        lrr.coefs_paths_ = None  # Remove the paths before saving the model
        print("Finished training, chosen C {}".format(lrr.C_), file=sys.stderr)


# Define a function for training a model with the ability to handle development set splitting
def train_model(train_data: List[List[str]], train_target: List[int], args: argparse.Namespace):
    if args.dev:
        train_data, dev_data, train_target, dev_target = train_test_split(
            train_data, train_target, test_size=args.dev, random_state=args.seed, shuffle=False)

    # Create and train the model
    model = create_model(args)
    model.fit(train_data, train_target)
    postprocess_model(model, args)

    # Evaluate model on the development set if provided
    if args.dev:
        dev_accuracy = model.score(dev_data, dev_target)
        print("Development accuracy: {:.2f}%".format(100 * dev_accuracy), file=sys.stderr)

    return model


def predict_with_model(test_data: List[List[str]], model: sklearn.pipeline.Pipeline, args: argparse.Namespace) -> List[
    int]:
    return model.predict(test_data)


def main(args: argparse.Namespace) -> Optional[str]:
    np.random.seed(args.seed)

    if args.predict is None:  # Training mode
        # Load and preprocess training dataset
        train = Dataset()
        train_data, train_target, _ = train.get_features(args)
        train_data = prune_features(train_data, args.prune)

        # Train model according to the specified model kind
        if args.model_kind == "single":
            model = train_model(train_data, train_target, args)
            # Serialize the model
            with lzma.open(args.model_path, "wb") as model_file:
                pickle.dump((model, args), model_file)
        else:
            # Train per-letter model logic
            raise NotImplementedError("Per-letter model training is not implemented in the refactoring scope.")
    else:  # Prediction mode
        # Load and preprocess test dataset
        test = Dataset(args.predict)
        test_data, _, test_indices = test.get_features(args)

        # Deserialize the trained model
        with lzma.open(args.model_path, "rb") as model_file:
            model, model_args = pickle.load(model_file)

        # Generate predictions using the deserialized model
        test_target = predict_with_model(test_data, model, model_args)

        # Post-process predictions to form the output string
        predictions = list(test.data)
        for i, target in zip(test_indices, test_target):
            predictions[i] = test.target_to_letter(target, test.data[i].lower(), model_args.target_mode)
            if test.data[i].isupper():
                predictions[i] = predictions[i].upper()
        predictions = "".join(predictions)

        return predictions



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
