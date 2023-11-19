#!/usr/bin/env python3
import argparse
import lzma
import multiprocessing
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import scipy.ndimage
import sklearn.compose
import sklearn.ensemble
import sklearn.kernel_approximation
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--augment", default=False, action="store_true", help="Augment during training")
parser.add_argument("--epochs", default=15, type=int, help="Training epochs")
parser.add_argument("--models", default=1, type=int, help="Model to train")
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")


class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


def augment(x):
    x = x.reshape(28, 28)
    x = scipy.ndimage.zoom(x.reshape(28, 28), (np.random.uniform(0.86, 1.2), np.random.uniform(0.86, 1.2)))
    x = np.pad(x, [(2, 2), (2, 2)])
    os = [np.random.randint(size - 28 + 1) for size in x.shape]
    x = x[os[0]:os[0] + 28, os[1]:os[1] + 28]
    x = scipy.ndimage.rotate(x, np.random.uniform(-15, 15), reshape=False)
    x = np.clip(x, 0, 1)
    return x.reshape(-1)


def train_and_save_model(train_data, train_target, args):
    # Define the machine learning pipeline
    model = sklearn.pipeline.Pipeline([
        ("scaler", sklearn.preprocessing.MinMaxScaler()),
        ("MLPs", sklearn.ensemble.VotingClassifier(
            [
                ("MLP{}".format(i), sklearn.neural_network.MLPClassifier(
                    tol=0, verbose=1, alpha=0, hidden_layer_sizes=(500), max_iter=1 if args.augment else args.epochs))
                for i in range(args.models)
            ],
            voting="soft")),
    ])

    # Fit the model with or without augmentation
    model.fit(train_data, train_target)

    if args.augment:
        # Perform data augmentation in parallel using multiprocessing
        pool = multiprocessing.Pool()
        for mlp in model["MLPs"].estimators_:
            for epoch in range(args.epochs - 1):
                print(f"Augmenting data for epoch {epoch}...", end="", flush=True)
                augmented_data = pool.map(augment, train_data)
                print("Done", flush=True)
                mlp.partial_fit(augmented_data, train_target)

    # Compress the model by converting the numpy arrays to float16 and removing the optimizer
    compress_model(model["MLPs"].estimators_)

    # Serialize the model to the disk using LZMA compression
    with lzma.open(args.model_path, "wb") as model_file:
        pickle.dump(model, model_file)


def load_and_predict(model_path, test_data):
    # Deserialize the model from the disk
    with lzma.open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # Predict using the loaded model
    return model.predict(test_data)


def compress_model(mlp_estimators):
    # Remove the optimizer and convert arrays to float16 for compression
    for mlp in mlp_estimators:
        mlp._optimizer = None  # Not needed for prediction
        mlp.coefs_ = [c.astype(np.float16) for c in mlp.coefs_]
        mlp.intercepts_ = [i.astype(np.float16) for i in mlp.intercepts_]


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    # Train or predict based on the presence of the --predict flag
    if args.predict is None:
        # We are training a model
        np.random.seed(args.seed)
        train = Dataset()  # Load the training dataset
        train_and_save_model(train.data, train.target, args)  # Train and save the model
    else:
        # We are predicting using a pre-trained model
        test = Dataset(args.predict)  # Load the testing dataset
        predictions = load_and_predict(args.model_path, test.data)  # Make predictions
        return predictions


# Entry point of the script
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
