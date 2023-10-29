#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")


class Dataset:
    def __init__(self, name="mnist.train.npz", data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28 * 28]).astype(float)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        # Preprocess the data using MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(train.data)

        # Create and train the model
        model = MLPClassifier(hidden_layer_sizes=(500,), max_iter=200, verbose=True)
        model.fit(scaled_data, train.target)

        # Serialize the model
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump((model, scaler), model_file)

    else:
        # Load the test dataset
        test = Dataset(args.predict)

        # Load the model and the scaler
        with lzma.open(args.model_path, "rb") as model_file:
            model, scaler = pickle.load(model_file)

        # Preprocess the test data
        scaled_test_data = scaler.transform(test.data)

        # Generate predictions
        predictions = model.predict(scaled_test_data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
