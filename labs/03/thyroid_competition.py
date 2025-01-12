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
from sklearn.linear_model import LogisticRegression  # Importing Logistic Regression

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")

class Dataset:
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        # Training a Logistic Regression model
        model = LogisticRegression()
        model.fit(train.data, train.target)

        # Serialize the model
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
    else:
        # Use the model and return test set predictions
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Generating predictions with the test set predictions
        predictions = model.predict(test.data)

        return predictions

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
