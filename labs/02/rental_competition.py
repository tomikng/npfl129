#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
import numpy as np
import sklearn.compose
import sklearn.dummy
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import urllib.request

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")
parser.add_argument("--cv", default=5, type=int, help="Cross-validate with given number of folds")
parser.add_argument("--model", default="lr", type=str, help="Model to use")


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """

    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def select_model(args: argparse.Namespace, train):
    if args.model == "mean" or args.model == "median":
        return sklearn.dummy.DummyRegressor(strategy=args.model)
    elif args.model == "gbt":
        return sklearn.ensemble.GradientBoostingRegressor(max_depth=6, n_estimators=200)
    else:
        return create_complex_model(args, train)


def create_complex_model(args: argparse.Namespace, train):
    int_columns = np.all(train.data.astype(int) == train.data, axis=0)
    model_steps = [
        ("preprocess", sklearn.compose.ColumnTransformer([
            ("onehot", sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"), int_columns),
            ("scaler", sklearn.preprocessing.StandardScaler(), ~int_columns),
        ]))
    ]

    # Extend with model specific steps
    if args.model == "lr":
        model_steps.extend([
            ("poly", sklearn.preprocessing.PolynomialFeatures(2)),
            ("lr_cv", sklearn.linear_model.RidgeCV(alphas=np.arange(0.1, 10.1, 0.1))),
        ])
    else:
        raise ValueError("Unknown model '{}'".format(args.model))

    return sklearn.pipeline.Pipeline(model_steps)


def cross_validate(model, train, args):
    scores = sklearn.model_selection.cross_val_score(model, train.data, train.target,
                                                     scoring="neg_root_mean_squared_error", cv=args.cv)
    print(f"Cross-validation with {args.cv} folds: {(-scores.mean()):.2f} +/-{scores.std():.2f}")


def train_model(args):
    np.random.seed(args.seed)
    train = Dataset()
    model = select_model(args, train)
    if args.cv:
        cross_validate(model, train, args)
    model.fit(train.data, train.target)
    with lzma.open(args.model_path, "wb") as model_file:
        pickle.dump(model, model_file)


def predict_model(args):
    test = Dataset(args.predict)
    with lzma.open(args.model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model.predict(test.data)


def main(args):
    if args.predict is None:
        train_model(args)
    else:
        return predict_model(args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
