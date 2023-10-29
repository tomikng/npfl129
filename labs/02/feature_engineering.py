#!/usr/bin/env python3
import argparse

import numpy as np
from sklearn.compose import ColumnTransformer
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()
    X, y = dataset.data, dataset.target

    # Split the dataset into a train set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    # Identify categorical and numerical columns
    is_categorical = np.array([np.all(np.mod(X[:, i], 1) == 0) for i in range(X.shape[1])])
    categorical_indices = np.where(is_categorical)[0]
    numerical_indices = np.where(~is_categorical)[0]

    # Column-wise transformations
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_indices),
            ('num', StandardScaler(), numerical_indices)
        ],
        remainder='passthrough'
    )

    # Combine all transformations into a pipeline
    pipeline = Pipeline([
        ('preprocess', column_transformer),
        ('poly', PolynomialFeatures(2, include_bias=False))
    ])

    # Fit the feature preprocessing steps on the training data
    train_data = pipeline.fit_transform(X_train)

    # Transform the testing data
    test_data = pipeline.transform(X_test)

    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))),
                  *["..."] if dataset.shape[1] > 140 else [])
