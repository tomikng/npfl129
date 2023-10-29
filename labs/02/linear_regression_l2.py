#!/usr/bin/env python3
import argparse
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=13, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")


def main(args: argparse.Namespace) -> tuple[float, float]:
    dataset = sklearn.datasets.load_diabetes()
    X, y = dataset.data, dataset.target

    # Split the dataset into a train set and a test set
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=args.test_size,
                                                                                random_state=args.seed)

    lambdas = np.geomspace(0.01, 10, num=500)
    rmses = []
    best_lambda = None
    best_rmse = float('inf')

    for l in lambdas:
        model = sklearn.linear_model.Ridge(alpha=l)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))

        rmses.append(rmse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_lambda = l

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength $\\lambda$")
        plt.ylabel("RMSE")
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))
