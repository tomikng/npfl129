#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

# Argument parser setup.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--with_reference", default=False, action="store_true", help="Use reference implementation")


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Setup random generator, create dataset, n-hot encode target, and split data.
    generator = np.random.RandomState(args.seed)
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    target = np.zeros((args.data_size, args.classes), dtype=int)  # Correct the shape to be 2D
    for i, label_indices in enumerate(target_list):
        # Ensure we handle both single class cases and multiple classes per example
        target[i, label_indices if isinstance(label_indices, list) else [label_indices]] = 1

    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)
    weights = generator.uniform(size=(train_data.shape[1], args.classes), low=-0.1, high=0.1)

    # Training with SGD and weight update per batch.
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        for i in range(0, len(permutation), args.batch_size):
            batch_indices = permutation[i:i + args.batch_size]
            batch_data, batch_targets = train_data[batch_indices], train_target[batch_indices]
            predictions = 1 / (1 + np.exp(-batch_data @ weights))
            weights -= args.learning_rate * (batch_data.T @ (predictions - batch_targets) / args.batch_size)

        # Manual computation of micro/macro F1-scores.
        train_predictions = (train_data @ weights) > 0
        test_predictions = (test_data @ weights) > 0
        train_f1_micro, train_f1_macro, test_f1_micro, test_f1_macro = compute_f1_scores(train_target, test_target, train_predictions, test_predictions)

        if args.with_reference:
            # Using sklearn metrics for reference if requested.
            train_f1_micro = sklearn.metrics.f1_score(train_target, train_predictions, average="micro", zero_division=0)
            train_f1_macro = sklearn.metrics.f1_score(train_target, train_predictions, average="macro", zero_division=0)
            test_f1_micro = sklearn.metrics.f1_score(test_target, test_predictions, average="micro", zero_division=0)
            test_f1_macro = sklearn.metrics.f1_score(test_target, test_predictions, average="macro", zero_division=0)

        # Output the epoch results.
        print_epoch_results(epoch, train_f1_micro, train_f1_macro, test_f1_micro, test_f1_macro)

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


def compute_f1_scores(train_target, test_target, train_predictions, test_predictions):
    train_f1_micro = f1_score(train_target, train_predictions, average='micro')
    train_f1_macro = f1_score(train_target, train_predictions, average='macro')
    test_f1_micro = f1_score(test_target, test_predictions, average='micro')
    test_f1_macro = f1_score(test_target, test_predictions, average='macro')
    return train_f1_micro, train_f1_macro, test_f1_micro, test_f1_macro


def f1_score(y_true, y_pred, average):
    if average not in ['micro', 'macro']:
        raise ValueError("Average must be either 'micro' or 'macro'")

    def f1_calc(true, pred):
        tp = np.sum((true == 1) & (pred == 1))
        fp = np.sum((true == 0) & (pred == 1))
        fn = np.sum((true == 1) & (pred == 0))
        return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0

    if average == 'micro':
        return f1_calc(y_true.ravel(), y_pred.ravel())
    else:  # macro
        return np.mean([f1_calc(t, p) for t, p in zip(y_true.T, y_pred.T)])


def print_epoch_results(epoch, train_f1_micro, train_f1_macro, test_f1_micro, test_f1_macro):
    print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
        epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, f1_scores = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")

