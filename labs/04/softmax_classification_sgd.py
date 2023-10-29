#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    n = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[range(n), y_true])
    return np.sum(log_likelihood) / n

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    metrics = []

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        for i in range(0, train_data.shape[0], args.batch_size):
            indices = permutation[i:i + args.batch_size]
            batch_x, batch_y = train_data[indices], train_target[indices]

            # Forward pass to calculate predictions
            logits = np.dot(batch_x, weights)

            # Stable softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Compute the gradient
            batch_y_one_hot = np.eye(args.classes)[batch_y]
            gradient = np.dot(batch_x.T, (probs - batch_y_one_hot)) / args.batch_size

            # Update the weights
            weights -= args.learning_rate * gradient

        # Evaluation
        train_logits = np.dot(train_data, weights)
        train_probs = softmax(train_logits)
        train_loss = cross_entropy_loss(train_probs, train_target)
        train_accuracy = np.mean(np.argmax(train_probs, axis=1) == train_target)

        test_logits = np.dot(test_data, weights)
        test_probs = softmax(test_logits)
        test_loss = cross_entropy_loss(test_probs, test_target)
        test_accuracy = np.mean(np.argmax(test_probs, axis=1) == test_target)

        metrics.append((train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        print(
            f"After epoch {epoch + 1}: train loss {train_loss:.4f} acc {train_accuracy * 100:.1f}%, test loss {test_loss:.4f} acc {test_accuracy * 100:.1f}%")

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
