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
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where $ReLU(x) = max(x, 0)$, and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as `ReLU(inputs @ weights[0] + biases[0])`.
        # The value of the output layer is computed as `softmax(hidden_layer @ weights[1] + biases[1])`.
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate values which are non-positive, and overflow does not occur.
        # Hidden layer
        hidden_input = inputs @ weights[0] + biases[0]
        hidden_output = np.maximum(0, hidden_input)  # ReLU activation

        # Output layer with softmax
        output_input = hidden_output @ weights[1] + biases[1]
        output_input -= np.max(output_input, axis=1, keepdims=True)  # for numerical stability
        exps = np.exp(output_input)
        softmax_output = exps / np.sum(exps, axis=1, keepdims=True)

        return hidden_output, softmax_output

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        for batch_start in range(0, len(train_data), args.batch_size):
            batch_indices = permutation[batch_start:batch_start + args.batch_size]
            batch_data = train_data[batch_indices]
            batch_target = train_target[batch_indices]

            # Forward pass
            hidden_output, softmax_output = forward(batch_data)

            # Compute the gradient
            grad_output = softmax_output.copy()
            grad_output[range(len(batch_target)), batch_target] -= 1  # derivative of loss w.r.t softmax input

            grad_weights1 = hidden_output.T @ grad_output
            grad_biases1 = np.sum(grad_output, axis=0)

            grad_hidden = grad_output @ weights[1].T
            grad_hidden[hidden_output <= 0] = 0  # backprop through ReLU

            grad_weights0 = batch_data.T @ grad_hidden
            grad_biases0 = np.sum(grad_hidden, axis=0)

            # Update weights and biases
            weights[1] -= args.learning_rate * grad_weights1 / args.batch_size
            biases[1] -= args.learning_rate * grad_biases1 / args.batch_size

            weights[0] -= args.learning_rate * grad_weights0 / args.batch_size
            biases[0] -= args.learning_rate * grad_biases0 / args.batch_size

        # Compute training and testing accuracy
        _, train_preds = forward(train_data)
        train_accuracy = np.mean(np.argmax(train_preds, axis=1) == train_target)

        _, test_preds = forward(test_data)
        test_accuracy = np.mean(np.argmax(test_preds, axis=1) == test_target)

        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [100 * train_accuracy, 100 * test_accuracy]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(args)
    print("Learned parameters:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:12]] + ["..."]) for ws in parameters), sep="\n")
