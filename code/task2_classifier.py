#!/usr/bin/env python3
"""Task 2: classify the Ionosphere data with a NumPy neural network.

The model, softmax cross-entropy loss, backpropagation, and SGD updates are
implemented directly with NumPy.  No machine-learning package is required.

Example:
    python task2_classifier.py --data 2026_ionosphere_data.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="2026_ionosphere_data.csv",
                        help="Ionosphere CSV file")
    parser.add_argument("--output-dir", default="task2_results",
                        help="Directory for plots, metrics, and predictions")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--activation", choices=["relu", "tanh"],
                        default="relu")
    parser.add_argument("--hidden", type=int, nargs="+", default=[16, 16],
                        help="Hidden-layer widths for the main model")
    parser.add_argument("--compare-widths", type=int, nargs="+",
                        default=[2, 8, 16, 32],
                        help="Last-hidden-layer widths for latent comparison")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0.0 <= args.momentum < 1.0:
        parser.error("momentum must be in [0, 1)")
    if any(width < 1 for width in args.hidden + args.compare_widths):
        parser.error("hidden widths must be positive")
    return args


def load_ionosphere(path):
    """Load 34 numeric features and the final b/g class label."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Could not find dataset: {path}")

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    if not rows or any(len(row) != 35 for row in rows):
        raise ValueError("Expected every row to contain 34 features and 1 label")

    features = np.asarray(
        [[float(value) for value in row[:-1]] for row in rows],
        dtype=np.float32,
    )
    label_names = np.asarray([row[-1].strip() for row in rows])
    classes = np.array(["b", "g"])
    if not np.all(np.isin(label_names, classes)):
        raise ValueError("Labels must be exactly 'b' or 'g'")

    label_indices = np.asarray([0 if label == "b" else 1 for label in label_names])
    targets = np.eye(2, dtype=np.float32)[label_indices]
    return features, targets, label_indices, classes


def split_and_standardize(features, targets, labels, seed):
    """Shuffle into an 80/20 split and standardize using training data only."""
    split_rng = np.random.default_rng(seed)
    indices = split_rng.permutation(len(features))
    train_count = int(0.8 * len(indices))
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]

    mean = features[train_indices].mean(axis=0, keepdims=True)
    scale = features[train_indices].std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale

    return (
        standardized.astype(np.float32), targets,
        labels, train_indices, test_indices, mean, scale,
    )


def cross_entropy(targets, probabilities):
    """Mean two-class cross-entropy."""
    clipped = np.clip(probabilities, 1e-8, 1.0)
    return float(-np.sum(targets * np.log(clipped)) / len(targets))


def classification_error(labels, predicted_labels):
    return float(np.mean(labels != predicted_labels))


class MLP:
    """An MLP with configurable hidden activation and softmax output."""

    def __init__(self, input_size, hidden_sizes, output_size, seed,
                 activation="relu"):
        rng = np.random.default_rng(seed)
        self.activation_name = activation
        layer_sizes = [input_size, *hidden_sizes, output_size]
        self.weights = []
        self.biases = []

        for layer_index, (fan_in, fan_out) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            is_hidden = layer_index < len(layer_sizes) - 2
            if is_hidden and activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(
                rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)
            )
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

        self.weight_velocity = [np.zeros_like(weight) for weight in self.weights]
        self.bias_velocity = [np.zeros_like(bias) for bias in self.biases]

    def forward(self, inputs, return_cache=False):
        activation = inputs
        activations = [inputs]
        pre_activations = []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            pre_activation = activation @ weight + bias
            pre_activations.append(pre_activation)
            if self.activation_name == "relu":
                activation = np.maximum(pre_activation, 0.0)
            else:
                activation = np.tanh(pre_activation)
            activations.append(activation)

        logits = activation @ self.weights[-1] + self.biases[-1]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponential = np.exp(shifted)
        probabilities = exponential / np.sum(exponential, axis=1, keepdims=True)
        if return_cache:
            return probabilities, (activations, pre_activations)
        return probabilities

    def latent(self, inputs):
        """Return the activation immediately before the output layer."""
        activation = inputs
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            pre_activation = activation @ weight + bias
            if self.activation_name == "relu":
                activation = np.maximum(pre_activation, 0.0)
            else:
                activation = np.tanh(pre_activation)
        return activation

    def backward(self, probabilities, targets, cache):
        activations, pre_activations = cache
        delta = (probabilities - targets) / len(targets)
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)

        for layer_index in range(len(self.weights) - 1, -1, -1):
            weight_gradients[layer_index] = activations[layer_index].T @ delta
            bias_gradients[layer_index] = np.sum(delta, axis=0, keepdims=True)

            if layer_index > 0:
                delta = delta @ self.weights[layer_index].T
                if self.activation_name == "relu":
                    delta *= pre_activations[layer_index - 1] > 0.0
                else:
                    delta *= 1.0 - activations[layer_index] ** 2

        return weight_gradients, bias_gradients

    def update(self, weight_gradients, bias_gradients, learning_rate, momentum):
        for index in range(len(self.weights)):
            self.weight_velocity[index] = (
                momentum * self.weight_velocity[index] + weight_gradients[index]
            )
            self.bias_velocity[index] = (
                momentum * self.bias_velocity[index] + bias_gradients[index]
            )
            self.weights[index] -= learning_rate * self.weight_velocity[index]
            self.biases[index] -= learning_rate * self.bias_velocity[index]

    def predict(self, inputs):
        return np.argmax(self.forward(inputs), axis=1)


def evaluate(model, features, targets, labels):
    probabilities = model.forward(features)
    predicted = np.argmax(probabilities, axis=1)
    return {
        "loss": cross_entropy(targets, probabilities),
        "error": classification_error(labels, predicted),
        "predicted": predicted,
        "probabilities": probabilities,
    }


def train_model(features, targets, labels, train_indices, test_indices,
                hidden_sizes, args, checkpoints, seed):
    model = MLP(
        features.shape[1], hidden_sizes, 2, seed,
        activation=args.activation,
    )
    # Log-prior initialization gives the softmax a sensible starting point.
    class_prior = targets[train_indices].mean(axis=0)
    model.biases[-1][0] = np.log(np.maximum(class_prior, 1e-8))

    batch_rng = np.random.default_rng(seed + 100)
    history = {"train_loss": [], "test_loss": [],
               "train_error": [], "test_error": []}
    latent_snapshots = {}

    for epoch in range(1, args.epochs + 1):
        order = batch_rng.permutation(train_indices)
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start:start + args.batch_size]
            probabilities, cache = model.forward(
                features[batch_indices], return_cache=True
            )
            weight_gradients, bias_gradients = model.backward(
                probabilities, targets[batch_indices], cache
            )
            model.update(
                weight_gradients, bias_gradients,
                args.learning_rate, args.momentum
            )

        train_metrics = evaluate(
            model, features[train_indices], targets[train_indices], labels[train_indices]
        )
        test_metrics = evaluate(
            model, features[test_indices], targets[test_indices], labels[test_indices]
        )
        history["train_loss"].append(train_metrics["loss"])
        history["test_loss"].append(test_metrics["loss"])
        history["train_error"].append(train_metrics["error"])
        history["test_error"].append(test_metrics["error"])

        if epoch in checkpoints:
            latent_snapshots[epoch] = model.latent(features).copy()

        if epoch == 1 or epoch in checkpoints:
            print(
                f"hidden={hidden_sizes!s:>12} | epoch {epoch:4d}/{args.epochs} "
                f"| train loss {train_metrics['loss']:.4f} "
                f"| test error {test_metrics['error']:.3f}"
            )

    train_metrics = evaluate(
        model, features[train_indices], targets[train_indices], labels[train_indices]
    )
    test_metrics = evaluate(
        model, features[test_indices], targets[test_indices], labels[test_indices]
    )
    if args.epochs not in latent_snapshots:
        latent_snapshots[args.epochs] = model.latent(features).copy()

    return {
        "model": model,
        "hidden_sizes": hidden_sizes,
        "activation": args.activation,
        "history": history,
        "latent_snapshots": latent_snapshots,
        "train": train_metrics,
        "test": test_metrics,
    }


def pca_projection(values):
    """Project latent features to two dimensions without sklearn."""
    centered = values - values.mean(axis=0, keepdims=True)
    if centered.shape[1] == 1:
        return np.concatenate((centered, np.zeros_like(centered)), axis=1)
    _, _, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    components = right_singular_vectors[:2].T
    return centered @ components


def save_learning_curve(result, epochs, output_dir):
    epoch_numbers = np.arange(1, epochs + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epoch_numbers, result["history"]["train_loss"], label="train")
    axes[0].plot(epoch_numbers, result["history"]["test_loss"], label="test")
    axes[0].set_title("Cross-entropy learning curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epoch_numbers, result["history"]["train_error"], label="train")
    axes[1].plot(epoch_numbers, result["history"]["test_error"], label="test")
    axes[1].set_title("Classification error rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_dir / "learning_curve.png", dpi=160)
    plt.close(figure)


def save_latent_comparison(results, labels, classes, checkpoints, output_dir):
    widths = list(results)
    figure, axes = plt.subplots(
        len(widths), len(checkpoints),
        figsize=(4.0 * len(checkpoints), 3.4 * len(widths)),
        squeeze=False,
    )
    colors = ["tab:blue", "tab:orange"]

    for row, width in enumerate(widths):
        result = results[width]
        for column, epoch in enumerate(checkpoints):
            axis = axes[row][column]
            projected = pca_projection(result["latent_snapshots"][epoch])
            for class_index, class_name in enumerate(classes):
                selected = labels == class_index
                axis.scatter(
                    projected[selected, 0], projected[selected, 1],
                    s=18, alpha=0.75, c=colors[class_index], label=class_name,
                )
            axis.set_title(f"{width} latent nodes, epoch {epoch}")
            axis.set_xlabel("latent PC 1")
            axis.set_ylabel("latent PC 2")
            axis.grid(True, alpha=0.2)
            if row == 0 and column == 0:
                axis.legend(title="class")

    figure.suptitle("Latent-feature distributions (PCA projection)", y=1.0)
    figure.tight_layout()
    figure.savefig(output_dir / "latent_feature_distribution.png", dpi=160)
    plt.close(figure)


def save_metrics(results, args, train_indices, test_indices, output_dir):
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model", "hidden_layers", "hidden_units", "activation", "epochs",
            "learning_rate", "momentum", "batch_size", "training_loss",
            "test_loss", "training_error_rate", "test_error_rate",
            "training_samples", "test_samples",
        ])
        for name, result in results.items():
            writer.writerow([
                name,
                len(result["hidden_sizes"]),
                " ".join(map(str, result["hidden_sizes"])),
                result["activation"],
                args.epochs,
                args.learning_rate,
                args.momentum,
                args.batch_size,
                f'{result["train"]["loss"]:.8f}',
                f'{result["test"]["loss"]:.8f}',
                f'{result["train"]["error"]:.8f}',
                f'{result["test"]["error"]:.8f}',
                len(train_indices),
                len(test_indices),
            ])


def save_predictions(result, features, labels, classes, output_dir):
    probabilities = result["model"].forward(features)
    predicted = np.argmax(probabilities, axis=1)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample", "true_label", "predicted_label", "p_b", "p_g"])
        for index, (true_index, predicted_index, probability) in enumerate(
            zip(labels, predicted, probabilities)
        ):
            writer.writerow([
                index, classes[true_index], classes[predicted_index],
                f"{probability[0]:.8f}", f"{probability[1]:.8f}",
            ])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, targets, labels, classes = load_ionosphere(args.data)
    features, targets, labels, train_indices, test_indices, mean, scale = (
        split_and_standardize(features, targets, labels, args.seed)
    )
    checkpoints = sorted({epoch for epoch in (10, 100, 500, args.epochs)
                          if epoch <= args.epochs})

    print(f"Samples: {len(features)}; features: {features.shape[1]}")
    print(f"Classes: {classes.tolist()} ({len(classes)} softmax outputs)")
    print(f"Train samples: {len(train_indices)}; test samples: {len(test_indices)}")
    print(
        f"Main architecture: 34 -> {args.hidden} -> 2 ({args.activation}); "
        f"epochs={args.epochs}, batch={args.batch_size}, "
        f"learning_rate={args.learning_rate}, momentum={args.momentum}"
    )

    main_result = train_model(
        features, targets, labels, train_indices, test_indices,
        args.hidden, args, checkpoints, args.seed + 1,
    )
    comparison_results = {}
    for offset, width in enumerate(args.compare_widths, start=10):
        comparison_results[width] = train_model(
            features, targets, labels, train_indices, test_indices,
            [width], args, checkpoints, args.seed + offset,
        )

    save_learning_curve(main_result, args.epochs, output_dir)
    save_latent_comparison(
        comparison_results, labels, classes, checkpoints, output_dir
    )
    all_results = {"main": main_result}
    all_results.update({f"latent_width_{width}": result
                        for width, result in comparison_results.items()})
    save_metrics(all_results, args, train_indices, test_indices, output_dir)
    save_predictions(main_result, features, labels, classes, output_dir)

    with (output_dir / "architecture.txt").open("w", encoding="utf-8") as file:
        file.write("Task 2 Ionosphere binary classifier\n")
        file.write(f"Input features: {features.shape[1]}\n")
        file.write(f"Hidden layers: {args.hidden}\n")
        file.write(f"Hidden activation: {args.activation}\n")
        file.write("Output: 2 logits with softmax\n")
        file.write("Loss: mean softmax cross-entropy\n")
        file.write(f"Training samples: {len(train_indices)}\n")
        file.write(f"Test samples: {len(test_indices)}\n")
        file.write(f"Standardization mean: {mean.ravel().tolist()}\n")
        file.write(f"Standardization scale: {scale.ravel().tolist()}\n")

    print("\nFinal main-model metrics:")
    print(f"training loss: {main_result['train']['loss']:.6f}")
    print(f"test loss:     {main_result['test']['loss']:.6f}")
    print(f"training error: {main_result['train']['error']:.2%}")
    print(f"test error:     {main_result['test']['error']:.2%}")
    print("\nLatent-width comparison (final test error):")
    for width, result in comparison_results.items():
        print(f"  {width:>2} nodes: {result['test']['error']:.2%}")
    print(f"\nSaved Task 2 outputs in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
