#!/usr/bin/env python3
"""Task 1: fit an image with a manually implemented neural network.

Only NumPy is used for the model, loss, backpropagation, and SGD.
Pillow loads the image; Matplotlib saves the required figures.

Example:
    python task1_inr.py --image target.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="2026_NYCU.png",
                        help="Input image (default: 2026_NYCU.png)")
    parser.add_argument("--output-dir", default="task1_results",
                        help="Directory for plots and metrics")
    parser.add_argument(
        "--epochs", type=int, default=800,
        help="Training epochs; default 800 includes the requested checkpoints"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--learning-rate", type=float, default=0.01,
        help="SGD learning rate; 0.01 works well with momentum"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9,
        help="Momentum coefficient for SGD; default 0.9"
    )
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128],
                        help="Hidden-layer widths (default: 128 128)")
    parser.add_argument("--fourier-bands", type=int, default=6,
                        help="Number of Fourier frequency bands")
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
    if args.fourier_bands < 1:
        parser.error("--fourier-bands must be positive")
    if any(width < 1 for width in args.hidden):
        parser.error("hidden-layer widths must be positive")
    return args


def load_image(image_path):
    """Load an image as float RGB values in [0, 1]."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Could not find {image_path}. Put target.png beside this script "
            "or pass its location with --image."
        )
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    return image


def make_coordinates(height, width):
    """Make one normalized (x, y) coordinate for every image pixel."""
    x_values = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y_values = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    return np.stack((xx.ravel(), yy.ravel()), axis=1).astype(np.float32)


def fourier_encode(coordinates, number_of_bands):
    """Append sine/cosine features at increasing spatial frequencies."""
    features = [coordinates]
    for band in range(number_of_bands):
        frequency = np.pi * (2 ** band)
        features.append(np.sin(frequency * coordinates))
        features.append(np.cos(frequency * coordinates))
    return np.concatenate(features, axis=1).astype(np.float32)


def mse_loss(targets, predictions):
    """Mean squared error over all pixels and all three color channels."""
    return float(np.mean((predictions - targets) ** 2))


def rms_error(targets, predictions):
    """RMS over all RGB values, as requested for evaluation."""
    return float(np.sqrt(mse_loss(targets, predictions)))

class MLP:
    """A tanh multilayer perceptron with a linear RGB output layer."""

    def __init__(self, input_size, hidden_sizes, output_size, seed):
        rng = np.random.default_rng(seed)
        layer_sizes = [input_size, *hidden_sizes, output_size]
        self.weights = []
        self.biases = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier/Glorot initialization is appropriate for tanh layers
            # and keeps the linear RGB output in a useful range at startup.
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            weight = rng.normal(0.0, scale, size=(fan_in, fan_out))
            self.weights.append(weight.astype(np.float32))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

        self.weight_velocity = [np.zeros_like(weight) for weight in self.weights]
        self.bias_velocity = [np.zeros_like(bias) for bias in self.biases]

    def forward(self, inputs, return_cache=False):
        """Compute predictions; optionally retain values needed for backprop."""
        activation = inputs
        activations = [inputs]
        pre_activations = []

        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = activation @ weight + bias
            pre_activations.append(z)
            activation = np.tanh(z)  # tanh
            activations.append(activation)

        # Regression output is linear; RGB predictions are not clipped here.
        predictions = activation @ self.weights[-1] + self.biases[-1]
        if return_cache:
            cache = (activations, pre_activations)
            return predictions, cache
        return predictions

    def backward(self, predictions, targets, cache):
        """Manually differentiate MSE through the network."""
        activations, pre_activations = cache
        # d(MSE)/d(predictions), averaging over batch rows and RGB channels.
        delta = (2.0 / targets.size) * (predictions - targets)
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)

        for layer_index in range(len(self.weights) - 1, -1, -1):
            weight_gradients[layer_index] = (
                activations[layer_index].T @ delta
            )
            bias_gradients[layer_index] = np.sum(
                delta, axis=0, keepdims=True
            )

            if layer_index > 0:
                delta = delta @ self.weights[layer_index].T
                delta *= 1.0 - activations[layer_index] ** 2

        return weight_gradients, bias_gradients

    def update(self, weight_gradients, bias_gradients, learning_rate,
               momentum=0.0):
        """One SGD-with-momentum parameter update."""
        for index in range(len(self.weights)):
            self.weight_velocity[index] = (
                momentum * self.weight_velocity[index] + weight_gradients[index]
            )
            self.bias_velocity[index] = (
                momentum * self.bias_velocity[index] + bias_gradients[index]
            )
            self.weights[index] -= learning_rate * self.weight_velocity[index]
            self.biases[index] -= learning_rate * self.bias_velocity[index]

    def predict(self, inputs, chunk_size=8192):
        """Predict in chunks to keep memory use modest for larger images."""
        outputs = []
        for start in range(0, len(inputs), chunk_size):
            outputs.append(self.forward(inputs[start:start + chunk_size]))
        return np.concatenate(outputs, axis=0)


def save_reconstruction(path, predictions, height, width):
    """Save predictions as an image; clip only for display, not for metrics."""
    rgb = predictions.reshape(height, width, 3)
    plt.imsave(path, np.clip(rgb, 0.0, 1.0))


def train_model(name, features, colors, train_indices, test_indices,
                image_shape, args, output_dir, checkpoints):
    height, width = image_shape
    model = MLP(
        input_size=features.shape[1],
        hidden_sizes=args.hidden,
        output_size=3,
        seed=args.seed,
    )
    # Start from the best constant RGB predictor.  This avoids spending the
    # first several epochs moving the output bias away from zero and makes the
    # relatively larger learning rate useful from the first mini-batch.
    model.biases[-1][0] = colors[train_indices].mean(axis=0)
    # Use a reproducible shuffled mini-batch order for each model.
    shuffle_rng = np.random.default_rng(args.seed + 100)
    history = {"train_mse": [], "test_mse": []}
    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        shuffled_train_indices = shuffle_rng.permutation(train_indices)

        for start in range(0, len(shuffled_train_indices), args.batch_size):
            batch_indices = shuffled_train_indices[start:start + args.batch_size]
            batch_inputs = features[batch_indices]
            batch_targets = colors[batch_indices]

            predictions, cache = model.forward(batch_inputs, return_cache=True)
            weight_gradients, bias_gradients = model.backward(
                predictions, batch_targets, cache
            )
            model.update(
                weight_gradients, bias_gradients,
                args.learning_rate, args.momentum
            )

        # Evaluation does not update the weights. Keep the test pixels held out
        # from all gradient updates.
        train_predictions = model.predict(features[train_indices])
        test_predictions = model.predict(features[test_indices])
        train_mse = mse_loss(colors[train_indices], train_predictions)
        test_mse = mse_loss(colors[test_indices], test_predictions)
        history["train_mse"].append(train_mse)
        history["test_mse"].append(test_mse)

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(
                f"{name:>8} | epoch {epoch:4d}/{args.epochs} "
                f"| train MSE {train_mse:.6f} "
                f"| test MSE {test_mse:.6f}"
            )

        if epoch in checkpoints:
            full_predictions = model.predict(features)
            save_reconstruction(
                model_dir / f"reconstruction_epoch_{epoch}.png",
                full_predictions, height, width
            )

    # Final metrics use the unclipped network predictions.
    train_predictions = model.predict(features[train_indices])
    test_predictions = model.predict(features[test_indices])
    final_train_rms = rms_error(colors[train_indices], train_predictions)
    final_test_rms = rms_error(colors[test_indices], test_predictions)
    full_predictions = model.predict(features)
    save_reconstruction(
        model_dir / "reconstruction_final.png",
        full_predictions, height, width
    )

    return {
        "name": name,
        "model": model,
        "history": history,
        "train_rms": final_train_rms,
        "test_rms": final_test_rms,
        "full_predictions": full_predictions,
        "input_size": features.shape[1],
    }


def save_learning_curve(results, epochs, output_dir):
    """Plot train/test MSE for both coordinate representations."""
    figure, axis = plt.subplots(figsize=(9, 5))
    colors_by_model = {"raw": "tab:blue", "fourier": "tab:orange"}
    epoch_numbers = np.arange(1, epochs + 1)

    for result in results:
        color = colors_by_model[result["name"]]
        axis.plot(
            epoch_numbers, result["history"]["train_mse"],
            color=color, label=f'{result["name"]} train'
        )
        axis.plot(
            epoch_numbers, result["history"]["test_mse"],
            color=color, linestyle="--",
            label=f'{result["name"]} test'
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE")
    axis.set_title("Learning curves: raw vs Fourier coordinates")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curve.png", dpi=160)
    plt.close(figure)


def save_final_comparison(original, raw_predictions, fourier_predictions,
                          height, width, output_dir):
    raw_image = np.clip(raw_predictions.reshape(height, width, 3), 0.0, 1.0)
    fourier_image = np.clip(
        fourier_predictions.reshape(height, width, 3), 0.0, 1.0
    )

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        (original, "Ground truth"),
        (raw_image, "Raw coordinates: final"),
        (fourier_image, "Fourier coordinates: final"),
    ]
    for axis, (pixels, title) in zip(axes, panels):
        axis.imshow(pixels)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "final_comparison.png", dpi=160)
    plt.close(figure)


def save_metrics(results, args, train_count, test_count, output_dir):
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model", "input_features", "hidden_layers", "hidden_units",
            "epochs", "learning_rate", "batch_size",
            "momentum",
            "training_rms", "test_rms", "training_pixels", "test_pixels"
        ])
        for result in results:
            writer.writerow([
                result["name"],
                result["input_size"],
                len(args.hidden),
                " ".join(map(str, args.hidden)),
                args.epochs,
                args.learning_rate,
                args.batch_size,
                args.momentum,
                f'{result["train_rms"]:.8f}',
                f'{result["test_rms"]:.8f}',
                train_count,
                test_count,
            ])


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(args.image)
    height, width, _ = image.shape
    coordinates = make_coordinates(height, width)
    colors = image.reshape(-1, 3).astype(np.float32)

    # Shuffle pixel indices once, then use the same split for both models.
    split_rng = np.random.default_rng(args.seed)
    shuffled_indices = split_rng.permutation(len(coordinates))
    train_count = int(0.9 * len(shuffled_indices))
    train_count = min(max(train_count, 1), len(shuffled_indices) - 1)
    train_indices = shuffled_indices[:train_count]
    test_indices = shuffled_indices[train_count:]

    print(f"Image: {width} x {height}")
    print(f"Coordinate array: {coordinates.shape}")
    print(f"RGB array:         {colors.shape}")
    print(f"Train pixels: {len(train_indices)}; test pixels: {len(test_indices)}")
    print(
        f"Architecture: input -> {args.hidden} -> 3 linear RGB outputs; "
        f"epochs={args.epochs}, batch={args.batch_size}, "
        f"learning_rate={args.learning_rate}, momentum={args.momentum}"
    )

    plt.imsave(output_dir / "ground_truth.png", image)
    checkpoints = {
        epoch for epoch in (100, 200, 400, 500, 600, 700, args.epochs)
        if epoch <= args.epochs
    }

    raw_result = train_model(
        "raw", coordinates, colors, train_indices, test_indices,
        (height, width), args, output_dir, checkpoints
    )

    encoded_coordinates = fourier_encode(
        coordinates, args.fourier_bands
    )
    fourier_result = train_model(
        "fourier", encoded_coordinates, colors, train_indices, test_indices,
        (height, width), args, output_dir, checkpoints
    )
    results = [raw_result, fourier_result]

    save_learning_curve(results, args.epochs, output_dir)
    save_final_comparison(
        image,
        raw_result["full_predictions"],
        fourier_result["full_predictions"],
        height, width, output_dir
    )
    save_metrics(results, args, len(train_indices), len(test_indices), output_dir)

    print("\nFinal RMS errors (computed before display clipping):")
    for result in results:
        print(
            f'{result["name"]:>8}: '
            f'train RMS={result["train_rms"]:.6f}, '
            f'test RMS={result["test_rms"]:.6f}'
        )
    print(f"\nSaved plots and metrics in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
