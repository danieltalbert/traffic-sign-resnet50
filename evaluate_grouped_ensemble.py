#!/usr/bin/env python3
"""Evaluate a fixed equal ensemble on the zero-overlap grouped validation set."""

# argparse accepts an explicit list of compatible grouped checkpoints.
import argparse
# csv records individual and ensemble metrics for the experiment report.
import csv
# Path resolves local model and output locations.
from pathlib import Path

# NumPy stores labels and probability matrices.
import numpy as np
# TensorFlow reproduces the validation decode/resize/preprocess pipeline.
import tensorflow as tf
# sklearn computes exact accuracy and macro F1 consistently with training reports.
from sklearn.metrics import accuracy_score, f1_score

# Shared rules guarantee the same grouped split and ResNet preprocessing.
from roadsign_pipeline import (
    DEFAULT_SEED,
    NUM_CLASSES,
    build_grouped_split,
    collect_training_samples,
    preprocess_resnet_image,
    resize_rgb_image,
)


def build_validation_dataset(paths: np.ndarray, image_size: int, batch_size: int):
    """Build a deterministic, unaugmented validation image dataset."""

    def decode(path):
        # Force three RGB channels before the same bilinear resize and ResNet transform.
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = resize_rgb_image(image, image_size=(image_size, image_size))
        return preprocess_resnet_image(image)

    # No shuffle call appears here; prediction rows stay aligned with split.val_labels.
    dataset = tf.data.Dataset.from_tensor_slices(paths.astype(str))
    dataset = dataset.map(decode, num_parallel_calls=8)
    return dataset.batch(batch_size).prefetch(2)


def parse_args() -> argparse.Namespace:
    """Define grouped data, compatible models, and the comparison output."""
    parser = argparse.ArgumentParser(description="Evaluate a grouped model ensemble.")
    parser.add_argument("--training-dir", default="data/roadsigns")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="experiments/grouped_ensemble_validation.csv")
    return parser.parse_args()


def main() -> None:
    """Score each checkpoint and their fixed equal-probability average."""
    args = parse_args()

    # Rebuild the exact same seed-123 physical-sign split used by every experiment.
    paths, labels, groups, _ = collect_training_samples(Path(args.training_dir).resolve())
    split = build_grouped_split(paths, labels, groups, seed=DEFAULT_SEED)
    validation_dataset = build_validation_dataset(
        split.val_paths,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    true_labels = split.val_labels.astype(np.int64)

    # Models are named explicitly on the command line; no checkpoint search or cherry-picking occurs.
    probability_matrices = []
    rows = []
    for model_argument in args.models:
        model_path = Path(model_argument).resolve()
        model = tf.keras.models.load_model(model_path, compile=False)
        assert tuple(model.input_shape[-3:]) == (args.image_size, args.image_size, 3)
        assert int(model.output_shape[-1]) == NUM_CLASSES

        probabilities = np.asarray(model.predict(validation_dataset, verbose=0))
        predictions = np.argmax(probabilities, axis=1)
        probability_matrices.append(probabilities)
        rows.append(
            {
                "candidate": model_path.name,
                "models": 1,
                "validation_accuracy": float(accuracy_score(true_labels, predictions)),
                "macro_f1": float(f1_score(true_labels, predictions, average="macro")),
            }
        )

    # The ensemble is a fixed equal average across all supplied grouped checkpoints.
    ensemble_probabilities = np.mean(np.stack(probability_matrices, axis=0), axis=0)
    ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)
    rows.append(
        {
            "candidate": "equal_probability_ensemble",
            "models": len(probability_matrices),
            "validation_accuracy": float(accuracy_score(true_labels, ensemble_predictions)),
            "macro_f1": float(f1_score(true_labels, ensemble_predictions, average="macro")),
        }
    )

    # Persist the complete individual/ensemble comparison before printing it.
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate", "models", "validation_accuracy", "macro_f1"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Grouped validation candidate comparison:")
    for row in rows:
        print(
            f"  {row['candidate']}: accuracy={row['validation_accuracy']:.6f}, "
            f"macro_f1={row['macro_f1']:.6f}"
        )
    print(f"Comparison table: {output}")


# Direct execution performs the fixed validation comparison.
if __name__ == "__main__":
    main()
