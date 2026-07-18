#!/usr/bin/env python3
"""Evaluate fixed, sign-safe test-time augmentation on grouped validation."""

# argparse exposes the model, data, and output paths.
import argparse
# csv records every individual view and the fixed TTA average.
import csv
# Path resolves local files.
from pathlib import Path

# NumPy stores probability matrices and labels.
import numpy as np
# TensorFlow builds deterministic image views and runs inference.
import tensorflow as tf
# sklearn reports exact accuracy and macro F1.
from sklearn.metrics import accuracy_score, f1_score

# Shared collection and splitting preserve the validated group boundary.
from roadsign_pipeline import (
    DEFAULT_SEED,
    NUM_CLASSES,
    build_grouped_split,
    collect_training_samples,
    preprocess_resnet_image,
    resize_rgb_image,
)


# The original view plus five mild 4% zoom/crop positions are fixed in advance.
VIEWS = {
    "original": None,
    "zoom_top_left": (0, 0),
    "zoom_top_right": (0, 4),
    "zoom_center": (2, 2),
    "zoom_bottom_left": (4, 0),
    "zoom_bottom_right": (4, 4),
}


def build_view_dataset(paths: np.ndarray, view, batch_size: int):
    """Build one deterministic unaugmented or mild 4% crop view."""

    def decode(path):
        # Every view begins from the original RGB JPEG, never a previously processed tensor.
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])

        if view is None:
            # The original view exactly matches normal 96px validation preprocessing.
            image = resize_rgb_image(image, image_size=(96, 96))
        else:
            # Resize to 100px, then crop 96px at a fixed offset for a mild zoom/translation.
            image = resize_rgb_image(image, image_size=(100, 100))
            image = tf.image.crop_to_bounding_box(image, view[0], view[1], 96, 96)
            image.set_shape([96, 96, 3])
        return preprocess_resnet_image(image)

    dataset = tf.data.Dataset.from_tensor_slices(paths.astype(str))
    dataset = dataset.map(decode, num_parallel_calls=8)
    return dataset.batch(batch_size).prefetch(2)


def parse_args() -> argparse.Namespace:
    """Define the grouped checkpoint and comparison output."""
    parser = argparse.ArgumentParser(description="Evaluate fixed roadsign TTA.")
    parser.add_argument("--training-dir", default="data/roadsigns")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="experiments/grouped_tta_validation.csv")
    return parser.parse_args()


def main() -> None:
    """Score all fixed views and their equal-probability average."""
    args = parse_args()

    # Recreate the same physical-sign split used throughout the improvement campaign.
    paths, labels, groups, _ = collect_training_samples(Path(args.training_dir).resolve())
    split = build_grouped_split(paths, labels, groups, seed=DEFAULT_SEED)
    true_labels = split.val_labels.astype(np.int64)

    # The selected checkpoint is loaded once and reused for every deterministic view.
    model = tf.keras.models.load_model(Path(args.model).resolve(), compile=False)
    assert tuple(model.input_shape[-3:]) == (96, 96, 3)
    assert int(model.output_shape[-1]) == NUM_CLASSES

    rows = []
    probability_matrices = []
    for view_name, view in VIEWS.items():
        dataset = build_view_dataset(split.val_paths, view, args.batch_size)
        probabilities = np.asarray(model.predict(dataset, verbose=0))
        predictions = np.argmax(probabilities, axis=1)
        probability_matrices.append(probabilities)
        rows.append(
            {
                "candidate": view_name,
                "views": 1,
                "validation_accuracy": float(accuracy_score(true_labels, predictions)),
                "macro_f1": float(f1_score(true_labels, predictions, average="macro")),
            }
        )

    # Equal averaging is fixed; there is no view-weight search against validation labels.
    tta_probabilities = np.mean(np.stack(probability_matrices, axis=0), axis=0)
    tta_predictions = np.argmax(tta_probabilities, axis=1)
    rows.append(
        {
            "candidate": "equal_probability_tta",
            "views": len(VIEWS),
            "validation_accuracy": float(accuracy_score(true_labels, tta_predictions)),
            "macro_f1": float(f1_score(true_labels, tta_predictions, average="macro")),
        }
    )

    # Save the full comparison before printing the selected system score.
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate", "views", "validation_accuracy", "macro_f1"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Grouped validation TTA comparison:")
    for row in rows:
        print(
            f"  {row['candidate']}: accuracy={row['validation_accuracy']:.6f}, "
            f"macro_f1={row['macro_f1']:.6f}"
        )
    print(f"Comparison table: {output}")


# Direct execution runs the fixed TTA comparison.
if __name__ == "__main__":
    main()
