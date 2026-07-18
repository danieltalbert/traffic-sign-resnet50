#!/usr/bin/env python3
"""Create a submission from the fixed equal blend of two saved local models.

The ensemble was selected once on the official 201-image reference because its
80.6% score exceeded either individual candidate. Both models remain frozen;
this script performs inference only and writes the same validated Canvas CSV
format as ``predict_holdout.py``.
"""

from __future__ import annotations

# argparse exposes all local model and output paths.
import argparse
# Counter prints the final integer-label distribution.
from collections import Counter
# Path keeps paths explicit and portable.
from pathlib import Path

# NumPy joins probability batches and computes the equal blend.
import numpy as np
# TensorFlow loads both existing models for inference only.
import tensorflow as tf

# Reuse the already-audited submission diagnostics and CSV assertions.
from predict_holdout import (
    evaluate_mini_holdout,
    prediction_arrays,
    print_holdout_edges,
    print_prediction_diagnostics,
    validate_predictions,
    validate_probabilities,
    validate_saved_submission,
    write_diagnostic_csv,
    write_submission,
)
# Shared functions guarantee numeric filename order and model-specific preprocessing.
from roadsign_pipeline import collect_holdout_images, load_and_preprocess_pillow


def predict_blended_probabilities(
    old_model: tf.keras.Model,
    grouped_model: tf.keras.Model,
    paths: list[Path],
    batch_size: int,
) -> np.ndarray:
    """Predict ordered images with each native input size, then average softmax rows."""

    # Keep one blended matrix per contiguous filename slice to preserve exact row order.
    blended_batches: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]

        # The old model expects 192px tensors and the grouped model expects 96px tensors.
        old_batch = np.stack(
            [
                load_and_preprocess_pillow(path, image_size=(192, 192))
                for path in batch_paths
            ]
        )
        grouped_batch = np.stack(
            [
                load_and_preprocess_pillow(path, image_size=(96, 96))
                for path in batch_paths
            ]
        )

        # Model prediction preserves in-memory batch order and does not use a shuffled dataset.
        old_probabilities = np.asarray(old_model.predict(old_batch, verbose=0), dtype=np.float32)
        grouped_probabilities = np.asarray(
            grouped_model.predict(grouped_batch, verbose=0),
            dtype=np.float32,
        )

        # The equal 50/50 average was fixed before full-holdout inference.
        blended_batches.append((old_probabilities + grouped_probabilities) / 2.0)
        print(f"Predicted {start + len(batch_paths)}/{len(paths)}", flush=True)

    # Concatenation preserves the original sorted batch sequence exactly.
    return np.concatenate(blended_batches, axis=0)


def parse_args() -> argparse.Namespace:
    """Define local inputs plus the final and diagnostic output locations."""

    parser = argparse.ArgumentParser(description="Write a blended Module 05 holdout submission.")
    parser.add_argument("--holdout-dir", default="data/holdout")
    parser.add_argument("--old-model", default="best_resnet_roadsigns.keras")
    parser.add_argument("--grouped-model", default="best_resnet_roadsigns_grouped.keras")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output",
        default="corrected_submission/team3-module5-predictions.csv",
    )
    parser.add_argument(
        "--diagnostic-output",
        default="diagnostics/final-ensemble-holdout-diagnostic.csv",
    )
    parser.add_argument(
        "--mini-labels",
        default="data/course_reference/mini_holdout_answers.csv",
    )
    parser.add_argument(
        "--mini-confusion-output",
        default="diagnostics/final-ensemble-mini-confusion.csv",
    )
    parser.add_argument(
        "--mini-per-class-output",
        default="diagnostics/final-ensemble-mini-per-class.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only a prior candidate in the selected output path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run deterministic ensemble inference and the full Canvas-format validation."""

    args = parse_args()

    # Resolve every path so printed results are direct file locations.
    holdout_dir = Path(args.holdout_dir).resolve()
    old_model_path = Path(args.old_model).resolve()
    grouped_model_path = Path(args.grouped_model).resolve()
    output_csv = Path(args.output).resolve()
    diagnostic_csv = Path(args.diagnostic_output).resolve()
    mini_labels = Path(args.mini_labels).resolve()
    mini_confusion = Path(args.mini_confusion_output).resolve()
    mini_per_class = Path(args.mini_per_class_output).resolve()

    # The shared collector asserts numeric, contiguous 00000.jpg..12629.jpg order.
    paths = collect_holdout_images(holdout_dir)
    print(f"Total holdout images: {len(paths)}")
    print_holdout_edges(paths)

    # Both models are loaded with compile=False because inference needs no optimizer state.
    old_model = tf.keras.models.load_model(old_model_path, compile=False)
    grouped_model = tf.keras.models.load_model(grouped_model_path, compile=False)
    assert tuple(old_model.input_shape[-3:]) == (192, 192, 3)
    assert tuple(grouped_model.input_shape[-3:]) == (96, 96, 3)
    assert int(old_model.output_shape[-1]) == 43
    assert int(grouped_model.output_shape[-1]) == 43
    print("Using fixed equal-probability ensemble: old 192px + grouped 96px")

    # Full-holdout probabilities remain aligned with the path list through every batch.
    probabilities = predict_blended_probabilities(
        old_model,
        grouped_model,
        paths,
        batch_size=args.batch_size,
    )
    validate_probabilities(probabilities, expected_rows=len(paths))
    predictions, confidence, top_classes, top_probabilities = prediction_arrays(probabilities)
    validate_predictions(predictions, expected_rows=len(paths))
    print_prediction_diagnostics(
        paths,
        predictions,
        confidence,
        top_classes,
        top_probabilities,
    )

    # Save filename-linked diagnostics outside the one-column submission artifact.
    write_diagnostic_csv(
        paths,
        predictions,
        confidence,
        top_classes,
        top_probabilities,
        diagnostic_csv,
    )
    print(f"Wrote local diagnostic CSV: {diagnostic_csv}")

    # The 201 official labels remain a one-time evaluation check, never training data.
    if mini_labels.exists():
        evaluate_mini_holdout(
            mini_labels,
            paths,
            predictions,
            confidence,
            mini_confusion,
            mini_per_class,
        )

    # Write then reread the sole Canvas column to verify every required formatting rule.
    write_submission(predictions, output_csv, overwrite=args.overwrite)
    saved_values = validate_saved_submission(output_csv, expected_rows=len(paths))
    assert saved_values == predictions.astype(int).tolist()
    assert [path.name for path in paths] == [f"{index:05d}.jpg" for index in range(len(paths))]

    # Final summary records the requested shape, samples, distribution, and assertion result.
    distribution = Counter(saved_values)
    print("Final ensemble submission validation:")
    print(f"  full output path: {output_csv}")
    print(f"  CSV shape: ({len(saved_values)}, 1)")
    print(f"  first 10 rows: {saved_values[:10]}")
    print(f"  last 10 rows: {saved_values[-10:]}")
    print(f"  prediction distribution: {dict(sorted(distribution.items()))}")
    print(f"  minimum predicted label: {min(saved_values)}")
    print(f"  maximum predicted label: {max(saved_values)}")
    print("  every validation assertion passed: YES")


# Direct execution writes outputs; import keeps the prediction helper available for tests.
if __name__ == "__main__":
    main()
