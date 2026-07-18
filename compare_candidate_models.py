#!/usr/bin/env python3
"""Compare the two saved candidates on the official 201-image mini holdout.

This is intentionally a single, limited model-selection check: the original
192px model, the corrected grouped 96px model, and their equal-probability
ensemble. It does not train or tune either model.
"""

from __future__ import annotations

# argparse keeps all model and reference-data paths explicit.
import argparse
# csv writes the compact comparison table used in the final report.
import csv
# Path makes file access portable across WSL and Windows paths.
from pathlib import Path

# NumPy stores probability matrices and calculates accuracy.
import numpy as np
# TensorFlow loads the two local Keras models for inference only.
import tensorflow as tf

# Shared helpers enforce the same numeric holdout order and preprocessing contract.
from predict_holdout import load_mini_holdout_labels
from roadsign_pipeline import collect_holdout_images, load_and_preprocess_pillow


def predict_probabilities(
    model: tf.keras.Model,
    paths: list[Path],
    image_size: int,
    batch_size: int,
) -> np.ndarray:
    """Predict a fixed ordered path list with the input size expected by one model."""

    # Each model receives its own correctly resized tensors but the same filenames.
    batches: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = np.stack(
            [
                load_and_preprocess_pillow(path, image_size=(image_size, image_size))
                for path in batch_paths
            ]
        )
        batches.append(np.asarray(model.predict(batch, verbose=0), dtype=np.float32))
    return np.concatenate(batches, axis=0)


def parse_args() -> argparse.Namespace:
    """Define the small, fixed candidate comparison."""

    parser = argparse.ArgumentParser(description="Compare saved roadsign candidates on mini holdout.")
    parser.add_argument("--holdout-dir", default="data/holdout")
    parser.add_argument("--mini-labels", default="data/course_reference/mini_holdout_answers.csv")
    parser.add_argument("--old-model", default="best_resnet_roadsigns.keras")
    parser.add_argument("--grouped-model", default="best_resnet_roadsigns_grouped.keras")
    parser.add_argument("--output", default="diagnostics/candidate_model_comparison.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    """Evaluate two individual models and one fixed equal-weight ensemble."""

    args = parse_args()

    # The official labels specify both the exact count and filename order to evaluate.
    label_rows = load_mini_holdout_labels(Path(args.mini_labels).resolve())
    paths = collect_holdout_images(Path(args.holdout_dir).resolve())[: len(label_rows)]
    expected_names = [filename for filename, _ in label_rows]
    assert [path.name for path in paths] == expected_names
    true_labels = np.asarray([label for _, label in label_rows], dtype=np.int64)

    # Load both local checkpoints without optimizer state because no training occurs here.
    old_model = tf.keras.models.load_model(Path(args.old_model).resolve(), compile=False)
    grouped_model = tf.keras.models.load_model(Path(args.grouped_model).resolve(), compile=False)

    # Their input shapes prove the intended 192px and 96px preprocessing choices.
    assert tuple(old_model.input_shape[-3:]) == (192, 192, 3)
    assert tuple(grouped_model.input_shape[-3:]) == (96, 96, 3)

    old_probabilities = predict_probabilities(old_model, paths, 192, args.batch_size)
    grouped_probabilities = predict_probabilities(grouped_model, paths, 96, args.batch_size)
    ensemble_probabilities = (old_probabilities + grouped_probabilities) / 2.0

    # The equal blend is fixed before seeing the result; no weight search is performed.
    candidates = {
        "old_192px": old_probabilities,
        "grouped_96px": grouped_probabilities,
        "equal_probability_ensemble": ensemble_probabilities,
    }
    rows = []
    for name, probabilities in candidates.items():
        prediction = np.argmax(probabilities, axis=1)
        correct = int(np.sum(prediction == true_labels))
        rows.append(
            {
                "candidate": name,
                "mini_holdout_accuracy": f"{correct / len(true_labels):.8f}",
                "correct": correct,
                "total": len(true_labels),
            }
        )

    # Persist the result so model selection is transparent and reproducible.
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate", "mini_holdout_accuracy", "correct", "total"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Official 201-image candidate comparison:")
    for row in rows:
        print(
            f"  {row['candidate']}: {row['mini_holdout_accuracy']} "
            f"({row['correct']}/{row['total']})"
        )
    print(f"Comparison table: {output}")


# Importing this file exposes the helper for tests; direct execution runs the comparison.
if __name__ == "__main__":
    main()
