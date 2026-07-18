#!/usr/bin/env python3
"""Diagnose holdout inference and create the one-column Module 05 submission.

This script never trains on holdout data. It loads a saved Keras model, predicts
images in deterministic numeric filename order, writes a detailed local-only
diagnostic CSV, evaluates the official 201-image mini holdout when labels are
available, and separately writes the exact one-column Canvas submission.
"""

from __future__ import annotations

# argparse exposes all input and output paths without requiring code edits.
import argparse
# csv writes and rereads outputs without adding a dataframe index column.
import csv
# Counter summarizes predicted labels and mini-holdout confusion pairs.
from collections import Counter
# Path keeps WSL and native Windows path handling explicit.
from pathlib import Path
# statistics supplies the requested confidence median.
from statistics import median

# NumPy stores probability matrices and performs deterministic top-k selection.
import numpy as np
# TensorFlow loads the saved model and performs inference.
import tensorflow as tf

# Shared utilities prevent training and holdout preprocessing/order rules from drifting.
from roadsign_pipeline import (
    DEFAULT_IMAGE_SIZE,
    NUM_CLASSES,
    class_index_mapping,
    collect_holdout_images,
    load_and_preprocess_pillow,
)


def print_holdout_edges(paths: list[Path], edge_count: int = 20) -> None:
    """Print the first and last filenames that define final CSV row order."""

    # The beginning verifies that prediction row zero starts at 00000.jpg.
    print(f"First {min(edge_count, len(paths))} holdout filenames:")
    for index, path in enumerate(paths[:edge_count]):
        print(f"  row {index:05d}: {path.name}")

    # The ending catches truncation or an incorrectly sorted final batch.
    print(f"Last {min(edge_count, len(paths))} holdout filenames:")
    start_index = max(0, len(paths) - edge_count)
    for index, path in enumerate(paths[start_index:], start=start_index):
        print(f"  row {index:05d}: {path.name}")


def predict_in_batches(
    model: tf.keras.Model,
    paths: list[Path],
    batch_size: int,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Predict every holdout image without shuffling or reordering samples."""

    # Each appended matrix corresponds to one contiguous slice of the ordered path list.
    probability_batches: list[np.ndarray] = []
    total = len(paths)

    # A simple sequential range makes row ordering transparent and deterministic.
    for start in range(0, total, batch_size):
        batch_paths = paths[start : start + batch_size]

        # Pillow reads each path, while shared TensorFlow functions resize and preprocess it.
        batch = np.stack(
            [
                load_and_preprocess_pillow(path, image_size=image_size)
                for path in batch_paths
            ]
        )

        # model.predict receives an in-memory batch, so no generator can shuffle filenames.
        batch_probabilities = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
        probability_batches.append(batch_probabilities)

        # Progress output makes a full 12,630-image run observable.
        end = start + len(batch_paths)
        print(f"Predicted {end}/{total}", flush=True)

    # Concatenation preserves the same batch and within-batch order used above.
    return np.concatenate(probability_batches, axis=0)


def validate_probabilities(probabilities: np.ndarray, expected_rows: int) -> None:
    """Assert that model output is a valid probability matrix for 43 classes."""

    # One row per image and one column per class are required before argmax is meaningful.
    assert probabilities.shape == (expected_rows, NUM_CLASSES), (
        f"Expected probability shape {(expected_rows, NUM_CLASSES)}, "
        f"received {probabilities.shape}"
    )
    assert np.isfinite(probabilities).all(), "Probabilities contain NaN or infinity"

    # Softmax rows should sum to one; a loose tolerance allows normal float32 rounding.
    row_sums = probabilities.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3), "Model outputs do not sum to one"

    # Probabilities outside 0..1 indicate the wrong output layer or a corrupted model.
    assert float(probabilities.min()) >= -1e-6
    assert float(probabilities.max()) <= 1.0 + 1e-6


def prediction_arrays(
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert softmax output into integer predictions, confidence, and top three."""

    # axis=1 selects the best class independently for each image row.
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    confidence = np.max(probabilities, axis=1).astype(np.float32)

    # argsort followed by reversal produces descending top-three class indices per row.
    top_three_classes = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1].astype(np.int64)
    top_three_probabilities = np.take_along_axis(
        probabilities,
        top_three_classes,
        axis=1,
    ).astype(np.float32)
    return predictions, confidence, top_three_classes, top_three_probabilities


def validate_predictions(predictions: np.ndarray, expected_rows: int) -> None:
    """Assert complete integer class predictions in the valid 0..42 range."""

    # The final vector must align one-for-one with the sorted filename list.
    assert predictions.ndim == 1
    assert len(predictions) == expected_rows
    assert np.isfinite(predictions).all()

    # argmax returns integers, and this explicit check protects future code changes.
    assert np.issubdtype(predictions.dtype, np.integer)
    assert int(predictions.min()) >= 0
    assert int(predictions.max()) < NUM_CLASSES


def print_prediction_diagnostics(
    paths: list[Path],
    predictions: np.ndarray,
    confidence: np.ndarray,
    top_three_classes: np.ndarray,
    top_three_probabilities: np.ndarray,
) -> None:
    """Print the requested sample rows, label distribution, warnings, and confidence stats."""

    # First-20 top-three output is the fastest way to spot a mapping or preprocessing failure.
    print("First 20 predictions with confidence and top three classes:")
    for index in range(min(20, len(paths))):
        top_three_text = ", ".join(
            f"{int(label)} ({float(score):.4f})"
            for label, score in zip(
                top_three_classes[index],
                top_three_probabilities[index],
            )
        )
        print(
            f"  {paths[index].name}: prediction={int(predictions[index])}, "
            f"confidence={float(confidence[index]):.4f}, top3=[{top_three_text}]"
        )

    # A complete sorted distribution exposes class collapse or an unexpected label range.
    distribution = Counter(int(value) for value in predictions.tolist())
    print(f"Prediction label distribution: {dict(sorted(distribution.items()))}")

    # The four requested summary statistics describe certainty without exposing probabilities in Canvas.
    print("Confidence statistics:")
    print(f"  minimum: {float(np.min(confidence)):.6f}")
    print(f"  maximum: {float(np.max(confidence)):.6f}")
    print(f"  mean: {float(np.mean(confidence)):.6f}")
    print(f"  median: {float(median(confidence.tolist())):.6f}")

    # More than a quarter of all predictions in one class is suspicious for this 43-class set.
    dominant_label, dominant_count = distribution.most_common(1)[0]
    dominant_fraction = dominant_count / len(predictions)
    if dominant_fraction > 0.25:
        print(
            f"WARNING: class {dominant_label} dominates {dominant_fraction:.1%} "
            "of predictions."
        )

    # Nearly constant confidence often indicates broken preprocessing or a degenerate model.
    confidence_std = float(np.std(confidence))
    rounded_unique = len(np.unique(np.round(confidence, decimals=4)))
    if confidence_std < 0.01 or rounded_unique < 10:
        print(
            "WARNING: confidence values are suspiciously uniform "
            f"(std={confidence_std:.6f}, rounded_unique={rounded_unique})."
        )

    # Duplicate class labels are expected; duplicate filenames or missing rows are not.
    if len({path.name for path in paths}) != len(paths):
        print("WARNING: duplicate holdout filenames were detected.")
    if len(predictions) != len(paths):
        print("WARNING: one or more holdout images has no prediction.")


def write_diagnostic_csv(
    paths: list[Path],
    predictions: np.ndarray,
    confidence: np.ndarray,
    top_three_classes: np.ndarray,
    top_three_probabilities: np.ndarray,
    output_csv: Path,
) -> None:
    """Write local filename/confidence/top-three details that must not be submitted."""

    # Diagnostic artifacts live separately from the one-column Canvas file.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "filename",
            "prediction",
            "confidence",
            "top1_class",
            "top1_probability",
            "top2_class",
            "top2_probability",
            "top3_class",
            "top3_probability",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        # zip follows the already verified path order and cannot shuffle rows.
        for path, prediction, score, top_classes, top_scores in zip(
            paths,
            predictions,
            confidence,
            top_three_classes,
            top_three_probabilities,
        ):
            writer.writerow(
                {
                    "filename": path.name,
                    "prediction": int(prediction),
                    "confidence": f"{float(score):.8f}",
                    "top1_class": int(top_classes[0]),
                    "top1_probability": f"{float(top_scores[0]):.8f}",
                    "top2_class": int(top_classes[1]),
                    "top2_probability": f"{float(top_scores[1]):.8f}",
                    "top3_class": int(top_classes[2]),
                    "top3_probability": f"{float(top_scores[2]):.8f}",
                }
            )


def load_mini_holdout_labels(label_csv: Path) -> list[tuple[str, int]]:
    """Load the official 201 filename/class pairs from the BYU-I course repository."""

    # DictReader enforces named columns instead of assuming a fragile positional layout.
    with label_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Filename", "ClassId"]:
            raise AssertionError(
                f"Unexpected mini-holdout columns: {reader.fieldnames}"
            )
        rows = [(row["Filename"], int(row["ClassId"])) for row in reader]

    # Course reference labels must also stay inside the same 43-class mapping.
    if not rows:
        raise RuntimeError(f"No labels found in {label_csv}")
    if any(label < 0 or label >= NUM_CLASSES for _, label in rows):
        raise AssertionError("Mini-holdout labels fall outside 0..42")
    return rows


def save_mini_evaluation_tables(
    confusion: np.ndarray,
    class_totals: np.ndarray,
    class_correct: np.ndarray,
    confusion_csv: Path,
    per_class_csv: Path,
) -> None:
    """Save mini-holdout confusion matrix and per-class accuracy tables."""

    # The confusion matrix rows are true classes and columns are predicted classes.
    confusion_csv.parent.mkdir(parents=True, exist_ok=True)
    with confusion_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label"] + list(range(NUM_CLASSES)))
        for label, row in enumerate(confusion.tolist()):
            writer.writerow([label] + row)

    # Classes absent from the 201-image sample receive a blank accuracy, not a fake zero.
    per_class_csv.parent.mkdir(parents=True, exist_ok=True)
    with per_class_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class", "correct", "total", "accuracy"],
        )
        writer.writeheader()
        for label in range(NUM_CLASSES):
            total = int(class_totals[label])
            correct = int(class_correct[label])
            writer.writerow(
                {
                    "class": label,
                    "correct": correct,
                    "total": total,
                    "accuracy": f"{correct / total:.8f}" if total else "",
                }
            )


def evaluate_mini_holdout(
    label_csv: Path,
    paths: list[Path],
    predictions: np.ndarray,
    confidence: np.ndarray,
    confusion_csv: Path,
    per_class_csv: Path,
) -> float:
    """Evaluate the official first 201 images without using them for training."""

    # Labels include filenames so alignment can be proven before any metric is calculated.
    labeled_rows = load_mini_holdout_labels(label_csv)
    labeled_count = len(labeled_rows)
    expected_names = [filename for filename, _ in labeled_rows]
    actual_names = [path.name for path in paths[:labeled_count]]
    assert actual_names == expected_names, (
        "Mini-holdout labels do not align with the first sorted full-holdout filenames"
    )

    # Slice only the officially labeled prefix; the remaining holdout labels stay unknown.
    true_labels = np.asarray([label for _, label in labeled_rows], dtype=np.int64)
    mini_predictions = predictions[:labeled_count]
    mini_confidence = confidence[:labeled_count]
    accuracy = float(np.mean(mini_predictions == true_labels))

    # Build a full 43x43 confusion matrix so class indices remain visually unambiguous.
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true_label, predicted_label in zip(true_labels, mini_predictions):
        confusion[int(true_label), int(predicted_label)] += 1

    # Per-class counts expose classes hidden by the aggregate 201-image accuracy.
    class_totals = np.bincount(true_labels, minlength=NUM_CLASSES)
    class_correct = np.bincount(
        true_labels[mini_predictions == true_labels],
        minlength=NUM_CLASSES,
    )
    save_mini_evaluation_tables(
        confusion,
        class_totals,
        class_correct,
        confusion_csv,
        per_class_csv,
    )

    # Exclude the diagonal when ranking the most common true-to-predicted mistakes.
    confusion_pairs = Counter(
        (int(true_label), int(predicted_label))
        for true_label, predicted_label in zip(true_labels, mini_predictions)
        if true_label != predicted_label
    )

    # Low-confidence errors are useful diagnostics without repeatedly tuning on the mini set.
    incorrect_indices = np.flatnonzero(mini_predictions != true_labels)
    lowest_confidence_errors = sorted(
        incorrect_indices.tolist(),
        key=lambda index: float(mini_confidence[index]),
    )[:10]

    print("Official 201-image mini-holdout evaluation:")
    print(f"  filename alignment assertion: PASSED ({labeled_count} images)")
    print(f"  accuracy: {accuracy:.6f} ({int(np.sum(mini_predictions == true_labels))}/{labeled_count})")
    print("  most frequent confused class pairs (true -> predicted):")
    for (true_label, predicted_label), count in confusion_pairs.most_common(10):
        print(f"    {true_label} -> {predicted_label}: {count}")
    print("  lowest-confidence incorrect predictions:")
    for index in lowest_confidence_errors:
        print(
            f"    {paths[index].name}: true={int(true_labels[index])}, "
            f"predicted={int(mini_predictions[index])}, "
            f"confidence={float(mini_confidence[index]):.6f}"
        )
    print(f"  confusion matrix: {confusion_csv}")
    print(f"  per-class accuracy: {per_class_csv}")
    return accuracy


def write_submission(
    predictions: np.ndarray,
    output_csv: Path,
    overwrite: bool,
) -> None:
    """Write exactly one ``predictions`` column and no dataframe index."""

    # A new corrected directory avoids overwriting the earlier submitted file.
    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing submission {output_csv}; "
            "choose a new path or pass --overwrite."
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # csv.writer emits one header plus one integer row per image.
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["predictions"])
        writer.writerows([[int(prediction)] for prediction in predictions])


def validate_saved_submission(
    output_csv: Path,
    expected_rows: int,
) -> list[int]:
    """Reread the saved CSV and assert every Canvas formatting requirement."""

    # Reading the file back catches newline, header, blank, and accidental extra-column bugs.
    with output_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    assert rows and rows[0] == ["predictions"], "CSV must have one predictions header"
    data_rows = rows[1:]
    assert len(data_rows) == expected_rows, "CSV row count does not match holdout"
    assert all(len(row) == 1 for row in data_rows), "CSV contains extra columns"
    assert all(row[0].strip() != "" for row in data_rows), "CSV contains blank values"

    # int(text) rejects probabilities, one-hot vectors, and non-integer labels.
    values = [int(row[0]) for row in data_rows]
    assert all(0 <= value < NUM_CLASSES for value in values)
    return values


def parse_args() -> argparse.Namespace:
    """Define command-line options for reproducible inference and output paths."""

    parser = argparse.ArgumentParser(
        description="Diagnose Module 05 holdout predictions and write the Canvas CSV."
    )

    # Inputs identify the full unlabeled set, trained model, and shared input resolution.
    parser.add_argument("--holdout-dir", default="data/holdout")
    parser.add_argument("--model", default="best_resnet_roadsigns_grouped.keras")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square override; by default infer the saved model input size.",
    )
    parser.add_argument("--batch-size", type=int, default=128)

    # The final submission stays isolated from every local-only diagnostic artifact.
    parser.add_argument(
        "--output",
        default="corrected_submission/team3-module5-predictions.csv",
    )
    parser.add_argument(
        "--diagnostic-output",
        default="diagnostics/holdout_predictions_diagnostic.csv",
    )

    # Official mini-holdout labels are used only for evaluation, never model fitting.
    parser.add_argument(
        "--mini-labels",
        default="data/course_reference/mini_holdout_answers.csv",
    )
    parser.add_argument(
        "--mini-confusion-output",
        default="diagnostics/mini_holdout_confusion_matrix.csv",
    )
    parser.add_argument(
        "--mini-per-class-output",
        default="diagnostics/mini_holdout_per_class_accuracy.csv",
    )
    parser.add_argument(
        "--skip-mini-evaluation",
        action="store_true",
        help="Skip the official 201-image check even if the labels file exists.",
    )

    # Overwrite is opt-in so an earlier submission cannot disappear silently.
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run ordered inference, diagnostics, mini evaluation, and submission checks."""

    args = parse_args()

    # Resolve paths once so every printed artifact location is absolute and unambiguous.
    holdout_dir = Path(args.holdout_dir).resolve()
    model_path = Path(args.model).resolve()
    output_csv = Path(args.output).resolve()
    diagnostic_csv = Path(args.diagnostic_output).resolve()
    mini_label_csv = Path(args.mini_labels).resolve()
    mini_confusion_csv = Path(args.mini_confusion_output).resolve()
    mini_per_class_csv = Path(args.mini_per_class_output).resolve()

    # Numeric sorting plus contiguous-ID assertions define the submission row order.
    paths = collect_holdout_images(holdout_dir)
    print(f"Total holdout images: {len(paths)}")
    print_holdout_edges(paths)

    # compile=False loads only inference state and avoids optimizer/loss compatibility issues.
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Input and output dimensions prove this is a square RGB, 43-class model.
    model_input_shape = tuple(model.input_shape)
    model_output_shape = tuple(model.output_shape)
    model_height, model_width, model_channels = model_input_shape[-3:]
    assert model_channels == 3, f"Model must accept RGB input, received {model_input_shape}"
    assert model_height == model_width, f"Model input must be square, received {model_input_shape}"

    # Infer the required resolution from the saved model unless the caller asks
    # for a square override; this prevents a 96px model from being fed 192px data.
    image_size = (
        (int(model_height), int(model_width))
        if args.image_size is None
        else (args.image_size, args.image_size)
    )
    assert model_input_shape[-3:] == (image_size[0], image_size[1], 3), (
        f"Model input shape {model_input_shape} does not match {image_size + (3,)}"
    )
    mapping = class_index_mapping(int(model_output_shape[-1]))
    print(f"Class index -> label mapping: {mapping}")
    print("Class mapping assertions: PASSED (index 0 -> 0, 1 -> 1, ..., 42 -> 42)")

    # Prediction follows the same list order printed above and never enables shuffle.
    probabilities = predict_in_batches(
        model,
        paths,
        batch_size=args.batch_size,
        image_size=image_size,
    )
    validate_probabilities(probabilities, expected_rows=len(paths))
    predictions, confidence, top_three_classes, top_three_probabilities = prediction_arrays(
        probabilities
    )
    validate_predictions(predictions, expected_rows=len(paths))
    print_prediction_diagnostics(
        paths,
        predictions,
        confidence,
        top_three_classes,
        top_three_probabilities,
    )

    # The detailed file proves every prediction remains paired with the sorted filename.
    write_diagnostic_csv(
        paths,
        predictions,
        confidence,
        top_three_classes,
        top_three_probabilities,
        diagnostic_csv,
    )
    print(f"Wrote local diagnostic CSV: {diagnostic_csv}")

    # Evaluate the official labeled prefix once as a real-generalization diagnostic.
    if not args.skip_mini_evaluation and mini_label_csv.exists():
        evaluate_mini_holdout(
            mini_label_csv,
            paths,
            predictions,
            confidence,
            mini_confusion_csv,
            mini_per_class_csv,
        )
    elif not args.skip_mini_evaluation:
        print(f"Mini-holdout labels not found; skipped evaluation: {mini_label_csv}")

    # The submission contains predictions only and intentionally omits all debug fields.
    write_submission(predictions, output_csv, overwrite=args.overwrite)
    saved_values = validate_saved_submission(output_csv, expected_rows=len(paths))

    # Internal order proof: the diagnostic filenames and predictions came from the same loop.
    assert saved_values == predictions.astype(int).tolist()
    assert [path.name for path in paths] == [f"{index:05d}.jpg" for index in range(len(paths))]

    # Final output mirrors the assignment checklist so the file can be submitted confidently.
    distribution = Counter(saved_values)
    print("Final submission validation:")
    print(f"  full output path: {output_csv}")
    print(f"  CSV shape: ({len(saved_values)}, 1)")
    print(f"  first 10 rows: {saved_values[:10]}")
    print(f"  last 10 rows: {saved_values[-10:]}")
    print(f"  prediction distribution: {dict(sorted(distribution.items()))}")
    print(f"  minimum predicted label: {min(saved_values)}")
    print(f"  maximum predicted label: {max(saved_values)}")
    print("  every validation assertion passed: YES")


# Running as a script executes inference; importing it exposes testable helpers only.
if __name__ == "__main__":
    main()
