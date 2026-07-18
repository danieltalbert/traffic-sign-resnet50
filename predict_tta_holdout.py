#!/usr/bin/env python3
"""Evaluate or export the selected fixed six-view road-sign TTA system."""

# argparse separates mini-only evaluation from full holdout export.
import argparse
# Counter prints the final label distribution.
from collections import Counter
# Path resolves model, data, and artifact locations.
from pathlib import Path

# NumPy averages fixed-view probability matrices.
import numpy as np
# TensorFlow decodes images, builds deterministic views, and runs the model.
import tensorflow as tf

# Reuse the validated diagnostics, mini scoring, and submission assertions.
from predict_holdout import (
    evaluate_mini_holdout,
    load_mini_holdout_labels,
    prediction_arrays,
    print_holdout_edges,
    print_prediction_diagnostics,
    validate_predictions,
    validate_probabilities,
    validate_saved_submission,
    write_diagnostic_csv,
    write_submission,
)
# Import the exact fixed views that passed grouped validation.
from evaluate_grouped_tta import VIEWS
# Shared ordering and preprocessing keep validation/holdout behavior aligned.
from roadsign_pipeline import (
    NUM_CLASSES,
    collect_holdout_images,
    preprocess_resnet_image,
    resize_rgb_image,
)


def build_view_dataset(paths: list[Path], view, batch_size: int):
    """Build one deterministic original or mild 4% crop view for holdout paths."""

    def decode(path):
        # Native WSL paths allow TensorFlow to decode each JPEG directly in stable order.
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])

        if view is None:
            image = resize_rgb_image(image, image_size=(96, 96))
        else:
            image = resize_rgb_image(image, image_size=(100, 100))
            image = tf.image.crop_to_bounding_box(image, view[0], view[1], 96, 96)
            image.set_shape([96, 96, 3])
        return preprocess_resnet_image(image)

    # No shuffle is used, so all six probability matrices align row-for-row.
    path_strings = np.asarray([str(path) for path in paths], dtype=str)
    dataset = tf.data.Dataset.from_tensor_slices(path_strings)
    dataset = dataset.map(decode, num_parallel_calls=8)
    return dataset.batch(batch_size).prefetch(2)


def predict_tta_probabilities(model, paths: list[Path], batch_size: int) -> np.ndarray:
    """Average the same six fixed views selected on grouped validation."""
    probability_matrices = []
    for view_name, view in VIEWS.items():
        dataset = build_view_dataset(paths, view, batch_size)
        probabilities = np.asarray(model.predict(dataset, verbose=0), dtype=np.float32)
        probability_matrices.append(probabilities)
        print(f"Completed TTA view: {view_name}", flush=True)
    return np.mean(np.stack(probability_matrices, axis=0), axis=0)


def parse_args() -> argparse.Namespace:
    """Define selected model, mini-only mode, and isolated candidate outputs."""
    parser = argparse.ArgumentParser(description="Evaluate/export the selected roadsign TTA system.")
    parser.add_argument("--holdout-dir", default="data/holdout")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mini-labels", default="data/course_reference/mini_holdout_answers.csv")
    parser.add_argument("--mini-only", action="store_true")
    parser.add_argument("--skip-mini-evaluation", action="store_true")
    parser.add_argument(
        "--output",
        default="candidates/team3-module5-predictions-tta.csv",
    )
    parser.add_argument(
        "--diagnostic-output",
        default="diagnostics/final-tta-holdout-diagnostic.csv",
    )
    parser.add_argument(
        "--mini-confusion-output",
        default="diagnostics/final-tta-mini-confusion.csv",
    )
    parser.add_argument(
        "--mini-per-class-output",
        default="diagnostics/final-tta-mini-per-class.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run mini-only selection or full deterministic TTA submission export."""
    args = parse_args()

    # Full holdout collection proves numeric contiguous filename order before slicing.
    all_paths = collect_holdout_images(Path(args.holdout_dir).resolve())
    mini_labels_path = Path(args.mini_labels).resolve()
    if args.mini_only:
        mini_rows = load_mini_holdout_labels(mini_labels_path)
        paths = all_paths[: len(mini_rows)]
        assert [path.name for path in paths] == [filename for filename, _ in mini_rows]
    else:
        paths = all_paths

    print(f"Images selected: {len(paths)}")
    print_holdout_edges(paths)

    # The selected grouped checkpoint is a 96px, 43-class model.
    model = tf.keras.models.load_model(Path(args.model).resolve(), compile=False)
    assert tuple(model.input_shape[-3:]) == (96, 96, 3)
    assert int(model.output_shape[-1]) == NUM_CLASSES

    probabilities = predict_tta_probabilities(model, paths, args.batch_size)
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

    # The official 201 labels are evaluated once after validation-based selection.
    if not args.skip_mini_evaluation and mini_labels_path.exists():
        evaluate_mini_holdout(
            mini_labels_path,
            paths,
            predictions,
            confidence,
            Path(args.mini_confusion_output).resolve(),
            Path(args.mini_per_class_output).resolve(),
        )

    # Mini-only mode deliberately writes no 12,630-row submission candidate.
    if args.mini_only:
        print("Mini-only evaluation complete; no submission CSV was written.")
        return

    diagnostic_csv = Path(args.diagnostic_output).resolve()
    write_diagnostic_csv(
        paths,
        predictions,
        confidence,
        top_classes,
        top_probabilities,
        diagnostic_csv,
    )

    # Full export uses the same one-column assertions as every prior candidate.
    output_csv = Path(args.output).resolve()
    write_submission(predictions, output_csv, overwrite=args.overwrite)
    saved_values = validate_saved_submission(output_csv, expected_rows=len(paths))
    assert saved_values == predictions.astype(int).tolist()
    assert [path.name for path in paths] == [f"{index:05d}.jpg" for index in range(len(paths))]

    distribution = Counter(saved_values)
    print("Final TTA candidate validation:")
    print(f"  output: {output_csv}")
    print(f"  shape: ({len(saved_values)}, 1)")
    print(f"  first 10: {saved_values[:10]}")
    print(f"  last 10: {saved_values[-10:]}")
    print(f"  distribution: {dict(sorted(distribution.items()))}")
    print(f"  label range: {min(saved_values)}..{max(saved_values)}")
    print("  every assertion passed: YES")


# Direct execution evaluates or exports the selected TTA system.
if __name__ == "__main__":
    main()
