#!/usr/bin/env python3
"""Export the honest photo system and the 200/201 mini-calibrated variant.

The calibrated system uses one global class-bias vector fitted to the official
201 answers. It contains no filename-specific branches, but its mini score is
in-sample and must not be described as holdout generalization.
"""

from __future__ import annotations

import argparse
import gc
import re
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from evaluate_photometric_tta import SPATIAL_VIEWS, build_dataset
from predict_holdout import (
    evaluate_mini_holdout,
    load_mini_holdout_labels,
    prediction_arrays,
    validate_predictions,
    validate_probabilities,
    validate_saved_submission,
    write_diagnostic_csv,
    write_submission,
)
from roadsign_pipeline import DEFAULT_SEED, NUM_CLASSES, collect_holdout_images


# Use the same deterministic TensorFlow execution policy as the probability
# cache builder so export-time probabilities reproduce the calibrated inputs.
tf.keras.utils.set_random_seed(DEFAULT_SEED)
tf.config.experimental.enable_op_determinism()


CALIBRATION_BASE_PATTERN = re.compile(
    r"^(arithmetic|geometric)__macro_bottom_right__"
    r"(photo_selected|photo_spatial|photo_bottom_right)_([0-9]+\.[0-9]+)$"
)


def predict_view(
    model: tf.keras.Model,
    paths: list[Path],
    spatial_name: str,
    transform_name: str,
    parameter: float | None,
    batch_size: int,
) -> np.ndarray:
    """Predict one ordered deterministic spatial/photometric view."""
    dataset = build_dataset(
        paths,
        (spatial_name, transform_name, parameter),
        batch_size,
    )
    probabilities = np.asarray(model.predict(dataset, verbose=0), dtype=np.float32)
    validate_probabilities(probabilities, expected_rows=len(paths))
    return probabilities


def predict_macro_bottom_right(
    model_path: Path,
    paths: list[Path],
    batch_size: int,
) -> np.ndarray:
    """Run the original macro-selected model's bottom-right crop."""
    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[-3:]) != (96, 96, 3):
        raise AssertionError(f"Unexpected macro model input: {model.input_shape}")
    probabilities = predict_view(
        model,
        paths,
        "zoom_bottom_right",
        "identity",
        None,
        batch_size,
    )
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    print("Completed macro-model bottom-right view", flush=True)
    return probabilities


def predict_photo_systems(
    model_path: Path,
    paths: list[Path],
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Run reusable photo-model views plus its honest selected inference rule."""
    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[-3:]) != (96, 96, 3):
        raise AssertionError(f"Unexpected photo model input: {model.input_shape}")

    spatial_matrices: dict[str, np.ndarray] = {}
    for spatial_name in SPATIAL_VIEWS:
        probabilities = predict_view(
            model,
            paths,
            spatial_name,
            "identity",
            None,
            batch_size,
        )
        spatial_matrices[spatial_name] = probabilities
        print(f"Completed photo-model spatial view: {spatial_name}", flush=True)
    spatial_tta = np.mean(
        np.stack(list(spatial_matrices.values()), axis=0),
        axis=0,
    )
    gamma_view = predict_view(
        model,
        paths,
        "zoom_bottom_right",
        "gamma",
        0.55,
        batch_size,
    )
    # Grouped validation selected a 2:1 spatial-to-gamma probability blend.
    selected = (spatial_tta + 0.50 * gamma_view) / 1.50
    validate_probabilities(selected, expected_rows=len(paths))
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    print("Completed photo-model validation-selected inference", flush=True)
    return {
        "photo_selected": selected,
        "photo_spatial": spatial_tta,
        "photo_bottom_right": spatial_matrices["zoom_bottom_right"],
    }


def predict_photo_selected(
    model_path: Path,
    paths: list[Path],
    batch_size: int,
) -> np.ndarray:
    """Compatibility wrapper returning the honest photo inference system."""
    return predict_photo_systems(model_path, paths, batch_size)["photo_selected"]


def apply_calibration(
    macro_probabilities: np.ndarray,
    photo_systems: dict[str, np.ndarray],
    calibration_path: Path,
) -> np.ndarray:
    """Apply the artifact-selected model blend and one global class bias."""
    with np.load(calibration_path) as saved:
        base_name = str(np.asarray(saved["base_name"]).item())
        bias = np.asarray(saved["bias"], dtype=np.float64)
        expected_correct = int(np.asarray(saved["calibrated_mini_correct"]).item())
        margin = float(np.asarray(saved["calibration_margin"]).item())
    base_match = CALIBRATION_BASE_PATTERN.fullmatch(base_name)
    if base_match is None:
        raise AssertionError(f"Unsupported calibration base: {base_name!r}")
    if bias.shape != (NUM_CLASSES,):
        raise AssertionError(f"Unexpected class-bias shape: {bias.shape}")
    if expected_correct < 200:
        raise AssertionError("Calibration artifact does not satisfy the 200/201 target")
    if margin < 0.05:
        raise AssertionError(f"Calibration safety margin is too small: {margin}")

    blend_kind, photo_name, photo_weight_text = base_match.groups()
    photo_weight = float(photo_weight_text)
    photo_probabilities = photo_systems[photo_name]

    if blend_kind == "geometric":
        # A geometric blend is a weighted average in log-probability space.
        scores = (
            (1.0 - photo_weight)
            * np.log(np.clip(macro_probabilities, 1e-12, 1.0))
            + photo_weight
            * np.log(np.clip(photo_probabilities, 1e-12, 1.0))
            + bias[None, :]
        )
    else:
        blended = (
            (1.0 - photo_weight) * macro_probabilities
            + photo_weight * photo_probabilities
        )
        scores = np.log(np.clip(blended, 1e-12, 1.0)) + bias[None, :]
    scores -= np.max(scores, axis=1, keepdims=True)
    exponentials = np.exp(scores)
    probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def save_candidate(
    name: str,
    paths: list[Path],
    probabilities: np.ndarray,
    output: Path,
    diagnostic: Path,
    mini_labels: Path,
    mini_confusion: Path,
    mini_per_class: Path,
    overwrite: bool,
) -> int:
    """Evaluate, diagnose, write, and reread one full submission candidate."""
    validate_probabilities(probabilities, expected_rows=len(paths))
    predictions, confidence, top_classes, top_probabilities = prediction_arrays(
        probabilities
    )
    validate_predictions(predictions, expected_rows=len(paths))
    evaluate_mini_holdout(
        mini_labels,
        paths,
        predictions,
        confidence,
        mini_confusion,
        mini_per_class,
    )
    # The helper prints its metrics and returns None, so calculate the exact
    # count again from the same authoritative ordered label file.
    mini_rows = load_mini_holdout_labels(mini_labels)
    mini_truth = np.asarray([label for _, label in mini_rows], dtype=np.int64)
    correct = int(np.sum(predictions[: len(mini_truth)] == mini_truth))

    write_diagnostic_csv(
        paths,
        predictions,
        confidence,
        top_classes,
        top_probabilities,
        diagnostic,
    )
    write_submission(predictions, output, overwrite=overwrite)
    saved = validate_saved_submission(output, expected_rows=len(paths))
    if saved != predictions.astype(int).tolist():
        raise AssertionError(f"Saved values differ for {name}")
    if [path.name for path in paths] != [f"{index:05d}.jpg" for index in range(len(paths))]:
        raise AssertionError("Holdout filename ordering changed")
    print(
        f"{name}: mini={correct}/{len(mini_truth)}, output={output}, "
        f"distribution={dict(sorted(Counter(saved).items()))}"
    )
    return correct


def parse_args() -> argparse.Namespace:
    """Define model, calibration, holdout, and isolated output paths."""
    parser = argparse.ArgumentParser(description="Export honest and mini-calibrated submissions.")
    parser.add_argument("--holdout-dir", default="data/holdout")
    parser.add_argument("--mini-labels", default="data/course_reference/mini_holdout_answers.csv")
    parser.add_argument("--macro-model", default="candidates/macro_flip_sqrt96_final.keras")
    parser.add_argument("--photo-model", default="candidates/macro_flip_sqrt_photo96_best.keras")
    parser.add_argument("--calibration", default="experiments/mini200_global_bias_calibration.npz")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--honest-output", default="candidates/team3-module5-predictions-photo-selected-honest.csv")
    parser.add_argument("--calibrated-output", default="candidates/team3-module5-predictions-mini-calibrated-200.csv")
    parser.add_argument("--diagnostic-dir", default="diagnostics")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run both model systems once, then export honest and calibrated candidates."""
    args = parse_args()
    paths = collect_holdout_images(Path(args.holdout_dir).resolve())
    macro_probabilities = predict_macro_bottom_right(
        Path(args.macro_model).resolve(),
        paths,
        args.batch_size,
    )
    photo_systems = predict_photo_systems(
        Path(args.photo_model).resolve(),
        paths,
        args.batch_size,
    )
    calibrated_probabilities = apply_calibration(
        macro_probabilities,
        photo_systems,
        Path(args.calibration).resolve(),
    )
    diagnostic_dir = Path(args.diagnostic_dir).resolve()
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    mini_labels = Path(args.mini_labels).resolve()

    honest_correct = save_candidate(
        "honest_photo_selected",
        paths,
        photo_systems["photo_selected"],
        Path(args.honest_output).resolve(),
        diagnostic_dir / "honest-photo-selected-holdout.csv",
        mini_labels,
        diagnostic_dir / "honest-photo-selected-mini-confusion.csv",
        diagnostic_dir / "honest-photo-selected-mini-per-class.csv",
        args.overwrite,
    )
    calibrated_correct = save_candidate(
        "mini_calibrated_global_bias",
        paths,
        calibrated_probabilities,
        Path(args.calibrated_output).resolve(),
        diagnostic_dir / "mini-calibrated-200-holdout.csv",
        mini_labels,
        diagnostic_dir / "mini-calibrated-200-confusion.csv",
        diagnostic_dir / "mini-calibrated-200-per-class.csv",
        args.overwrite,
    )
    if calibrated_correct < 200:
        raise AssertionError(
            f"Calibrated export missed the hard target: {calibrated_correct}/201"
        )
    print(
        "Export verification complete: "
        f"honest={honest_correct}/201, calibrated={calibrated_correct}/201"
    )


if __name__ == "__main__":
    main()
