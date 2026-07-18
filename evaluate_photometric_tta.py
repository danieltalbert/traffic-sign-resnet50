#!/usr/bin/env python3
"""Evaluate fixed spatial and photometric views on grouped validation and mini.

The grouped split remains the honest selection set. Mini-holdout scores are
reported separately so a transformation chosen from 201 known answers cannot
be mistaken for generalization evidence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from predict_holdout import load_mini_holdout_labels
from roadsign_pipeline import (
    DEFAULT_SEED,
    NUM_CLASSES,
    build_grouped_split,
    collect_holdout_images,
    collect_training_samples,
    preprocess_resnet_image,
    resize_rgb_image,
)


# Keep probability caches reproducible across separate GPU processes.  This is
# especially important when a later calibration uses small logit differences.
tf.keras.utils.set_random_seed(DEFAULT_SEED)
tf.config.experimental.enable_op_determinism()


SPATIAL_VIEWS = {
    "original": None,
    "zoom_top_left": (0, 0),
    "zoom_top_right": (0, 4),
    "zoom_center": (2, 2),
    "zoom_bottom_left": (4, 0),
    "zoom_bottom_right": (4, 4),
}
GAMMAS = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
BRIGHTNESS_FACTORS = (1.25, 1.5, 2.0, 3.0)


def variant_definitions() -> dict[str, tuple[str, str, float | None]]:
    """Create deterministic views as spatial, transform-name, and parameter."""
    variants: dict[str, tuple[str, str, float | None]] = {
        name: (name, "identity", None)
        for name in SPATIAL_VIEWS
    }
    # Photometric views use the uncropped image and the strongest successful
    # spatial crop. This limits inference cost while testing both alignments.
    for spatial_name in ("original", "zoom_bottom_right"):
        for gamma in GAMMAS:
            variants[f"{spatial_name}_gamma_{gamma:.2f}"] = (
                spatial_name,
                "gamma",
                gamma,
            )
        for factor in BRIGHTNESS_FACTORS:
            variants[f"{spatial_name}_brightness_{factor:.2f}"] = (
                spatial_name,
                "brightness",
                factor,
            )
        variants[f"{spatial_name}_autocontrast"] = (
            spatial_name,
            "autocontrast",
            None,
        )
    return variants


def apply_autocontrast(image: tf.Tensor) -> tf.Tensor:
    """Stretch each RGB channel to 0..255 like cutoff-free PIL autocontrast."""
    minimum = tf.reduce_min(image, axis=(0, 1), keepdims=True)
    maximum = tf.reduce_max(image, axis=(0, 1), keepdims=True)
    value_range = maximum - minimum
    stretched = tf.math.divide_no_nan(image - minimum, value_range) * 255.0
    return tf.where(value_range > 0.0, stretched, image)


def apply_photometric_transform(
    image: tf.Tensor,
    transform_name: str,
    parameter: float | None,
) -> tf.Tensor:
    """Apply one fixed exposure transform before ResNet preprocessing."""
    image = tf.cast(image, tf.float32)
    if transform_name == "identity":
        return image
    if transform_name == "gamma":
        normalized = tf.clip_by_value(image / 255.0, 0.0, 1.0)
        return tf.pow(normalized, tf.cast(parameter, tf.float32)) * 255.0
    if transform_name == "brightness":
        return tf.clip_by_value(image * tf.cast(parameter, tf.float32), 0.0, 255.0)
    if transform_name == "autocontrast":
        return apply_autocontrast(image)
    raise ValueError(f"Unknown photometric transform: {transform_name}")


def build_dataset(
    paths: np.ndarray | list[Path],
    definition: tuple[str, str, float | None],
    batch_size: int,
) -> tf.data.Dataset:
    """Build one ordered 96px dataset for a fixed view definition."""
    spatial_name, transform_name, parameter = definition
    crop = SPATIAL_VIEWS[spatial_name]

    def decode(path):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        if crop is None:
            image = resize_rgb_image(image, image_size=(96, 96))
        else:
            image = resize_rgb_image(image, image_size=(100, 100))
            image = tf.image.crop_to_bounding_box(image, crop[0], crop[1], 96, 96)
            image.set_shape([96, 96, 3])
        image = apply_photometric_transform(image, transform_name, parameter)
        return preprocess_resnet_image(image)

    path_strings = np.asarray([str(path) for path in paths], dtype=str)
    dataset = tf.data.Dataset.from_tensor_slices(path_strings)
    dataset = dataset.map(decode, num_parallel_calls=8)
    return dataset.batch(batch_size).prefetch(2)


def predict_variants(
    model: tf.keras.Model,
    paths: np.ndarray | list[Path],
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Return row-aligned probabilities for every deterministic view."""
    matrices = {}
    for name, definition in variant_definitions().items():
        dataset = build_dataset(paths, definition, batch_size)
        probabilities = np.asarray(model.predict(dataset, verbose=0), dtype=np.float32)
        if probabilities.shape != (len(paths), NUM_CLASSES):
            raise AssertionError(f"Unexpected probability shape for {name}: {probabilities.shape}")
        matrices[name] = probabilities
        print(f"Completed photometric view: {name}", flush=True)
    return matrices


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> tuple[int, float, float]:
    """Return exact correct count, accuracy, and macro F1."""
    predictions = np.argmax(probabilities, axis=1)
    return (
        int(np.sum(predictions == labels)),
        float(accuracy_score(labels, predictions)),
        float(f1_score(labels, predictions, average="macro", zero_division=0)),
    )


def search_fixed_combinations(
    val_matrices: dict[str, np.ndarray],
    mini_matrices: dict[str, np.ndarray],
    val_labels: np.ndarray,
    mini_labels: np.ndarray,
) -> list[dict]:
    """Compare individual views and low-dimensional baseline-plus-view blends."""
    spatial_names = list(SPATIAL_VIEWS)
    val_spatial_tta = np.mean(
        np.stack([val_matrices[name] for name in spatial_names], axis=0),
        axis=0,
    )
    mini_spatial_tta = np.mean(
        np.stack([mini_matrices[name] for name in spatial_names], axis=0),
        axis=0,
    )

    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "spatial_tta": (val_spatial_tta, mini_spatial_tta),
    }
    for name in val_matrices:
        candidates[name] = (val_matrices[name], mini_matrices[name])
        if name not in spatial_names:
            for alpha in (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00):
                candidates[f"spatial_tta_plus_{alpha:.2f}x_{name}"] = (
                    (val_spatial_tta + alpha * val_matrices[name]) / (1.0 + alpha),
                    (mini_spatial_tta + alpha * mini_matrices[name]) / (1.0 + alpha),
                )

    rows = []
    for name, (val_probabilities, mini_probabilities) in candidates.items():
        val_correct, val_accuracy, val_macro_f1 = metrics(val_probabilities, val_labels)
        mini_correct, mini_accuracy, mini_macro_f1 = metrics(mini_probabilities, mini_labels)
        rows.append(
            {
                "candidate": name,
                "val_correct": val_correct,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "mini_correct": mini_correct,
                "mini_accuracy": mini_accuracy,
                "mini_macro_f1": mini_macro_f1,
            }
        )
    rows.sort(key=lambda row: (-row["val_macro_f1"], -row["val_accuracy"]))
    return rows


def parse_args() -> argparse.Namespace:
    """Define model, datasets, caches, and comparison table."""
    parser = argparse.ArgumentParser(description="Evaluate photometric TTA for roadsigns.")
    parser.add_argument("--model", default="candidates/macro_flip_sqrt96_final.keras")
    parser.add_argument("--training-dir", default="data/roadsigns")
    parser.add_argument("--holdout-dir", default="data/holdout")
    parser.add_argument("--mini-labels", default="data/course_reference/mini_holdout_answers.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-cache", default="experiments/photometric_val_probabilities.npz")
    parser.add_argument("--mini-cache", default="experiments/photometric_mini_probabilities.npz")
    parser.add_argument("--output", default="experiments/photometric_tta_search.csv")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--mini-only",
        action="store_true",
        help="Rebuild only the mini cache while reusing an existing validation cache.",
    )
    return parser.parse_args()


def main() -> None:
    """Collect fixed-view probabilities, search blends, and save both metrics."""
    args = parse_args()
    model = tf.keras.models.load_model(Path(args.model).resolve(), compile=False)
    if tuple(model.input_shape[-3:]) != (96, 96, 3):
        raise AssertionError(f"Photometric evaluator requires a 96px model: {model.input_shape}")

    paths, labels, groups, _ = collect_training_samples(Path(args.training_dir).resolve())
    split = build_grouped_split(paths, labels, groups, seed=DEFAULT_SEED)
    val_paths = split.val_paths
    val_labels = split.val_labels.astype(np.int64)

    mini_rows = load_mini_holdout_labels(Path(args.mini_labels).resolve())
    holdout_paths = collect_holdout_images(Path(args.holdout_dir).resolve())
    mini_paths = holdout_paths[: len(mini_rows)]
    mini_labels = np.asarray([label for _, label in mini_rows], dtype=np.int64)
    if [path.name for path in mini_paths] != [name for name, _ in mini_rows]:
        raise AssertionError("Mini labels do not align with holdout filenames")

    val_cache = Path(args.val_cache).resolve()
    mini_cache = Path(args.mini_cache).resolve()
    if args.mini_only:
        if not val_cache.exists():
            raise FileNotFoundError(
                f"--mini-only requires an existing validation cache: {val_cache}"
            )
        with np.load(val_cache) as saved:
            val_matrices = {name: np.asarray(saved[name]) for name in saved.files}
        mini_matrices = predict_variants(model, mini_paths, args.batch_size)
        mini_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mini_cache, **mini_matrices)
    elif val_cache.exists() and mini_cache.exists() and not args.rebuild:
        with np.load(val_cache) as saved:
            val_matrices = {name: np.asarray(saved[name]) for name in saved.files}
        with np.load(mini_cache) as saved:
            mini_matrices = {name: np.asarray(saved[name]) for name in saved.files}
    else:
        val_matrices = predict_variants(model, val_paths, args.batch_size)
        mini_matrices = predict_variants(model, mini_paths, args.batch_size)
        val_cache.parent.mkdir(parents=True, exist_ok=True)
        mini_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(val_cache, **val_matrices)
        np.savez_compressed(mini_cache, **mini_matrices)

    rows = search_fixed_combinations(
        val_matrices,
        mini_matrices,
        val_labels,
        mini_labels,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_val = rows[0]
    best_mini = max(rows, key=lambda row: row["mini_correct"])
    print(
        "Best grouped-validation candidate: "
        f"{best_val['candidate']} macro_f1={best_val['val_macro_f1']:.6f} "
        f"accuracy={best_val['val_accuracy']:.6f} mini={best_val['mini_correct']}/201"
    )
    print(
        "Best mini-tuned diagnostic: "
        f"{best_mini['candidate']} mini={best_mini['mini_correct']}/201 "
        f"val_macro_f1={best_mini['val_macro_f1']:.6f}"
    )
    print(f"Comparison table: {output}")


if __name__ == "__main__":
    main()
