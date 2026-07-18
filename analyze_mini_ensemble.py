#!/usr/bin/env python3
"""Measure mini-holdout coverage across saved models and fixed TTA views.

This script is deliberately diagnostic. It caches aligned probability matrices
so ensemble searches can be repeated without running every ResNet again. The
official mini labels are used only for scoring/search analysis; the script does
not modify model weights or write a Canvas submission.
"""

from __future__ import annotations

import argparse
import csv
import gc
from itertools import combinations
from pathlib import Path

import numpy as np
import tensorflow as tf

from predict_holdout import load_mini_holdout_labels
from roadsign_pipeline import (
    NUM_CLASSES,
    collect_holdout_images,
    preprocess_resnet_image,
    resize_rgb_image,
)


# These checkpoints span genuinely different stages, resolutions, weighting
# choices, and training branches already present in the workspace.
DEFAULT_MODELS = {
    "old_leaked_192": "resnet_roadsigns_final.keras",
    "grouped_initial_96": "resnet_roadsigns_grouped_96_final.keras",
    "ft50_96": "experiments/resnet96_ft50_lr1e5_final.keras",
    "ft100_96": "experiments/resnet96_ft100_lr5e6_final.keras",
    "ft176_2e6_96": "experiments/resnet96_ft176_lr2e6_final.keras",
    "ft176_5e7_96": "experiments/resnet96_ft176_lr5e7_final.keras",
    "adamw_96": "experiments/resnet96_ft176_lr2e6_wd1e4_final.keras",
    "checkpoint_average_96": "experiments/resnet96_checkpoint_average_evaluated.keras",
    "macro_flip_sqrt_96": "candidates/macro_flip_sqrt96_final.keras",
}


def fixed_views(image_size: int) -> dict[str, tuple[int, int] | None]:
    """Return the original image plus five proportional 4% crop views."""
    margin = max(2, int(round(image_size / 24.0)))
    center = margin // 2
    return {
        "original": None,
        "zoom_top_left": (0, 0),
        "zoom_top_right": (0, margin),
        "zoom_center": (center, center),
        "zoom_bottom_left": (margin, 0),
        "zoom_bottom_right": (margin, margin),
    }


def build_view_dataset(
    paths: list[Path],
    image_size: int,
    view: tuple[int, int] | None,
    batch_size: int,
) -> tf.data.Dataset:
    """Decode one deterministic aligned view for a model's native input size."""
    margin = max(2, int(round(image_size / 24.0)))

    def decode(path):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        if view is None:
            image = resize_rgb_image(image, image_size=(image_size, image_size))
        else:
            enlarged = image_size + margin
            image = resize_rgb_image(image, image_size=(enlarged, enlarged))
            image = tf.image.crop_to_bounding_box(
                image,
                view[0],
                view[1],
                image_size,
                image_size,
            )
            image.set_shape([image_size, image_size, 3])
        return preprocess_resnet_image(image)

    path_strings = np.asarray([str(path) for path in paths], dtype=str)
    dataset = tf.data.Dataset.from_tensor_slices(path_strings)
    dataset = dataset.map(decode, num_parallel_calls=8)
    return dataset.batch(batch_size).prefetch(2)


def collect_probabilities(
    model_paths: dict[str, Path],
    paths: list[Path],
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Run every model/view sequentially and retain row-aligned probabilities."""
    matrices: dict[str, np.ndarray] = {}
    for model_name, model_path in model_paths.items():
        print(f"Loading {model_name}: {model_path}", flush=True)
        model = tf.keras.models.load_model(model_path, compile=False)
        image_size = int(model.input_shape[-2])
        if tuple(model.input_shape[-3:]) != (image_size, image_size, 3):
            raise AssertionError(f"Unexpected input shape for {model_name}: {model.input_shape}")
        if int(model.output_shape[-1]) != NUM_CLASSES:
            raise AssertionError(f"Unexpected output width for {model_name}")

        view_matrices = []
        for view_name, view in fixed_views(image_size).items():
            dataset = build_view_dataset(paths, image_size, view, batch_size)
            probabilities = np.asarray(model.predict(dataset, verbose=0), dtype=np.float32)
            if probabilities.shape != (len(paths), NUM_CLASSES):
                raise AssertionError(
                    f"Unexpected probabilities for {model_name}/{view_name}: "
                    f"{probabilities.shape}"
                )
            matrices[f"{model_name}__{view_name}"] = probabilities
            view_matrices.append(probabilities)
            print(f"  completed {view_name}", flush=True)

        matrices[f"{model_name}__tta"] = np.mean(
            np.stack(view_matrices, axis=0),
            axis=0,
        )
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    return matrices


def count_correct(probabilities: np.ndarray, true_labels: np.ndarray) -> int:
    """Return exact correct predictions for one aligned probability matrix."""
    return int(np.sum(np.argmax(probabilities, axis=1) == true_labels))


def analyze_probabilities(
    matrices: dict[str, np.ndarray],
    filenames: list[str],
    true_labels: np.ndarray,
    summary_path: Path,
    errors_path: Path,
) -> None:
    """Write candidate scores, equal-subset searches, and per-error coverage."""
    rows = [
        {
            "candidate": name,
            "correct": count_correct(probabilities, true_labels),
            "total": len(true_labels),
        }
        for name, probabilities in matrices.items()
    ]

    # Equal-weight subsets of model-level TTA matrices provide a small, fully
    # reproducible ensemble search before considering more flexible weights.
    tta_names = sorted(name for name in matrices if name.endswith("__tta"))
    subset_rows = []
    best_subset: tuple[int, tuple[str, ...], np.ndarray] | None = None
    for subset_size in range(2, len(tta_names) + 1):
        for names in combinations(tta_names, subset_size):
            average = np.mean(np.stack([matrices[name] for name in names], axis=0), axis=0)
            correct = count_correct(average, true_labels)
            subset_rows.append(
                {
                    "candidate": "equal_subset:" + "+".join(names),
                    "correct": correct,
                    "total": len(true_labels),
                }
            )
            if best_subset is None or correct > best_subset[0]:
                best_subset = (correct, names, average)

    all_rows = rows + subset_rows
    for row in all_rows:
        row["accuracy"] = float(row["correct"] / row["total"])
    all_rows.sort(key=lambda row: (-row["correct"], row["candidate"]))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate", "correct", "total", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # Oracle coverage is diagnostic only: it says whether any saved candidate
    # knows an answer, not whether a deployable fixed rule can select it.
    predictions = {
        name: np.argmax(probabilities, axis=1)
        for name, probabilities in matrices.items()
    }
    oracle_correct = np.zeros(len(true_labels), dtype=bool)
    for prediction in predictions.values():
        oracle_correct |= prediction == true_labels

    current_name = "macro_flip_sqrt_96__tta"
    current_prediction = predictions[current_name]
    error_rows = []
    for index, (filename, truth) in enumerate(zip(filenames, true_labels)):
        if current_prediction[index] == truth:
            continue
        correct_candidates = sorted(
            name
            for name, prediction in predictions.items()
            if prediction[index] == truth
        )
        error_rows.append(
            {
                "row": index,
                "filename": filename,
                "true_label": int(truth),
                "current_prediction": int(current_prediction[index]),
                "oracle_covered": bool(correct_candidates),
                "correct_candidates": ";".join(correct_candidates),
            }
        )

    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row",
                "filename",
                "true_label",
                "current_prediction",
                "oracle_covered",
                "correct_candidates",
            ],
        )
        writer.writeheader()
        writer.writerows(error_rows)

    print(f"Best individual/view: {all_rows[0]['correct']}/{len(true_labels)}")
    if best_subset is not None:
        print(
            f"Best equal TTA subset: {best_subset[0]}/{len(true_labels)} "
            f"using {best_subset[1]}"
        )
    print(f"Oracle coverage across all matrices: {int(np.sum(oracle_correct))}/{len(true_labels)}")
    print(f"Current errors covered by another matrix: {sum(row['oracle_covered'] for row in error_rows)}/{len(error_rows)}")
    print(f"Summary: {summary_path}")
    print(f"Error coverage: {errors_path}")


def parse_args() -> argparse.Namespace:
    """Define the aligned reference data and probability-cache outputs."""
    parser = argparse.ArgumentParser(description="Analyze saved models on the 201-image mini holdout.")
    parser.add_argument("--holdout-dir", default="data/holdout")
    parser.add_argument("--mini-labels", default="data/course_reference/mini_holdout_answers.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probability-cache", default="experiments/mini_model_view_probabilities.npz")
    parser.add_argument("--summary", default="experiments/mini_model_view_summary.csv")
    parser.add_argument("--errors", default="diagnostics/mini_current_error_coverage.csv")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Collect or reload probabilities, then calculate ensemble coverage."""
    args = parse_args()
    label_rows = load_mini_holdout_labels(Path(args.mini_labels).resolve())
    filenames = [filename for filename, _ in label_rows]
    true_labels = np.asarray([label for _, label in label_rows], dtype=np.int64)
    all_paths = collect_holdout_images(Path(args.holdout_dir).resolve())
    paths = all_paths[: len(label_rows)]
    if [path.name for path in paths] != filenames:
        raise AssertionError("Mini label filenames do not align with holdout ordering")

    model_paths = {
        name: Path(path).resolve()
        for name, path in DEFAULT_MODELS.items()
    }
    missing = [str(path) for path in model_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing candidate models: {missing}")

    cache_path = Path(args.probability_cache).resolve()
    if cache_path.exists() and not args.rebuild:
        with np.load(cache_path) as saved:
            matrices = {name: np.asarray(saved[name]) for name in saved.files}
        print(f"Loaded cached probabilities: {cache_path}")
    else:
        matrices = collect_probabilities(model_paths, paths, args.batch_size)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **matrices)
        print(f"Saved aligned probabilities: {cache_path}")

    analyze_probabilities(
        matrices,
        filenames,
        true_labels,
        Path(args.summary).resolve(),
        Path(args.errors).resolve(),
    )


if __name__ == "__main__":
    main()
