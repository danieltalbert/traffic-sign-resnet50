#!/usr/bin/env python3
"""Audit data leakage, preprocessing, class mapping, and holdout ordering.

The script is intentionally read-only except for the explicit split manifest.
It recreates the original image-level split to quantify leakage, builds the
corrected group-level split, and checks the shared training/inference contract.
"""

from __future__ import annotations

# argparse makes every audited path visible and reproducible.
import argparse
# csv reads the official filename-aligned 201-image answer file.
import csv
# Path provides consistent filesystem handling.
from pathlib import Path

# NumPy powers masks and concise numeric assertions.
import numpy as np
# sklearn recreates the exact random image-level split used by the original code.
from sklearn.model_selection import train_test_split

# Shared functions are the same ones used by training and full holdout prediction.
from roadsign_pipeline import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEED,
    NUM_CLASSES,
    assert_preprocessing_contract,
    build_grouped_split,
    class_index_mapping,
    collect_holdout_images,
    collect_training_samples,
    load_and_preprocess_pillow,
    preprocess_resnet_image,
    preprocessing_summary,
    print_split_statistics,
    resize_rgb_image,
    write_split_manifest,
)


def quantify_original_split_leakage(
    paths: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> None:
    """Recreate the former split and report physical-sign overlap exactly."""

    # Splitting indices preserves access to labels and groups for the same samples.
    all_indices = np.arange(len(paths))
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )

    # The original code ignored groups, so nearly every sign contributes frames to both sets.
    train_groups = set(groups[train_indices].tolist())
    val_groups = set(groups[val_indices].tolist())
    overlap = train_groups & val_groups
    all_groups = set(groups.tolist())

    print("Original random image-level split leakage:")
    print(f"  total images: {len(paths)}")
    print(f"  total physical-sign groups: {len(all_groups)}")
    print(f"  train images: {len(train_indices)}")
    print(f"  validation images: {len(val_indices)}")
    print(f"  train groups: {len(train_groups)}")
    print(f"  validation groups: {len(val_groups)}")
    print(f"  groups present in both: {len(overlap)}")
    print(f"  overlap percentage: {100.0 * len(overlap) / len(all_groups):.2f}%")
    print(
        "  validation frames whose physical sign also appears in training: "
        f"{sum(group in overlap for group in groups[val_indices])}/{len(val_indices)}"
    )
    print(f"  example leaked groups: {sorted(overlap)[:10]}")
    print("  verdict: the previous random validation score is not a reliable generalization estimate")


def inspect_raw_image(path: Path) -> dict[str, object]:
    """Read raw width, height, mode, and channel count without altering pixels."""

    # Pillow exposes source metadata clearly for both training and holdout JPEGs.
    from PIL import Image

    with Image.open(path) as image_file:
        width, height = image_file.size
        mode = image_file.mode
        channels = len(image_file.getbands())
    return {
        "filename": path.name,
        "width": width,
        "height": height,
        "mode": mode,
        "channels": channels,
    }


def audit_preprocessing(
    training_path: Path,
    holdout_path: Path,
    image_size: tuple[int, int],
) -> None:
    """Compare representative validation-style and holdout inference tensors."""

    # TensorFlow decoding mirrors the training/validation tf.data path.
    import tensorflow as tf

    # channels=3 guarantees RGB even if source image metadata differs.
    training_bytes = tf.io.read_file(str(training_path))
    training_rgb = tf.io.decode_image(
        training_bytes,
        channels=3,
        expand_animations=False,
    )
    training_rgb.set_shape([None, None, 3])

    # Validation receives resize then exactly one ResNet50 preprocess call.
    training_processed = preprocess_resnet_image(
        resize_rgb_image(training_rgb, image_size=image_size)
    ).numpy()

    # Holdout uses Pillow only for RGB decode, then the same shared TensorFlow transforms.
    holdout_processed = load_and_preprocess_pillow(
        holdout_path,
        image_size=image_size,
    )

    # Shape, float32 dtype, finite values, and expected ResNet range must all match.
    assert_preprocessing_contract(
        training_processed,
        holdout_processed,
        image_size=image_size,
    )

    print("Preprocessing audit:")
    print(f"  representative training source: {inspect_raw_image(training_path)}")
    print(f"  representative holdout source: {inspect_raw_image(holdout_path)}")
    print(f"  processed training tensor: {preprocessing_summary(training_processed)}")
    print(f"  processed holdout tensor: {preprocessing_summary(holdout_processed)}")
    print("  channel handling: source RGB -> ResNet50 preprocess converts internally to BGR")
    print(f"  geometry: bilinear resize to {image_size}, with no crop, pad, or aspect-ratio preservation")
    print("  scaling: no 1/255 normalization and no duplicate preprocessing")
    print("  augmentation: training-only; validation and holdout use no random transforms")
    print("  preprocessing contract assertions: PASSED")


def audit_mini_holdout_alignment(label_csv: Path, holdout_paths: list[Path]) -> None:
    """Prove the official 201 answers match the same prefix of the full holdout."""

    # The answer CSV names every row, allowing exact order verification rather than assumption.
    with label_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Filename", "ClassId"]:
            raise AssertionError(f"Unexpected mini label columns: {reader.fieldnames}")
        rows = list(reader)

    label_names = [row["Filename"] for row in rows]
    holdout_prefix = [path.name for path in holdout_paths[: len(rows)]]
    assert label_names == holdout_prefix
    assert all(0 <= int(row["ClassId"]) < NUM_CLASSES for row in rows)

    print("Official mini-holdout alignment:")
    print(f"  labeled rows: {len(rows)}")
    print(f"  first pair: {rows[0]}")
    print(f"  last pair: {rows[-1]}")
    print("  labels align with the first sorted full-holdout filenames: PASSED")


def audit_model(model_path: Path, image_size: tuple[int, int]) -> None:
    """Verify saved model input shape, output width, and identity class mapping."""

    # compile=False avoids loading training-only optimizer state during a structural audit.
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)

    # The saved model must accept the shared RGB tensor and emit one score per actual class.
    assert input_shape[-3:] == (image_size[0], image_size[1], 3)
    mapping = class_index_mapping(int(output_shape[-1]))

    print("Saved model/class mapping audit:")
    print(f"  input shape: {input_shape}")
    print(f"  output shape: {output_shape}")
    print(f"  complete index -> label mapping: {mapping}")
    print("  mapping assertions: PASSED")


def parse_args() -> argparse.Namespace:
    """Define audit inputs and the one intentionally written manifest output."""

    parser = argparse.ArgumentParser(description="Audit the Module 05 roadsign pipeline.")
    parser.add_argument("--training-dir", default="data/roadsigns")
    parser.add_argument("--holdout-dir", default="data/holdout")
    parser.add_argument(
        "--mini-labels",
        default="data/course_reference/mini_holdout_answers.csv",
    )
    parser.add_argument("--model", default="best_resnet_roadsigns.keras")
    parser.add_argument("--manifest", default="diagnostics/group_split_manifest.csv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip loading the Keras model while still auditing all data rules.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all leakage, ordering, preprocessing, label, and model checks."""

    args = parse_args()

    # Absolute paths make the audit report independent of the caller's working directory.
    training_dir = Path(args.training_dir).resolve()
    holdout_dir = Path(args.holdout_dir).resolve()
    mini_labels = Path(args.mini_labels).resolve()
    model_path = Path(args.model).resolve()
    manifest_path = Path(args.manifest).resolve()
    image_size = (args.image_size, args.image_size)

    # Training collection parses actual folder labels and filename-derived group IDs.
    paths, labels, groups, found_classes = collect_training_samples(training_dir)
    print(f"Training classes found: {sorted(found_classes)}")
    print(f"Training label range: {int(labels.min())}..{int(labels.max())}")
    assert sorted(found_classes) == list(range(NUM_CLASSES))

    # First quantify the defect, then produce the corrected reusable split.
    quantify_original_split_leakage(paths, labels, groups, seed=args.seed)
    grouped_split = build_grouped_split(paths, labels, groups, seed=args.seed)
    print_split_statistics(grouped_split)
    write_split_manifest(grouped_split, manifest_path)
    print(f"Grouped split manifest: {manifest_path}")

    # Holdout collection uses numeric order and asserts contiguous IDs 00000..12629.
    holdout_paths = collect_holdout_images(holdout_dir)
    print(f"Holdout image count: {len(holdout_paths)}")
    print(f"First 20 holdout filenames: {[path.name for path in holdout_paths[:20]]}")
    print(f"Last 20 holdout filenames: {[path.name for path in holdout_paths[-20:]]}")
    print("Holdout numeric ordering assertions: PASSED")

    # Class mapping is explicit even before loading a model.
    print(f"Expected class index -> label mapping: {class_index_mapping()}")

    # The official labels are optional, but present labels must align exactly by filename.
    if mini_labels.exists():
        audit_mini_holdout_alignment(mini_labels, holdout_paths)
    else:
        print(f"Mini-holdout labels are missing: {mini_labels}")

    # Representative tensors verify the same model interface for validation and prediction.
    audit_preprocessing(
        Path(grouped_split.train_paths[0]),
        holdout_paths[0],
        image_size=image_size,
    )

    # Model loading is separable so split audits can still run in a lightweight environment.
    if not args.skip_model:
        audit_model(model_path, image_size=image_size)

    print("Road-sign pipeline audit completed successfully.")


# The audit runs only when invoked directly.
if __name__ == "__main__":
    main()
