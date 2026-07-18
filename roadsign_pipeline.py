#!/usr/bin/env python3
"""Shared, testable data rules for the Module 05 road-sign project.

Training, validation, and holdout inference import this module so that class
IDs, image size, file ordering, resizing, and ResNet50 preprocessing cannot
quietly drift apart between scripts.
"""

from __future__ import annotations

# csv is used for the reproducible train/validation split manifest.
import csv
# Counter makes the printed class distributions compact and deterministic.
from collections import Counter, defaultdict
# dataclass gives the grouped split a named, self-documenting return value.
from dataclasses import dataclass
# Path keeps file handling consistent across WSL, Linux, and Windows paths.
from pathlib import Path
# Sequence documents inputs that can be lists or NumPy arrays.
from typing import Sequence

# NumPy handles labels, masks, deterministic shuffling, and validation checks.
import numpy as np


# The BYU-I dataset uses the 43 German traffic-sign labels 0 through 42.
NUM_CLASSES = 43
# All scripts use the same square input shape unless a training CLI override is supplied.
DEFAULT_IMAGE_SIZE = (192, 192)
# A fixed seed makes the grouped split reproducible across every experiment.
DEFAULT_SEED = 123
# Twenty percent of physical sign groups are reserved for validation per class.
DEFAULT_VALIDATION_FRACTION = 0.20
# Only these suffixes are interpreted as road-sign images.
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".ppm"})

# Horizontal mirroring changes the meaning of directional signs.  These are
# the only class pairs in this dataset for which a mirror is still a valid
# training example after the label is swapped at the same time.  Every other
# class remains ineligible for horizontal-flip augmentation.
HORIZONTAL_FLIP_CLASS_PAIRS = (
    (19, 20),  # dangerous curve left <-> dangerous curve right
    (33, 34),  # turn right <-> turn left
    (36, 37),  # straight or right <-> straight or left
    (38, 39),  # keep/pass right <-> keep/pass left
)


def horizontal_flip_label_map() -> np.ndarray:
    """Return an involutive 43-class lookup for label-safe horizontal flips.

    Applying this mapping twice returns the original label.  Non-directional
    classes map to themselves so callers can identify eligible rows by testing
    whether the mapped label differs from the input label.
    """

    mapping = np.arange(NUM_CLASSES, dtype=np.int32)
    for left_label, right_label in HORIZONTAL_FLIP_CLASS_PAIRS:
        mapping[left_label] = right_label
        mapping[right_label] = left_label
    return mapping


@dataclass(frozen=True)
class GroupedSplit:
    """Hold every array needed for a zero-overlap grouped split."""

    # Indices preserve the original sample ordering for manifest generation.
    train_indices: np.ndarray
    val_indices: np.ndarray
    # Paths and labels feed directly into tf.data.
    train_paths: np.ndarray
    val_paths: np.ndarray
    train_labels: np.ndarray
    val_labels: np.ndarray
    # Group IDs are retained so overlap can be asserted and reported.
    train_groups: np.ndarray
    val_groups: np.ndarray


def training_group_id(image_path: str | Path, label: int) -> str:
    """Return a class-qualified physical-sign group ID from a training filename.

    A filename such as ``00006_00019.jpg`` is frame 19 from sign/track 6.
    Track numbers restart inside each class folder, so the class ID must be part
    of the group key (for example, ``22:00006``).
    """

    # Split the stem once so the first token is the physical sign/track number.
    stem = Path(image_path).stem
    track_id, separator, frame_id = stem.partition("_")

    # Fail loudly if the dataset naming convention changes or a stray file appears.
    if not separator or not track_id.isdigit() or not frame_id.isdigit():
        raise ValueError(
            f"Training image does not match TRACK_FRAME naming: {image_path}"
        )

    # Qualifying by class prevents class 0 track 00001 from merging with class 1 track 00001.
    return f"{int(label):02d}:{track_id}"


def collect_training_samples(
    dataset_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]:
    """Collect training paths, integer labels, and physical-sign group IDs."""

    # Numeric sorting makes the class-to-label mapping explicit instead of relying on strings.
    class_dirs = sorted(
        (
            path
            for path in dataset_root.iterdir()
            if path.is_dir() and path.name.isdigit() and 0 <= int(path.name) < NUM_CLASSES
        ),
        key=lambda path: int(path.name),
    )

    # Lists are filled in stable class/path order before conversion to compact arrays.
    paths: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    found_classes: set[int] = set()

    # Every class folder name is the actual target label, not an enumerated surrogate.
    for class_dir in class_dirs:
        label = int(class_dir.name)
        found_classes.add(label)

        # Recursive scanning supports both flat and nested course-data layouts.
        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        )
        for image_path in image_paths:
            paths.append(str(image_path))
            labels.append(label)
            groups.append(training_group_id(image_path, label))

    # An empty scan almost always means the wrong dataset directory was supplied.
    if not paths:
        raise RuntimeError(f"No training images found beneath {dataset_root}")

    # All 43 classes are required because the saved model has exactly 43 outputs.
    expected_classes = set(range(NUM_CLASSES))
    if found_classes != expected_classes:
        missing = sorted(expected_classes - found_classes)
        extra = sorted(found_classes - expected_classes)
        raise RuntimeError(f"Expected classes 0..42; missing={missing}, extra={extra}")

    # Object strings avoid truncating long paths or group identifiers.
    return (
        np.asarray(paths, dtype=object),
        np.asarray(labels, dtype=np.int32),
        np.asarray(groups, dtype=object),
        found_classes,
    )


def build_grouped_split(
    paths: Sequence[str],
    labels: Sequence[int],
    groups: Sequence[str],
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = DEFAULT_SEED,
) -> GroupedSplit:
    """Split whole physical-sign groups while balancing each class by group count."""

    # Convert inputs once so masks and indexed outputs have predictable dtypes.
    path_array = np.asarray(paths, dtype=object)
    label_array = np.asarray(labels, dtype=np.int32)
    group_array = np.asarray(groups, dtype=object)

    # Parallel arrays must describe exactly the same samples.
    if not (len(path_array) == len(label_array) == len(group_array)):
        raise ValueError("paths, labels, and groups must have identical lengths")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    # Every group must belong to one class; otherwise a group-wise class split is ambiguous.
    group_labels: dict[str, set[int]] = defaultdict(set)
    for label, group in zip(label_array.tolist(), group_array.tolist()):
        group_labels[str(group)].add(int(label))
    mixed_groups = sorted(group for group, values in group_labels.items() if len(values) != 1)
    if mixed_groups:
        raise AssertionError(f"Groups span multiple classes: {mixed_groups[:10]}")

    # One seeded generator is shared across classes so the full split is reproducible.
    random_generator = np.random.default_rng(seed)
    validation_groups: set[str] = set()

    # Selecting validation groups independently inside each class preserves class balance.
    for label in range(NUM_CLASSES):
        class_groups = np.asarray(
            sorted(set(group_array[label_array == label].tolist())),
            dtype=object,
        )
        if len(class_groups) < 2:
            raise RuntimeError(f"Class {label} needs at least two physical-sign groups")

        # Shuffle group IDs, never individual frames from the same sign.
        random_generator.shuffle(class_groups)

        # Round to the nearest group, keep at least one group in each split.
        validation_group_count = int(round(len(class_groups) * validation_fraction))
        validation_group_count = max(1, validation_group_count)
        validation_group_count = min(validation_group_count, len(class_groups) - 1)
        validation_groups.update(str(group) for group in class_groups[:validation_group_count])

    # Membership in a validation group determines the split for every frame in that group.
    validation_mask = np.asarray(
        [str(group) in validation_groups for group in group_array],
        dtype=bool,
    )
    train_indices = np.flatnonzero(~validation_mask)
    val_indices = np.flatnonzero(validation_mask)

    # Build the named return value before running the final safety assertions.
    split = GroupedSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        train_paths=path_array[train_indices],
        val_paths=path_array[val_indices],
        train_labels=label_array[train_indices],
        val_labels=label_array[val_indices],
        train_groups=group_array[train_indices],
        val_groups=group_array[val_indices],
    )

    # This is the core leakage guard: no physical sign may occur in both partitions.
    overlap = set(split.train_groups.tolist()) & set(split.val_groups.tolist())
    assert not overlap, f"Group leakage detected: {sorted(overlap)[:10]}"

    # Every class should remain represented in both train and validation.
    expected_labels = set(range(NUM_CLASSES))
    assert set(split.train_labels.tolist()) == expected_labels
    assert set(split.val_labels.tolist()) == expected_labels
    assert len(split.train_indices) + len(split.val_indices) == len(path_array)
    return split


def print_split_statistics(split: GroupedSplit) -> None:
    """Print image, unique-group, and per-class counts for both partitions."""

    # Convert NumPy scalar keys to ordinary integers for readable output.
    train_counts = {
        int(label): int(count)
        for label, count in sorted(Counter(split.train_labels.tolist()).items())
    }
    val_counts = {
        int(label): int(count)
        for label, count in sorted(Counter(split.val_labels.tolist()).items())
    }

    # Group sets are recomputed here to make the reported overlap independently visible.
    train_group_set = set(split.train_groups.tolist())
    val_group_set = set(split.val_groups.tolist())
    overlap = train_group_set & val_group_set

    print("Grouped train/validation split:")
    print(f"  train images: {len(split.train_paths)}")
    print(f"  validation images: {len(split.val_paths)}")
    print(f"  train unique groups: {len(train_group_set)}")
    print(f"  validation unique groups: {len(val_group_set)}")
    print(f"  overlapping groups: {len(overlap)}")
    print(f"  train class counts: {train_counts}")
    print(f"  validation class counts: {val_counts}")

    # Keep the assertion beside the printed statistic so failures cannot be overlooked.
    assert len(overlap) == 0, "The grouped split must have zero overlap"
    print("  zero group overlap assertion: PASSED")


def write_split_manifest(split: GroupedSplit, target: Path) -> None:
    """Save one auditable row per image with its label, group, and partition."""

    # Map original sample indices to split names without changing sample order.
    split_by_index = {int(index): "train" for index in split.train_indices.tolist()}
    split_by_index.update({int(index): "validation" for index in split.val_indices.tolist()})

    # Reconstruct full arrays in original order from the two indexed partitions.
    total = len(split.train_indices) + len(split.val_indices)
    paths_by_index: dict[int, str] = {}
    labels_by_index: dict[int, int] = {}
    groups_by_index: dict[int, str] = {}
    for indices, paths, labels, groups in (
        (split.train_indices, split.train_paths, split.train_labels, split.train_groups),
        (split.val_indices, split.val_paths, split.val_labels, split.val_groups),
    ):
        for index, path, label, group in zip(indices, paths, labels, groups):
            paths_by_index[int(index)] = str(path)
            labels_by_index[int(index)] = int(label)
            groups_by_index[int(index)] = str(group)

    # The parent directory may not exist on a first run.
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group", "split"])
        writer.writeheader()
        for index in range(total):
            writer.writerow(
                {
                    "path": paths_by_index[index],
                    "label": labels_by_index[index],
                    "group": groups_by_index[index],
                    "split": split_by_index[index],
                }
            )


def numeric_image_sort_key(path: Path) -> tuple[int, str]:
    """Sort holdout images by numeric stem, with filename as a stable tie-breaker."""

    # Numeric parsing avoids the classic 1, 10, 100, 2 lexical-order bug.
    if not path.stem.isdigit():
        raise ValueError(f"Holdout filename stem must be numeric: {path.name}")
    return int(path.stem), path.name


def collect_holdout_images(holdout_dir: Path) -> list[Path]:
    """Collect holdout files in the exact deterministic course submission order."""

    # The course zip stores all prediction images directly inside one folder.
    paths = [
        path
        for path in holdout_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]
    paths.sort(key=numeric_image_sort_key)

    # Missing extraction or a wrong path should fail before model inference begins.
    if not paths:
        raise RuntimeError(f"No holdout images found in {holdout_dir}")

    # The expected course order is contiguous 00000, 00001, ... with no duplicate IDs.
    numeric_ids = [int(path.stem) for path in paths]
    expected_ids = list(range(len(paths)))
    if numeric_ids != expected_ids:
        raise AssertionError(
            "Holdout numeric IDs must be unique and contiguous from 0; "
            f"first mismatch near {next((i for i, pair in enumerate(zip(numeric_ids, expected_ids)) if pair[0] != pair[1]), 'end')}"
        )
    if len({path.name for path in paths}) != len(paths):
        raise AssertionError("Duplicate holdout filenames were found")
    return paths


def class_index_mapping(output_width: int = NUM_CLASSES) -> dict[int, int]:
    """Return and validate the identity mapping from model index to class label."""

    # The folder label is used directly during training, so no LabelEncoder is involved.
    if output_width != NUM_CLASSES:
        raise AssertionError(
            f"Model must have {NUM_CLASSES} outputs, but has {output_width}"
        )
    mapping = {index: index for index in range(NUM_CLASSES)}
    assert mapping[0] == 0 and mapping[1] == 1 and mapping[42] == 42
    return mapping


def resize_rgb_image(image, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE):
    """Cast an RGB image to float32 and resize it without cropping or padding."""

    # TensorFlow is imported lazily so pure split/order audits do not require it.
    import tensorflow as tf

    # The decoder or Pillow loader must already have produced exactly three RGB channels.
    image_tensor = tf.convert_to_tensor(image)
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = tf.image.resize(
        image_tensor,
        image_size,
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    image_tensor.set_shape([image_size[0], image_size[1], 3])
    return image_tensor


def preprocess_resnet_image(image):
    """Apply ResNet50's one and only model-specific preprocessing step."""

    # TensorFlow's ResNet50 function converts RGB to BGR and subtracts ImageNet means.
    import tensorflow as tf

    # Input must remain on the ordinary 0..255 pixel scale; no 1/255 rescaling is used.
    image_tensor = tf.cast(image, tf.float32)
    return tf.keras.applications.resnet50.preprocess_input(image_tensor)


def load_and_preprocess_pillow(
    path: Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Read a holdout image as RGB, then use the shared resize/preprocess functions."""

    # Pillow avoids native-Windows TensorFlow failures when paths use the WSL UNC form.
    from PIL import Image

    # convert("RGB") makes grayscale, palette, or alpha images consistently three-channel.
    with Image.open(path) as image_file:
        rgb_array = np.asarray(image_file.convert("RGB"), dtype=np.float32)

    # Validation and prediction now share the same TensorFlow resize and ResNet transform.
    resized = resize_rgb_image(rgb_array, image_size=image_size)
    return preprocess_resnet_image(resized).numpy()


def preprocessing_summary(image: np.ndarray) -> dict[str, object]:
    """Return the shape, dtype, range, and mean used by preprocessing diagnostics."""

    # NumPy conversion handles either eager TensorFlow tensors or existing arrays.
    array = np.asarray(image)
    return {
        "shape": tuple(int(value) for value in array.shape),
        "dtype": str(array.dtype),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "all_finite": bool(np.isfinite(array).all()),
    }


def assert_preprocessing_contract(
    training_image: np.ndarray,
    holdout_image: np.ndarray,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> None:
    """Assert matching inference shape, dtype, finiteness, and ResNet value bounds."""

    # The model contract is identical for validation and holdout tensors.
    expected_shape = (image_size[0], image_size[1], 3)
    for name, image in (("training", training_image), ("holdout", holdout_image)):
        array = np.asarray(image)
        assert array.shape == expected_shape, f"{name} shape is {array.shape}"
        assert array.dtype == np.float32, f"{name} dtype is {array.dtype}"
        assert np.isfinite(array).all(), f"{name} preprocessing produced NaN/Inf"

        # RGB values 0..255 become roughly -123.68..151.06 after ResNet preprocessing.
        assert float(array.min()) >= -124.0, f"{name} minimum is outside ResNet bounds"
        assert float(array.max()) <= 152.0, f"{name} maximum is outside ResNet bounds"

    # Matching dtype and shape prove both sources enter the same model interface.
    assert np.asarray(training_image).shape == np.asarray(holdout_image).shape
    assert np.asarray(training_image).dtype == np.asarray(holdout_image).dtype
