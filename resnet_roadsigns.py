#!/usr/bin/env python3
"""Train and evaluate a ResNet50 transfer-learning model on BYUI road signs.

This file is intentionally written as both a training script and a study guide.
Most comments explain not only what the code does, but why that choice matters
for model quality, GPU usage, or project scoring.

High-level workflow:
1. Download/extract the BYUI road-sign image folders.
2. Build a memory-conscious tf.data pipeline for train/validation images.
3. Load ImageNet-pretrained ResNet50 as the visual feature extractor.
4. Train a new 43-class classifier head while ResNet is frozen.
5. Fine-tune the last ResNet layers with a small learning rate.
6. Report both exact class accuracy and broad assignment category accuracy.

The important ML distinction in this project:
- exact accuracy asks, "Did the model predict the exact sign ID?"
- category accuracy asks, "Did the model at least predict the right broad
  group, such as rules-of-road vs cautionary?"
"""

import argparse
import csv
import os
import random
import shutil
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen
from datetime import datetime, timezone

import gc
import numpy as np

# Ask TensorFlow to use a CUDA allocator that is less likely to fragment VRAM
# during long fine-tuning runs. TensorFlow ignores this on unsupported setups.
# This must be set before importing TensorFlow because TensorFlow reads many GPU
# configuration values during import.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, recall_score)

# Shared data rules keep group IDs, class labels, resizing, and model
# preprocessing identical between training, auditing, and holdout inference.
from roadsign_pipeline import (
    ALLOWED_EXTENSIONS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEED,
    NUM_CLASSES,
    build_grouped_split,
    collect_training_samples,
    horizontal_flip_label_map,
    preprocess_resnet_image,
    print_split_statistics,
    resize_rgb_image,
    write_split_manifest,
)

def configure_training_gpu() -> list:
    """Require and configure a GPU only when the training entrypoint runs.

    Keeping this check out of module import makes pure metric and augmentation
    tests CPU-safe while still preventing an accidental multi-hour CPU run.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No GPU detected. Stop and fix TensorFlow before training.")
    for gpu in gpus:
        # Memory growth means TensorFlow asks for VRAM as needed instead of
        # reserving nearly the entire device immediately.
        tf.config.experimental.set_memory_growth(gpu, True)
    try:
        # TF32 speeds up supported float32 GPU operations without extra VRAM.
        tf.config.experimental.enable_tensor_float_32_execution(True)
    except Exception:
        pass
    print("Using GPU:", gpus)
    return gpus


# The course data is split across two zip files. Later we merge them into one
# folder structure where each class ID has exactly one directory.
DATA_URLS = [
    (
        "training1.zip",
        "https://github.com/byui-cse/cse450-course/raw/master/data/roadsigns/training1.zip",
    ),
    (
        "training2.zip",
        "https://github.com/byui-cse/cse450-course/raw/master/data/roadsigns/training2.zip",
    ),
]

# 192x192 is a speed-focused input size. ResNet50 can accept this because the
# top classifier is removed, and road signs are usually simple enough that the
# smaller image may preserve the important shape/color cues. If accuracy drops,
# rerun with --img-size 224 to use the classic ResNet training size.
# If training crashes with an out-of-memory error, batch size is the first knob
# to lower because activations scale roughly with batch size.
IMG_SIZE = DEFAULT_IMAGE_SIZE
BATCH_SIZE = 64
SEED = DEFAULT_SEED

# These keep the input pipeline useful without letting it build a huge queue of
# decoded images in system RAM.
# Larger prefetch/parallel values can improve GPU utilization, but also increase
# CPU and RAM pressure. These defaults feed the GPU harder while staying
# reasonable for a 32 GB system that is also doing normal desktop work.
SHUFFLE_BUFFER = 2048
PREFETCH_BUFFER = 4
MAX_PARALLEL_CALLS = 16
TRAINING_DATASET_DETERMINISTIC = True

# Named performance profiles make it easy to restart with a safer or faster
# setting without editing code. "turbo" is intended for your 16 GB VRAM target.
BALANCED_BATCH_SIZE = 16
BALANCED_PARALLEL_CALLS = 8
BALANCED_PREFETCH_BUFFER = 2
LOW_MEMORY_BATCH_SIZE = 8
LOW_MEMORY_PARALLEL_CALLS = 4
LOW_MEMORY_PREFETCH_BUFFER = 1
TURBO_BATCH_SIZE = 128
TURBO_PARALLEL_CALLS = 24
TURBO_PREFETCH_BUFFER = 4

# XLA can fuse TensorFlow ops and reduce overhead on GPU. It may spend extra
# time compiling at the start, but can make longer runs faster afterward.
USE_XLA = True
USE_DISK_CACHE = True
# The grouped split must never reuse v1/v2 caches built from the leaked random split.
CACHE_VERSION = "v3_grouped"

# Class weights help rare sign classes, but clipping avoids making training
# unstable for very small classes.
# Without clipping, a tiny class could receive a very large weight and make the
# optimizer chase rare-class mistakes too aggressively.
MAX_CLASS_WEIGHT = 5.0

# Initial frozen training: learn only the new classifier head first.
# This is usually safer than unfreezing everything immediately because the new
# Dense layer starts random while ResNet's pretrained features are already good.
FROZEN_EPOCHS = 15
INITIAL_LR = 1e-3

# Fine-tuning: unfreeze only the last part of ResNet and use a much smaller LR.
# A negative index means "count from the end"; -50 keeps most early layers
# frozen and only adapts the last 50-ish layers to road signs.
FINE_TUNE_AT = -50
FINE_TUNE_BLOCK_EPOCHS = 5
FINE_TUNE_LR = 2e-5
MAX_FINE_TUNE_ROUNDS = 8

# Category targets from the assignment. These are broad category accuracies,
# not exact 43-class sign accuracies.
# Keep this distinction clear when reading the terminal output: Keras' built-in
# accuracy is exact class accuracy, while these targets are calculated after
# each training stage by compute_category_metrics().
TARGET_CAUTIONARY = 0.95
TARGET_RULES_OF_ROAD = 0.98

# GTSRB-style label groups used by the BYUI roadsigns folders.
# 0..17 and 32..42 are regulatory/priority/mandatory road-rule signs.
# 18..31 are cautionary/warning signs.
# The original code tried to infer groups from label text, but labels here are
# numeric folder names. Explicit numeric sets are less clever and much safer.
RULES_OF_ROAD_CLASSES = frozenset(list(range(0, 18)) + list(range(32, 43)))
CAUTIONARY_CLASSES = frozenset(range(18, 32))
CATEGORY_GROUPS = {
    "rules_of_road": RULES_OF_ROAD_CLASSES,
    "cautionary": CAUTIONARY_CLASSES,
}
CATEGORY_TARGETS = {
    "rules_of_road": TARGET_RULES_OF_ROAD,
    "cautionary": TARGET_CAUTIONARY,
}


@tf.keras.utils.register_keras_serializable(package="Roadsigns")
class SparseMacroF1(tf.keras.metrics.Metric):
    """Accumulate exact multiclass macro F1 for sparse integer labels.

    Keras' built-in accuracy is useful, but the course grader is driven by
    macro F1.  Keeping one confusion matrix for the whole epoch lets callbacks
    select the right checkpoint without running a second validation pass.
    """

    def __init__(self, num_classes=NUM_CLASSES, name="macro_f1", dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.num_classes = int(num_classes)
        self.confusion = self.add_weight(
            name="confusion_matrix",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
            dtype=self.dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Add one batch using XLA-friendly one-hot matrix multiplication."""
        true_labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        predicted_labels = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        true_one_hot = tf.one_hot(true_labels, self.num_classes, dtype=self.dtype)
        predicted_one_hot = tf.one_hot(
            predicted_labels,
            self.num_classes,
            dtype=self.dtype,
        )
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1, 1]), self.dtype)
            true_one_hot = true_one_hot * weights
        batch_confusion = tf.matmul(
            true_one_hot,
            predicted_one_hot,
            transpose_a=True,
        )
        self.confusion.assign_add(batch_confusion)

    def result(self):
        """Return the unweighted mean of all 43 per-class F1 values."""
        true_positives = tf.linalg.diag_part(self.confusion)
        false_positives = tf.reduce_sum(self.confusion, axis=0) - true_positives
        false_negatives = tf.reduce_sum(self.confusion, axis=1) - true_positives
        per_class_f1 = tf.math.divide_no_nan(
            2.0 * true_positives,
            2.0 * true_positives + false_positives + false_negatives,
        )
        return tf.reduce_mean(per_class_f1)

    def reset_state(self):
        """Clear the epoch accumulator before Keras begins the next epoch."""
        self.confusion.assign(tf.zeros_like(self.confusion))

    def get_config(self):
        """Preserve the class count when a compiled model is serialized."""
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


class GuardedMacroF1(tf.keras.callbacks.Callback):
    """Expose macro F1 to later callbacks only above an accuracy floor."""

    def __init__(self, accuracy_floor=0.0):
        super().__init__()
        self.accuracy_floor = float(accuracy_floor)

    def on_epoch_end(self, epoch, logs=None):
        """Add ``val_guarded_macro_f1`` before checkpoint callbacks run."""
        logs = logs if logs is not None else {}
        macro_f1 = logs.get("val_macro_f1")
        accuracy = logs.get("val_accuracy")
        if macro_f1 is None or accuracy is None:
            raise RuntimeError(
                "Guarded macro F1 requires val_macro_f1 and val_accuracy logs."
            )
        guarded_value = float(macro_f1) if float(accuracy) >= self.accuracy_floor else -np.inf
        logs["val_guarded_macro_f1"] = guarded_value
        if not np.isfinite(guarded_value):
            print(
                "Guarded macro F1 rejected this epoch: "
                f"val_accuracy={float(accuracy):.4f} < floor={self.accuracy_floor:.4f}"
            )


def apply_horizontal_flip_mask(images, labels, requested_mask, label_map):
    """Flip eligible images and remap their directional labels atomically."""
    labels = tf.cast(labels, tf.int32)
    remapped_labels = tf.gather(label_map, labels)
    eligible = tf.not_equal(remapped_labels, labels)
    applied_mask = tf.logical_and(tf.cast(requested_mask, tf.bool), eligible)
    flipped_images = tf.image.flip_left_right(images)
    output_images = tf.where(
        applied_mask[:, None, None, None],
        flipped_images,
        images,
    )
    output_labels = tf.where(applied_mask, remapped_labels, labels)
    return output_images, output_labels


def apply_paired_random_horizontal_flip(images, labels, probability, label_map):
    """Randomly mirror only directional classes with a valid partner label."""
    if probability <= 0.0:
        return images, labels
    requested_mask = tf.random.uniform(tf.shape(labels), 0.0, 1.0) < probability
    return apply_horizontal_flip_mask(images, labels, requested_mask, label_map)


def apply_random_photometric_augmentation(
    images,
    probability,
    gamma_min=0.55,
    gamma_max=1.45,
    brightness_min=0.70,
    brightness_max=1.50,
):
    """Randomize exposure for selected training images without changing labels.

    Gamma changes shadow visibility while a multiplicative brightness factor
    covers under- and over-exposure. Values remain in the original 0..255 range
    expected by the later ResNet preprocessing call.
    """
    if probability <= 0.0:
        return images
    batch_size = tf.shape(images)[0]
    requested = tf.random.uniform([batch_size, 1, 1, 1]) < probability
    gammas = tf.random.uniform(
        [batch_size, 1, 1, 1],
        minval=gamma_min,
        maxval=gamma_max,
    )
    brightness = tf.random.uniform(
        [batch_size, 1, 1, 1],
        minval=brightness_min,
        maxval=brightness_max,
    )
    normalized = tf.clip_by_value(tf.cast(images, tf.float32) / 255.0, 0.0, 1.0)
    transformed = tf.pow(normalized, gammas) * 255.0 * brightness
    transformed = tf.clip_by_value(transformed, 0.0, 255.0)
    return tf.where(requested, transformed, tf.cast(images, tf.float32))


def download_file(url: str, target: Path) -> None:
    """Download a dataset zip only when it is not already present."""
    if target.exists():
        # Reusing the existing zip keeps repeated experiments fast and avoids
        # relying on the network after the first successful download.
        print(f"Skipping download, file already exists: {target}")
        return

    print(f"Downloading {url} -> {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, open(target, "wb") as out_file:
        out_file.write(response.read())


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a dataset zip while ignoring macOS metadata folders."""
    print(f"Extracting {zip_path} to {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if member.filename.startswith("__MACOSX"):
                # These folders are metadata from macOS archives, not training
                # data, and should not be interpreted as classes.
                continue
            target_path = target_dir.joinpath(Path(member.filename))
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, open(target_path, "wb") as dest:
                    dest.write(source.read())


def merge_training_dirs(root_dir: Path) -> None:
    """Move training1/training2 class folders into one shared dataset root."""
    for source_dir in list(root_dir.iterdir()):
        if not source_dir.is_dir() or source_dir.name not in {"training1", "training2"}:
            continue
        for class_dir in source_dir.iterdir():
            if not class_dir.is_dir():
                continue
            dest_dir = root_dir / class_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for image_file in class_dir.iterdir():
                if image_file.is_file():
                    target_file = dest_dir / image_file.name
                    if not target_file.exists():
                        # Move rather than copy so we do not keep duplicate
                        # images and waste disk space.
                        image_file.rename(target_file)
                    else:
                        # If a file name already exists from the other zip,
                        # delete the duplicate rather than training on it twice.
                        image_file.unlink()
        try:
            source_dir.rmdir()
        except OSError:
            pass


def _collect_image_paths(dataset_root: Path):
    """Compatibility helper that preserves actual numeric folder labels."""
    # Numeric sorting and direct integer conversion prevent directory enumeration
    # from silently creating a different class-index mapping.
    class_dirs = sorted(
        [d for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda path: int(path.name),
    )
    class_names = [str(int(d.name)) for d in class_dirs]
    image_paths = []
    image_labels = []
    for class_dir in class_dirs:
        label = int(class_dir.name)
        for image_file in sorted(class_dir.iterdir()):
            if image_file.suffix.lower() in ALLOWED_EXTENSIONS and image_file.is_file():
                image_paths.append(str(image_file))
                image_labels.append(label)
    if not image_paths:
        raise RuntimeError(f"No image files found under dataset root {dataset_root}")
    return np.array(image_paths, dtype=object), np.array(image_labels, dtype=np.int32), class_names


def _decode_image(path, label):
    """Read an image file, force RGB, and resize before model preprocessing."""
    # tf.data runs this function inside TensorFlow's input pipeline, so images
    # stream from disk in batches instead of being loaded all at once.
    image = tf.io.read_file(path)
    # decode_image handles jpg/png/ppm-like formats while channels=3 guarantees
    # the model always receives RGB tensors.
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    # TensorFlow needs a known channel dimension so later layers can build.
    image.set_shape([None, None, 3])
    # The shared resize function is also used by holdout inference.
    image = resize_rgb_image(image, image_size=IMG_SIZE)
    return image, label


def find_dataset_root(extract_root: Path) -> Path:
    """Find the folder whose children are class directories containing images."""
    def is_image_file(path: Path) -> bool:
        return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".ppm"}

    candidate_roots = [extract_root]
    candidate_roots.extend(
        [extract_root / child for child in extract_root.iterdir() if child.is_dir()]
    )
    for candidate in candidate_roots:
        # The zips may extract directly into data/roadsigns or into an extra
        # nested folder. This scan accepts either layout.
        if not candidate.exists():
            continue
        class_dirs = [d for d in candidate.iterdir() if d.is_dir()]
        if not class_dirs:
            continue
        if any(
            any(
                is_image_file(file)
                for file in class_dir.iterdir()
                if file.is_file()
            )
            for class_dir in class_dirs
        ):
            return candidate
    raise RuntimeError(
        f"Could not find a usable image dataset root beneath {extract_root}."
    )


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
    power: float = 1.0,
) -> np.ndarray:
    """Give rare classes configurable influence during loss calculation.

    ``power=1`` preserves full inverse-frequency weighting, ``power=0.5``
    applies the milder square-root form, and ``power=0`` disables weighting.
    """
    if not 0.0 <= power <= 1.0:
        raise ValueError("class-weight power must be between 0 and 1")
    # Formula intuition:
    # - common classes get lower weight because the model sees many examples
    # - rare classes get higher weight so their mistakes are not drowned out
    # - clipping keeps the rare-class boost from becoming extreme
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = float(np.sum(counts))
    weights = np.ones(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    weights = np.power(weights, power).astype(np.float32)
    return np.clip(weights, 1.0 / MAX_CLASS_WEIGHT, MAX_CLASS_WEIGHT)


def clear_tf_cache_prefix(cache_prefix: Path) -> None:
    """Remove an existing tf.data disk cache prefix and its sidecar files."""
    # tf.data.cache("prefix") may create files such as prefix.data-00000-of-00001
    # and prefix.index. Removing by glob lets us cleanly rebuild stale caches.
    for path in cache_prefix.parent.glob(f"{cache_prefix.name}*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def cache_prefix(cache_dir: Path, split_name: str) -> Path:
    """Build a cache prefix unique to split, image size, and script cache version."""
    return cache_dir / f"{CACHE_VERSION}_{split_name}_{IMG_SIZE[0]}x{IMG_SIZE[1]}_seed{SEED}"


def prepare_datasets(
    root_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_parallel_calls: int = MAX_PARALLEL_CALLS,
    prefetch_buffer: int = PREFETCH_BUFFER,
    training_deterministic: bool = TRAINING_DATASET_DETERMINISTIC,
    disk_cache: bool = USE_DISK_CACHE,
    cache_dir: Path | None = None,
    rebuild_cache: bool = False,
    augment: bool = True,
    class_weight_power: float = 1.0,
    paired_hflip_probability: float = 0.0,
    photometric_augmentation_probability: float = 0.0,
    split_manifest: Path | None = None,
):
    """Build memory-conscious, group-separated train/validation pipelines."""
    print(f"Preparing datasets from {root_dir}")
    dataset_root = find_dataset_root(root_dir)
    print(f"Using dataset root: {dataset_root}")

    # Report folders ignored by the strict 0..42 numeric class mapping.
    all_class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    ignored = [
        directory.name
        for directory in all_class_dirs
        if not directory.name.isdigit()
        or not 0 <= int(directory.name) < NUM_CLASSES
    ]
    if ignored:
        print(f"Ignoring non-class or out-of-range folders: {ignored}")

    # Parse the actual numeric label and physical-sign group from every image.
    paths, labels, groups, found_classes = collect_training_samples(dataset_root)

    # Split whole signs/tracks per class. Frames from one sign can no longer leak
    # into both training and validation as they did in the original image split.
    grouped_split = build_grouped_split(
        paths,
        labels,
        groups,
        validation_fraction=0.20,
        seed=SEED,
    )
    print_split_statistics(grouped_split)
    if split_manifest is not None:
        write_split_manifest(grouped_split, split_manifest)
        print(f"Saved grouped split manifest: {split_manifest}")

    # Named arrays keep the remaining tf.data construction easy to audit.
    train_paths = grouped_split.train_paths
    val_paths = grouped_split.val_paths
    train_labels = grouped_split.train_labels
    val_labels = grouped_split.val_labels

    # Class weights are calculated from the corrected training partition only.
    class_weights = compute_class_weights(
        train_labels,
        NUM_CLASSES,
        power=class_weight_power,
    )
    # Store class weights as a Tensor so tf.data can attach the right weight to
    # each training example without Python work during every batch.
    class_weight_tensor = tf.constant(class_weights, dtype=tf.float32)
    flip_label_tensor = tf.constant(horizontal_flip_label_map(), dtype=tf.int32)

    # At this point the dataset contains only file paths and integer labels.
    # That is intentionally lightweight: no full images are loaded into RAM yet.
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    train_options = tf.data.Options()
    # Let parallel map calls return batches as soon as they are ready. This is
    # often faster and still fine for stochastic training.
    train_options.experimental_deterministic = training_deterministic
    train_ds = train_ds.with_options(train_options)

    train_cache_prefix = None
    val_cache_prefix = None
    if disk_cache:
        if cache_dir is None:
            cache_dir = root_dir / ".tf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_cache_split = "train_aug" if augment else "train_noaug"
        train_cache_prefix = cache_prefix(cache_dir, train_cache_split)
        val_cache_prefix = cache_prefix(cache_dir, "val")
        if rebuild_cache:
            clear_tf_cache_prefix(train_cache_prefix)
            clear_tf_cache_prefix(val_cache_prefix)

    # preprocess_input converts RGB pixel values into the format ResNet50 was
    # trained with. This must match the pretrained model family.
    preprocess = preprocess_resnet_image
    data_augmentation = tf.keras.Sequential(
        [
            # Horizontal flips are handled separately because only eight
            # directional labels have a valid mirrored partner.  These generic
            # transforms remain mild enough to preserve sign identity.
            tf.keras.layers.RandomRotation(0.06, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.08, fill_mode="nearest"),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    def augment_batch(images, labels):
        """Apply atomic paired flips, generic transforms, and preprocessing."""
        images, labels = apply_paired_random_horizontal_flip(
            images,
            labels,
            paired_hflip_probability,
            flip_label_tensor,
        )
        images = apply_random_photometric_augmentation(
            images,
            probability=photometric_augmentation_probability,
        )
        images = data_augmentation(images, training=True)
        # Contrast and interpolation may move pixels outside the source range.
        images = tf.clip_by_value(images, 0.0, 255.0)
        return preprocess(images), labels

    def prepare(ds, training=False, decoded_cache_prefix=None):
        # Training data is shuffled so batches are mixed. Validation data is not
        # shuffled because stable ordering makes evaluation/reports easier.
        ds = ds.map(_decode_image, num_parallel_calls=num_parallel_calls)
        if training and augment:
            if disk_cache and decoded_cache_prefix is not None:
                # With augmentation enabled, cache decoded/resized images before
                # random transforms so each epoch still sees fresh augmentations.
                ds = ds.cache(str(decoded_cache_prefix))
            # Shuffle after caching so later epochs can still reshuffle cached
            # tensors instead of freezing the first epoch's order.
            shuffle_size = min(SHUFFLE_BUFFER, len(train_paths))
            ds = ds.shuffle(shuffle_size, seed=SEED, reshuffle_each_iteration=True)

            # Batch before augmentation so TensorFlow transforms a vectorized
            # batch instead of invoking four random layers once per image.
            ds = ds.batch(batch_size)
            ds = ds.map(
                augment_batch,
                num_parallel_calls=num_parallel_calls,
            )
        else:
            # Validation and optional no-augmentation training use the same
            # single ResNet preprocessing call with no random transformations.
            ds = ds.map(
                lambda x, y: (preprocess(x), y),
                num_parallel_calls=num_parallel_calls,
            )
            if disk_cache and decoded_cache_prefix is not None:
                # Without augmentation, cache after preprocessing. This avoids
                # decode, resize, and ResNet preprocessing on later epochs/runs.
                ds = ds.cache(str(decoded_cache_prefix))
            if training:
                shuffle_size = min(SHUFFLE_BUFFER, len(train_paths))
                ds = ds.shuffle(shuffle_size, seed=SEED, reshuffle_each_iteration=True)
            # Non-augmented data is batched after per-image cache/preprocessing.
            ds = ds.batch(batch_size)
        if training:
            # Sample weights are attached per batched label so rare classes count more.
            # Keras automatically treats a 3-tuple as (inputs, labels, weights).
            ds = ds.map(
                lambda x, y: (x, y, tf.gather(class_weight_tensor, y)),
                num_parallel_calls=num_parallel_calls,
            )
        # Prefetch overlaps CPU input work with GPU training. Keeping this small
        # protects system RAM while still reducing GPU idle time.
        return ds.prefetch(buffer_size=prefetch_buffer)

    train_ds = prepare(train_ds, training=True, decoded_cache_prefix=train_cache_prefix)
    val_ds = prepare(val_ds, decoded_cache_prefix=val_cache_prefix)

    # Build exact 43 class names (0..42) for model outputs
    # These are still numeric strings because the folder names are numeric; the
    # category mapping above supplies the semantic grouping.
    class_names = [str(i) for i in range(NUM_CLASSES)]
    print(f"Found {len(found_classes)} class folders, {len(paths)} images after filtering")
    print(f"Train images: {len(train_paths)}, validation images: {len(val_paths)}")
    print(
        f"Batch size: {batch_size}, prefetch buffer: {prefetch_buffer}, "
        f"parallel calls: {num_parallel_calls}, "
        f"training deterministic: {training_deterministic}"
    )
    if disk_cache:
        print(f"Disk cache enabled: {cache_dir}")
        print("Cache note: the first full epoch may build the cache; later epochs/runs can reuse it.")
    else:
        print("Disk cache disabled.")
    print(f"Augmentation enabled: {augment}")
    print(f"Class-weight power: {class_weight_power:.3f}")
    print(f"Paired horizontal-flip probability: {paired_hflip_probability:.3f}")
    print(
        "Photometric augmentation probability: "
        f"{photometric_augmentation_probability:.3f}"
    )
    missing = sorted(set(range(NUM_CLASSES)) - found_classes)
    if missing:
        raise RuntimeError(
            f"Dataset must contain exactly 43 classes 0..42; missing classes: {missing}"
        )
    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    """Create a ResNet50 transfer-learning model for road-sign classes."""
    # include_top=False removes ImageNet's original 1000-class classifier.
    # We only want the convolutional feature extractor, then we add our own
    # 43-class road-sign classifier below.
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        # Global average pooling turns the final feature maps into one vector
        # per image, which is a compact input for the Dense classifier.
        pooling="avg",
    )
    # Freeze the pretrained feature extractor for the first training stage.
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    # training=False keeps BatchNorm layers in inference mode while frozen.
    # This avoids disturbing pretrained BatchNorm statistics during head-only
    # training, which can otherwise hurt transfer learning.
    x = base_model(inputs, training=False)
    # Dropout discourages the new classifier from relying too heavily on any one
    # feature. It is a small regularization step for generalization.
    x = tf.keras.layers.Dropout(0.5)(x)
    # Keep logits/probabilities in float32 even if mixed precision is enabled.
    # Softmax gives a probability distribution over the 43 sign classes.
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs, name="resnet_roadsigns")
    return model, base_model


def find_resnet_base_model(model: tf.keras.Model) -> tf.keras.Model:
    """Locate the nested ResNet feature extractor inside a loaded checkpoint."""
    # Saved Keras models preserve the nested base model as a top-level layer.
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("resnet"):
            return layer
    raise RuntimeError("Loaded checkpoint does not contain a nested ResNet base model")


def make_training_callbacks(
    checkpoint_path: Path | None = None,
    accuracy_floor: float = 0.0,
    initial_best_macro_f1: float | None = None,
):
    """Select epochs by macro F1 without accepting an accuracy collapse."""
    callbacks = [GuardedMacroF1(accuracy_floor=accuracy_floor)]
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_guarded_macro_f1",
                mode="max",
                save_best_only=True,
                verbose=1,
                initial_value_threshold=initial_best_macro_f1,
            )
        )
    callbacks.extend([
        tf.keras.callbacks.ReduceLROnPlateau(
            # Scheduling follows the raw grading metric. A temporary accuracy
            # miss should not look like an infinite regression to the optimizer.
            monitor="val_macro_f1",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            # Stop when raw macro F1 genuinely plateaus. The guarded checkpoint
            # is reloaded explicitly after fit, so this callback never decides
            # which weights become the experiment result.
            monitor="val_macro_f1",
            mode="max",
            patience=5,
            min_delta=0.001,
            restore_best_weights=False,
            verbose=1,
        ),
    ])
    return callbacks


def restore_selected_checkpoint(model, checkpoint_path: Path) -> None:
    """Load the accuracy-guarded macro-F1 winner back into the live model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Guarded macro-F1 checkpoint was not created: {checkpoint_path}"
        )
    # Keras can read weights directly from a full .keras archive. This avoids
    # constructing a second ResNet and keeps the existing nested base-model
    # reference valid for a later fine-tuning stage.
    model.load_weights(str(checkpoint_path))
    print(f"Restored guarded macro-F1 checkpoint from {checkpoint_path}")


def compile_and_train(
    model,
    train_ds,
    val_ds,
    initial_lr=INITIAL_LR,
    epochs=FROZEN_EPOCHS,
    callbacks=None,
    jit_compile=USE_XLA,
):
    """Train the new classifier head while ResNet remains frozen."""
    # SparseCategoricalCrossentropy is correct when labels are integer class IDs
    # like 0, 1, 2 rather than one-hot vectors.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            SparseMacroF1(NUM_CLASSES),
        ],
        jit_compile=jit_compile,
    )
    # verbose=2 prints one stable summary line per epoch without a massive
    # carriage-return progress log in WSL or redirected experiment output.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks or [],
        verbose=2,
    )
    return history


def fine_tune_block(
    model,
    base_model,
    train_ds,
    val_ds,
    epochs=FINE_TUNE_BLOCK_EPOCHS,
    lr=FINE_TUNE_LR,
    trainable_base_layers=abs(FINE_TUNE_AT),
    weight_decay=0.0,
    callbacks=None,
    jit_compile=USE_XLA,
):
    """Fine-tune a configurable number of final ResNet layers at a small LR."""
    if not 1 <= trainable_base_layers <= len(base_model.layers):
        raise ValueError(
            f"trainable_base_layers must be 1..{len(base_model.layers)}, "
            f"received {trainable_base_layers}"
        )

    base_model.trainable = True
    # Early layers learn generic edges/textures, while later layers contain the
    # task-specific features that are safest to adapt first.
    freeze_before = len(base_model.layers) - trainable_base_layers
    for layer_index, layer in enumerate(base_model.layers):
        should_train = layer_index >= freeze_before
        # Keep BatchNorm statistics fixed during transfer learning because small
        # road-sign batches can otherwise destabilize pretrained normalization.
        layer.trainable = should_train and not isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        )
    actual_trainable_layers = sum(layer.trainable for layer in base_model.layers)
    print(
        f"Fine-tuning final {trainable_base_layers} ResNet layers "
        f"({actual_trainable_layers} non-BatchNorm layers trainable)."
    )
    # Fine-tuning uses a smaller learning rate because pretrained weights are
    # already useful. Optional AdamW decay addresses late-stage overfitting.
    optimizer = (
        tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
        if weight_decay > 0.0
        else tf.keras.optimizers.Adam(learning_rate=lr)
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            SparseMacroF1(NUM_CLASSES),
        ],
        jit_compile=jit_compile,
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks or [],
        verbose=2,
    )
    return history


def flatten_dataset(dataset):
    """Debug helper that turns a small dataset into NumPy arrays."""
    # This is not used in normal training because flattening the whole dataset
    # would use much more RAM. Keep it for small sanity checks only.
    images = []
    labels = []
    for batch in dataset:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        images.append(x.numpy())
        labels.append(y.numpy())
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)


def compute_category_metrics(y_true, y_pred):
    """Measure broad assignment categories separately from exact sign labels."""
    category_metrics = {}
    for group_name, class_ids in CATEGORY_GROUPS.items():
        class_id_array = np.array(sorted(class_ids), dtype=np.int32)
        # Select only validation examples whose true label belongs to this broad
        # assignment category.
        mask = np.isin(y_true, class_id_array)
        support = int(np.sum(mask))
        if support == 0:
            category_metrics[group_name] = {
                "group_accuracy": float("nan"),
                "exact_accuracy": float("nan"),
                "support": 0,
            }
            continue

        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        category_metrics[group_name] = {
            # Broad category metric: predicting any class in the same group counts.
            # Example: if true class 20 and predicted class 24 are both cautionary,
            # category accuracy counts it correct even though exact accuracy does not.
            "group_accuracy": float(np.mean(np.isin(y_pred_group, class_id_array))),
            # Exact metric: prediction must match the true sign class.
            "exact_accuracy": float(np.mean(y_pred_group == y_true_group)),
            "support": support,
        }
    return category_metrics


def category_targets_met(category_metrics, targets=CATEGORY_TARGETS):
    """Return True only when both assignment category goals are satisfied."""
    for group_name, target in targets.items():
        # np.nan guards against accidental empty categories or broken metric
        # computation. A missing metric should never be treated as success.
        group_accuracy = category_metrics.get(group_name, {}).get("group_accuracy", float("nan"))
        if np.isnan(group_accuracy) or group_accuracy < target:
            return False
    return True


def category_score(category_metrics):
    """Average the broad category accuracies for best-model selection."""
    values = [
        category_metrics.get(group_name, {}).get("group_accuracy", float("nan"))
        for group_name in CATEGORY_GROUPS
    ]
    return (
        float(np.nanmean(values))
        if not np.all(np.isnan(values))
        else -1.0
    )


def get_dataset_labels(dataset):
    """Collect labels from a dataset, whether or not it includes sample weights."""
    labels = []
    for batch in dataset:
        # Training batches are (x, y, sample_weight). Validation batches are
        # (x, y). Supporting both makes this helper reusable.
        if len(batch) == 3:
            _, y, _ = batch
        else:
            _, y = batch
        labels.append(y.numpy())
    return np.concatenate(labels, axis=0)


def evaluate_model_on_dataset(model, val_ds, class_names):
    """Run validation predictions and compute exact plus category metrics."""
    # Prediction returns one probability vector per image. argmax picks the
    # class with the highest predicted probability.
    predictions = model.predict(val_ds, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = get_dataset_labels(val_ds)

    num_classes = len(class_names)
    labels_param = list(range(num_classes))
    # sklearn's classification_report gives per-class precision/recall/F1. This
    # is useful when overall accuracy hides weak classes.
    report_dict = classification_report(
        true_labels,
        pred_labels,
        labels=labels_param,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    # The confusion matrix is the best way to see which signs the model confuses
    # with each other, such as class 7 often being predicted as class 2.
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_param)
    # Macro metrics give every class equal importance, which helps reveal
    # problems on rare classes that overall accuracy can hide.
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    macro_recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    category_metrics = compute_category_metrics(true_labels, pred_labels)
    return report_dict, cm, float(macro_f1), float(macro_recall), category_metrics


def run_learning_rate_experiment(
    train_ds,
    val_ds,
    class_names,
    result_csv: str,
    best_model_path: str,
    epochs: int,
):
    """Optional unguarded screening sweep for frozen-head learning rates."""
    # This trains several short-lived models from scratch. It is useful for
    # exploration, but optional because it adds time before the real run.
    lr_values = [1e-3, 1e-4, 1e-5, 1e-6]
    best_validation_accuracy = -1.0
    best_macro_f1 = -1.0
    best_lr = None
    for lr in lr_values:
        # Clear old models between experiments so GPU memory is released as much
        # as TensorFlow allows.
        tf.keras.backend.clear_session()
        gc.collect()
        print(f"Learning rate sweep: testing lr={lr:.0e} for {epochs} epochs")
        model, _ = build_model(len(class_names))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                SparseMacroF1(NUM_CLASSES),
            ],
            jit_compile=USE_XLA,
        )
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)

        report_dict, cm, macro_f1, macro_recall, category_metrics = evaluate_model_on_dataset(
            model, val_ds, class_names
        )
        # The learning-rate sweep records the same final metrics as the main
        # training path so results are comparable.
        val_accuracy_overall = float(
            report_dict.get("accuracy", 0.0)
            if not isinstance(report_dict.get("accuracy", 0.0), dict)
            else report_dict.get("accuracy", {}).get("recall", 0.0)
        )
        rules_acc = category_metrics["rules_of_road"]["group_accuracy"]
        caution_acc = category_metrics["cautionary"]["group_accuracy"]
        rules_exact = category_metrics["rules_of_road"]["exact_accuracy"]
        caution_exact = category_metrics["cautionary"]["exact_accuracy"]
        current_category_score = category_score(category_metrics)

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "learning_rate": lr,
            "epochs": epochs,
            "val_accuracy": val_accuracy_overall,
            "category_rules_of_road": rules_acc,
            "category_cautionary": caution_acc,
            "exact_rules_of_road": rules_exact,
            "exact_cautionary": caution_exact,
            "macro_f1": float(macro_f1),
            "macro_recall": float(macro_recall),
            "category_score": float(current_category_score),
        }
        append_progress_row(Path(result_csv), row)

        if macro_f1 > best_macro_f1:
            # Canvas grades macro F1, so the sweep must select on the same
            # objective as the final checkpoint. Accuracy remains visible as a
            # guardrail in the experiment table and summary message.
            best_macro_f1 = float(macro_f1)
            best_validation_accuracy = val_accuracy_overall
            best_lr = lr
            model.save(best_model_path)
            print(
                f"Saved best lr model to {best_model_path} for lr={lr} "
                f"(macro_f1={macro_f1:.4f}, "
                f"group_validation_accuracy={val_accuracy_overall:.4f})"
            )

        tf.keras.backend.clear_session()
        gc.collect()

    print(
        "Learning rate sweep complete. "
        f"Best lr={best_lr} with macro_f1={best_macro_f1:.4f} and "
        f"group_validation_accuracy={best_validation_accuracy:.4f}"
    )
    return best_lr, best_validation_accuracy, best_macro_f1


def save_csv_report(report_dict, class_names, target: Path):
    """Save per-class precision/recall/F1 for later inspection."""
    # Map numeric label keys to class names when report_dict uses string keys
    rows = []
    num = len(class_names)
    for i, cname in enumerate(class_names):
        key = str(i)
        metrics = report_dict.get(key, {}) if isinstance(report_dict, dict) else {}
        if not metrics:
            # Some formats return numeric keys, so support both "0" and 0.
            metrics = report_dict.get(i, {}) if isinstance(report_dict, dict) else {}
        rows.append(
            {
                "label": cname,
                "precision": metrics.get("precision", ""),
                "recall": metrics.get("recall", ""),
                "f1-score": metrics.get("f1-score", ""),
                "support": metrics.get("support", ""),
            }
        )
    # Accuracy is a single scalar in sklearn's report, not a per-class row, so
    # it gets written in the recall column to preserve the simple CSV shape.
    if "accuracy" in report_dict:
        acc = report_dict.get("accuracy")
        rows.append({"label": "accuracy", "precision": "", "recall": acc if not isinstance(acc, dict) else acc.get("recall",""), "f1-score": "", "support": ""})

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["label", "precision", "recall", "f1-score", "support"])
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(cm, class_names, target: Path):
    """Save which classes are being confused with which other classes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # First row is predicted labels; first column of later rows is true label.
        writer.writerow([""] + class_names)
        for row_label, row in zip(class_names, cm.tolist()):
            writer.writerow([row_label] + row)


def save_summary(summary: dict, target: Path):
    """Save the headline metrics in a compact two-column CSV."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def append_progress_row(target: Path, row: dict):
    """Append progress while preserving older CSV columns from previous runs."""
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    existing_fieldnames = []
    if target.exists():
        # Older project runs may have different columns. Read/rewrite with the
        # union of old and new columns so previous logs are not destroyed.
        with open(target, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)

    fieldnames = list(dict.fromkeys(existing_fieldnames + list(row.keys())))
    with open(target, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)


def overall_accuracy_from_report(report_dict):
    """Extract sklearn's overall exact accuracy from its report dictionary."""
    accuracy = report_dict.get("accuracy", 0.0)
    if isinstance(accuracy, dict):
        return float(accuracy.get("recall", 0.0))
    return float(accuracy)


def print_category_metrics(category_metrics):
    """Print broad category accuracy next to exact-in-category accuracy."""
    for group_name, target in CATEGORY_TARGETS.items():
        metrics = category_metrics[group_name]
        # category = broad assignment metric; exact = exact sign ID metric.
        print(
            f"{group_name}: "
            f"category={metrics['group_accuracy']:.4f} "
            f"(target={target:.4f}), "
            f"exact={metrics['exact_accuracy']:.4f}, "
            f"support={metrics['support']}"
        )


def summarize_training_history(history):
    """Return metrics from the best guarded-macro-F1 epoch of one stage."""
    # Keras stores one value per completed epoch under these standard metric keys.
    values = history.history
    if "val_macro_f1" not in values:
        raise RuntimeError("Training history is missing required val_macro_f1 values.")
    selection_values = np.asarray(
        values.get("val_guarded_macro_f1", values["val_macro_f1"]),
        dtype=np.float64,
    )
    eligible_indices = np.flatnonzero(np.isfinite(selection_values))
    if eligible_indices.size == 0:
        return {
            "best_epoch": "",
            "training_loss": "",
            "training_accuracy": "",
            "validation_loss": "",
            "history_validation_accuracy": "",
            "history_validation_macro_f1": "",
            "history_guarded_macro_f1": "",
        }

    # Macro F1 is the grading objective; the guarded series excludes epochs that
    # fell below the stage's fixed validation-accuracy floor. Restricting argmax
    # to finite values prevents an ineligible ``-inf`` epoch from being reported.
    best_index = int(
        eligible_indices[np.argmax(selection_values[eligible_indices])]
    )

    def value_at(key):
        # Missing optional metrics remain blank instead of inventing a value.
        series = values.get(key, [])
        return float(series[best_index]) if len(series) > best_index else ""

    return {
        "best_epoch": best_index + 1,
        "training_loss": value_at("loss"),
        "training_accuracy": value_at("accuracy"),
        "validation_loss": value_at("val_loss"),
        "history_validation_accuracy": value_at("val_accuracy"),
        "history_validation_macro_f1": value_at("val_macro_f1"),
        "history_guarded_macro_f1": value_at("val_guarded_macro_f1"),
    }


def evaluate_and_log(
    model,
    val_ds,
    class_names,
    stage,
    epochs_this_stage,
    progress_csv,
    history,
    training_seconds,
    learning_rate,
    exact_validation_target=None,
    trainable_base_layers="",
    weight_decay=0.0,
    selection_accuracy_floor=0.0,
    class_weight_power=1.0,
    paired_hflip_probability=0.0,
    photometric_augmentation_probability=0.0,
):
    """Evaluate one stage and preserve the best accuracy-guarded macro F1."""
    report_dict, cm, macro_f1, macro_recall, category_metrics = evaluate_model_on_dataset(
        model, val_ds, class_names
    )
    val_accuracy_overall = overall_accuracy_from_report(report_dict)
    current_category_score = category_score(category_metrics)
    targets_met = category_targets_met(category_metrics)
    history_summary = summarize_training_history(history)

    print(f"\nEvaluation after {stage}:")
    print(f"overall exact accuracy: {val_accuracy_overall:.4f}")
    print_category_metrics(category_metrics)
    print(f"macro_f1={macro_f1:.4f}, macro_recall={macro_recall:.4f}")

    # This row is the "training diary" for later analysis. It records both the
    # assignment target metrics and the stricter exact metrics.
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "epochs_this_stage": epochs_this_stage,
        "best_epoch": history_summary["best_epoch"],
        "training_loss": history_summary["training_loss"],
        "training_accuracy": history_summary["training_accuracy"],
        "validation_loss": history_summary["validation_loss"],
        "history_validation_accuracy": history_summary["history_validation_accuracy"],
        "history_validation_macro_f1": history_summary["history_validation_macro_f1"],
        "history_guarded_macro_f1": history_summary["history_guarded_macro_f1"],
        "training_seconds": float(training_seconds),
        "learning_rate": float(learning_rate),
        "trainable_base_layers": trainable_base_layers,
        "weight_decay": float(weight_decay),
        "class_weight_power": float(class_weight_power),
        "paired_hflip_probability": float(paired_hflip_probability),
        "photometric_augmentation_probability": float(
            photometric_augmentation_probability
        ),
        "split_strategy": "class_stratified_physical_sign_groups",
        "random_seed": SEED,
        "val_accuracy": val_accuracy_overall,
        "category_rules_of_road": category_metrics["rules_of_road"]["group_accuracy"],
        "category_cautionary": category_metrics["cautionary"]["group_accuracy"],
        "exact_rules_of_road": category_metrics["rules_of_road"]["exact_accuracy"],
        "exact_cautionary": category_metrics["cautionary"]["exact_accuracy"],
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_recall),
        "selection_accuracy_floor": float(selection_accuracy_floor),
        "category_score": float(current_category_score),
        "targets_met": targets_met,
        "exact_validation_target": (
            float(exact_validation_target)
            if exact_validation_target is not None
            else ""
        ),
        "exact_validation_target_met": (
            val_accuracy_overall >= exact_validation_target
            if exact_validation_target is not None
            else ""
        ),
    }
    append_progress_row(Path(progress_csv), row)

    # The caller explicitly restores the guarded checkpoint before evaluation,
    # so these two values describe one concrete model rather than independent
    # maxima observed at different epochs.
    selected_validation_accuracy = val_accuracy_overall
    selected_macro_f1 = macro_f1

    return (
        report_dict,
        cm,
        macro_f1,
        macro_recall,
        category_metrics,
        selected_validation_accuracy,
        selected_macro_f1,
        targets_met,
    )


def save_final_outputs(
    model,
    val_ds,
    class_names,
    model_path,
    report_csv,
    confusion_csv,
    summary_csv,
):
    """Write the final model and all evaluation artifacts to disk."""
    # Re-evaluate at the end so final CSVs match the model that will be saved.
    report_dict, cm, macro_f1, macro_recall, category_metrics = evaluate_model_on_dataset(
        model, val_ds, class_names
    )
    val_accuracy_overall = overall_accuracy_from_report(report_dict)

    save_csv_report(report_dict, class_names, Path(report_csv))
    save_confusion_matrix(cm, class_names, Path(confusion_csv))
    save_summary(
        {
            # val_accuracy is the exact 43-class accuracy Keras users usually
            # expect. The next two category_* fields are the assignment targets.
            "val_accuracy": val_accuracy_overall,
            "category_rules_of_road": category_metrics["rules_of_road"]["group_accuracy"],
            "category_cautionary": category_metrics["cautionary"]["group_accuracy"],
            "exact_rules_of_road": category_metrics["rules_of_road"]["exact_accuracy"],
            "exact_cautionary": category_metrics["cautionary"]["exact_accuracy"],
            "macro_f1": macro_f1,
            "macro_recall": macro_recall,
            "targets_met": category_targets_met(category_metrics),
        },
        Path(summary_csv),
    )
    model.save(model_path)

    print("\nFinal evaluation:")
    print(f"overall exact accuracy: {val_accuracy_overall:.4f}")
    print_category_metrics(category_metrics)
    print(f"Saved final model to {model_path}")
    print(f"Saved report to {report_csv}")
    print(f"Saved confusion matrix to {confusion_csv}")
    print(f"Saved summary to {summary_csv}")


def train_until_targets(
    model,
    base_model,
    train_ds,
    val_ds,
    class_names,
    args,
):
    """Train the head, then run one stateful macro-F1-selected fine-tune stage."""
    best_validation_accuracy = -1.0
    best_macro_f1 = -1.0
    best_model_path = Path(args.best_model)

    print("\nStage 1: training the ResNet classification head")
    # Stage 1 is fast and stable: only the small classifier head learns while
    # ResNet stays frozen.
    stage_started = time.perf_counter()
    frozen_history = compile_and_train(
        model,
        train_ds,
        val_ds,
        initial_lr=args.initial_lr,
        epochs=args.frozen_epochs,
        callbacks=make_training_callbacks(
            checkpoint_path=best_model_path,
            accuracy_floor=0.0,
        ),
        jit_compile=args.xla,
    )
    frozen_seconds = time.perf_counter() - stage_started
    restore_selected_checkpoint(model, best_model_path)
    (
        _,
        _,
        _,
        _,
        category_metrics,
        best_validation_accuracy,
        best_macro_f1,
        targets_met,
    ) = evaluate_and_log(
        model,
        val_ds,
        class_names,
        "frozen_head",
        len(frozen_history.epoch),
        args.progress_csv,
        frozen_history,
        frozen_seconds,
        args.initial_lr,
        args.exact_validation_target,
        0,
        0.0,
        0.0,
        args.class_weight_power,
        args.paired_hflip_prob,
        args.photometric_aug_prob,
    )

    # Preserve the corrected frozen-head baseline before any fine-tuning changes it.
    baseline_model_path = Path(args.baseline_model)
    baseline_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(baseline_model_path)
    print(f"Saved grouped frozen-head baseline to {baseline_model_path}")

    # A zero-round run is an explicit baseline-only experiment.
    if args.fine_tune_rounds == 0:
        print("Baseline-only run requested; skipping fine-tuning.")
        return

    print("\nStage 2: continuous fine-tuning of the selected ResNet layers")
    total_fine_tune_epochs = args.fine_tune_rounds * args.fine_tune_block_epochs
    selection_accuracy_floor = max(
        0.0,
        best_validation_accuracy - args.max_accuracy_drop,
    )
    print(
        f"Macro-F1 selection accuracy floor: {selection_accuracy_floor:.4f} "
        f"(maximum allowed drop {args.max_accuracy_drop:.4f})"
    )
    # One compile and one fit preserve Adam moments plus LR/early-stop state for
    # the entire stage; the old block loop reset all three every few epochs.
    stage_started = time.perf_counter()
    fine_tune_history = fine_tune_block(
        model,
        base_model,
        train_ds,
        val_ds,
        epochs=total_fine_tune_epochs,
        lr=args.fine_tune_lr,
        trainable_base_layers=args.fine_tune_layers,
        weight_decay=args.weight_decay,
        callbacks=make_training_callbacks(
            checkpoint_path=best_model_path,
            accuracy_floor=selection_accuracy_floor,
            initial_best_macro_f1=best_macro_f1,
        ),
        jit_compile=args.xla,
    )
    fine_tune_seconds = time.perf_counter() - stage_started
    restore_selected_checkpoint(model, best_model_path)
    (
        _,
        _,
        _,
        _,
        category_metrics,
        best_validation_accuracy,
        best_macro_f1,
        targets_met,
    ) = evaluate_and_log(
        model,
        val_ds,
        class_names,
        "fine_tune_continuous",
        len(fine_tune_history.epoch),
        args.progress_csv,
        fine_tune_history,
        fine_tune_seconds,
        args.fine_tune_lr,
        args.exact_validation_target,
        args.fine_tune_layers,
        args.weight_decay,
        selection_accuracy_floor,
        args.class_weight_power,
        args.paired_hflip_prob,
        args.photometric_aug_prob,
    )

    if (
        best_validation_accuracy >= args.exact_validation_target
        and best_macro_f1 >= args.macro_f1_target
    ):
        print(
            "Both validation targets met: "
            f"accuracy={best_validation_accuracy:.4f}, macro_f1={best_macro_f1:.4f}"
        )
        return

    print(
        "Fine-tuning finished with best validation "
        f"accuracy={best_validation_accuracy:.4f} and "
        f"guarded_macro_f1={best_macro_f1:.4f}."
    )


def continue_fine_tuning_to_target(
    model,
    base_model,
    train_ds,
    val_ds,
    class_names,
    args,
):
    """Resume a grouped checkpoint and improve it toward the exact target."""
    best_model_path = Path(args.best_model)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Establish the checkpoint's starting score on the same fixed validation
    # split before changing any weights.
    (
        report_dict,
        _,
        macro_f1,
        macro_recall,
        category_metrics,
    ) = evaluate_model_on_dataset(model, val_ds, class_names)
    best_validation_accuracy = overall_accuracy_from_report(report_dict)
    best_macro_f1 = float(macro_f1)
    model.save(best_model_path)

    # Record the untouched starting point in the same experiment table as later
    # fine-tuning blocks so improvements remain directly comparable.
    append_progress_row(
        Path(args.progress_csv),
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": "resume_start",
            "epochs_this_stage": 0,
            "best_epoch": "",
            "training_loss": "",
            "training_accuracy": "",
            "validation_loss": "",
            "history_validation_accuracy": "",
            "history_validation_macro_f1": "",
            "history_guarded_macro_f1": "",
            "training_seconds": 0.0,
            "learning_rate": float(args.fine_tune_lr),
            "trainable_base_layers": args.fine_tune_layers,
            "weight_decay": float(args.weight_decay),
            "class_weight_power": float(args.class_weight_power),
            "paired_hflip_probability": float(args.paired_hflip_prob),
            "photometric_augmentation_probability": float(
                args.photometric_aug_prob
            ),
            "split_strategy": "class_stratified_physical_sign_groups",
            "random_seed": SEED,
            "val_accuracy": best_validation_accuracy,
            "category_rules_of_road": category_metrics["rules_of_road"]["group_accuracy"],
            "category_cautionary": category_metrics["cautionary"]["group_accuracy"],
            "exact_rules_of_road": category_metrics["rules_of_road"]["exact_accuracy"],
            "exact_cautionary": category_metrics["cautionary"]["exact_accuracy"],
            "macro_f1": float(macro_f1),
            "macro_recall": float(macro_recall),
            "selection_accuracy_floor": max(
                0.0,
                best_validation_accuracy - args.max_accuracy_drop,
            ),
            "category_score": float(category_score(category_metrics)),
            "targets_met": category_targets_met(category_metrics),
            "exact_validation_target": float(args.exact_validation_target),
            "exact_validation_target_met": (
                best_validation_accuracy >= args.exact_validation_target
            ),
        },
    )
    print(
        "Resume checkpoint validation: "
        f"accuracy={best_validation_accuracy:.4f}, macro_f1={best_macro_f1:.4f}"
    )

    # Avoid unnecessary work only when the checkpoint meets both the accuracy
    # safety goal and the macro-F1 grade target.
    if (
        best_validation_accuracy >= args.exact_validation_target
        and best_macro_f1 >= args.macro_f1_target
    ):
        print("Resume checkpoint already meets both validation targets.")
        return

    selection_accuracy_floor = max(
        0.0,
        best_validation_accuracy - args.max_accuracy_drop,
    )
    total_fine_tune_epochs = args.fine_tune_rounds * args.fine_tune_block_epochs
    print(
        f"Continuous resume for up to {total_fine_tune_epochs} epochs; "
        f"macro-F1 selection accuracy floor={selection_accuracy_floor:.4f}"
    )
    stage_started = time.perf_counter()
    history = fine_tune_block(
        model,
        base_model,
        train_ds,
        val_ds,
        epochs=total_fine_tune_epochs,
        lr=args.fine_tune_lr,
        trainable_base_layers=args.fine_tune_layers,
        weight_decay=args.weight_decay,
        callbacks=make_training_callbacks(
            checkpoint_path=best_model_path,
            accuracy_floor=selection_accuracy_floor,
            initial_best_macro_f1=best_macro_f1,
        ),
        jit_compile=args.xla,
    )
    training_seconds = time.perf_counter() - stage_started
    restore_selected_checkpoint(model, best_model_path)

    (
        _,
        _,
        _,
        _,
        _,
        best_validation_accuracy,
        best_macro_f1,
        _,
    ) = evaluate_and_log(
        model,
        val_ds,
        class_names,
        "resume_fine_tune_continuous",
        len(history.epoch),
        args.progress_csv,
        history,
        training_seconds,
        args.fine_tune_lr,
        args.exact_validation_target,
        args.fine_tune_layers,
        args.weight_decay,
        selection_accuracy_floor,
        args.class_weight_power,
        args.paired_hflip_prob,
        args.photometric_aug_prob,
    )

    print(
        "Resume run finished with best validation "
        f"accuracy={best_validation_accuracy:.4f} and "
        f"guarded_macro_f1={best_macro_f1:.4f}."
    )


def main():
    """Parse options, prepare data, train, then save final outputs."""
    global IMG_SIZE

    # Enforce GPU training here rather than during module import so helper tests
    # can run safely on machines without a configured accelerator.
    configure_training_gpu()

    parser = argparse.ArgumentParser(description="Train and evaluate ResNet50 on BYUI roadsigns")

    # Output/input paths. These defaults keep all generated artifacts beside the
    # script so they are easy to inspect after training.
    parser.add_argument("--data-dir", default="data/roadsigns", help="Directory to store downloaded dataset")
    parser.add_argument("--model-name", default="resnet_roadsigns_final.keras", help="Final saved Keras model file")
    parser.add_argument("--summary-csv", default="resnet_results_summary.csv", help="Summary results CSV")
    parser.add_argument("--report-csv", default="classification_report.csv", help="Classification report CSV")
    parser.add_argument("--confusion-csv", default="confusion_matrix.csv", help="Confusion matrix CSV")
    parser.add_argument("--progress-csv", default="training_progress.csv", help="Training progress log CSV")
    parser.add_argument("--best-model", default="best_resnet_roadsigns.keras", help="Path to save best model during fine-tuning")
    parser.add_argument(
        "--resume-model",
        default=None,
        help="Grouped Keras checkpoint to continue fine-tuning instead of starting fresh",
    )
    parser.add_argument(
        "--baseline-model",
        default="baseline_resnet_roadsigns_grouped.keras",
        help="Frozen-head grouped baseline saved before controlled fine-tuning",
    )
    parser.add_argument(
        "--split-manifest",
        default="diagnostics/group_split_manifest.csv",
        help="CSV proving each physical-sign group belongs to exactly one split",
    )

    # Performance knobs. Increase these only when CPU/RAM/VRAM have headroom.
    # Batch size mainly affects VRAM; parallel calls and prefetch mainly affect
    # CPU/RAM and how well the GPU stays fed.
    parser.add_argument("--img-size", type=int, default=IMG_SIZE[0], help="Square image size; 192 is faster, 224 may preserve more detail")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size; lower this if GPU memory is tight")
    parser.add_argument("--num-parallel-calls", type=int, default=MAX_PARALLEL_CALLS, help="Parallel image decode/augment calls")
    parser.add_argument("--prefetch-buffer", type=int, default=PREFETCH_BUFFER, help="Number of batches to prefetch")
    parser.add_argument(
        "--deterministic-data",
        dest="deterministic_data",
        action="store_true",
        default=TRAINING_DATASET_DETERMINISTIC,
        help="Use deterministic tf.data and TensorFlow operations",
    )
    parser.add_argument(
        "--non-deterministic-data",
        dest="deterministic_data",
        action="store_false",
        help="Allow nondeterministic input/ops for maximum throughput",
    )
    parser.add_argument("--disk-cache", dest="disk_cache", action="store_true", default=USE_DISK_CACHE, help="Cache decoded/resized images to disk")
    parser.add_argument("--no-disk-cache", dest="disk_cache", action="store_false", help="Disable tf.data disk cache")
    parser.add_argument("--cache-dir", default=None, help="Directory for tf.data cache files; defaults to data/roadsigns/.tf_cache")
    parser.add_argument("--rebuild-cache", action="store_true", help="Delete and rebuild cache files for this image size")
    parser.add_argument("--no-augment", action="store_true", help="Disable training augmentation so preprocessed training images can be cached")
    parser.add_argument(
        "--paired-hflip-prob",
        type=float,
        default=0.0,
        help="Training-only probability for horizontal flips with directional label remapping",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help="Inverse-frequency strength: 1=full, 0.5=square-root, 0=off",
    )
    parser.add_argument(
        "--photometric-aug-prob",
        type=float,
        default=0.0,
        help="Training-only probability for random gamma and exposure augmentation",
    )
    parser.add_argument("--turbo", action="store_true", help="Turbo profile: batch 128, parallel calls 24, prefetch 4")
    parser.add_argument("--balanced", action="store_true", help="Balanced profile: batch 16, parallel calls 8, prefetch 2")
    parser.add_argument("--low-memory", action="store_true", help="Conservative fallback: batch 8, parallel calls 4, prefetch 1")

    # Training schedule knobs. These are useful if you want to experiment with
    # longer/shorter runs without editing the file.
    parser.add_argument("--frozen-epochs", type=int, default=FROZEN_EPOCHS, help="Epochs for frozen ResNet head training")
    parser.add_argument("--fine-tune-rounds", type=int, default=MAX_FINE_TUNE_ROUNDS, help="Maximum fine-tune blocks")
    parser.add_argument("--fine-tune-block-epochs", type=int, default=FINE_TUNE_BLOCK_EPOCHS, help="Epochs per fine-tune block")
    parser.add_argument("--initial-lr", type=float, default=INITIAL_LR, help="Learning rate for frozen-head training")
    parser.add_argument("--fine-tune-lr", type=float, default=FINE_TUNE_LR, help="Learning rate for fine-tuning")
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=abs(FINE_TUNE_AT),
        help="Number of final ResNet layers eligible for fine-tuning",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW decoupled weight decay for fine-tuning; zero uses Adam",
    )
    parser.add_argument(
        "--exact-validation-target",
        type=float,
        default=0.88,
        help="Exact zero-overlap validation accuracy goal reported with macro F1",
    )
    parser.add_argument(
        "--macro-f1-target",
        type=float,
        default=0.886,
        help="Grouped macro-F1 goal; 0.886 corresponds to roughly an 80 grader score",
    )
    parser.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=0.01,
        help="Largest validation-accuracy drop allowed for macro-F1 checkpoint selection",
    )

    # Assignment target knobs. These affect the broad category stopping rule,
    # not the exact 43-class Keras accuracy shown during each epoch.
    parser.add_argument("--target-rules", type=float, default=TARGET_RULES_OF_ROAD, help="Rules-of-road broad category target")
    parser.add_argument("--target-caution", type=float, default=TARGET_CAUTIONARY, help="Cautionary broad category target")

    # Mixed precision uses float16 for many GPU operations and float32 where
    # precision matters. It often improves speed and lowers VRAM use on modern
    # NVIDIA GPUs, so it is on by default here.
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true", default=True, help="Use mixed precision to reduce GPU memory on supported NVIDIA GPUs")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false", help="Disable mixed precision and use float32")
    parser.add_argument("--xla", dest="xla", action="store_true", default=USE_XLA, help="Enable XLA/JIT compilation for potentially faster GPU execution")
    parser.add_argument("--no-xla", dest="xla", action="store_false", help="Disable XLA/JIT compilation")

    # Optional experiment mode. This is separate from the main training loop
    # because it trains several short models and can add noticeable time.
    parser.add_argument("--lr-sweep", action="store_true", help="Run a short frozen-head learning-rate sweep before full training")
    parser.add_argument("--lr-sweep-only", action="store_true", help="Run only the learning-rate sweep and exit")
    args = parser.parse_args()

    # Seed every random source used by Python, NumPy, Keras, augmentation, and splitting.
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    if args.deterministic_data:
        # Deterministic kernels make repeated controlled comparisons more reproducible.
        tf.config.experimental.enable_op_determinism()
    print(f"Reproducibility seed: {SEED}")
    print(f"Deterministic TensorFlow operations: {args.deterministic_data}")

    if args.turbo:
        args.batch_size = max(args.batch_size, TURBO_BATCH_SIZE)
        args.num_parallel_calls = max(args.num_parallel_calls, TURBO_PARALLEL_CALLS)
        args.prefetch_buffer = max(args.prefetch_buffer, TURBO_PREFETCH_BUFFER)

    if args.balanced:
        args.batch_size = min(args.batch_size, BALANCED_BATCH_SIZE)
        args.num_parallel_calls = min(args.num_parallel_calls, BALANCED_PARALLEL_CALLS)
        args.prefetch_buffer = min(args.prefetch_buffer, BALANCED_PREFETCH_BUFFER)

    if args.low_memory:
        # Low-memory mode is intentionally conservative. It caps the knobs rather
        # than overwriting them upward, so custom smaller values still work.
        args.batch_size = min(args.batch_size, LOW_MEMORY_BATCH_SIZE)
        args.num_parallel_calls = min(args.num_parallel_calls, LOW_MEMORY_PARALLEL_CALLS)
        args.prefetch_buffer = min(args.prefetch_buffer, LOW_MEMORY_PREFETCH_BUFFER)

    if args.img_size < 96:
        raise ValueError("--img-size should be at least 96 for this transfer-learning setup.")
    if not 0.0 < args.exact_validation_target <= 1.0:
        raise ValueError("--exact-validation-target must be in the interval (0, 1].")
    if args.fine_tune_layers < 1:
        raise ValueError("--fine-tune-layers must be at least 1.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be zero or positive.")
    if not 0.0 <= args.paired_hflip_prob <= 1.0:
        raise ValueError("--paired-hflip-prob must be between 0 and 1.")
    if not 0.0 <= args.class_weight_power <= 1.0:
        raise ValueError("--class-weight-power must be between 0 and 1.")
    if not 0.0 <= args.photometric_aug_prob <= 1.0:
        raise ValueError("--photometric-aug-prob must be between 0 and 1.")
    if not 0.0 < args.macro_f1_target <= 1.0:
        raise ValueError("--macro-f1-target must be in the interval (0, 1].")
    if not 0.0 <= args.max_accuracy_drop < 1.0:
        raise ValueError("--max-accuracy-drop must be in the interval [0, 1).")
    if args.resume_model and args.lr_sweep:
        raise ValueError("Learning-rate sweep is only supported for a fresh model.")
    IMG_SIZE = (args.img_size, args.img_size)

    # Let command-line targets update the global target dictionary used by the
    # metric printing and early-stopping checks.
    CATEGORY_TARGETS["rules_of_road"] = args.target_rules
    CATEGORY_TARGETS["cautionary"] = args.target_caution

    if args.mixed_precision:
        # If this causes driver/hardware issues, rerun with --no-mixed-precision.
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled.")
    print(f"XLA enabled: {args.xla}")
    print(
        "Performance settings: "
        f"img_size={IMG_SIZE[0]}x{IMG_SIZE[1]}, "
        f"batch_size={args.batch_size}, "
        f"parallel_calls={args.num_parallel_calls}, "
        f"prefetch={args.prefetch_buffer}, "
        f"disk_cache={args.disk_cache}, "
        f"augment={not args.no_augment}, "
        f"paired_hflip={args.paired_hflip_prob}, "
        f"photometric_aug={args.photometric_aug_prob}, "
        f"class_weight_power={args.class_weight_power}"
    )
    print(
        "Speed note: frozen-head training uses less VRAM than fine-tuning; "
        "compare seconds per epoch after restarting with the new batch size."
    )

    data_dir = Path(args.data_dir).resolve()
    os.makedirs(data_dir, exist_ok=True)

    for filename, url in DATA_URLS:
        # These functions are idempotent enough for repeated runs: existing zips
        # are reused, and extraction/merge tolerate data already being present.
        download_file(url, data_dir / filename)
        extract_zip(data_dir / filename, data_dir)

    # The downloaded zips may unpack into separate training folders; merge them
    # so each numeric class ID has one folder.
    merge_training_dirs(data_dir)
    train_ds, val_ds, classes = prepare_datasets(
        data_dir,
        batch_size=args.batch_size,
        num_parallel_calls=args.num_parallel_calls,
        prefetch_buffer=args.prefetch_buffer,
        training_deterministic=args.deterministic_data,
        disk_cache=args.disk_cache,
        cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else None,
        rebuild_cache=args.rebuild_cache,
        augment=not args.no_augment,
        class_weight_power=args.class_weight_power,
        paired_hflip_probability=args.paired_hflip_prob,
        photometric_augmentation_probability=args.photometric_aug_prob,
        split_manifest=Path(args.split_manifest).resolve(),
    )

    if args.lr_sweep:
        # This is optional because it trains several small models and adds time.
        lr_results_csv = "learning_rate_results.csv"
        best_lr_model = Path(args.best_model).with_suffix(".lr_best.keras")
        best_lr, best_validation_accuracy, best_lr_macro_f1 = run_learning_rate_experiment(
            train_ds,
            val_ds,
            classes,
            lr_results_csv,
            str(best_lr_model),
            epochs=3,
        )
        print("Learning rate search concluded.")
        print(f"Best learning rate: {best_lr}")
        print(f"Best grouped macro F1: {best_lr_macro_f1:.4f}")
        print(f"Best grouped exact validation accuracy: {best_validation_accuracy:.4f}")
        print(f"Results written to {lr_results_csv}")
        if best_lr is not None:
            # If the sweep found a winner, use it for the real frozen-head stage.
            args.initial_lr = best_lr
        if args.lr_sweep_only:
            return

    if args.resume_model:
        # Resume only a checkpoint trained on this corrected grouped split. Input
        # and output assertions prevent accidental use of an incompatible model.
        resume_path = Path(args.resume_model).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")
        print(f"Loading grouped checkpoint for continuation: {resume_path}")
        model = tf.keras.models.load_model(resume_path, compile=False)
        if tuple(model.input_shape[-3:]) != (*IMG_SIZE, 3):
            raise AssertionError(
                f"Resume model input {model.input_shape} does not match {IMG_SIZE}"
            )
        if int(model.output_shape[-1]) != len(classes):
            raise AssertionError("Resume model output width does not match class count")
        base_model = find_resnet_base_model(model)
        continue_fine_tuning_to_target(
            model,
            base_model,
            train_ds,
            val_ds,
            classes,
            args,
        )
    else:
        # A fresh run starts from ImageNet ResNet weights and trains its head
        # before entering the controlled fine-tuning stages.
        model, base_model = build_model(len(classes))
        train_until_targets(model, base_model, train_ds, val_ds, classes, args)

    best_model_path = Path(args.best_model)
    if best_model_path.exists():
        # Use the checkpoint with the best accuracy-guarded macro F1, not
        # necessarily the weights from the final epoch.
        print(f"Loading best guarded-macro-F1 model from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path, compile=False)

    save_final_outputs(
        model,
        val_ds,
        classes,
        args.model_name,
        args.report_csv,
        args.confusion_csv,
        args.summary_csv,
    )


if __name__ == "__main__":
    main()
