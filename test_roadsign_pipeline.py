#!/usr/bin/env python3
"""Focused regression tests for the corrected road-sign data contract."""

# tempfile creates isolated fake holdout directories without touching project data.
import tempfile
# unittest provides a dependency-free test runner.
import unittest
# Path creates representative training and holdout filenames.
from pathlib import Path

# NumPy compares deterministic split indices exactly.
import numpy as np

# Test the same shared helpers used by training, auditing, and prediction.
from roadsign_pipeline import (
    HORIZONTAL_FLIP_CLASS_PAIRS,
    NUM_CLASSES,
    build_grouped_split,
    class_index_mapping,
    collect_holdout_images,
    horizontal_flip_label_map,
    training_group_id,
)


class GroupedSplitTests(unittest.TestCase):
    """Prove physical-sign groups remain intact and reproducible."""

    @staticmethod
    def synthetic_samples():
        """Build five three-frame groups for every one of the 43 classes."""
        paths = []
        labels = []
        groups = []
        for label in range(NUM_CLASSES):
            for group_number in range(5):
                group = f"{label:02d}:{group_number:05d}"
                for frame_number in range(3):
                    paths.append(
                        f"/{label:05d}/{group_number:05d}_{frame_number:05d}.jpg"
                    )
                    labels.append(label)
                    groups.append(group)
        return paths, labels, groups

    def test_grouped_split_has_zero_overlap_and_all_classes(self):
        """Every frame from one synthetic sign must enter the same partition."""
        paths, labels, groups = self.synthetic_samples()
        split = build_grouped_split(paths, labels, groups, seed=123)

        self.assertFalse(set(split.train_groups) & set(split.val_groups))
        self.assertEqual(set(split.train_labels.tolist()), set(range(NUM_CLASSES)))
        self.assertEqual(set(split.val_labels.tolist()), set(range(NUM_CLASSES)))
        self.assertEqual(len(split.train_paths) + len(split.val_paths), len(paths))

    def test_grouped_split_is_reproducible(self):
        """The same seed must produce exactly the same train/validation rows."""
        paths, labels, groups = self.synthetic_samples()
        first = build_grouped_split(paths, labels, groups, seed=123)
        second = build_grouped_split(paths, labels, groups, seed=123)

        np.testing.assert_array_equal(first.train_indices, second.train_indices)
        np.testing.assert_array_equal(first.val_indices, second.val_indices)


class MappingAndOrderingTests(unittest.TestCase):
    """Prove filename parsing, class mapping, and holdout order assumptions."""

    def test_training_group_includes_class_and_track(self):
        """Track numbers restart per class, so both pieces belong in the key."""
        path = Path("/training/00022/00006_00019.jpg")
        self.assertEqual(training_group_id(path, 22), "22:00006")

    def test_class_mapping_is_identity_zero_through_42(self):
        """No encoder or off-by-one conversion may alter actual class labels."""
        mapping = class_index_mapping()
        self.assertEqual(mapping, {index: index for index in range(NUM_CLASSES)})

    def test_horizontal_flip_mapping_swaps_only_directional_pairs(self):
        """Mirroring must atomically exchange only known left/right labels."""
        mapping = horizontal_flip_label_map()
        paired_labels = {
            label
            for pair in HORIZONTAL_FLIP_CLASS_PAIRS
            for label in pair
        }

        for left_label, right_label in HORIZONTAL_FLIP_CLASS_PAIRS:
            self.assertEqual(int(mapping[left_label]), right_label)
            self.assertEqual(int(mapping[right_label]), left_label)

        for label in set(range(NUM_CLASSES)) - paired_labels:
            self.assertEqual(int(mapping[label]), label)

    def test_horizontal_flip_mapping_is_involutive(self):
        """Applying the label remap twice must recover every original class."""
        mapping = horizontal_flip_label_map()
        np.testing.assert_array_equal(mapping[mapping], np.arange(NUM_CLASSES))

    def test_holdout_collection_uses_numeric_contiguous_order(self):
        """Creation order must not affect the final prediction row order."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for filename in ("00002.jpg", "00000.jpg", "00003.jpg", "00001.jpg"):
                (root / filename).touch()

            ordered = collect_holdout_images(root)
            self.assertEqual(
                [path.name for path in ordered],
                ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg"],
            )

    def test_holdout_collection_rejects_missing_numeric_rows(self):
        """A filename gap must fail before predictions can shift against Canvas rows."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "00000.jpg").touch()
            (root / "00002.jpg").touch()

            with self.assertRaises(AssertionError):
                collect_holdout_images(root)


# Direct execution runs the same suite as ``python -m unittest``.
if __name__ == "__main__":
    unittest.main()
