"""Focused tests for calibration artifact dispatch and probability math."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from predict_mini_calibrated_submission import apply_calibration
from roadsign_pipeline import NUM_CLASSES


class CalibrationExportTests(unittest.TestCase):
    """Protect the dynamic calibration rule used by the promoted export."""

    def setUp(self) -> None:
        # Use strictly positive normalized probabilities so expected log blends
        # can be computed without depending on clipping edge cases.
        macro = np.arange(1, NUM_CLASSES + 1, dtype=np.float32)
        spatial = np.arange(NUM_CLASSES, 0, -1, dtype=np.float32)
        self.macro = (macro / macro.sum())[None, :]
        self.spatial = (spatial / spatial.sum())[None, :]
        self.photo_systems = {
            "photo_selected": self.spatial,
            "photo_spatial": self.spatial,
            "photo_bottom_right": self.spatial,
        }

    def _artifact(self, directory: str, base_name: str, margin: float = 0.05) -> Path:
        path = Path(directory) / "calibration.npz"
        np.savez(
            path,
            base_name=np.asarray(base_name),
            bias=np.zeros(NUM_CLASSES, dtype=np.float64),
            calibrated_mini_correct=np.asarray(200),
            calibration_margin=np.asarray(margin),
        )
        return path

    def test_geometric_base_dispatch_matches_weighted_log_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(
                directory,
                "geometric__macro_bottom_right__photo_spatial_0.30",
            )
            actual = apply_calibration(self.macro, self.photo_systems, artifact)

        scores = 0.70 * np.log(self.macro) + 0.30 * np.log(self.spatial)
        expected = np.exp(scores - scores.max(axis=1, keepdims=True))
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_artifact_without_safety_margin_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(
                directory,
                "geometric__macro_bottom_right__photo_spatial_0.30",
                margin=0.0,
            )
            with self.assertRaisesRegex(AssertionError, "margin"):
                apply_calibration(self.macro, self.photo_systems, artifact)


if __name__ == "__main__":
    unittest.main()
