#!/usr/bin/env python3
"""Focused TensorFlow tests for macro-F1 selection and safe augmentation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from roadsign_pipeline import horizontal_flip_label_map
from resnet_roadsigns import (
    GuardedMacroF1,
    SparseMacroF1,
    apply_horizontal_flip_mask,
    apply_paired_random_horizontal_flip,
    apply_random_photometric_augmentation,
    compute_class_weights,
    make_training_callbacks,
    restore_selected_checkpoint,
)


class SparseMacroF1Tests(unittest.TestCase):
    """Keep the epoch metric numerically aligned with sklearn and the grader."""

    def test_sparse_macro_f1_matches_sklearn(self):
        true_labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        predicted_labels = np.array([0, 1, 1, 1, 2, 1], dtype=np.int32)
        probabilities = tf.one_hot(predicted_labels, depth=3)

        metric = SparseMacroF1(num_classes=3)
        metric.update_state(true_labels, probabilities)

        expected = f1_score(true_labels, predicted_labels, average="macro")
        self.assertAlmostEqual(float(metric.result().numpy()), expected, places=6)

    def test_guarded_macro_f1_enforces_accuracy_floor(self):
        callback = GuardedMacroF1(accuracy_floor=0.85)
        accepted = {"val_macro_f1": 0.80, "val_accuracy": 0.85}
        callback.on_epoch_end(0, accepted)
        self.assertEqual(accepted["val_guarded_macro_f1"], 0.80)

        rejected = {"val_macro_f1": 0.90, "val_accuracy": 0.84}
        callback.on_epoch_end(1, rejected)
        self.assertEqual(rejected["val_guarded_macro_f1"], -np.inf)

    def test_guarded_metric_is_recorded_and_checkpoint_serializes(self):
        # Exercise the real callback order on a tiny model before committing a
        # long GPU run to a checkpoint monitor that Keras might not recognize.
        inputs = tf.keras.Input(shape=(2,))
        outputs = tf.keras.layers.Dense(3, activation="softmax")(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-2),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                SparseMacroF1(num_classes=3),
            ],
        )
        features = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, -1.0], [-0.9, -1.1]],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)

        with TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "guarded.keras"
            history = model.fit(
                features,
                labels,
                validation_data=(features, labels),
                epochs=2,
                verbose=0,
                callbacks=make_training_callbacks(
                    checkpoint_path=checkpoint,
                    accuracy_floor=0.0,
                ),
            )

            self.assertIn("val_guarded_macro_f1", history.history)
            self.assertTrue(checkpoint.exists())
            restored = tf.keras.models.load_model(checkpoint, compile=False)
            self.assertEqual(restored.output_shape[-1], 3)

            # Prove that a full .keras checkpoint can safely restore only the
            # weights into the existing live model used by later stages.
            expected = restored.predict(features, verbose=0)
            model.set_weights([np.zeros_like(value) for value in model.get_weights()])
            restore_selected_checkpoint(model, checkpoint)
            actual = model.predict(features, verbose=0)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

            callbacks = make_training_callbacks(accuracy_floor=0.8)
            self.assertEqual(callbacks[-2].monitor, "val_macro_f1")
            self.assertEqual(callbacks[-1].monitor, "val_macro_f1")


class DirectionalFlipTests(unittest.TestCase):
    """Ensure pixels and paired labels always change together."""

    def setUp(self):
        self.label_map = tf.constant(horizontal_flip_label_map(), dtype=tf.int32)
        # Left and right columns differ so an unintended flip is easy to detect.
        self.images = tf.constant(
            [
                [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
                [[[7.0], [8.0], [9.0]], [[10.0], [11.0], [12.0]]],
            ]
        )

    def test_requested_mask_flips_only_eligible_rows(self):
        labels = tf.constant([19, 0], dtype=tf.int32)
        requested = tf.constant([True, True])

        images, output_labels = apply_horizontal_flip_mask(
            self.images,
            labels,
            requested,
            self.label_map,
        )

        np.testing.assert_array_equal(images[0].numpy(), self.images[0, :, ::-1, :].numpy())
        np.testing.assert_array_equal(images[1].numpy(), self.images[1].numpy())
        np.testing.assert_array_equal(output_labels.numpy(), [20, 0])

    def test_probability_zero_changes_nothing(self):
        labels = tf.constant([33, 38], dtype=tf.int32)
        images, output_labels = apply_paired_random_horizontal_flip(
            self.images,
            labels,
            0.0,
            self.label_map,
        )
        np.testing.assert_array_equal(images.numpy(), self.images.numpy())
        np.testing.assert_array_equal(output_labels.numpy(), labels.numpy())


class PhotometricAugmentationTests(unittest.TestCase):
    """Keep exposure augmentation optional, bounded, and shape-preserving."""

    def test_probability_zero_preserves_pixels_exactly(self):
        images = tf.constant([[[[0.0], [64.0], [255.0]]]], dtype=tf.float32)
        output = apply_random_photometric_augmentation(images, probability=0.0)
        np.testing.assert_array_equal(output.numpy(), images.numpy())

    def test_probability_one_preserves_shape_and_pixel_range(self):
        images = tf.fill([4, 3, 3, 3], 64.0)
        output = apply_random_photometric_augmentation(images, probability=1.0)
        self.assertEqual(output.shape, images.shape)
        self.assertGreaterEqual(float(tf.reduce_min(output).numpy()), 0.0)
        self.assertLessEqual(float(tf.reduce_max(output).numpy()), 255.0)


class ClassWeightTests(unittest.TestCase):
    """Verify the sweep can move safely between full, mild, and no weighting."""

    def test_class_weight_power_zero_returns_ones(self):
        labels = np.array([0, 1, 1, 1], dtype=np.int32)
        weights = compute_class_weights(labels, num_classes=2, power=0.0)
        np.testing.assert_allclose(weights, np.ones(2), rtol=0, atol=0)

    def test_square_root_weights_are_milder_than_full_inverse(self):
        labels = np.array([0, 1, 1, 1], dtype=np.int32)
        full = compute_class_weights(labels, num_classes=2, power=1.0)
        mild = compute_class_weights(labels, num_classes=2, power=0.5)
        self.assertLess(mild[0], full[0])
        self.assertGreater(mild[1], full[1])


if __name__ == "__main__":
    unittest.main()
