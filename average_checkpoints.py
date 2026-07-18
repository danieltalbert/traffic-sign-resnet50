#!/usr/bin/env python3
"""Create one model by averaging two compatible fine-tuned checkpoints.

Checkpoint averaging is a fixed, non-training stabilization step. It can smooth
small weight oscillations near a validation plateau without using holdout labels
or taking additional gradient steps.
"""

# argparse keeps both source checkpoints, blend weight, and output explicit.
import argparse
# Path handles model locations consistently in WSL.
from pathlib import Path

# NumPy performs deterministic elementwise weight averaging.
import numpy as np
# TensorFlow loads and saves the compatible Keras model architecture.
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    """Define the two checkpoints and fixed averaging ratio."""
    parser = argparse.ArgumentParser(description="Average two compatible Keras checkpoints.")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weight-a", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Validate compatibility, average every tensor, and save one inference model."""
    args = parse_args()
    if not 0.0 <= args.weight_a <= 1.0:
        raise ValueError("--weight-a must be between 0 and 1")

    # compile=False ignores optimizer state because only learned model weights matter.
    model_a = tf.keras.models.load_model(Path(args.model_a).resolve(), compile=False)
    model_b = tf.keras.models.load_model(Path(args.model_b).resolve(), compile=False)
    if model_a.input_shape != model_b.input_shape or model_a.output_shape != model_b.output_shape:
        raise AssertionError("Checkpoint input/output shapes do not match")

    weights_a = model_a.get_weights()
    weights_b = model_b.get_weights()
    if len(weights_a) != len(weights_b):
        raise AssertionError("Checkpoint weight counts do not match")

    # A fixed 50/50 default treats the two adjacent validation peaks symmetrically.
    weight_b = 1.0 - args.weight_a
    averaged_weights = []
    for index, (array_a, array_b) in enumerate(zip(weights_a, weights_b)):
        if array_a.shape != array_b.shape:
            raise AssertionError(f"Weight tensor {index} shapes do not match")
        averaged_weights.append(
            np.asarray(array_a * args.weight_a + array_b * weight_b, dtype=array_a.dtype)
        )

    # Reuse model A's identical architecture and replace it with the averaged tensors.
    model_a.set_weights(averaged_weights)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model_a.save(output)
    print(f"Saved averaged checkpoint: {output}")
    print(f"Model A weight: {args.weight_a:.2f}; model B weight: {weight_b:.2f}")
    print(f"Weight tensors averaged: {len(averaged_weights)}")


# Direct execution creates the averaged model; import leaves helpers available for tests.
if __name__ == "__main__":
    main()
