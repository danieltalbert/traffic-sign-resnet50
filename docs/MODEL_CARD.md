# Model card: leakage-aware traffic-sign ResNet50

## Model details

- Task: single-label image classification
- Classes: 43 numeric traffic-sign categories
- Backbone: ImageNet-pretrained ResNet50
- Input: 96 x 96 RGB images
- Head: global average pooling, dropout, and a 43-way softmax classifier
- Framework: TensorFlow 2.21 / Keras 3
- Selection target: macro F1 on a physical-sign-grouped validation partition, guarded against a large accuracy regression

## Intended use

This project demonstrates transfer learning, class-imbalance handling, leakage-aware validation, deterministic inference, and sign-safe augmentation in an educational computer-vision setting. It is not designed or validated for safety-critical driving decisions.

## Evaluation design

Frames from one physical sign are highly correlated. The original image-level split leaked 1,306 of 1,307 sign groups across train and validation. The corrected evaluation assigns each `CLASS:TRACK` group wholly to one partition with seed 123.

| Partition | Images | Physical-sign groups |
|---|---:|---:|
| Train | 31,379 | 1,046 |
| Validation | 7,830 | 261 |

The split contains all 43 classes in both partitions and has zero group overlap.

## Honest performance

| System | Grouped accuracy | Grouped macro F1 |
|---|---:|---:|
| Initial corrected model | 70.45% | 64.63% |
| Guarded macro-F1 checkpoint | 91.03% | 89.26% |
| Guarded checkpoint + six spatial views | 92.12% | 90.39% |
| Exposure-trained honest system | **92.90%** | **91.29%** |

An additional 201-image reference check produced 190/201 correct for the honest system. This small reference set is supplementary and is not a substitute for the 7,830-image grouped validation partition.

## Calibration disclosure

A research-only post-hoc experiment optimized a single 43-class score-bias vector directly against the 201 known reference labels. It achieved 200/201 on those same labels, but its grouped validation performance was lower at 91.63% accuracy and 89.91% macro F1.

The 200/201 number is in-sample. It must not be interpreted as expected performance on unseen images or as a known score for the remaining holdout set.

## Training and inference safeguards

- Deterministic group splitting and TensorFlow seeds
- Exact class mapping from numeric folders `0..42`
- One shared RGB, bilinear-resize, and ResNet preprocessing path
- Direction-aware horizontal flips with paired label remapping
- Square-root class weighting for imbalance
- Fixed multi-view inference with probability averaging
- Assertions for image order, output shape, class range, and CSV length
- Tests for split isolation, preprocessing, metrics, callbacks, and calibration math

## Limitations

- The validation partition comes from the same overall dataset distribution as training.
- Traffic signs are cropped classification inputs; the project does not detect signs in full road scenes.
- Real-world shifts in weather, geography, camera hardware, occlusion, and sign standards were not measured.
- The full holdout ground truth was unavailable, so full-holdout accuracy and macro F1 cannot be claimed.
- Model checkpoints and datasets are omitted because of size, source licensing, and course-integrity constraints.

## Reproducibility

The verified environment used Python 3.12.13, TensorFlow 2.21.0, Keras 3.15.0, NumPy 2.5.0, SciPy 1.18.0, scikit-learn 1.9.0, and Pillow 12.2.0. Dependency pins are recorded in `requirements.txt`; exact GPU kernels and floating-point reductions may still cause small platform-specific differences.
