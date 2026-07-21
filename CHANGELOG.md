# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-07-20

### Added

- Leakage-aware, class-stratified physical-sign group validation.
- Reproducible ResNet50 training, progressive fine-tuning, and deterministic inference.
- Sign-safe spatial and photometric test-time augmentation.
- Honest grouped accuracy and macro-F1 reporting across all 43 classes.
- Model card, publication-safe diagrams, aggregate experiment results, and 19 focused tests.
- CPU GitHub Actions verification that uses synthetic fixtures and requires no private dataset.

### Publication boundary

- Raw images, holdout data, course answer files, model checkpoints, probability caches,
  and submission exports are intentionally excluded.
- The MIT license covers original project code and documentation only.

[1.0.0]: https://github.com/danieltalbert/traffic-sign-resnet50/releases/tag/v1.0.0
