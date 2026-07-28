# Contributing

Thanks for helping improve this leakage-aware traffic-sign case study. Small,
focused pull requests with reproducible evidence are easiest to review.

## Before you start

1. Read the [model card](docs/MODEL_CARD.md) and the README's
   [data and artifact policy](README.md#data-and-artifact-policy).
2. Open an issue before proposing a new model family, evaluation target, or data
   source so the experiment boundary is explicit.
3. Never commit raw course data, holdout images, answer files, checkpoints,
   prediction exports, credentials, or machine-specific absolute paths.

## Local checks

Create the documented Python environment, then run:

```bash
python -m unittest -v \
  test_roadsign_pipeline.py \
  test_resnet_training_tools.py \
  test_mini_calibrated_submission.py
python audit_roadsigns.py --help
git diff --check
```

Changes to splitting, preprocessing, label remapping, or selection logic should
include a focused regression test. Performance claims must name the exact
evaluation boundary and distinguish grouped validation from any fitted
calibration reference.

## Pull requests

Explain the problem, the change, and how you verified it. Include aggregate
metrics or publication-safe visuals when behavior changes. Keep unrelated
cleanup out of the same pull request.
