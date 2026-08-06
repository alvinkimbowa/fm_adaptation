# Progress

- Added a YAML-driven 2D nnU-Net dataset framework for SAM3 Perception Encoder linear and nonlinear probing.
- Added training logs, best/final probe-only checkpoints, original-resolution inference, Dice/MASD metrics, and cross-dataset HTML reports.
- Verified preprocessing, prediction restoration, metrics, and CLI imports in the SAM3 environment.
- Added reusable float16 PE feature caching and validation-based early stopping for low-compute probing.
