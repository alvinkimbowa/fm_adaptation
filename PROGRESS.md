# Progress

- Added a YAML-driven 2D nnU-Net dataset framework for SAM3 Perception Encoder linear and nonlinear probing.
- Added training logs, best/final probe-only checkpoints, original-resolution inference, Dice/MASD metrics, and cross-dataset HTML reports.
- Verified preprocessing, prediction restoration, metrics, and CLI imports in the SAM3 environment.
- Added reusable float16 PE feature caching and validation-based early stopping for low-compute probing.
- Completed fold-0 PE linear-probe training for BUSBRA, KidneyUS, and MMOTU; retained per-epoch loss/Dice histories and best/final probe checkpoints.
- Baseline cross-dataset analysis identified small-lesion oversegmentation as the main MMOTU failure; selected a minimal nonlinear decoder as one targeted test.
- Added full PE fine-tuning from the best linear probe with bf16, activation checkpointing, gradient accumulation, separate encoder/probe learning rates, and full best/final checkpoints.
- Corrected reports to include in-domain validation and macro-average cross-domain dataset statistics; added machine-readable summary CSV output.
- MMOTU nonlinear probe improved in-domain Dice 0.726→0.795 and CEUS Dice 0.653→0.686 while reducing MASD; propagated the unchanged decoder test to BUSBRA and KidneyUS.
- Completed all LP, nonlinear, and full-fine-tuning runs plus cross-dataset inference. Full fine-tuning achieved cross-domain Dice 0.795 BUSBRA macro, 0.931 KidneyUS, and 0.744 MMOTU CEUS.
- Final tables are in `models/cross_dataset_report.html` and `.csv`; all nine runs passed checkpoint/history audits.
- Split the HTML report into separate BUSBRA, KidneyUS, and MMOTU table sections.
- Added available nnU-Net runs to the corresponding cross-dataset comparison tables.
- Renamed PE report rows to SAM3 and ordered LP, NLP, then LP + FT.
