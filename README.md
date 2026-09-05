# Foundation-model adaptation

This repository evaluates frozen foundation-model features for 2D medical image
segmentation. Datasets use the nnU-Net raw-data layout, but training and inference
are independent of nnU-Net.

The first model adapter uses SAM3's **Perception Encoder (PE)** through SAM3's
official `build_sam3_image_model` entry point. A null checkpoint in the YAML lets
SAM3 download and cache its official Hugging Face checkpoint. Images retain their
aspect ratio, are padded to PE's 1008×1008 input, and predictions are restored to
their original pixel size.

Run a complete linear-probe experiment:

```bash
bash scripts/run_exp.sh configs/sam3_linear.yaml
```

Run these commands from an environment containing SAM3 and its training extras
(OpenCV, PyYAML, SciPy). The linked SAM3 checkout's `.venv` is suitable; activate
it before launching an experiment.

Or run stages separately:

```bash
PYTHONPATH=src python -m fm_adaptation.training --config configs/sam3_linear.yaml
PYTHONPATH=src python -m fm_adaptation.predict --config configs/sam3_linear.yaml
PYTHONPATH=src python -m fm_adaptation.report \
    --results-dir models \
    --nnunet-results-dir ../nnUNet_fork/data/nnUNet_results
```

Full fine-tuning is the same command against a config that sets `train_encoder` and
names the probe run it starts from in `init_from`. These write `linear_finetune` and
`nonlinear_finetune` runs beside the probes they continue:

```bash
PYTHONPATH=src python -m fm_adaptation.training --config configs/sam3_linear_finetune_busbra.yaml
PYTHONPATH=src python -m fm_adaptation.training --config configs/sam3_nonlinear_finetune_busbra.yaml
```

Runs are stored under
`models/{foundation_model}/{training_dataset}/{probe}/fold_{fold}`. Cross-dataset
predictions and per-case metrics live below each run's `test/{dataset}` directory;
the report command creates `models/cross_dataset_report.html`. Pass an nnU-Net
results directory to include its available `test/*/metrics.csv` results as
comparison rows in each dataset-family table.

Neurite reports include independent annotator checkboxes for Yvonne (Coco, Yvonne,
Tanya) and Yvonne_b2 (Queena, Sarah). All start checked. Changes update Dice,
clDice, annotation counts, averages and ranking highlights directly in the HTML,
including the row's own Test results. Selecting none shows no measurements for
that source. The controls work offline; selections reset on reload. The generated
CSV always contains all annotators. Counts include pooled folds and scale variants.
Dataset203's historical image IDs belong to Yvonne_b2 and follow that control.

The separate neurite interrater table always shows all available human pairs,
independently of the checkboxes. It displays Dice and the same clDice tolerance
as the model table (`--cldice-tolerance 0` through `4`; `run_report.sh` currently
defaults to 4). It excludes scale copies and pairs native annotations across
label splits. Yvonne has three paired images; Yvonne_b2 has none.

Populate the human-agreement caches once, without rescoring model predictions:

```bash
PYTHONPATH=src python -m fm_adaptation.neurite_agreement \
    --raw-data-dir /home/ultrai/GAA/spinal_cord_injury/data/nnUNet_raw \
    --results-dir models
params=0 bash scripts/run_report.sh
```

The cache command defaults to datasets 203, 300, 301, 302, 304 and 306; pass
`--datasets` to narrow the list. It stores all five clDice tolerances, reuses
current caches and refreshes them when masks, mappings or required columns change.
Missing caches/tolerances are identified in the report with a regeneration command.

Annotator identities and workbook provenance are stored in
`src/fm_adaptation/assets/neurite_annotators.json`. To refresh them after workbook
changes, run `PYTHONPATH=src python scripts/import_neurite_annotators.py --yvonne
"/path/to/Manual Tracing Image List.xlsx" --yvonne-b2
"/path/to/Manual Tracing Samples-2026.xlsx"`, then refresh the agreement caches.
Unknown target image IDs fail report generation with an actionable error instead
of silently assigning an annotator. “Cocco” is accepted as an alias for “Coco”.
