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
as the model table (`--cldice-tolerance 0` through `6`; `run_report.sh` currently
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
`--datasets` to narrow the list. It stores every clDice tolerance, reuses
current caches and refreshes them when masks, mappings or required columns change.
Missing caches/tolerances are identified in the report with a regeneration command.

Annotator identities and workbook provenance are stored in
`src/fm_adaptation/assets/neurite_annotators.json`. To refresh them after workbook
changes, run `PYTHONPATH=src python scripts/import_neurite_annotators.py --yvonne
"/path/to/Manual Tracing Image List.xlsx" --yvonne-b2
"/path/to/Manual Tracing Samples-2026.xlsx"`, then refresh the agreement caches.
Unknown target image IDs fail report generation with an actionable error instead
of silently assigning an annotator. “Cocco” is accepted as an alias for “Coco”.

## Yvonne_b2 instance Mask2Former

The two instance experiments are `configs/m2f_inj_ft_aug_ours.yaml` (ViT-L with
injector) and `configs/convnextt_m2f_ft_aug_p256_red_ours.yaml` (ConvNeXt-T).
`model.task` defaults to `semantic`; the new configurations explicitly select
`instance` and `probe: mask2former`. They use MMDetection's instance head, native
256 px SMI-in-red inputs, 128 queries, FP32, 64 patches per case, and the same
Dataset301 fold. Existing semantic configurations/checkpoints are unchanged.

Prepare annotations without copying image pixels:

```bash
PYTHONPATH=src .venv-mm/bin/python -m data.instance_data \
  --source ../../../GAA/spinal_cord_injury/data/nnUNet_raw/Dataset301_neurite_yvonne_b2_smi \
  --raw ../../../GAA/spinal_cord_injury/data/raw_data/Yvonne_2026-05-16
PYTHONPATH=src .venv-mm/bin/python -m data.audit_instances
PYTHONPATH=src .venv-mm/bin/python -m data.audit_instance_epochs --epochs 100
PYTHONPATH=src .venv-mm/bin/python -m pytest tests/test_instances.py -q
```

Preparation copies the original split bytes, verifies every symlink and original
mask union, computes the training-only fifth-percentile length (16 px for fold 0),
and writes bbox-local RLE, original polylines, stable source IDs, and
`data/instances_yvonne_b2/instances.coco.json`. ROI helper attribution and the
upstream MIT license are in `src/data/vendor/`. The source project is
needed only as a data location, never as a runtime Python import.

Training generates patches in memory. Geometry transforms the image and original
polylines together; clipped re-entry runs are separate targets. Crop-created short
fragments are ignored while original contained short fibers remain positive.
Positive masks override ignore pixels at crossings. Hungarian matching and every
auxiliary decoder loss sample positive support from each fiber, along with random
and uncertain valid pixels. Classification/BCE/Dice weights are 2/5/5. Query
budget overflow is an error and never truncates targets.

After checking GPU placement, use the project launcher (through the pooria run
ledger when remote):

```bash
overfit_steps=30 gpu_id=1 bash scripts/run_instance_exp.sh configs/m2f_inj_ft_aug_ours.yaml
gpu_id=1 bash scripts/run_instance_exp.sh configs/m2f_inj_ft_aug_ours.yaml
# resume=1 continues from last.pt, including optimizer, scheduler and RNG state.
```

`best.pt` is selected by deterministic validation-grid mask AP; `last.pt` carries
resume state. Each completed epoch refreshes `history.png` with training loss and
validation mask AP/AP50, including all pre-resume history. Fixed validation
examples are saved every epoch to `qualitative/epoch_NNN.png` and
`qualitative/latest.png`: native SMI input, overlapping ground-truth fibers,
and predicted instances with confidence scores (display threshold 0.5). The launcher then performs validation and held-out slide inference,
using stride 128, conservative shared-region stitching, and overlapping instance
COCO exports. Predictions are disk-backed and probability fusion uses 256 px
buffers. Original short duplicates can merge at high IoU; continuations require
16 px support, 70% mutual coverage within 2 px, and directions within 30 degrees.
Touching alone does not match. Ambiguous alternatives stay separate.

Each `instance_predictions_{val,test}/metrics.json` reports mask AP/AP50/AP75/AR,
the explicit 10,000-prediction evaluation limit, tile saturation and ambiguous
matches, union Dice/clDice, ASSD/HD95 in pixels, and peak allocated GPU memory.
These are instance metrics, distinct from the existing semantic report pipeline.
Native channel/geometry QC crops and the count audit are written under
`results/instance_preflight/`. Stitch thresholds are fixed initially; any tuning
must use validation cases only.
