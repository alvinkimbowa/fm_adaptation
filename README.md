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

Full fine-tuning starts from the corresponding best linear probe:

```bash
PYTHONPATH=src python -m fm_adaptation.finetune --config configs/sam3_finetune_busbra.yaml
```

Full fine-tuning from the best nonlinear probe uses the corresponding nonlinear model
and writes a separate `nonlinear_finetune` run:

```bash
PYTHONPATH=src python -m fm_adaptation.finetune --config configs/sam3_nonlinear_finetune_busbra.yaml
```

Runs are stored under
`models/{foundation_model}/{training_dataset}/{probe}/fold_{fold}`. Cross-dataset
predictions and per-case metrics live below each run's `test/{dataset}` directory;
the report command creates `models/cross_dataset_report.html`. Pass an nnU-Net
results directory to include its available `test/*/metrics.csv` results as
comparison rows in each dataset-family table.
