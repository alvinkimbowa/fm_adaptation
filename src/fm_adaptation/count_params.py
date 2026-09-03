"""Collect total and trainable parameter counts for every configuration in the report.

The counts come from wherever each family already records them, so nothing is estimated:

* foundation-model runs -- built from their own config and counted directly;
* mmseg adapter runs -- built from `dinov3_mmseg.head_cfg`, which is what trained them;
* nnU-Net -- its own `model_stats.json`, and nothing where a run has not written one;
* MonoUNet -- its own `model_analysis.json`, written per training dataset;

Written once to `models/parameter_counts.json`; `report.py` only reads that file, so generating the
tables stays fast and needs no model building.
"""

import argparse
import json
from pathlib import Path

import yaml

MMSEG_RUNS = {"upernet", "upernet_inj", "m2f", "m2f_inj"}
# Every dataset in the study is binary, and the head is the only part whose size depends on this.
NUM_CLASSES = 2
CROP_SIZE = 896


def _counts(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _foundation_counts(model_name, run_name, probe, injector, train_encoder=False, variant="vitl16"):
    from .models import build_model

    if run_name in MMSEG_RUNS:
        from mmengine.registry import init_default_scope

        init_default_scope("mmseg")
        import mmdet.models  # noqa: F401
        import mmseg.models  # noqa: F401
        from mmseg.registry import MODELS

        from .dinov3_mmseg import head_cfg

        head = "m2f" if run_name.startswith("m2f") else "upernet"
        cfg = head_cfg(head, NUM_CLASSES, CROP_SIZE, injector=run_name.endswith("_inj"))
        return _counts(MODELS.build(cfg))

    model = build_model(
        model_name, probe, NUM_CLASSES, None,
        # The two-stage finetuning runs say so in their name; the adapter runs say so in their config.
        train_encoder=train_encoder or "finetune" in run_name, injector=injector, variant=variant,
    )
    return _counts(model)


def _nnunet_counts(results_dirs):
    """Read from each trainer's `model_stats.json`; a run without one is left uncounted.

    nnU-Net writes that file itself, and each plans variant is a row of its own -- the ResEnc presets
    are scaled to the dataset, so two runs of the same plans name legitimately differ in size. No
    checkpoint is opened here: a run whose stats file is missing reports no size rather than costing a
    load of every checkpoint on every report.
    """
    from .report import nnunet_label

    found = {}
    for results_dir in results_dirs:
        for path in sorted(Path(results_dir).glob("nnunet/Dataset*/*/model_stats.json")):
            stats = json.loads(path.read_text())
            key = f"{nnunet_label(path.parent.name)}||{path.parents[1].name}"
            found[key] = {
                "total": stats["num_parameters"],
                "trainable": stats["num_parameters_trainable"],
            }
    return found


def _monounet_counts(results_dirs):
    """Read from each training dataset's `model_analysis.json`, which MonoUNet writes itself.

    The count sits one level below the architecture directory because the head is sized to the
    dataset's classes, so two datasets of the same architecture differ by a few parameters. A dataset
    without the file is left uncounted rather than built here, which would need the other project.
    """
    from .report import monounet_label

    found = {}
    for results_dir in results_dirs:
        results_dir = Path(results_dir)
        for path in sorted(results_dir.glob("Dataset*/model_analysis.json")):
            parameters = json.loads(path.read_text())["parameters"]
            key = f"{monounet_label(results_dir.name)}||{path.parent.name}"
            found[key] = {
                "total": parameters["total"],
                "trainable": parameters["trainable"],
            }
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="models")
    parser.add_argument("--nnunet-results-dir", nargs="*", default=[])
    parser.add_argument("--monounet-results-dir", nargs="*", default=[])
    parser.add_argument("--output", default="models/parameter_counts.json")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Keep the counts already written and build only the runs the file does not cover.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    counts = {}
    if args.only_missing and output.exists():
        counts = json.loads(output.read_text())
    seen = dict(counts)
    for config_path in sorted(Path(args.results_dir).glob("*/Dataset*/*/fold_*/config.yaml")):
        cfg = yaml.safe_load(config_path.read_text())
        model_name = cfg["model"]["name"]
        run_name = cfg["model"].get("run_name", cfg["model"]["probe"])
        key = f"{model_name}|{run_name}|"
        if key in seen:
            continue
        probe = cfg["model"]["probe"]
        injector = bool(cfg["model"].get("injector", False))
        train_encoder = bool(cfg["model"].get("train_encoder", False))
        variant = str(cfg["model"].get("variant", "vitl16"))
        print(f"counting {model_name}/{run_name} ...", flush=True)
        try:
            seen[key] = _foundation_counts(model_name, run_name, probe, injector, train_encoder, variant)
        except Exception as error:  # a missing optional dependency should not lose the rest
            print(f"  skipped: {type(error).__name__}: {error}")
            continue
        counts[key] = seen[key]
        print(f"  total {seen[key]['total']/1e6:.1f}M  trainable {seen[key]['trainable']/1e6:.2f}M")

    counts.update(_nnunet_counts(args.nnunet_results_dir))
    counts.update(_monounet_counts(args.monounet_results_dir))

    output.write_text(json.dumps(counts, indent=2, sort_keys=True))
    print(f"wrote {output} ({len(counts)} entries)")


if __name__ == "__main__":
    main()
