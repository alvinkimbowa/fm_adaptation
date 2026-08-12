"""Collect total and trainable parameter counts for every configuration in the report.

The counts come from wherever each family already records them, so nothing is estimated:

* foundation-model runs -- built from their own config and counted directly;
* mmseg adapter runs -- built from `dinov3_mmseg.head_cfg`, which is what trained them;
* nnU-Net -- its own `model_stats.json`;
* MonoUNet -- summed from the saved checkpoint, which holds exactly the trained weights.

Written once to `models/parameter_counts.json`; `report.py` only reads that file, so generating the
tables stays fast and needs no model building.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

MMSEG_RUNS = {"upernet", "upernet_inj", "m2f", "m2f_inj"}
# Every dataset in the study is binary, and the head is the only part whose size depends on this.
NUM_CLASSES = 2
CROP_SIZE = 896


def _counts(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _foundation_counts(model_name, run_name, probe, injector, train_encoder=False):
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
        train_encoder=train_encoder or "finetune" in run_name, injector=injector,
    )
    return _counts(model)


def _nnunet_counts(results_dirs):
    """nnU-Net records its own stats per trainer; keep the plan that the report actually reads."""
    found = {}
    for results_dir in results_dirs:
        for path in sorted(Path(results_dir).glob("nnunet/Dataset*/*/model_stats.json")):
            stats = json.loads(path.read_text())
            key = f"nnU-Net||{path.parents[1].name}"
            found[key] = {
                "total": stats["num_parameters"],
                "trainable": stats["num_parameters_trainable"],
            }
    return found


def _monounet_counts(results_dirs, names):
    """No stats file, so count the checkpoint: MonoUNet trains every weight it saves."""
    found = {}
    for results_dir in results_dirs:
        results_dir = Path(results_dir)
        label = names.get(results_dir.name, results_dir.name)
        for path in sorted(results_dir.glob("Dataset*/fold_*/model_final.pth")):
            state = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                continue
            total = sum(v.numel() for v in state.values() if torch.is_tensor(v) and v.is_floating_point())
            found[f"{label}||{path.parents[1].name}"] = {"total": total, "trainable": total}
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

    from .report import MONOUNET_NAMES

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
        print(f"counting {model_name}/{run_name} ...", flush=True)
        try:
            seen[key] = _foundation_counts(model_name, run_name, probe, injector, train_encoder)
        except Exception as error:  # a missing optional dependency should not lose the rest
            print(f"  skipped: {type(error).__name__}: {error}")
            continue
        counts[key] = seen[key]
        print(f"  total {seen[key]['total']/1e6:.1f}M  trainable {seen[key]['trainable']/1e6:.2f}M")

    counts.update(_nnunet_counts(args.nnunet_results_dir))
    counts.update(_monounet_counts(args.monounet_results_dir, MONOUNET_NAMES))

    output.write_text(json.dumps(counts, indent=2, sort_keys=True))
    print(f"wrote {output} ({len(counts)} entries)")


if __name__ == "__main__":
    main()
