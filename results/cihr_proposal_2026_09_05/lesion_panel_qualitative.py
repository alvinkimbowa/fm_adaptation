"""One high-Dice lesion case: image with GT contour and two prediction masks."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fm_adaptation.data import load_dataset_json, rgb_planes
from fm_adaptation.metrics import read_case_metrics
from fm_adaptation.qualitative import COLORS, _to_rgb, draw, read_channel_planes, read_image


def select_case(raw, dino, nnunet, seed, source_split="Ts", case_prefix="Katie_contusion__",
                min_dice=0.88):
    dino_scores = {r["case_id"]: r["dice"] for r in read_case_metrics(dino / "metrics.csv")}
    nnunet_scores = {r["case_id"]: r["dice"] for r in read_case_metrics(nnunet / "metrics.csv")}
    channels = load_dataset_json(raw)["channel_names"]
    candidates = []
    for case, score in sorted(dino_scores.items()):
        if not case.startswith(case_prefix) or not np.isfinite(score) or score <= min_dice:
            continue
        if not np.isfinite(nnunet_scores.get(case, np.nan)):
            continue
        required = [raw / f"images{source_split}" / f"{case}_{int(c):04d}.png" for c in channels]
        required += [raw / f"labels{source_split}" / f"{case}.png",
                     dino / "predictions" / f"{case}.png", nnunet / "preds" / f"{case}.png"]
        if all(p.is_file() for p in required):
            candidates.append(case)
    if not candidates:
        raise ValueError(f"No matching case with DINOv3 Dice >{min_dice:.0%}, both scores, and all images/masks; the threshold was not relaxed")
    case = candidates[int(np.random.default_rng(None if seed < 0 else seed).integers(len(candidates)))]
    return case, nnunet_scores[case], dino_scores[case], len(candidates)


def render(image, masks, scores, output, rotate=90, line_width=1.5, pred_style="mask"):
    for mask in masks:
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} differs from image {image.shape[:2]}")
    image = np.rot90(image, rotate // 90)
    height, width = image.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4 * height / width))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.01)
    # Paint masks at native resolution, then reduce coverage to the displayed panel.
    import cv2

    target = (800, max(1, round(800 * height / width)))
    backdrop = cv2.resize(_to_rgb(image), target, interpolation=cv2.INTER_AREA)
    text_box = dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3)
    for ax, mask, title, score in zip(
        axes, masks, ("a) Image", "b) nnU-Net", "c) DINOv3"), (None, *scores)
    ):
        as_mask = score is not None and pred_style == "mask"
        panel = np.zeros_like(backdrop) if as_mask else backdrop.copy()
        style = "overlay" if as_mask else "contour"
        draw(panel, np.rot90(mask, rotate // 90) == 1, style, COLORS["white"],
             width=line_width, alpha=1.0, size=target)
        ax.imshow(panel.clip(0, 255).astype(np.uint8))
        ax.axis("off")
        ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top", color="white",
                fontsize=16, bbox=text_box)
        if score is not None:
            ax.text(0.02, 0.02, f"Dice: {score:.1%}", transform=ax.transAxes, va="bottom",
                    color="white", fontsize=16, bbox=text_box)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--dino-results", type=Path, required=True)
    parser.add_argument("--nnunet-results", type=Path, required=True)
    parser.add_argument("--source-split", choices=("Tr", "Ts"), default="Ts")
    parser.add_argument("--case-prefix", default="Katie_contusion__")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-dice", type=float, default=0.88)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=90)
    parser.add_argument("--line-width", type=float, default=1.5)
    parser.add_argument("--pred-style", choices=("mask", "contour"), default="mask")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cihr_proposal_2026_09_05/figures/lesion"))
    args = parser.parse_args()
    case, nn_score, dino_score, count = select_case(
        args.raw_dataset, args.dino_results, args.nnunet_results, args.seed,
        args.source_split, args.case_prefix, args.min_dice
    )
    planes = rgb_planes(load_dataset_json(args.raw_dataset)["channel_names"])
    image = read_channel_planes(args.raw_dataset / f"images{args.source_split}", case, planes)
    masks = [read_image(directory / f"{case}.png") for directory in (
        args.raw_dataset / f"labels{args.source_split}", args.nnunet_results / "preds",
        args.dino_results / "predictions",
    )]
    output = args.output_dir / f"test__{args.raw_dataset.name}__{case}.png"
    render(image, masks, (nn_score, dino_score), output, args.rotate, args.line_width,
           args.pred_style)
    print(f"Selected {case} from {count} eligible cases (seed={args.seed})")
    print(f"nnU-Net Dice: {nn_score:.1%}; DINOv3 Dice: {dino_score:.1%}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
