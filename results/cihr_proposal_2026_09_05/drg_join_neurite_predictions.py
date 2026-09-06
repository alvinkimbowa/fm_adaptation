#!/usr/bin/env python3
"""Reconnect fragmented neurite predictions using geometry and SMI ridge evidence.

The input prediction is never modified. Three outputs are produced: a directional
closing baseline, conservative endpoint bridging, and longer endpoint-to-endpoint paths
routed along image ridges. Ground-truth labels are deliberately not accepted by this tool.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import line as raster_line
from skimage.filters import frangi
from skimage.graph import route_through_array
from skimage.morphology import skeletonize


NEIGHBOUR_KERNEL = np.ones((3, 3), dtype=np.uint8)
DEFAULT_MAX_GAP = 12.0
DEFAULT_MAX_ANGLE = 35.0
DEFAULT_MIN_SUPPORT = 0.08
DEFAULT_GUIDED_MAX_GAP = 22.0
DEFAULT_GUIDED_MAX_ANGLE = 25.0
DEFAULT_GUIDED_MAX_DETOUR = 1.15
DEFAULT_GUIDED_CORRIDOR_WIDTH = 2


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read {path}")
    return image


def crop(array: np.ndarray, top: int | None, left: int | None, size: int | None) -> np.ndarray:
    if top is None and left is None and size is None:
        return array
    if top is None or left is None or size is None:
        raise ValueError("--top, --left and --size must be supplied together")
    if top < 0 or left < 0 or top + size > array.shape[0] or left + size > array.shape[1]:
        raise ValueError(f"Crop {(top, left, size)} is outside image shape {array.shape}")
    return array[top:top + size, left:left + size]


def normalise(array: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(array, (1, 99.5))
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array.astype(np.float32) - lo) / (hi - lo), 0, 1)


def ridge_evidence(image: np.ndarray) -> np.ndarray:
    """Bright, thin SMI structures scored on a robust 0..1 scale."""
    intensity = normalise(image)
    ridges = frangi(intensity, sigmas=(1, 2, 3), black_ridges=False)
    ridges = normalise(ridges)
    # Frangi suppresses puncta; intensity retains very faint neurites that Frangi misses.
    return np.maximum(ridges, 0.35 * intensity)


def endpoints(skeleton: np.ndarray) -> np.ndarray:
    neighbours = ndimage.convolve(skeleton.astype(np.uint8), NEIGHBOUR_KERNEL, mode="constant")
    return np.argwhere(skeleton & (neighbours == 2))  # count includes the centre pixel


def outward_tangent(skeleton: np.ndarray, endpoint: np.ndarray, steps: int = 7) -> np.ndarray | None:
    """Unit vector pointing out of a skeleton endpoint, estimated along its local branch."""
    current = tuple(int(v) for v in endpoint)
    previous: tuple[int, int] | None = None
    trail = [np.asarray(current, dtype=np.float32)]
    for _ in range(steps):
        row, col = current
        candidates: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                candidate = (row + dr, col + dc)
                if (dr or dc) and candidate != previous and 0 <= candidate[0] < skeleton.shape[0] \
                        and 0 <= candidate[1] < skeleton.shape[1] and skeleton[candidate]:
                    candidates.append(candidate)
        if not candidates:
            break
        # A junction makes the tangent ambiguous; stop at it instead of choosing a branch.
        if len(candidates) > 1:
            break
        previous, current = current, candidates[0]
        trail.append(np.asarray(current, dtype=np.float32))
    if len(trail) < 3:
        return None
    vector = trail[0] - trail[-1]
    length = float(np.linalg.norm(vector))
    return vector / length if length else None


def line_pixels(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    del shape  # endpoints already lie inside the prediction
    return raster_line(int(a[0]), int(a[1]), int(b[0]), int(b[1]))


def nearby_pairs(points: np.ndarray, distance: float) -> list[tuple[int, int]]:
    """Endpoint index pairs within a radius, without a quadratic all-pairs scan."""
    if len(points) < 2:
        return []
    return sorted(cKDTree(points).query_pairs(distance))


def conservative_bridging(
    prediction: np.ndarray,
    evidence: np.ndarray,
    max_gap: float,
    max_angle: float,
    min_support: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    skeleton = skeletonize(prediction)
    points = endpoints(skeleton)
    components, _ = ndimage.label(skeleton, structure=NEIGHBOUR_KERNEL)
    tangents = [outward_tangent(skeleton, point) for point in points]
    threshold = max(float(np.percentile(evidence[skeleton], 20)), min_support) if skeleton.any() else min_support
    candidates: list[tuple[float, int, int, float]] = []
    cos_limit = math.cos(math.radians(max_angle))

    for i, j in nearby_pairs(points, max_gap):
        a, b = points[i], points[j]
        if tangents[i] is None:
            continue
        if tangents[j] is None or components[tuple(a)] == components[tuple(b)]:
            continue
        vector = b.astype(np.float32) - a
        distance = float(np.linalg.norm(vector))
        if not 1 < distance:
            continue
        direction = vector / distance
        alignment = min(float(np.dot(tangents[i], direction)),
                        float(np.dot(tangents[j], -direction)))
        if alignment < cos_limit:
            continue
        rr, cc = line_pixels(a, b, prediction.shape)
        support = float(np.mean(evidence[rr, cc]))
        if support >= threshold:
            score = support + alignment - 0.25 * distance / max_gap
            candidates.append((score, i, j, support))

    # One endpoint can participate in only one bridge. Greedy selection is deterministic.
    result = prediction.copy()
    used: set[int] = set()
    accepted: list[dict[str, float]] = []
    for _, i, j, support in sorted(candidates, reverse=True):
        if i in used or j in used:
            continue
        a, b = points[i], points[j]
        rr, cc = line_pixels(a, b, prediction.shape)
        result[rr, cc] = True
        used.update((i, j))
        accepted.append({"row_a": int(a[0]), "col_a": int(a[1]),
                         "row_b": int(b[0]), "col_b": int(b[1]),
                         "length": float(np.linalg.norm(b - a)), "support": support})
    return result, accepted


def guided_endpoint_bridging(
    bridged: np.ndarray,
    evidence: np.ndarray,
    max_gap: float,
    max_angle: float,
    min_support: float,
    max_detour: float,
    corridor_width: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Join longer aligned endpoint pairs along an image-supported ridge path."""
    skeleton = skeletonize(bridged)
    points = endpoints(skeleton)
    components, _ = ndimage.label(skeleton, structure=NEIGHBOUR_KERNEL)
    tangents = [outward_tangent(skeleton, point) for point in points]
    threshold = max(float(np.percentile(evidence[skeleton], 20)), min_support) if skeleton.any() else min_support
    cos_limit = math.cos(math.radians(max_angle))
    candidates: list[tuple[float, int, int, np.ndarray, float, float]] = []

    for i, j in nearby_pairs(points, max_gap):
        a, b = points[i], points[j]
        if tangents[i] is None:
            continue
        if tangents[j] is None or components[tuple(a)] == components[tuple(b)]:
            continue
        vector = b.astype(np.float32) - a
        distance = float(np.linalg.norm(vector))
        if not 2 < distance:
            continue
        direction = vector / distance
        alignment = min(float(np.dot(tangents[i], direction)),
                        float(np.dot(tangents[j], -direction)))
        if alignment < cos_limit:
            continue

        # Route only inside a narrow corridor between endpoints. This lets a bridge
        # bend with a neurite but prevents it from wandering onto a nearby fibre.
        margin = corridor_width + 2
        top = max(0, int(min(a[0], b[0])) - margin)
        bottom = min(skeleton.shape[0], int(max(a[0], b[0])) + margin + 1)
        left = max(0, int(min(a[1], b[1])) - margin)
        right = min(skeleton.shape[1], int(max(a[1], b[1])) + margin + 1)
        local_evidence = evidence[top:bottom, left:right]
        local_a = (int(a[0]) - top, int(a[1]) - left)
        local_b = (int(b[0]) - top, int(b[1]) - left)
        centreline = np.zeros(local_evidence.shape, dtype=np.uint8)
        cv2.line(centreline, (local_a[1], local_a[0]), (local_b[1], local_b[0]), 1, 1)
        corridor = ndimage.distance_transform_edt(~centreline.astype(bool)) <= corridor_width
        cost = 1 / (0.05 + local_evidence)
        cost[~corridor] = 1e6
        route, _ = route_through_array(cost, local_a, local_b,
                                       fully_connected=True, geometric=True)
        path = np.asarray(route, dtype=np.int32)
        support = float(np.mean(local_evidence[path[:, 0], path[:, 1]]))
        path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        detour = path_length / distance
        if support < threshold or detour > max_detour:
            continue
        path[:, 0] += top
        path[:, 1] += left
        score = support + alignment - 0.25 * detour
        candidates.append((score, i, j, path, support, path_length))

    result = bridged.copy()
    used: set[int] = set()
    accepted: list[dict[str, float]] = []
    for _, i, j, path, support, path_length in sorted(candidates, key=lambda item: item[0],
                                                       reverse=True):
        if i in used or j in used:
            continue
        result[path[:, 0], path[:, 1]] = True
        used.update((i, j))
        a, b = points[i], points[j]
        accepted.append({"row_a": int(a[0]), "col_a": int(a[1]),
                         "row_b": int(b[0]), "col_b": int(b[1]),
                         "distance": float(np.linalg.norm(b - a)),
                         "path_length": path_length, "support": support})
    return result, accepted


def directional_closing(prediction: np.ndarray, max_gap: int) -> np.ndarray:
    skeleton = skeletonize(prediction).astype(np.uint8)
    result = skeleton.copy()
    length = max(3, int(max_gap) + 1)
    for angle in range(0, 180, 15):
        kernel = np.zeros((length, length), dtype=np.uint8)
        centre = (length - 1) / 2
        radius = centre
        dx = radius * math.cos(math.radians(angle))
        dy = radius * math.sin(math.radians(angle))
        cv2.line(kernel, (round(centre - dx), round(centre - dy)),
                 (round(centre + dx), round(centre + dy)), 1, 1)
        result |= cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel)
    return prediction | skeletonize(result)


def red_image(image: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*image.shape, 3), dtype=np.uint8)
    canvas[..., 2] = image  # SMI stays red, matching the dataset visualizations.
    return canvas


def titled(canvas: np.ndarray, title: str) -> np.ndarray:
    header = np.zeros((34, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.vstack((header, canvas))


def overlay(image: np.ndarray, original: np.ndarray, result: np.ndarray, title: str) -> np.ndarray:
    canvas = red_image(image)
    canvas[original] = (255, 255, 0)  # cyan: original prediction
    canvas[result & ~original] = (0, 255, 255)  # yellow: added pixels
    return titled(canvas, title)


def stats(original: np.ndarray, result: np.ndarray) -> dict[str, int]:
    skeleton = skeletonize(result)
    return {
        "foreground_pixels": int(result.sum()),
        "added_pixels": int((result & ~original).sum()),
        "components": int(ndimage.label(skeleton, structure=NEIGHBOUR_KERNEL)[1]),
        "endpoints": int(len(endpoints(skeleton))),
    }


def postprocess_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    max_gap: float = DEFAULT_MAX_GAP,
    max_angle: float = DEFAULT_MAX_ANGLE,
    min_support: float = DEFAULT_MIN_SUPPORT,
    guided_max_gap: float = DEFAULT_GUIDED_MAX_GAP,
    guided_max_angle: float = DEFAULT_GUIDED_MAX_ANGLE,
    guided_max_detour: float = DEFAULT_GUIDED_MAX_DETOUR,
    guided_corridor_width: int = DEFAULT_GUIDED_CORRIDOR_WIDTH,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return 1 px baseline, conservative, and guided masks for one image."""
    prediction = skeletonize(prediction > 0)
    evidence = ridge_evidence(image)
    closing = directional_closing(prediction, round(max_gap))
    bridged, bridges = conservative_bridging(
        prediction, evidence, max_gap, max_angle, min_support
    )
    guided, guided_bridges = guided_endpoint_bridging(
        bridged, evidence, guided_max_gap, guided_max_angle, min_support,
        guided_max_detour, guided_corridor_width,
    )
    outputs = {
        "original": prediction,
        "directional_closing": skeletonize(closing),
        "conservative_bridge": skeletonize(bridged),
        "guided_endpoint_bridge": skeletonize(guided),
    }
    return outputs, {"bridges": bridges, "guided_bridges": guided_bridges,
                     "ridge_evidence": evidence}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int)
    parser.add_argument("--left", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP)
    parser.add_argument("--max-angle", type=float, default=DEFAULT_MAX_ANGLE)
    parser.add_argument("--min-support", type=float, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--guided-max-gap", type=float, default=DEFAULT_GUIDED_MAX_GAP)
    parser.add_argument("--guided-max-angle", type=float, default=DEFAULT_GUIDED_MAX_ANGLE)
    parser.add_argument("--guided-max-detour", type=float, default=DEFAULT_GUIDED_MAX_DETOUR)
    parser.add_argument(
        "--guided-corridor-width", type=int, default=DEFAULT_GUIDED_CORRIDOR_WIDTH
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = crop(read_gray(args.image), args.top, args.left, args.size)
    prediction = crop(read_gray(args.prediction), args.top, args.left, args.size) > 0
    if image.shape != prediction.shape:
        raise ValueError(f"Image shape {image.shape} != prediction shape {prediction.shape}")

    outputs, details = postprocess_prediction(
        image, prediction, args.max_gap, args.max_angle, args.min_support,
        args.guided_max_gap, args.guided_max_angle, args.guided_max_detour,
        args.guided_corridor_width,
    )
    prediction = outputs["original"]
    evidence = details["ridge_evidence"]
    bridges = details["bridges"]
    guided_bridges = details["guided_bridges"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_only = red_image(image)
    panels = [titled(image_only, "image only")]
    for name, result in outputs.items():
        cv2.imwrite(str(args.output_dir / f"{name}.png"), result.astype(np.uint8) * 255)
        panels.append(overlay(image, prediction, result, name.replace("_", " ")))
    original_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    original_mask[prediction] = (255, 255, 255)
    panels.append(titled(original_mask, "original prediction mask"))
    final_mask = np.zeros((*outputs["guided_endpoint_bridge"].shape, 3), dtype=np.uint8)
    final_mask[outputs["guided_endpoint_bridge"]] = (255, 255, 255)
    panels.append(titled(final_mask, "final postprocessed prediction"))
    cv2.imwrite(str(args.output_dir / "ridge_evidence.png"), np.round(evidence * 255).astype(np.uint8))
    cv2.imwrite(str(args.output_dir / "comparison.png"), np.hstack(panels))

    report = {name: stats(prediction, result) for name, result in outputs.items()}
    report["conservative_bridge"]["accepted_bridges"] = len(bridges)
    report["guided_endpoint_bridge"]["accepted_bridges"] = len(guided_bridges)
    report["parameters"] = {
        "top": args.top, "left": args.left, "size": args.size,
        "max_gap": args.max_gap, "max_angle": args.max_angle,
        "min_support": args.min_support, "guided_max_gap": args.guided_max_gap,
        "guided_max_angle": args.guided_max_angle,
        "guided_max_detour": args.guided_max_detour,
        "guided_corridor_width": args.guided_corridor_width,
    }
    report["bridges"] = bridges
    report["guided_bridges"] = guided_bridges
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
