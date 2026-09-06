"""Write DRG joining, free-end extension, and min12 variants directly to outputs."""
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
import drg_join_neurite_predictions as joining
from drg_extend_endpoints import extend_endpoints
from drg_inference import PANEL, check_outputs, overlay_image, sha

JOIN_PARAMETERS = {k: v.default for k, v in inspect.signature(joining.postprocess_prediction).parameters.items()
                   if v.default is not inspect.Parameter.empty}
EXTENSION_PARAMETERS = dict(max_length=12., max_angle=30., angle_step=5., min_support=.08, min_length=3.)


def min12(mask):
    labels, _ = ndimage.label(mask, np.ones((3, 3)))
    keep = np.bincount(labels.ravel()) >= 12
    keep[0] = False
    return keep[labels]


def process(mask, evidence, excluded):
    original = skeletonize(mask)
    original[excluded] = False
    p = JOIN_PARAMETERS
    bridged, bridges = joining.conservative_bridging(original, evidence, p['max_gap'], p['max_angle'], p['min_support'])
    bridged[excluded] = False
    joined, guided = joining.guided_endpoint_bridging(bridged, evidence, p['guided_max_gap'], p['guided_max_angle'],
                                                     p['min_support'], p['guided_max_detour'], p['guided_corridor_width'])
    joined = skeletonize(joined)
    joined[excluded] = False
    extended, details = extend_endpoints(joined, evidence, joining, **EXTENSION_PARAMETERS)
    extended[excluded] = False
    return original, joined, extended, dict(conservative_bridges=bridges, guided_bridges=guided, **details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full-image', 'masked-region'], default='full-image')
    parser.add_argument('--input-dir', type=Path, default=PANEL, help='Image directory; masked mode also reads exclusion.png.')
    parser.add_argument('--predictions-dir', type=Path, help='Contains preds and preds_TTA; defaults to input directory.')
    parser.add_argument('--output-dir', type=Path, help='Defaults to predictions directory.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    base = args.input_dir.resolve()
    predictions = (args.predictions_dir or base).resolve()
    out = (args.output_dir or predictions).resolve()
    groups = ['preds', 'preds_TTA'] if args.mode == 'masked-region' else ['preds', 'preds_ensemble', 'preds_TTA']
    suffixes = ['', '_min12', '_joined', '_joined_min12', '_joined_extended', '_joined_extended_min12']
    destinations = [out/(g+s) for g in groups for s in suffixes if s or out != predictions]
    check_outputs(destinations, args.overwrite)
    rgb = np.array(Image.open(base/'images/drg_composite.png').convert('RGB'))
    assert not rgb[..., 1:].any()
    excluded = np.zeros(rgb.shape[:2], bool)
    if args.mode == 'masked-region':
        excluded = np.array(Image.open(base/'exclusion.png')) > 0
        assert excluded.shape == rgb.shape[:2] and not rgb[excluded].any()
    inputs = {g: sorted((predictions/g).glob('*.png')) for g in groups}
    if args.mode == 'masked-region':
        inputs = {g: [predictions/g/'Dataset304__convnextt_red__fold0.png'] for g in groups}
    expected = [1, 1] if args.mode == 'masked-region' else [11, 14, 15]
    for (group, files), count in zip(inputs.items(), expected):
        assert len(files) == count and all(p.is_file() for p in files), (group, count)
    evidence = joining.ridge_evidence(rgb[..., 0])
    evidence[excluded] = 0
    records = {g+s: [] for g in groups for s in suffixes}
    for group, files in inputs.items():
        for path in files:
            encoded = np.array(Image.open(path))
            assert encoded.shape == rgb.shape[:2] and set(np.unique(encoded)) <= {0, 255}
            assert not encoded[excluded].any()
            original, joined, extended, details = process(encoded > 0, evidence, excluded)
            for suffix, line in [('', original), ('_joined', joined), ('_joined_extended', extended)]:
                for minimum in [False, True]:
                    target_group = group+suffix+('_min12' if minimum else '')
                    mask = min12(line) if minimum else line
                    assert np.array_equal(mask, skeletonize(mask)) and not mask[excluded].any()
                    dest = out/target_group
                    info = dict(input=str(path), input_sha256=sha(path), file=path.name,
                                stats=joining.stats(original, mask), minimum_component_pixels=12 if minimum else None)
                    if suffix:
                        info.update(details)
                    records[target_group].append(info)
                    # Source predictions already have their inference manifest and overlays.
                    if dest == predictions/group:
                        assert np.array_equal(mask, encoded > 0)
                        continue
                    (dest/'overlays').mkdir(parents=True, exist_ok=True)
                    saved = mask.astype(np.uint8)*255
                    Image.fromarray(saved).save(dest/path.name)
                    assert np.array_equal(np.array(Image.open(dest/path.name)), saved)
                    if not suffix and (predictions/group/'raw'/path.name).exists():
                        (dest/'raw').mkdir(exist_ok=True)
                        Image.fromarray(saved).save(dest/'raw'/path.name)
                    overlay_image(rgb, mask, path.name, group == 'preds_TTA', path.stem.startswith('ensemble_'),
                                  joined=joined, original=original).save(dest/'overlays'/path.name)
                    info['output_sha256'] = sha(dest/path.name)
            print(group, path.name, flush=True)
    for group, rows in records.items():
        if out/group == predictions/group and group in groups:
            continue
        metadata = dict(status='complete', count=len(rows), image=str(base/'images/drg_composite.png'),
                        image_sha256=sha(base/'images/drg_composite.png'), join_parameters=JOIN_PARAMETERS,
                        extension_parameters=EXTENSION_PARAMETERS, predictions=rows,
                        joining_sha256=sha(Path(joining.__file__)),
                        extension_sha256=sha(Path(__file__).with_name('drg_extend_endpoints.py')))
        (out/group/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
