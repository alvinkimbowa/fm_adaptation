"""One-off DRG plain/TTA inference and ensembles; run from the repository root."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

PROJECT = Path(__file__).resolve().parents[2]
PANEL = Path('/home/ultrai/GAA/spinal_cord_injury/data/raw_data/DRG-panels')
VIEWS = [(k, flip) for k in range(4) for flip in (False, True)]

def transform(a,k,mirror):
 b=np.rot90(a,k)
 return np.ascontiguousarray(b[:,::-1] if mirror else b)

def inverse(a,k,mirror):
 return np.ascontiguousarray(np.rot90(a[:,::-1] if mirror else a,-k))

def predict_probs(model,cfg,plane):
 import torch
 from fm_adaptation.patching import CaseIndex, grid_anchors, _importance_map, _restore_probabilities, BLOCKS_PER_PATCH
 pc=cfg.patching; assert pc is not None
 layout=((0,0),) if pc.respect_channels else None
 case=CaseIndex(Path('/tmp'),'Ts','oneoff','.png',max(1,pc.patch_size//BLOCKS_PER_PATCH),layout)
 case._images=((0 if layout else None,plane),)
 case.build(pc.roi_threshold)
 anchors=grid_anchors(case,pc)
 accum=np.zeros(plane.shape,np.float32); weights=np.zeros_like(accum)
 importance=_importance_map(pc.patch_size)
 empty=np.zeros((pc.patch_size,pc.patch_size),np.uint8)
 with torch.inference_mode():
  for y,x in anchors:
   image,_=case.crop(int(y),int(x),pc.patch_size,require_label=False)
   tensor,_,geom=model.encoder.preprocess(image,empty)
   with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(tensor[None].to('cuda'))
   probs=logits.float().softmax(1)[0].cpu().numpy()
   fg=_restore_probabilities(probs,geom,pc.patch_size)[1]
   window=(slice(int(y),int(y)+pc.patch_size),slice(int(x),int(x)+pc.patch_size))
   accum[window]+=fg*importance; weights[window]+=importance
 covered=weights>0
 np.divide(accum,weights,out=accum,where=covered)
 assert np.isfinite(accum).all() and accum.min()>=0 and accum.max()<=1.00001
 return np.clip(accum,0,1),covered,len(anchors)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_outputs(paths, overwrite):
    existing = [str(p) for p in paths if p.exists() and (not p.is_dir() or any(p.iterdir()))]
    if existing and not overwrite:
        raise FileExistsError('Outputs already exist; choose another output directory or --overwrite: ' + ', '.join(existing))


def overlay_image(rgb, mask, name, tta=False, ensemble=False, joined=None, original=None):
    canvas = rgb.copy()
    canvas[mask] = (0, 255, 255)
    if original is not None and joined is not None:
        canvas[mask & joined & ~original] = (255, 255, 0)
        canvas[mask & ~joined & ~original] = (255, 0, 255)
    result = Image.new('RGB', (rgb.shape[1], rgb.shape[0] + 80), (24, 24, 24))
    result.paste(Image.fromarray(canvas), (0, 80))
    suffix = 'TTA Ensemble' if tta and ensemble else 'TTA' if tta else 'Ensemble' if ensemble else ''
    title = name + (' | ' + suffix if suffix else '')
    draw = ImageDraw.Draw(result)
    size = 30
    while True:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
        if draw.textbbox((0, 0), title, font=font)[2] <= rgb.shape[1] - 48 or size == 8:
            break
        size -= 1
    draw.text((24, 40), title, font=font, fill='white', anchor='lm')
    return result


def save_prediction(directory, name, prob, coverage, rgb, tta, info):
    assert prob.shape == rgb.shape[:2] and np.isfinite(prob).all()
    assert prob.min() >= 0 and prob.max() <= 1
    mask = skeletonize(prob > 0.5)
    encoded = mask.astype(np.uint8) * 255
    for folder in [directory, directory/'raw', directory/'overlays', directory/'probabilities']:
        folder.mkdir(parents=True, exist_ok=True)
    for p in [directory/(name+'.png'), directory/'raw'/(name+'.png')]:
        Image.fromarray(encoded).save(p)
        assert np.array_equal(np.array(Image.open(p)), encoded)
    overlay_image(rgb, mask, name+'.png', tta, name.startswith('ensemble_')).save(directory/'overlays'/(name+'.png'))
    np.savez_compressed(directory/'probabilities'/(name+'.npz'), foreground_probability=prob.astype(np.float32), coverage=coverage)
    return dict(info, status='complete', centerline_pixels=int(mask.sum()), mask=str(directory/(name+'.png')))


def model_runs(mode):
    dataset_names = {
        203: 'Dataset203_neurites_yvonne_smi_2px_scaleaug',
        301: 'Dataset301_neurite_yvonne_b2_smi',
        302: 'Dataset302_neurite_yvonne_b2_smi_1px',
        304: 'Dataset304_neurite_yvonne_b2_smi_1px_scaleaug',
        306: 'Dataset306_neurite_yvonne_b2_smi_topology_2px',
    }
    variants = ['red_distw10', 'red_distw', 'red', 'red_skelrec_distw10', 'red_skelrec_distw', 'red_skelrec', 'skelrec']
    rows = [(203, 'upernet_inj_ft_ours')]
    rows += [(301, 'convnextt_upernet_ft_aug_p512_red_ours')]
    rows += [(302, 'convnextt_upernet_ft_aug_p512_'+v+'_ours') for v in variants]
    rows += [(d, 'convnextt_upernet_ft_aug_p512_red_ours') for d in [304, 306]]
    records = []
    for dataset, config in rows:
        if mode == 'masked-region' and dataset != 304:
            continue
        variant = config.removesuffix('_ours').replace('convnextt_upernet_ft_aug_p512_', 'convnextt_')
        prefix = 'Dataset306_topology_2px' if dataset == 306 else f'Dataset{dataset}'
        records.append(dict(dataset=dataset, name=prefix+'__'+variant+'__fold0',
                            run=PROJECT/'models/dinov3'/dataset_names[dataset]/config/'fold_0'))
    return records


def ensemble_members(records):
    return {
        'ensemble_all10': [r for r in records if r['dataset'] != 306],
        'ensemble_302_only': [r for r in records if r['dataset'] == 302],
        'ensemble_304_only': [r for r in records if r['dataset'] == 304],
        'ensemble_302_304': [r for r in records if r['dataset'] in (302, 304)],
    }


def average_views(predict, plane, tta):
    accum = np.zeros(plane.shape, np.float32)
    counts = np.zeros(plane.shape, np.uint8)
    patches = []
    for k, flip in VIEWS if tta else VIEWS[:1]:
        prob, covered, n = predict(transform(plane, k, flip))
        prob, covered = inverse(prob, k, flip), inverse(covered, k, flip)
        accum += prob * covered
        counts += covered.astype(np.uint8)
        patches.append(n)
    np.divide(accum, counts, out=accum, where=counts > 0)
    accum[counts == 0] = 0
    return np.clip(accum, 0, 1), counts, patches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full-image', 'masked-region'], default='full-image')
    parser.add_argument('--input-dir', type=Path, default=PANEL, help='Contains images/drg_composite.png; masked mode also needs exclusion.png.')
    parser.add_argument('--output-dir', type=Path, help='Defaults to input directory; creates preds and preds_TTA.')
    parser.add_argument('--prediction', choices=['both', 'plain', 'tta'], default='both')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    base = args.input_dir.resolve()
    out = (args.output_dir or base).resolve()
    folders = ([] if args.prediction == 'tta' else ['preds']) + ([] if args.prediction == 'plain' else ['preds_TTA'])
    if args.mode == 'full-image' and 'preds' in folders:
        folders.append('preds_ensemble')
    check_outputs([out/f for f in folders], args.overwrite)
    source = base/'images/drg_composite.png'
    rgb = np.array(Image.open(source).convert('RGB'))
    assert not rgb[..., 1:].any(), 'Expected red-only DRG image'
    excluded = np.zeros(rgb.shape[:2], bool)
    if args.mode == 'masked-region':
        excluded = np.array(Image.open(base/'exclusion.png')) > 0
        assert excluded.shape == rgb.shape[:2] and not rgb[excluded].any()
    records = model_runs(args.mode)
    for r in records:
        for name in ['best.pt', 'config.yaml']:
            if not (r['run']/name).is_file():
                raise FileNotFoundError(r['run']/name)
    import torch
    from fm_adaptation.config import ExperimentConfig
    from fm_adaptation.models import load_trained_model
    torch.set_num_threads(4)
    manifests = {f: dict(source=str(source), source_sha256=sha(source), mode=args.mode,
                         postprocessing='probability >0.5, skeletonize; uint8 0/255; raw copies are centerlines',
                         aggregation='Equal covered-view probability means; equal model weights for ensembles',
                         results={}) for f in folders}
    for r in records:
        cfg = replace(ExperimentConfig.from_yaml(r['run']/'config.yaml'), results_dir=PROJECT/'models')
        model = load_trained_model(cfg, 'best', torch.device('cuda'), 2)
        model.eval()
        info = dict(checkpoint=str(r['run']/'best.pt'), checkpoint_sha256=sha(r['run']/'best.pt'),
                    config_sha256=sha(r['run']/'config.yaml'), patching=asdict(cfg.patching))
        for tta, folder in [(False, 'preds'), (True, 'preds_TTA')]:
            if folder not in folders:
                continue
            prob, cov, patches = average_views(lambda plane: predict_probs(model, cfg, plane), rgb[..., 0], tta)
            prob[excluded] = 0
            targets = [folder]
            if not tta and args.mode == 'full-image' and r['dataset'] != 306:
                targets.append('preds_ensemble')
            for target in targets:
                manifests[target]['results'][r['name']] = save_prediction(out/target, r['name'], prob, cov, rgb, tta,
                                                                          dict(info, views=8 if tta else 1, patch_counts=patches))
            print(folder, r['name'], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if args.mode == 'full-image':
        for folder in ['preds_ensemble', 'preds_TTA']:
            if folder not in folders:
                continue
            for name, members in ensemble_members(records).items():
                prob = np.zeros(rgb.shape[:2], np.float32)
                coverage = np.zeros(rgb.shape[:2], np.uint8)
                for member in members:
                    with np.load(out/folder/'probabilities'/(member['name']+'.npz')) as saved:
                        prob += saved['foreground_probability'] / len(members)
                        coverage += (saved['coverage'] > 0).astype(np.uint8)
                manifests[folder]['results'][name] = save_prediction(out/folder, name, np.clip(prob, 0, 1), coverage, rgb,
                                                                     folder == 'preds_TTA', dict(members=[m['name'] for m in members]))
    for folder, manifest in manifests.items():
        manifest['status'] = 'complete'
        (out/folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
