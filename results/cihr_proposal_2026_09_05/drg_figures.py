"""Generate final DRG comparison figures, with original spacing and alpha."""
import argparse
from pathlib import Path
import hashlib
import json
import shutil
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from drg_inference import PANEL, check_outputs

DEFAULT_OUTPUT = Path(__file__).resolve().parent/'figures/drg'

CAPTIONS = {'adaptive_threshold_vs_convnext_tta.txt': 'Fig. X. Adaptive-threshold and model-based neurite segmentation. (a) Confocal maximum-intensity projection; (b) manual tracing; (c) customized adaptive-threshold pipeline; (d) Dataset304-trained ConvNeXt-Tiny with test-time augmentation, gap joining, endpoint extension, and removal of components smaller than 12 pixels. Paired arrows highlight source-supported neurites missing or interrupted in (c) but retained in (d). Model traces are thickened for display.\n', 'neuritej_vs_convnext_tta.txt': 'Fig. X. Global-threshold and model-based neurite segmentation. (a) Confocal maximum-intensity projection; (b) manual tracing; (c) NeuriteJ global-threshold segmentation; (d) Dataset304-trained ConvNeXt-Tiny with test-time augmentation, gap joining, endpoint extension, and removal of components smaller than 12 pixels. Original arrows mark regions of divergent segmentation. Model traces are thickened for display.\n'}
ALIASES = {'neuritej_vs_convnext_tta.png': 'Fig1 - model vs global thresholding.png', 'adaptive_threshold_vs_convnext_tta.png': 'Fig2 - model vs adaptive thresholding.png'}

POINTS=[
    (569,597,(60,-35),'Left-middle region: a faint manually traced neurite is absent at the matched former-d point but retained by the model, with support in the red image.'),
    (1349,287,(60,-40),'Dense upper-right region: model retains a fine strand supported by the manual tracing and red signal, absent at the matched point in former d.'),
    (971,335,(-55,45),'Inner upper bundle: a short source-supported connection appears in model and manual tracing but is absent in former d.'),
    (1280,700,(60,35),'Right-lower branch: model follows a manually traced strand missing at the corresponding former-d location.'),
    (596,878,(65,20),'Lower-left bundle: model retains a manually traced, source-supported neurite segment missing at the matched former-d point.'),
]

def comparisons(panel, source, prediction_dir, output_dir):
    pred_path = prediction_dir / 'preds_TTA_joined_extended_min12/Dataset304__convnextt_red__fold0.png'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(p): sha(p) for p in [source, pred_path]}
    original = np.array(Image.open(source))
    assert original.shape == (2439, 4540, 4)
    pred = np.array(Image.open(pred_path))
    # Display-only 3-pixel strokes; keep the excluded region blank.
    display_pred = cv2.dilate(pred, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    excluded = np.array(Image.open(prediction_dir / 'exclusion.png')) > 0
    display_pred[excluded] = 0
    transforms = {key: json.loads((panel / 'registration' / f'transform_{key}.json').read_text()) for key in ['c', 'd']}
    crops = {key: tuple(info['source_crop_xyxy']) for key, info in transforms.items()}
    matrices = {key: np.array(info['mask_to_image'], np.float64) for key, info in transforms.items()}
    x0,y0,x1,y1 = crops['d']
    donor = original[y0:y1,x0:x1,:3]
    neutral = (donor[:,:,0] == donor[:,:,1]) & (donor[:,:,1] == donor[:,:,2])
    d_gray = np.where(neutral, donor[:,:,0], 0).astype(np.uint8)
    # Map directly between panel frames to avoid clipping through the image frame.
    c3 = np.vstack([matrices['c'], [0,0,1]])
    d3 = np.vstack([matrices['d'], [0,0,1]])
    d_to_c = (np.linalg.inv(c3) @ d3)[:2]
    d_in_c = cv2.warpAffine(d_gray, d_to_c, (1920,1220), flags=cv2.INTER_NEAREST)
    pred_in_d = cv2.warpAffine(display_pred, cv2.invertAffineTransform(matrices['d']), (1920,1220), flags=cv2.INTER_NEAREST)
    assert np.array_equal(d_to_c, np.array([[1,0,67],[0,1,0]]))
    assert set(np.unique(pred_in_d)) <= {0,255}
    # Exact integer translation check, including panel placement of every foreground pixel.
    yy,xx = np.where(pred > 0)
    inside = (xx-69>=0) & (xx-69<1920) & (yy+47<1220)
    assert np.all(pred_in_d[yy[inside]+47,xx[inside]-69] == 255)

    def replace_gray(canvas, key, grayscale):
        x0,y0,x1,y1 = crops[key]
        old = original[y0:y1,x0:x1]
        rgb = old[:,:,:3].astype(np.int16)
        orange = (rgb[:,:,0]>rgb[:,:,1]) & (rgb[:,:,1]>rgb[:,:,2])
        replacement = np.empty_like(old)
        replacement[:,:,:3] = grayscale[:,:,None]
        replacement[:,:,3] = 255
        replacement[orange] = old[orange]
        canvas[y0:y1,x0:x1] = replacement
        assert np.array_equal(canvas[y0:y1,x0:x1][orange], old[orange])

    output_dir.mkdir(parents=True, exist_ok=True)
    for other in ['d','c']:
        canvas = original.copy()
        if other == 'd':
            replace_gray(canvas,'c',d_in_c)
        replace_gray(canvas,'d',pred_in_d)
        assert np.array_equal(canvas[:1219],original[:1219])
        if other == 'c': assert np.array_equal(canvas[1219:,:2270],original[1219:,:2270])
        # Check all five arrows per slot, including antialiased orange edge pixels.
        for offset in [0,2270]:
            src = original[1219:,offset:offset+2270]
            rgb=src[:,:,:3].astype(np.int16)
            orange=(rgb[:,:,0]>rgb[:,:,1]) & (rgb[:,:,1]>rgb[:,:,2])
            assert np.array_equal(canvas[1219:,offset:offset+2270][orange],src[orange])
        filename = 'adaptive_threshold_vs_convnext_tta.png' if other == 'd' else 'neuritej_vs_convnext_tta.png'
        path=output_dir/filename
        canvas[:, :, 3] = original[:, :, 3]
        Image.fromarray(canvas).save(path)
        assert np.array_equal(np.array(Image.open(path)),canvas)
        print(path,flush=True)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    (output_dir/'verification.json').write_text(json.dumps(dict(source_sha256=hashes,prediction=str(pred_path),display_dilation='3x3 ellipse, one iteration; exclusion reapplied',d_to_c=d_to_c.tolist(),prediction_to_d=cv2.invertAffineTransform(matrices['d']).tolist(),top_row_unchanged=True,arrows_unchanged=True,output_size=[4540,2439]),indent=2)+'\n')

def arrows(panel, source, prediction_dir, output_dir):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    pred_path=prediction_dir/'preds_TTA_joined_extended_min12/Dataset304__convnextt_red__fold0.png'
    preserved={str(p):sha(p) for p in [source,pred_path]}
    original=np.array(Image.open(source))
    canvas=original.copy()
    pred=np.array(Image.open(pred_path))
    excluded=np.array(Image.open(prediction_dir/'exclusion.png'))>0
    display=cv2.dilate(pred,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    display[excluded]=0
    transforms={k:json.loads((panel/f'registration/transform_{k}.json').read_text()) for k in ['c','d']}
    matrices={k:np.array(v['mask_to_image'],float) for k,v in transforms.items()}
    crops={k:v['source_crop_xyxy'] for k,v in transforms.items()}
    x0,y0,x1,y1=crops['d']
    donor=original[y0:y1,x0:x1,:3]
    neutral=(donor[:,:,0]==donor[:,:,1]) & (donor[:,:,1]==donor[:,:,2])
    gray=np.where(neutral,donor[:,:,0],0).astype('uint8')
    to_c=(np.linalg.inv(np.vstack([matrices['c'],[0,0,1]])) @ np.vstack([matrices['d'],[0,0,1]]))[:2]
    clean={
        'c':cv2.warpAffine(gray,to_c,(1920,1220),flags=cv2.INTER_NEAREST),
        'd':cv2.warpAffine(display,cv2.invertAffineTransform(matrices['d']),(1920,1220),flags=cv2.INTER_NEAREST),
    }
    for k,img in clean.items():
        x0,y0,x1,y1=crops[k]
        canvas[y0:y1,x0:x1,:3]=img[:,:,None]
        canvas[y0:y1,x0:x1,3]=255
    before_arrows=canvas.copy()
    rgb=original[1219:,:,:3].astype(int)
    orange=(rgb[:,:,0]>rgb[:,:,1]) & (rgb[:,:,1]>rgb[:,:,2])
    colors,counts=np.unique(rgb[orange],axis=0,return_counts=True)
    color=tuple(int(v) for v in colors[counts.argmax()])
    image=np.array(Image.open(panel/'images/drg_composite.png'))[:,:,:3]
    manual=np.array(Image.open(panel/'masks/drg_composite.png'))>0
    former=np.array(Image.open(panel/'masks/drg_composite_d.png'))>0
    manual_dist=ndi.distance_transform_edt(~manual)
    former_dist=ndi.distance_transform_edt(~former)
    support=ndi.maximum_filter(image[:,:,0],3)
    annotations=[]
    footprints=np.zeros(canvas.shape[:2],bool)
    for x,y,offset,note in POINTS:
        assert pred[y,x]>0 and not excluded[y,x]
        assert manual_dist[y,x]<=4 and former_dist[y,x]>=7 and support[y,x]>80
        locations={}
        for k in ['c','d']:
            local=cv2.invertAffineTransform(matrices[k]) @ np.array([x,y,1.])
            origin=np.array(crops[k][:2])
            target=local+origin
            delta=np.array(offset,float)
            tip=tuple(np.rint(target+delta/np.linalg.norm(delta)*5).astype(int))
            tail=tuple(np.rint(target+delta).astype(int))
            alpha=np.zeros(canvas.shape[:2],np.uint8)
            # Match the original 26 px filled triangular head and 6 px shaft.
            direction=(np.array(tip,float)-tail)
            direction/=np.linalg.norm(direction)
            normal=np.array([-direction[1],direction[0]])
            head_base=np.array(tip)-26*direction
            outline=np.array([
                np.array(tail)+3*normal, head_base+3*normal,
                head_base+13*normal, tip, head_base-13*normal,
                head_base-3*normal, np.array(tail)-3*normal,
            ])
            cv2.fillPoly(alpha,[np.rint(outline*256).astype(np.int32)],255,
                         lineType=cv2.LINE_AA,shift=8)
            active=alpha>0
            weight=alpha[active,None]/255.
            canvas[active,:3]=np.rint(canvas[active,:3]*(1-weight)+np.array(color)*weight).astype('uint8')
            canvas[active,3]=255
            footprints|=active
            # Both panel targets must map back to the same source-image coordinate.
            assert np.allclose(matrices[k] @ np.r_[local,1.],[x,y])
            locations[k]=dict(target_xy=target.tolist(),tail_xy=list(map(int,tail)),tip_xy=list(map(int,tip)))
        annotations.append(dict(image_xy=[x,y],note=note,panels=locations,manual_distance_pixels=float(manual_dist[y,x]),former_d_distance_pixels=float(former_dist[y,x]),local_red_max=int(support[y,x])))
    assert np.array_equal(canvas[:1219],original[:1219])
    assert np.array_equal(canvas[~footprints],before_arrows[~footprints])
    output_dir.mkdir(parents=True, exist_ok=True)
    output=output_dir/'adaptive_threshold_vs_convnext_tta.png'
    canvas[:, :, 3] = original[:, :, 3]
    Image.fromarray(canvas).save(output)
    assert np.array_equal(np.array(Image.open(output)),canvas)
    (output_dir/'arrow_annotations.json').write_text(json.dumps(dict(arrow_color_rgb=color,annotations=annotations,preserved_sha256=preserved,description='Five matched local examples; qualitative source support, not a claim of global model superiority.'),indent=2)+'\n')
    assert all(sha(Path(p))==v for p,v in preserved.items())
    print('Verified five matched arrow pairs; top row, grayscale outside arrow footprints, model mask, and a/b/c figure unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PANEL, help='Original DRG images, masks, and registration.')
    parser.add_argument('--prediction-dir', type=Path, help='Masked region directory, including exclusion.png and processed predictions.')
    parser.add_argument('--source-composite', type=Path, help='Defaults to input-dir/../DRG-composite.png.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--operation', choices=['all', 'comparisons', 'arrows'], default='all')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    panel = args.input_dir.resolve()
    source = (args.source_composite or panel.parent/'DRG-composite.png').resolve()
    prediction_dir = (args.prediction_dir or panel/'masked_region').resolve()
    output = args.output_dir.resolve()
    names = ['adaptive_threshold_vs_convnext_tta.png', 'arrow_annotations.json'] if args.operation == 'arrows' else [
        'adaptive_threshold_vs_convnext_tta.png', 'neuritej_vs_convnext_tta.png', 'verification.json']
    if args.operation == 'all':
        names.append('arrow_annotations.json')
    figures = [name for name in names if name.endswith('.png')]
    names += [ALIASES[name] for name in figures] + [Path(name).with_suffix('.txt').name for name in figures]
    check_outputs([output/name for name in names], args.overwrite)
    if args.operation in ['all', 'comparisons']:
        comparisons(panel, source, prediction_dir, output)
    if args.operation in ['all', 'arrows']:
        arrows(panel, source, prediction_dir, output)
    for name in figures:
        shutil.copy2(output/name, output/ALIASES[name])
        caption = Path(name).with_suffix('.txt').name
        (output/caption).write_text(CAPTIONS[caption])


if __name__ == '__main__':
    main()
