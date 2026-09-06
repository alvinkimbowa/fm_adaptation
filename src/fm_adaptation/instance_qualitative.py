"""Fixed validation examples rendered from the already-computed validation pass."""
from pathlib import Path
import cv2
import numpy as np
from .models import DINOv3Encoder


def render_samples(run_dir, epoch, samples, threshold=.5):
    if not samples:return None
    output=Path(run_dir)/'qualitative';output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for image,target,prediction in samples:
        rgb=((image.numpy()*np.array(DINOv3Encoder.std)[:,None,None]+np.array(DINOv3Encoder.mean)[:,None,None])*255).round().clip(0,255).astype(np.uint8).transpose(1,2,0)
        def overlay(masks,scores=None):
            panel=np.ascontiguousarray(rgb*.3,dtype=np.uint8)
            rng=np.random.default_rng(7)
            for i,mask in enumerate(masks):
                color=tuple(int(x) for x in rng.integers(70,255,3))
                panel[mask]=np.maximum(panel[mask],color)
                if scores is not None and mask.any():
                    y,x=np.column_stack(np.nonzero(mask))[0]
                    cv2.putText(panel,f'{i}:{scores[i]:.2f}',(int(x),max(8,int(y))),cv2.FONT_HERSHEY_SIMPLEX,.24,(255,255,255),1,cv2.LINE_AA)
            panel[target['ignore']]=[255,210,0]
            return panel
        keep=prediction['scores']>=threshold
        masks=prediction['probabilities'][keep]>.5;scores=prediction['scores'][keep]
        panels=[rgb,overlay(target['masks']),overlay(masks,scores)]
        titles=['SMI input (red)',f"Ground truth: {len(target['masks'])} fibers",f'Predicted: {len(masks)} (score >= {threshold:g})']
        labeled=[]
        for panel,title in zip(panels,titles):
            header=np.full((24,256,3),245,np.uint8)
            cv2.putText(header,title,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.38,(25,25,25),1,cv2.LINE_AA)
            labeled.append(np.concatenate([header,panel]))
        row=np.concatenate(labeled,axis=1)
        caption=np.full((27,768,3),245,np.uint8)
        case=target['case'].removeprefix('Yvonne_b2__')
        cv2.putText(caption,f"Epoch {epoch} | {case} | x={target['left']}, y={target['top']} | yellow=ignored",(5,18),cv2.FONT_HERSHEY_SIMPLEX,.35,(25,25,25),1,cv2.LINE_AA)
        rows.append(np.concatenate([caption,row]))
    canvas=np.concatenate(rows)
    path=output/f'epoch_{epoch:03d}.png'
    if not cv2.imwrite(str(path),canvas[...,::-1]):raise OSError(f'Could not save {path}')
    latest=output/'latest.png';tmp=output/'latest.tmp.png'
    if not cv2.imwrite(str(tmp),canvas[...,::-1]):raise OSError(f'Could not save {tmp}')
    tmp.replace(latest)
    return path
