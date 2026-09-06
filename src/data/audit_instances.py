"""Exhaustive validation-grid and seeded augmentation audit, plus native QC crops."""
import json
from pathlib import Path
import cv2
import numpy as np
from fm_adaptation.config import ExperimentConfig
from data.instance_data import InstancePatches,red_image
from fm_adaptation.models import DINOv3Encoder

cfg=ExperimentConfig.from_yaml('configs/convnextt_m2f_ft_aug_p256_red_ours.yaml')
out=Path('results/instance_preflight');out.mkdir(parents=True,exist_ok=True)
report={}
for subset in ['train','val']:
    dataset=InstancePatches(cfg,subset)
    counts=[]
    for i in range(len(dataset)):
        image,t=dataset[i];counts.append(len(t['masks']))
        if i<8 or (len(t['masks'])>=50 and not (out/(subset+'_dense.png')).exists()):
            rgb=((image.numpy()*np.array(DINOv3Encoder.std)[:,None,None]+np.array(DINOv3Encoder.mean)[:,None,None])*255).round().clip(0,255).astype(np.uint8).transpose(1,2,0)
            assert not rgb[...,1:].any()
            overlay=rgb.copy()
            rng=np.random.default_rng(0)
            for mask in t['masks']:
                color=rng.integers(40,255,3)
                overlay[mask]=np.maximum(overlay[mask],color)
            overlay[t['ignore']]=[255,255,0]
            panel=np.concatenate([rgb,overlay],axis=1)
            filename=f'{subset}_{i:04d}.png' if i<8 else subset+'_dense.png'
            cv2.imwrite(str(out/filename),panel[...,::-1])
        if (i+1)%100==0:print(subset,i+1,'/',len(dataset),'max',max(counts),flush=True)
    report[subset]=dict(patches=len(counts),max_instances=max(counts),p99=float(np.percentile(counts,99)),empty=sum(n==0 for n in counts))
(out/'audit.json').write_text(json.dumps(report,indent=2));print(report)
