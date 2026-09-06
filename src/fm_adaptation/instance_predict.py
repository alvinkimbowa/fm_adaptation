"""Native tiling, instance stitching, slide COCO export and union metrics."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from .config import ExperimentConfig
from data.instance_data import positions,red_image,normalize
from .instance_model import InstanceModel,validate_config
from .instance_training import restore
from .instance_stitch import stitch
from .instance_metrics import InstanceEvaluator,global_rle,fiber_diagnostics


@torch.no_grad()
def predict(cfg, checkpoint='best', subset='test'):
    validate_config(cfg);torch.set_num_threads(4)
    checkpoint = 'last' if checkpoint == 'final' else checkpoint
    model=InstanceModel(cfg).to(cfg.device).eval();restore(cfg.run_dir/(checkpoint+'.pt'),model)
    root=cfg.instance_data_dir;folds=json.loads((root/'splits_final.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted((root/'annotations').glob('*.json'))]
    records=[r for r in records if (r['case'] in folds[int(cfg.fold)]['val'] if subset=='val' else r['image'].startswith('imagesTs/'))]
    output=cfg.run_dir/('instance_predictions_'+subset);output.mkdir(exist_ok=True,parents=True)
    evaluator=InstanceEvaluator();rows=[]
    for rec in records:
        case=rec['case'];case_dir=output/case;case_dir.mkdir(exist_ok=True)
        gray=cv2.imread(str(root/rec['image']),-1);shape=gray.shape
        predictions=[];tile_id=0;saturated=0;tile_provenance=[]
        for top in positions(shape[0],cfg.input_size,cfg.patching.stride):
            for left in positions(shape[1],cfg.input_size,cfg.patching.stride):
                tile=np.zeros((cfg.input_size,cfg.input_size),np.uint8)
                crop=gray[top:top+cfg.input_size,left:left+cfg.input_size];tile[:crop.shape[0],:crop.shape[1]]=crop
                pred=model.predict(normalize(red_image(tile))[None].to(cfg.device))[0]
                # Disk-backed probability maps bound resident memory independently of slide size.
                path=case_dir/f'tile_{tile_id:06d}.npy'
                np.save(path,pred['probabilities'].astype(np.float16))
                maps=np.load(path,mmap_mode='r')
                for i,score in enumerate(pred['scores']):
                    predictions.append(dict(tile=tile_id,origin=(left,top),query_id=int(pred['query_ids'][i]),probability=maps[i],score=float(score)))
                tile_provenance.append(dict(id=tile_id,origin=[left,top],probability_file=path.name,predictions=len(maps),query_saturation=pred['query_saturation']))
                saturated+=pred['query_saturation'];tile_id+=1
        instances,audit=stitch(predictions,shape)
        truth=[global_rle(f['rle'],f['offset'],shape) for f in rec['fibers']]
        evaluator.add(shape,truth,[(p['rle'],p['score']) for p in instances])
        union=np.zeros(shape,bool);target=np.zeros(shape,bool)
        for p in instances:union|=mu.decode(p['rle']).astype(bool)
        for rle in truth:target|=mu.decode(rle).astype(bool)
        from skimage.morphology import skeletonize
        from scipy.ndimage import distance_transform_edt,binary_erosion
        dice=float((2*(union&target).sum()+1e-8)/(union.sum()+target.sum()+1e-8))
        sp,st=skeletonize(union),skeletonize(target)
        precision=float((sp&target).sum()/max(1,sp.sum()));recall=float((st&union).sum()/max(1,st.sum()))
        cldice=2*precision*recall/max(1e-8,precision+recall)
        bp=union^binary_erosion(union);bt=target^binary_erosion(target)
        d1=distance_transform_edt(~bt)[bp];d2=distance_transform_edt(~bp)[bt]
        distances=np.r_[d1,d2] if bp.any() and bt.any() else np.array([np.inf])
        binary=dict(dice=dice,cldice=cldice,assd_px=float(distances.mean()),hd95_px=float(np.percentile(distances,95)))
        cv2.imwrite(str(case_dir/'binary_union.png'),union.astype(np.uint8))
        diagnostics=fiber_diagnostics(rec['fibers'],truth,instances)
        (case_dir/'fiber_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        # Native-resolution false-negative/false-positive QC, selected without resizing.
        errors=union^target
        best_box=(0,0);best_errors=-1
        for top in positions(shape[0]):
            for left in positions(shape[1]):
                count=int(errors[top:top+256,left:left+256].sum())
                if count>best_errors:best_errors=count;best_box=(left,top)
        left,top=best_box
        qc=red_image(gray[top:top+256,left:left+256]).copy()
        gt=target[top:top+256,left:left+256];pr=union[top:top+256,left:left+256]
        qc[gt & ~pr]=[0,255,0];qc[pr & ~gt]=[0,128,255];qc[gt & pr]=[255,255,255]
        cv2.imwrite(str(case_dir/'error_crop.png'),qc[...,::-1])
        row=dict(case=case,tiles=tile_id,saturated_tiles=saturated,
                 short_recall_iou50=diagnostics['short_recall_iou50'],
                 possible_fragmentation_fibers=len(diagnostics['possible_fragmentation_source_ids']),
                 **audit,**binary)
        (case_dir/'instances.json').write_text(json.dumps(dict(shape=shape,instances=instances,tiles=tile_provenance,audit=row)))
        rows.append(row);print(json.dumps(row),flush=True)
    result=dict(instance_metrics=evaluator.compute(),cases=rows,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                stitching=dict(stride=cfg.patching.stride,duplicate_iou=.7,shared_support_px=16,mutual_coverage=.7,distance_px=2,direction_degrees=30))
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    evaluator_data=dict(images=evaluator.images,annotations=evaluator.gt,categories=[dict(id=1,name='neurite')])
    (output/'ground_truth.coco.json').write_text(json.dumps(evaluator_data))
    (output/'predictions.coco.json').write_text(json.dumps(evaluator.pred))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',default='best',choices=['best','last','final']);p.add_argument('--subset',default='test',choices=['test','val'])
    a=p.parse_args();predict(ExperimentConfig.from_yaml(a.config),a.checkpoint,a.subset)
