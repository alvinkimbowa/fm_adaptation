"""COCO mask AP/AR, preserving overlaps and an explicit 10,000 prediction limit."""
import contextlib
import io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
from data.instance_data import rle_encode


def global_rle(local, offset, shape):
    """Translate bbox-local RLE without allocating a slide-sized mask."""
    mask = mu.decode(local)
    y, x = np.nonzero(mask)
    return pixels_rle(y+offset[1], x+offset[0], shape)


def pixels_rle(y, x, shape):
    indices = np.unique(np.asarray(x, np.int64)*shape[0]+np.asarray(y, np.int64))
    if not len(indices):
        return rle_encode(np.zeros((1,1), np.uint8)) if tuple(shape)==(1,1) else _compress([int(np.prod(shape))], shape)
    starts = indices[np.r_[True, np.diff(indices)>1]]
    ends = indices[np.r_[np.diff(indices)>1, True]]+1
    counts = []
    previous = 0
    for start, end in zip(starts, ends):
        counts.extend([int(start-previous), int(end-start)])
        previous = end
    counts.append(int(np.prod(shape)-previous))
    return _compress(counts, shape)


def _compress(counts, shape):
    rle = mu.frPyObjects(dict(size=list(shape),counts=counts), *shape)
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


class InstanceEvaluator:
    def __init__(self, max_dets=10000):
        self.images, self.gt, self.pred = [], [], []
        self.max_dets = max_dets

    def add(self, shape, targets, predictions):
        image_id = len(self.images)+1
        self.images.append(dict(id=image_id,height=shape[0],width=shape[1]))
        for rle in targets:
            self.gt.append(dict(id=len(self.gt)+1,image_id=image_id,category_id=1,
                                segmentation=rle,area=float(mu.area(rle)),bbox=mu.toBbox(rle).tolist(),iscrowd=0))
        for rle, score in predictions:
            self.pred.append(dict(image_id=image_id,category_id=1,segmentation=rle,score=float(score)))

    def compute(self):
        coco=COCO()
        coco.dataset=dict(images=self.images,annotations=self.gt,categories=[dict(id=1,name='neurite')], info={})
        with contextlib.redirect_stdout(io.StringIO()):
            coco.createIndex()
            if self.pred:
                dt=coco.loadRes(self.pred)
            else:
                dt=COCO(); dt.dataset=dict(images=self.images,annotations=[],categories=coco.dataset['categories']);dt.createIndex()
            ev=COCOeval(coco,dt,'segm')
            ev.params.maxDets=[1,100,self.max_dets]
            ev.evaluate();ev.accumulate()
        # COCO's default summarize assumes 100 detections for AP; index directly.
        precision=ev.eval['precision'][:,:,:,0,-1]
        recall=ev.eval['recall'][:,:,0,-1]
        mean=lambda a: float(a[a>=0].mean()) if (a>=0).any() else 0.
        return dict(mask_AP=mean(precision),mask_AP50=mean(precision[0]),mask_AP75=mean(precision[5]),
                    mask_AR=mean(recall),max_predictions_per_image=self.max_dets,
                    images=len(self.images),targets=len(self.gt),predictions=len(self.pred))


def fiber_diagnostics(fibers, truth, predictions, confidence=.5):
    """Bounded IoU/coverage audit; flags are review candidates, not proven errors."""
    selected=[p['rle'] for p in predictions if p['score']>=confidence]
    best=np.zeros(len(truth));fragments=np.zeros(len(truth),int)
    for start in range(0,len(truth),64):
        block=truth[start:start+64]
        if selected:
            iou=mu.iou(block,selected,[0]*len(selected))
            coverage=mu.iou(block,selected,[1]*len(selected))
            best[start:start+len(block)]=iou.max(1)
            fragments[start:start+len(block)]=(coverage>=.2).sum(1)
    lengths=np.array([f['length'] for f in fibers])
    short=lengths<16
    return dict(confidence_threshold=confidence,
                short_original_fibers=int(short.sum()),
                short_recall_iou50=float((best[short]>=.5).mean()) if short.any() else None,
                missed_source_fibers_iou50=[f['id'] for f,v in zip(fibers,best) if v<.5],
                possible_fragmentation_source_ids=[f['id'] for f,n in zip(fibers,fragments) if n>1])
