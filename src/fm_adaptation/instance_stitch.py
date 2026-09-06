"""Conservative shared-region matching and bounded spatial probability fusion."""
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import skeletonize
from .instance_metrics import pixels_rle


def shared_masks(a,b):
    ax,ay=a['origin'];bx,by=b['origin']; ah,aw=a['probability'].shape;bh,bw=b['probability'].shape
    x,y=max(ax,bx),max(ay,by);right,bottom=min(ax+aw,bx+bw),min(ay+ah,by+bh)
    if x>=right or y>=bottom:return None
    return (a['probability'][y-ay:bottom-ay,x-ax:right-ax]>.5,
            b['probability'][y-by:bottom-by,x-bx:right-bx]>.5)


def direction(skeleton):
    points=np.column_stack(np.nonzero(skeleton))
    if len(points)<2:return None
    _,_,v=np.linalg.svd(points-points.mean(0),full_matrices=False)
    return v[0]


def compatible(a,b):
    shared=shared_masks(a,b)
    if shared is None:return False
    ma,mb=shared
    if not ma.any() or not mb.any():return False
    iou=(ma&mb).sum()/(ma|mb).sum()
    # Fully visible high-IoU duplicates include original short fibers; the 16 px
    # support rule applies to continuation, not to exact contained duplicates.
    if iou >= .7 and ma.sum() == (a['probability'] > .5).sum() and mb.sum() == (b['probability'] > .5).sum():
        return True
    ca,na=label(ma,structure=np.ones((3,3)))
    cb,nb=label(mb,structure=np.ones((3,3)))
    for ia in range(1,na+1):
        sa=skeletonize(ca==ia)
        if sa.sum()<16:continue
        da=direction(sa)
        for ib in range(1,nb+1):
            sb=skeletonize(cb==ib)
            if sb.sum()<16:continue
            db=direction(sb)
            angle=np.degrees(np.arccos(np.clip(abs(da@db),0,1)))
            if angle>30:continue
            support_a=(sa & (distance_transform_edt(~sb)<=2)).sum()
            support_b=(sb & (distance_transform_edt(~sa)<=2)).sum()
            if min(support_a,support_b)>=16 and support_a/sa.sum()>=.7 and support_b/sb.sum()>=.7:
                return True
    return False


def groups(predictions):
    """Accept unique tile-pair matches; reject contradictory component merges.

    Disconnected runs from one tile can share identity only through a second tile
    whose prediction independently passes the continuation gates for each run.
    """
    parent=list(range(len(predictions))); members={i:{i} for i in parent}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    tiles={}
    for i,p in enumerate(predictions):tiles.setdefault(p['tile'],[]).append(i)
    edges=[]; ambiguous=0
    keys=sorted(tiles)
    for ti,ta in enumerate(keys):
        for tb in keys[ti+1:]:
            aa,bb=tiles[ta],tiles[tb]
            if shared_masks(predictions[aa[0]],predictions[bb[0]]) is None:continue
            candidates=[(i,j) for i in aa for j in bb if compatible(predictions[i],predictions[j])]
            for i,j in candidates:
                other_a=[u for u,v in candidates if v==j and u!=i]
                other_b=[v for u,v in candidates if u==i and v!=j]
                # Multiple disjoint pieces may reconnect; overlapping alternatives are ambiguous.
                def conflict(indices, base):
                    for k in indices:
                        ma,mb=shared_masks(predictions[k],predictions[base])
                        if (ma&mb).any():return True
                    return False
                if conflict(other_a,i) or conflict(other_b,j):ambiguous+=1;continue
                edges.append((i,j))
    for i,j in edges:
        ri,rj=root(i),root(j)
        if ri==rj:continue
        conflict=False
        for a in members[ri]:
            for b in members[rj]:
                if predictions[a]['tile']==predictions[b]['tile']:
                    ma,mb=shared_masks(predictions[a],predictions[b])
                    if (ma&mb).any():conflict=True
                elif shared_masks(predictions[a],predictions[b]) is not None:
                    ma,mb=shared_masks(predictions[a],predictions[b])
                    if ma.any() and mb.any() and not compatible(predictions[a],predictions[b]):conflict=True
        if conflict:ambiguous+=1;continue
        parent[rj]=ri;members[ri]|=members.pop(rj)
    return [sorted(v) for _,v in sorted(members.items())],ambiguous


def fuse(predictions, members, shape):
    """Fuse in 256x256 buffers; retain only sparse foreground coordinates."""
    selected=[predictions[i] for i in members]
    lo=np.min([p['origin'] for p in selected],0)
    hi=np.max([np.array(p['origin'])+p['probability'].shape[::-1] for p in selected],0)
    ys,xs=[],[]
    for y in range(int(lo[1]),int(hi[1]),256):
        for x in range(int(lo[0]),int(hi[0]),256):
            height,width=min(256,shape[0]-y),min(256,shape[1]-x)
            if height<=0 or width<=0:continue
            numerator=np.zeros((height,width),np.float32);denominator=np.zeros_like(numerator)
            by_tile = {}
            for p in selected:
                by_tile.setdefault(p['tile'], []).append(p)
            for pieces in by_tile.values():
                p = pieces[0]
                px,py=p['origin'];ph,pw=p['probability'].shape
                left,top=max(x,px),max(y,py);right,bottom=min(x+width,px+pw),min(y+height,py+ph)
                if left>=right or top>=bottom:continue
                yy=np.arange(top-py,bottom-py);xx=np.arange(left-px,right-px)
                weight=(.05+np.sin(np.pi*(yy+.5)/ph))[:,None]*(.05+np.sin(np.pi*(xx+.5)/pw))[None,:]
                sl=np.s_[top-y:bottom-y,left-x:right-x]
                probability=np.maximum.reduce([piece['probability'][top-py:bottom-py,left-px:right-px] for piece in pieces])
                numerator[sl]+=probability*weight
                denominator[sl]+=weight
            mask=(numerator/np.maximum(denominator,1e-8))>.5
            yy,xx=np.nonzero(mask);ys.append(yy+y);xs.append(xx+x)
    yy=np.concatenate(ys) if ys else np.array([],int);xx=np.concatenate(xs) if xs else np.array([],int)
    return dict(rle=pixels_rle(yy,xx,shape),score=float(np.mean([p['score'] for p in selected])),
                tiles=sorted(set(p['tile'] for p in selected)),members=members,
                provenance=[dict(tile=p['tile'],origin=list(p['origin']),query_id=p.get('query_id'),score=p['score']) for p in selected])


def stitch(predictions, shape):
    components,ambiguous=groups(predictions)
    output=[dict(id=i+1,**fuse(predictions,members,shape)) for i,members in enumerate(components)]
    return output,dict(local_predictions=len(predictions),slide_instances=len(output),ambiguous_matches=ambiguous)
