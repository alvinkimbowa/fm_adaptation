"""Conservative single-pass, image-supported extensions of joined centerlines."""
import math
import numpy as np
from scipy import ndimage
from skimage.draw import line
from skimage.morphology import skeletonize

def extend_endpoints(mask, evidence, joining, max_length=12., max_angle=30., angle_step=5., min_support=.08, min_length=3.):
    result = skeletonize(mask).copy()
    points = joining.endpoints(result)
    labels, _ = ndimage.label(result, structure=np.ones((3,3)))
    threshold = max(min_support, float(np.percentile(evidence[result],20))) if result.any() else min_support
    floor = max(min_support, .5*threshold)
    candidates=[]
    for endpoint in points:
        tangent=joining.outward_tangent(result, endpoint)
        if tangent is None: continue
        choices=[]
        for degrees in np.arange(-max_angle,max_angle+.1,angle_step):
            a=math.radians(degrees)
            direction=np.array([math.cos(a)*tangent[0]-math.sin(a)*tangent[1],math.sin(a)*tangent[0]+math.cos(a)*tangent[1]])
            target=np.rint(endpoint+max_length*direction).astype(int)
            rr,cc=line(*endpoint,*target)
            path=[]; distance=0.; prev=endpoint
            for r,c in zip(rr[1:],cc[1:]):
                if not (0<=r<result.shape[0] and 0<=c<result.shape[1]): break
                distance += float(np.linalg.norm(np.array([r,c])-prev)); prev=np.array([r,c])
                if distance>max_length or evidence[r,c]<floor: break
                neighbors=labels[max(0,r-1):r+2,max(0,c-1):c+2]
                if result[r,c]: break
                # Stop before intersecting any existing skeleton, except the starting branch nearby.
                if len(path)>=2 and np.any(neighbors): break
                if np.any((neighbors!=0)&(neighbors!=labels[tuple(endpoint)])): break
                path.append([int(r),int(c)])
            if not path: continue
            path=np.array(path)
            length=float(np.linalg.norm(np.diff(np.vstack([endpoint,path]),axis=0),axis=1).sum())
            support=float(evidence[path[:,0],path[:,1]].mean())
            if length<min_length or support<threshold: continue
            score=length*support*math.cos(a)
            choices.append((score,path,length,support,float(degrees)))
        if choices:
            best=max(choices,key=lambda x:x[0]); candidates.append((best[0],endpoint,*best[1:]))
    accepted=[]
    for _,endpoint,path,length,support,angle in sorted(candidates,key=lambda x:x[0],reverse=True):
        # Earlier extensions may now occupy this route or a neighboring pixel.
        added=result & ~mask
        if any(added[max(0,r-1):r+2,max(0,c-1):c+2].any() for r,c in path): continue
        result[path[:,0],path[:,1]]=True
        accepted.append({'endpoint':endpoint.tolist(),'path':path.tolist(),'length':length,'mean_support':support,'angle_degrees':angle})
    return skeletonize(result), {'extensions':accepted,'support_threshold':threshold,'per_pixel_support_floor':floor}
