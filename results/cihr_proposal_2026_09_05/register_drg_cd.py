from pathlib import Path
import cv2,json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from skimage.morphology import skeletonize
p=Path('/home/ultrai/GAA/spinal_cord_injury/data/raw_data/DRG-panels')
src=np.asarray(Image.open(p.parent/'DRG-composite.png').convert('RGB'))
a=np.asarray(Image.open(p/'images/drg_composite.png').convert('RGB'))
h,w=a.shape[:2]
r=a[:,:,0].astype(np.float32)/255
r=np.clip((r-gaussian_filter(r,8))/.4,0,1)
for name,x0 in [('c',0),('d',2270)]:
 # Keep all bottom-panel rows for registration; trim the label margin only.
 raw=src[1219:,x0+350:x0+2270]
 # White/gray traces only: orange arrows are excluded by their blue channel.
 mask=(raw.min(axis=2)>=128).astype(np.uint8)*255
 skel=skeletonize(mask>0)
 shifts=[]
 for sigma in [1.5,3]:
  field=gaussian_filter(r,sigma)
  cor=fftconvolve(field,skel[::-1,::-1],mode='full')
  cy,cx=np.array(mask.shape)-1
  window=cor[cy-180:cy+181,cx-180:cx+181]
  yy,xx=np.unravel_index(window.argmax(),window.shape)
  shifts.append((int(xx-180),int(yy-180)))
 tx,ty=shifts[0]
 transform=np.array([[1,0,tx],[0,1,ty]],dtype=np.float32)
 aligned=cv2.warpAffine(mask,transform,(w,h),flags=cv2.INTER_NEAREST)
 overlay=a.copy();overlay[:,:,1]=aligned
 Image.fromarray(mask).save('/tmp/drg_'+name+'_raw.png')
 Image.fromarray(aligned).save('/tmp/drg_'+name+'_aligned.png')
 Image.fromarray(overlay).save('/tmp/drg_'+name+'_overlay.png')
 print(name,'shifts at sigma 1.5 and 3:',shifts,'foreground before/after:',np.count_nonzero(mask),np.count_nonzero(aligned),flush=True)
 Path('/tmp/drg_'+name+'_transform.json').write_text(json.dumps({'source_panel':name,'source_crop_xyxy':[x0+350,1219,x0+2270,2439],'mask_to_image':transform.tolist(),'output_size':[w,h],'mask_extraction':'All RGB channels >=128; excludes orange arrows','registration':'Translation cross-correlation of skeleton against background-suppressed red signal','translation_checks':shifts},indent=2)+'\n')
