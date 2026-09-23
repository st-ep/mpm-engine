"""Cache recorded A/B pressing frames for the shaping slide; no new simulation."""
from pathlib import Path
import hashlib,json
import numpy as np
import cv2
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CROP=(33,180,1247,1052)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def specimen_mask(rgb):
    gray=rgb.mean(2).astype('float32')
    variance=np.maximum(0,cv2.blur(gray*gray,(7,7))-cv2.blur(gray,(7,7))**2)
    texture=(np.sqrt(variance)>17).astype('uint8')
    texture[:,:165]=0;texture[:,540:]=0
    texture=cv2.morphologyEx(texture,cv2.MORPH_OPEN,np.ones((17,17),'uint8'))
    count,labels,stats,_=cv2.connectedComponentsWithStats(texture)
    region=1+np.argmax(stats[1:,4]);yy,xx=np.where(labels==region)
    mask=np.zeros(gray.shape,'uint8')
    cv2.fillConvexPoly(mask,cv2.convexHull(np.c_[xx,yy].astype('int32')),255)
    # Continue the mask over the dark visible side face. The camera is fixed;
    # these small perspective offsets follow its two projected contact edges.
    xleft=int(xx.min()+3);ys=np.where(labels[:,xleft]==region)[0]
    top,bottom=int(yy.min()),int(yy.max())
    middle=(top+bottom)//2;left=int(np.where(gray[middle,:xleft]<120)[0].min())
    side=np.array([[left,top+16],[xleft,top],[xleft,bottom],[left,bottom-12]],'int32')
    cv2.fillConvexPoly(mask,side,255)
    return cv2.GaussianBlur(mask,(3,3),.45)


def prepare():
    result={};sources=[]
    for material in 'AB':
        candidates=[HERE/f'sources/stereo_press_{material}/camera_0.npy',
                    ROOT/f'out/press_separated_20260913/monotonic320/inputs_{material}/camera_0.npy',
                    Path(f'/dev/shm/press_separated_20260913/monotonic320/inputs_{material}/camera_0.npy')]
        source=next((p for p in candidates if p.exists()),None)
        if source is None:raise FileNotFoundError(f'Original material {material} stereo sequence unavailable')
        a=np.load(source,mmap_mode='r')
        indices=np.arange(0,201,4)
        frames=[np.asarray(Image.fromarray(a[i]).convert('RGB').crop(CROP).resize((608,437),Image.Resampling.LANCZOS)) for i in indices]
        result[material]=np.stack(frames)
        result[material+"_mask"]=np.stack([specimen_mask(frame) for frame in frames])
        sources.append({'material':material,'path':str(source),'sha256':sha(source),'indices':indices.tolist()})
    out=HERE/'assets/shaping_identification.npz'
    np.savez_compressed(out,**result,times=np.arange(51)*.04)
    (out.with_suffix('.json')).write_text(json.dumps({'sources':sources,'crop_xyxy':CROP,
        'operation':'Original independent A/B stereo image sequences, fixed crop, uniform resize, every fourth recorded frame. Persistent display cache, no tint or new physics.',
        'mask':'Display-only specimen tint mask from the visible textured face and fixed-camera side-face continuation. Plate/background unchanged; no measured geometry modified.',
        'time_s':[0,2],'cache_sha256':sha(out)},indent=2)+'\n')
    print(out)

if __name__=='__main__':prepare()
