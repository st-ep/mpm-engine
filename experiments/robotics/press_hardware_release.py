"""Check released silhouettes in image space, without material correspondences.

The selected 15 s frames have a visibly clear pad and precede hand contact.
Bounding extents are observations in pixels, not reconstructed 3D height.
Projection uses the fixed initial camera; no per-frame alignment is fitted.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from experiments.robotics.press_hardware_observe import read, save, jsonl

ROOT=Path(__file__).resolve().parents[2]


def bounds(mask):
    y,x=np.nonzero(mask)
    if len(x)<100:raise ValueError('Empty or tiny released silhouette')
    return [int(x.min()),int(y.min()),int(x.max()),int(y.max())]


def project(x,camera):
    forward=np.array(camera['camera_forward']);down=-np.array(camera['camera_up'])
    right=np.cross(down,forward);R=np.column_stack([right,down,forward])
    q=(x-np.array(camera['camera_position_m']))@R
    k=camera['intrinsics']
    return np.c_[q[:,0]/q[:,2]*k['fx']+k['ppx'],q[:,1]/q[:,2]*k['fy']+k['ppy']]


def observe(out):
    out.mkdir(parents=True,exist_ok=True);results=[];panels=[]
    for ep,material in [('ep0002','play_doh'),('ep0006','butter_slime'),('ep0010','plasticine')]:
        cv2.setRNGSeed(100+int(ep[2:]))
        raw=ROOT/'press_real_data'/ep
        a=read(ROOT/'out/press_observation_assessment_20260913'/ep/'assessment.json')
        hosts=np.array([f['t_host'] for f in jsonl(raw/'frames_hand.jsonl')])
        idx=int(np.argmin(abs(hosts-a['t0_host']-15)))
        cap=cv2.VideoCapture(str(raw/'hand_rgb.mp4'));cap.set(1,idx);ok,im=cap.read();cap.release()
        if not ok:raise RuntimeError(ep)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        yy,xx=np.mgrid[:480,:848]
        roi=(xx>370)&(xx<550)&(yy>170)&(yy<276)
        if material=='plasticine':
            # Image-assisted segmentation only: background is excluded around
            # the visibly released body, with a conservative interior seed.
            ell=((xx-462)/49)**2+((yy-230)/46)**2
            labels=np.full((480,848),cv2.GC_BGD,np.uint8)
            labels[(ell<1.35)&roi]=cv2.GC_PR_FGD
            labels[(ell<.35)&roi]=cv2.GC_FGD
            cv2.grabCut(im,labels,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
            mask=np.uint8((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD))
        else:
            hue=((hsv[...,0]<15)|(hsv[...,0]>165)) if material=='play_doh' else ((hsv[...,0]>15)&(hsv[...,0]<42))
            seed=np.uint8(hue&(hsv[...,1]>100)&(hsv[...,2]>45)&roi)
            # HSV alone excludes bright yellow highlights and the shaded
            # underside. Expand the candidate region in the RGB image, then
            # classify against surrounding background; no simulated contour
            # enters segmentation. Preserve the HSV result for audit.
            cv2.imwrite(str(out/f'{ep}_hsv_mask.png'),seed*255)
            ell=((xx-459)/78)**2+((yy-253)/23)**2
            labels=np.full((480,848),cv2.GC_BGD,np.uint8)
            labels[(ell<1.2)&(yy<276)]=cv2.GC_PR_FGD
            labels[(cv2.erode(seed,np.ones((7,7),np.uint8))>0)&(yy<266)]=cv2.GC_FGD
            cv2.grabCut(im,labels,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
            mask=np.uint8((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD))
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        component=1+int(np.argmax(stats[1:,4]));mask=np.uint8(labels==component)
        b=bounds(mask)
        cv2.imwrite(str(out/f'{ep}_mask.png'),mask*255)
        cv2.imwrite(str(out/f'{ep}_frame.jpg'),im)
        overlay=im.copy();contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(0,255,255),1)
        cv2.rectangle(overlay,tuple(b[:2]),tuple(b[2:]),(255,255,0),1)
        panel=cv2.resize(overlay[160:290,365:555],(570,390))
        cv2.putText(panel,f'{ep} released at {hosts[idx]-a["t0_host"]:.3f} s',(8,22),0,.6,(255,255,255),1)
        panels.append(panel)
        results.append(dict(episode=ep,material=material,frame_index=idx,time_s=float(hosts[idx]-a['t0_host']),
            bbox_xyxy_px=b,width_px=b[2]-b[0],vertical_extent_px=b[3]-b[1],
            uncertainty_scope='Image-assisted mask, finite pixels, contact shadows and perspective; no 3D height or material trajectory inferred. Small filaments excluded by morphology.',
            selection='Pad visibly clear; frame precedes hand contact with specimen.'))
    cv2.imwrite(str(out/'release_masks.jpg'),np.concatenate(panels,axis=1))
    save(out/'observations.json',dict(observations=results,scope='Direct RGB extents after full tool release. Earlier 11–13.4 s comparisons represent partial unloading under residual normal load.'))


def compare(run,observations,out):
    c=read(run/'completion.json');d=np.load(run/'trajectory.npz')
    observed=next(x for x in read(out/'observations.json')['observations'] if x['episode']==c['episode'])
    j=int(np.argmin(abs(d['time']-observed['time_s'])))
    if abs(d['time'][j]-observed['time_s'])>.06:raise ValueError('Simulation does not reach the observed release frame')
    camera=read(observations/c['episode']/'camera.json')
    uv=project(d['x'][j],camera);b=np.r_[uv.min(0),uv.max(0)]
    r=dict(observation=observed,simulated_time_s=float(d['time'][j]),predicted_bbox_xyxy_px=b.tolist(),
        predicted_width_px=float(b[2]-b[0]),predicted_vertical_extent_px=float(b[3]-b[1]),
        width_error_px=float(b[2]-b[0]-observed['width_px']),vertical_extent_error_px=float(b[3]-b[1]-observed['vertical_extent_px']),
        scope='Fixed-camera projection of saved particle centers, matching video hull support. Body extents, not plate gap. No per-frame translation, scale or time correction.',run=str(run.resolve()))
    save(run/'release_assessment.json',r)
    save(out/f'{c["material"]}_comparison.json',r);print(r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run',type=Path);p.add_argument('--observations',type=Path)
    a=p.parse_args()
    if a.run:compare(a.run,a.observations,a.out)
    else:observe(a.out)
