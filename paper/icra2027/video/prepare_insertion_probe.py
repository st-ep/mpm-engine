"""Color-code frozen A/B bending footage with geometry-derived visibility masks.

No dynamics or estimation are rerun. The original rendered RGB stays unchanged
outside the specimen; its existing texture/shading is preserved inside it.
"""
from pathlib import Path
import json, sys
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from experiments.elastic.strip_texture_observe import meshes
from experiments.elastic.strip_camera_observe import CAMERA
from experiments.elastic.strip_bending_study import pusher_position


def make():
    import pyvista as pv
    root=ROOT/'out/strip_texture_20260912'
    p=json.loads((root/'protocol.json').read_text())
    times=np.load(root/'media/video_time.npy')
    ids=np.array([np.argmin(abs(times-t)) for t in np.linspace(.30,.90,41)])
    output={}
    for material in 'AB':
        rgb=np.load(root/f'media/truth_{material}_frames.npy',mmap_mode='r')
        X=np.load(root/f'truth_{material}/x.npy',mmap_mode='r')
        state_times=np.load(root/f'truth_{material}/time.npy')
        axes=[np.unique(X[0,:,i]) for i in range(3)]
        shape=tuple(len(a) for a in axes)+(3,)
        (front,ref),others=meshes(p)
        pl=pv.Plotter(off_screen=True,window_size=(1280,1280))
        pl.set_background('black');pl.enable_anti_aliasing('ssaa')
        for mesh,_ in [(front,ref),*others]:
            pl.add_mesh(mesh,color='white',lighting=False)
        yf=p['center'][1]-p['size'][1]/2;yb=p['center'][1]+p['size'][1]/2
        for bounds in [(.101,.139,yf-.012,yf,.165,.193),(.101,.139,yb,yb+.012,.165,.193),(.101,.139,yf-.012,yb+.012,.193,.204)]:
            pl.add_mesh(pv.Box(bounds=bounds),color='black',lighting=False)
        pl.camera.position=CAMERA['centers'][0]
        pl.camera.focal_point=CAMERA['target'];pl.camera.up=(0,0,1)
        pl.camera.view_angle=float(np.degrees(2*np.arctan(1280/(2*CAMERA['focal_px']))))
        frames=[];masks=[]
        for j,i in enumerate(ids):
            t=times[i];si=int(np.argmin(abs(state_times-t)))
            deform=RegularGridInterpolator(axes,X[si].reshape(shape),bounds_error=False,fill_value=None)
            front.points=deform(ref)
            for mesh,q in others:mesh.points=deform(q)
            pl.add_mesh(pv.Cylinder(center=pusher_position(t,p),direction=(0,1,0),radius=p['pusher_radius'],height=p['pusher_length'],resolution=64),name='pusher',color='black',lighting=False)
            pl.reset_camera_clipping_range();pl.render()
            mask=Image.fromarray(pl.screenshot(return_img=True)[340:1230,300:960,0]).resize((360,486),Image.Resampling.LANCZOS)
            frame=Image.fromarray(rgb[i]).resize((360,486),Image.Resampling.LANCZOS)
            frames.append(np.asarray(frame));masks.append(np.asarray(mask))
        pl.close()
        output[material]=np.array(frames);output[material+'_mask']=np.array(masks)
        print('Prepared',material,flush=True)
    output['physical_time']=times[ids]
    np.savez_compressed(HERE/'assets/insertion_probe.npz',**output)
    (HERE/'assets/insertion_probe.json').write_text(json.dumps({
        'source':str(root.relative_to(ROOT)), 'materials':['A','B'],
        'physical_times_s':times[ids].tolist(),
        'rgb':'Unmodified frozen truth_A/B_frames.npy, resized with Lanczos',
        'mask':'Visible specimen geometry rendered with original camera; clamp and pusher occlude it',
        'editing':'Blue/orange display tint preserves original texture, geometry and luminance; tint returns to zero before tests'
    },indent=2)+'\n')

if __name__=='__main__':make()
