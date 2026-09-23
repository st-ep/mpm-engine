"""Display-only render of saved shaping particles, paper hand mesh and target outline."""
from pathlib import Path
import sys,json,subprocess,time,argparse,shutil,hashlib
import numpy as np
from PIL import Image,ImageDraw
import cv2
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import pyvista as pv
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.x_shaping import CONFIG,target_mesh
from experiments.robotics.x_shaping_franka import Panda,MOUNT_Z
OUT=HERE/'assets/shaping_hand';OUT.mkdir(exist_ok=True)
CAPTURE=ROOT/'out/paper_video_collection_20260918/shaping_capture'
XML=ROOT/'out/x_consistent_force_study_20260911/franka/panda_model_snapshot/panda.xml'
W,H=640,440
TARGET=OUT/'target.vtp'
if not TARGET.exists():shutil.copy2('/dev/shm/press_paper_update_20260913/execution80/inputs/target.vtp',TARGET)

def rotation(a):
 return np.array([[-np.sin(a),np.cos(a),0],[np.cos(a),np.sin(a),0],[0,0,-1.]])

def camera(p,t):
 u=np.clip((t-21.8)/1.5,0,1);u=u*u*(3-2*u)
 focus=np.array([.08,.08,.033-.006*u])
 offset=(1-u)*np.array([.20,-.20,.055])+u*np.array([0,-.13,.24])
 p.camera_position=[focus+offset,focus,(0,0,1)]
 p.enable_parallel_projection();p.camera.parallel_scale=.063*(1-u)+.048*u
 p.camera.clipping_range=(.01,2)
 return float(u)

def outline(mask):
 contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 image=Image.new('RGBA',(W,H));d=ImageDraw.Draw(image)
 for contour in contours:
  pts=contour[:,0,:].astype(float);pts=np.vstack([pts,pts[0]])
  lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
  for start in np.arange(0,lengths[-1],10):
   tt=np.linspace(start,min(start+6,lengths[-1]),8)
   points=list(zip(np.interp(tt,lengths,pts[:,0]),np.interp(tt,lengths,pts[:,1])))
   d.line(points,fill=(25,47,57,225),width=2)
 return image

def render(key,preview=False):
 folder=CAPTURE/key
 assert json.loads((folder/'display_release.json').read_text())['released_for_video']
 xs=np.load(folder/'positions.npy',mmap_mode='r');data=np.load(folder/'timing.npz')
 pl=pv.Plotter(off_screen=True,window_size=(W,H));pl.set_background('#f1f2f2');pl.enable_anti_aliasing('ssaa')
 body=pl.add_mesh(surface(xs[0],data['vol0'],h=CONFIG['voxel']),color='#2375aa' if key[0]=='A' else '#c96932',smooth_shading=True,ambient=.4,diffuse=.7,specular=.1)
 tools=[];adapters=[]
 for _ in range(2):
  tools.append(pl.add_mesh(pv.Cylinder(center=(0,0,0),direction=(0,0,1),radius=.014,height=.045,resolution=48),color='#697680',opacity=.28,smooth_shading=True))
  adapters.append(pl.add_mesh(pv.Cylinder(center=(0,0,0),direction=(0,0,1),radius=.007,height=.008),color='#c7cdd0'))
 centers=data['tool_centers'][0];angle=np.arctan2(*(centers[1]-centers[0])[1::-1]);r=rotation(angle);pos=centers.mean(0)+[0,0,MOUNT_Z]
 panda=Panda(XML);hand=[]
 for mesh,col in panda.meshes(centers,angle)[:-2]:
  mesh.points=(mesh.points-pos)@r
  mesh.compute_normals(inplace=True)
  hand.append(pl.add_mesh(mesh,color=col,smooth_shading=True,ambient=.6,diffuse=.5,specular=.1))
 mp=pv.Plotter(off_screen=True,window_size=(W,H));mp.set_background('white');mp.add_mesh(pv.read(TARGET),color='black',lighting=False)
 maskcache={};indices=[]
 times=np.linspace(0,23.72,400)
 if preview:times=np.array([0,2,7,15,21.9,23.72])
 writer=None
 if not preview:
  writer=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','25','-i','-','-an','-c:v','libx264','-crf','17','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/f'{key}.mp4')],stdin=subprocess.PIPE)
 start=time.monotonic()
 for j,t in enumerate(times):
  i=int(np.argmin(abs(data['time']-t)));indices.append(i)
  body.mapper.SetInputData(surface(xs[i],data['vol0'],h=CONFIG['voxel']).compute_normals(point_normals=True,cell_normals=False))
  centers=data['tool_centers'][i];a=np.arctan2(*(centers[1]-centers[0])[1::-1]);pose=np.eye(4);pose[:3,:3]=rotation(a);pose[:3,3]=centers.mean(0)+[0,0,MOUNT_Z]
  for actor in hand:actor.user_matrix=pose
  for actor,adapter,c in zip(tools,adapters,centers):actor.SetPosition(*c);adapter.SetPosition(*(c+[0,0,.045/2+.004]))
  tool_opacity=1-float(np.clip((t-21.8)/.4,0,1))
  for actor in hand+adapters:actor.prop.opacity=tool_opacity
  for actor in tools:actor.prop.opacity=.28*tool_opacity
  u=camera(pl,t)
  show_target = j == len(times)-1
  if show_target and u not in maskcache:
   camera(mp,t);mp.render();maskcache[u]=outline(mp.screenshot(return_img=True)[:,:,:3].mean(2)<128)
  pl.render();im=Image.fromarray(pl.screenshot(return_img=True)[:,:,:3]).convert('RGBA')
  if show_target:im.alpha_composite(maskcache[u])
  im=im.convert('RGB')
  if writer:writer.stdin.write(im.tobytes())
  if preview or j in [0,50,200,399]:im.save(OUT/f'{key}_{j:03d}.png')
  if j%50==0:print(key,j,'/',len(times),'elapsed',round(time.monotonic()-start,1),flush=True)
 if writer:writer.stdin.close();assert writer.wait()==0
 pl.close();mp.close()
 (OUT/f'{key}{"_preview" if preview else ""}.json').write_text(json.dumps({'case':key,'source':str(folder),'frames':len(times),'sample_indices':indices,'source_time_s':times.tolist(),'size':[W,H],'fps':25,'physics_rerun':False,'hand':'Same Panda hand mesh and adapter convention as the paper; rigid kinematic placement at saved collider positions. Actual translucent cylindrical contact geometry retained.','target_sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),'target':'Dashed projected silhouette only on the final held comparison frame, never during motion. Exact target mesh under the same camera and common scale. No per-case registration. Withdrawn tool illustrations fade out at source 21.8–22.2 s before the clean final comparison, as in the paper.','camera':'Paper-style low oblique contact view, transitioning identically across cases to the common final comparison view over source 21.8–23.3 s after finger withdrawal.'},indent=2)+'\n')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('cases');parser.add_argument('--preview',action='store_true');a=parser.parse_args()
 for case in a.cases.split(','):render(case,a.preview)
