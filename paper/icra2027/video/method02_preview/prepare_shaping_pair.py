"""Capture and render the frozen red hardware plan, without replanning/refitting."""
from pathlib import Path
import sys,json,hashlib,inspect,time,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
P=Path(__file__).resolve().parent
OUT=P/'shaping_pair';OUT.mkdir(exist_ok=True)
RUN=ROOT/'out/press_iou_independent_20260915/study/rounded_coarse/play_doh_01'
STUDY=RUN.parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def capture():
 from experiments.robotics import x_motion_shaping as core
 cfg=json.loads(RUN.with_suffix('.config.json').read_text())
 protocol=json.loads((STUDY/'protocol.json').read_text())
 for rel,h in protocol['source_sha256'].items():assert sha(ROOT/rel)==h,rel
 original=np.load(RUN.with_suffix('.npz'))
 assert sha(RUN.with_suffix('.npz'))==json.loads((STUDY.parent/'RESULTS.json').read_text())['selected_execution_sha256']
 assert np.allclose(original['gaps_mm'],[22,22,49.5,20])
 core.geometry.CONFIG.update(cfg['scene'])
 cmd={k:original[k] for k in ['time','tool_centers','phase_id','pinch_id','start_pose','gaps_mm']}
 phases=json.loads(RUN.with_suffix('.phases.json').read_text())
 ids=np.unique(np.r_[np.arange(9,len(cmd['time']),10),len(cmd['time'])-1])
 xs=np.lib.format.open_memmap(OUT/'positions.npy',mode='w+',dtype='float32',shape=(len(ids)+1,*original['initial'].shape))
 xs[0]=original['initial'];lookup={int(j):i+1 for i,j in enumerate(ids)}
 np.savez(OUT/'timing.npz',time=np.r_[0.,cmd['time'][ids]],tool_centers=np.concatenate([cmd['start_pose'][None],cmd['tool_centers'][ids]]),phase_id=np.r_[0,cmd['phase_id'][ids]],vol0=original['vol0'])
 start=time.monotonic()
 def capture_frame(solver,j):
  if j in lookup:xs[lookup[j]]=solver.x()
  if j%1000==0:print('capture',j,'/',len(cmd['time']),round(time.monotonic()-start,1),flush=True)
 core.capture_frame=capture_frame
 source=inspect.getsource(core.simulate)
 needle='        solver.step(dt, substeps=substeps)';assert source.count(needle)==1
 source=source.replace(needle,needle+'\n        capture_frame(solver,j)')
 (OUT/'instrumented_simulate.py').write_text(source)
 exec(compile(source,str(OUT/'instrumented_simulate.py'),'exec'),core.__dict__)
 replay=core.simulate(cfg['law'],cmd,phases,{'initial':original['initial'],'vol0':original['vol0']},'cuda:1',cfg['grid'],cfg['dt'])
 xs.flush();comparisons={}
 for k in [*[f'{i}_{p}' for i in range(4) for p in ['close','withdraw']],'x_after_1s']:
  delta=(replay[k].astype(float)-original[k].astype(float))*1000
  comparisons[k]={'rms_mm':float(np.sqrt(np.mean(np.sum(delta**2,axis=-1)))),'max_mm':float(np.linalg.norm(delta,axis=-1).max())}
 accepted=all(v['rms_mm']<=.02 and v['max_mm']<=.1 for v in comparisons.values())
 result={'source':str(RUN),'source_sha256':sha(RUN.with_suffix('.npz')),'config':cfg,'comparisons':comparisons,'accepted':accepted,'threshold_mm':{'rms':.02,'max':.1},'replanned':False,'refitted':False,'source_code_verified':True}
 (OUT/'replay_verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print('Replay accepted',accepted,comparisons,flush=True);assert accepted




def render_target():
 """Flat XY silhouette of the actual target, with its long axis vertical."""
 import pyvista as pv
 from PIL import Image
 target=pv.read(STUDY/'target.vtp');center=np.array(target.center)
 tp=pv.Plotter(off_screen=True,window_size=(480,600))
 tp.set_background('#eaf0f3');tp.enable_anti_aliasing('ssaa')
 tp.add_mesh(target,color='#677f8c',lighting=False)
 tp.camera_position=[center+np.array([0,0,1]),center,(0,1,0)]
 tp.enable_parallel_projection();tp.camera.parallel_scale=.042
 pixels=tp.screenshot(return_img=True)[:,:,:3];tp.close()
 mask=np.max(np.abs(pixels.astype(int)-[234,240,243]),axis=2)>20
 yy,xx=np.where(mask)
 Image.fromarray(pixels).crop((xx.min()-2,yy.min()-2,xx.max()+3,yy.max()+3)).save(OUT/'target_2d.png')


def render_pair():
 import cv2,pyvista as pv
 from PIL import Image,ImageDraw
 from experiments.robotics.plastic_shaping_figure import surface
 assert json.loads((OUT/'replay_verification.json').read_text())['accepted']
 xs=np.load(OUT/'positions.npy',mmap_mode='r');timing=np.load(OUT/'timing.npz')
 cfg=json.loads(RUN.with_suffix('.config.json').read_text())['scene']
 phases=json.loads(RUN.with_suffix('.phases.json').read_text())
 byname={ph['name']:ph for ph in phases}
 def st(name,end=False):
  ph=byname[name];return ph['start_s']+(ph['duration_s'] if end else 0)
 # Manually reviewed source-video events. Alignment changes display time only.
 # Repeated simulation endpoints display the actual pauses in the hardware clip.
 events=[(14.,0.,'start'),(15.,st('0:lower',True),'lower 1'),
  (18.6,st('0:close',True),'closed 1'),(21.3,st('0:close',True),'hold 1'),
  (22.,st('0:open',True),'open 1'),(24.7,st('0:withdraw',True),'withdraw 1'),
  (26.,st('1:reposition',True),'turn 2'),(27.,st('1:lower',True),'lower 2'),
  (30.5,st('1:close',True),'closed 2'),(34.8,st('1:close',True),'hold 2'),
  (36.5,st('1:open',True),'open 2'),(37.8,st('1:withdraw',True),'withdraw 2'),
  (38.5,st('2:reposition',True),'turn 3'),(39.2,st('2:lower',True),'lower 3'),
  (40.5,st('2:close',True),'closed 3'),(42.4,st('2:close',True),'hold 3'),
  (43.1,st('2:open',True),'open 3'),(44.3,st('2:withdraw',True),'withdraw 3'),
  (45.6,st('3:reposition',True),'turn 4'),(46.1,st('3:lower',True),'lower 4'),
  (49.,st('3:close',True),'closed 4'),(51.7,st('3:close',True),'hold 4'),
  (52.3,st('3:open',True),'open 4'),(53.2,st('3:withdraw',True),'withdraw 4'),
  (54.2,float(timing['time'][-1]),'released')]
 hwt=np.linspace(14,54.2,400)
 simt=np.interp(hwt,[e[0] for e in events],[e[1] for e in events])
 indices=np.array([np.argmin(abs(timing['time']-t)) for t in simt])
 for name in ('simulation_frames','hardware_frames'):(OUT/name).mkdir(exist_ok=True)
 W,H=520,340
 pl=pv.Plotter(off_screen=True,window_size=(W,H));pl.set_background('#eaf0f3');pl.enable_anti_aliasing('ssaa')
 focus=np.array([.08,.08,.034])
 pl.add_mesh(pv.Plane(center=(.08,.08,cfg['floor']-.0002),i_size=.3,j_size=.3),color='#eaf0f3',lighting=False)
 body=pl.add_mesh(surface(xs[0],timing['vol0'],h=cfg['voxel']),color='#2375aa',smooth_shading=True,ambient=.35,diffuse=.7,specular=.12)
 actors=[]
 for _ in range(2):
  actors.append(pl.add_mesh(pv.Cylinder(center=(0,0,0),direction=(0,0,1),radius=cfg['radius'],height=cfg['finger_height'],resolution=48),color='#6f818c',opacity=.38,smooth_shading=True))
 pl.camera_position=[focus+np.array([.15,-.24,.22]),focus,(0,0,1)]
 pl.enable_parallel_projection();pl.camera.parallel_scale=.052;pl.camera.clipping_range=(.01,2)
 cached={};start=time.monotonic()
 for n,idx in enumerate(indices):
  if int(idx) not in cached:
   mesh=surface(xs[idx],timing['vol0'],h=cfg['voxel']).compute_normals(point_normals=True,cell_normals=False)
   body.mapper.SetInputData(mesh)
   for actor,cp in zip(actors,timing['tool_centers'][idx]):actor.SetPosition(*cp)
   pl.render();cached[int(idx)]=pl.screenshot(return_img=True)[:,:,:3]
  Image.fromarray(cached[int(idx)]).save(OUT/'simulation_frames'/f'{n:04d}.png')
  if n%100==0:print('render matched simulation',n,'/400',round(time.monotonic()-start,1),flush=True)
 pl.close()
 render_target()
 video=ROOT/'paper/icra2027/video/sources/hardware_shaping/side_rgb.mp4'
 cap=cv2.VideoCapture(str(video));wanted=np.rint(hwt*30).astype(int);current=-1
 for n,idx in enumerate(wanted):
  while current<idx:
   ok,bgr=cap.read();assert ok;current+=1
  im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop((700,210,1440,660)).resize((534,324),Image.Resampling.LANCZOS)
  im.save(OUT/'hardware_frames'/f'{n:04d}.png')
 cap.release()
 metadata={'hardware_source':str(video),'hardware_confirmation':'Red matched-ID plan, user-confirmed','gaps_mm':[22,22,49.5,20],'axes':['Y','X','Y','X'],'offsets_mm':[0,0,0,0],
 'playback':'Phase-aligned illustrative comparison; not common physical-time playback. Hardware plays forward at a uniform rate; simulation time is piecewise aligned to manually reviewed gripper phases, including holds. No material trajectory is morphed or refitted.',
 'events':[{'hardware_s':a,'simulation_s':b,'event':c} for a,b,c in events],'frames':400,'fps':25,'hardware_times_s':hwt.tolist(),'simulation_times_s':simt.tolist(),'simulation_frame_indices':indices.tolist(),
 'camera':{'position':(focus+np.array([.15,-.24,.22])).tolist(),'focus':focus.tolist(),'parallel_scale':.052},'target':'Flat orthographic XY silhouette of the original target mesh, with the long axis vertical.','tools':'Actual cylindrical colliders; no illustrative robot hand. Fixed close view prioritizes contact and specimen deformation; raised fingers can leave the top of the view during withdrawal.',
 'hardware_crop':[700,210,1440,660]}
 (OUT/'pair_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
 sheet=Image.new('RGB',(1560,1020),'white');d=ImageDraw.Draw(sheet)
 for j,idx in enumerate([0,70,150,230,300,380]):
  x=j%3*520;y=j//3*510
  for folder,dy in [('simulation_frames',20),('hardware_frames',265)]:
   im=Image.open(OUT/folder/f'{idx:04d}.png');im.thumbnail((510,240));sheet.paste(im,(x,y+dy))
  d.text((x+8,y+3),f"video {idx/25:.1f}s | hardware {hwt[idx]:.1f}s | sim {simt[idx]:.2f}s",fill='black')
 sheet.save(OUT/'paired_review.png')
 print('Matched pair ready',flush=True)

if __name__=='__main__':
 if '--target-only' in sys.argv:render_target()
 elif '--render' in sys.argv:render_pair()
 else:capture()
