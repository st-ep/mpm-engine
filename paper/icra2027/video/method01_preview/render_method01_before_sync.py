"""Animated observation/reconstruction slide, using frozen experimental sources.
Public entry draw_method01(t) returns a 1280x720 PIL frame. Main video uses it.
"""
from pathlib import Path
from functools import lru_cache
import argparse,json,math,subprocess
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from matplotlib import colormaps
P=Path(__file__).resolve().parent;ROOT=P.parents[3];DURATION=14;FPS=25;S=2
BG='#f4f6f7';INK='#21333e';TEAL='#167b76';BLUE='#2375aa';MUTED='#566975';LINE='#d9e1e5'
@lru_cache(None)
def font(n,bold=False,math=False):
 p='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf' if math else '/usr/share/fonts/truetype/lato/Lato-'+('Bold' if bold else 'Regular')+'.ttf'
 return ImageFont.truetype(p,round(n*S))
class Canvas:
 def __init__(self,im):self.im=im;self.d=ImageDraw.Draw(im)
 def text(self,x,y,s,n=18,c=INK,bold=False,anchor=None):
  f=font(n,bold,any(ch in s for ch in 'ẋ∇Σφⱼ'));bb=self.d.textbbox((x*S,y*S),s,font=f,anchor=anchor)
  assert min(bb[:2])>=0 and bb[2]<=2560 and bb[3]<=1440,(s,bb)
  self.d.text((x*S,y*S),s,font=f,fill=c,anchor=anchor)
 def line(self,pts,c=TEAL,w=1):self.d.line([tuple(np.array(q)*S) for q in pts],fill=c,width=max(1,round(w*S)),joint='curve')
 def poly(self,p,c):self.d.polygon([tuple(np.array(q)*S) for q in p],fill=c)
 def dot(self,p,r,c=TEAL,edge=None):
  x,y=p;self.d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=c,outline=edge,width=S)
 def arrow(self,a,b,c=TEAL,w=1.5,h=6):
  a=np.asarray(a);b=np.asarray(b);length=np.linalg.norm(b-a)
  if length<.5:return
  v=(b-a)/length;n=np.array([-v[1],v[0]]);h=min(h,length*.5)
  self.line([a,b-v*h*.5],c,w);self.poly([b,b-h*v+n*h*.44,b-h*v-n*h*.44],c)
 def rect(self,box,c,r=5,edge=None):self.d.rounded_rectangle(tuple(v*S for v in box),radius=r*S,fill=c,outline=edge,width=S)
 def paste(self,src,box):
  x,y,w,h=box;self.im.paste(src.resize((round(w*S),round(h*S)),Image.Resampling.LANCZOS),(round(x*S),round(y*S)))

def basis(az=-60,el=23):
 az,el=np.deg2rad([az,el]);right=np.array([-np.sin(az),np.cos(az),0]);up=np.array([-np.cos(az)*np.sin(el),-np.sin(az)*np.sin(el),np.cos(el)]);return right,up,np.cross(right,up)
class Projection:
 def __init__(self,allpoints,box,az=-60,el=23):
  self.right,self.up,self.view=basis(az,el);self.origin=np.mean(allpoints,0)
  xy=self.raw(allpoints);self.mid=(xy.min(0)+xy.max(0))/2
  x,y,w,h=box;self.center=np.array([x+w/2,y+h/2]);self.scale=min(w/np.ptp(xy[:,0]),h/np.ptp(xy[:,1]))
 def raw(self,p):
  q=np.asarray(p)-self.origin;return np.c_[q@self.right,-q@self.up]
 def __call__(self,p):return (self.raw(p)-self.mid)*self.scale+self.center

@lru_cache(None)
def data():
 def eager(path):
  with np.load(path) as z:return {k:z[k] for k in z.files}
 hp=eager(ROOT/'press_real_data/handoff_ep0-7/ep0001/particles.npz')
 kin=eager(P/'animation_stereo_kinematics.npz')
 tr=eager(ROOT/'out/press_separated_20260913/monotonic320/fit_A/tracks.npz')
 force=eager(ROOT/'out/press_observation_assessment_20260913/ep0001/robot_force.npz')
 camera=np.load('/dev/shm/press_separated_20260913/monotonic320/inputs_A/camera_0.npy',mmap_mode='r')
 ids=json.loads((P/'middle_provenance.json').read_text())['stereo']['feature_ids']
 valid=tr['valid'][0:201,ids].all();assert valid
 # Static framing across time: deformation is never hidden by per-frame resizing.
 hsel=np.flatnonzero((hp['times']>=0)&(hp['times']<=2.05));hbound=hp['pos'][hsel[::3],::3].reshape(-1,3)
 hproj=[Projection(hbound,b) for b in [(370,254,171,142),(757,254,171,142),(1007,273,219,157)]]
 X=kin['X'];sproj=Projection(X[::10].reshape(-1,3),(766,523,160,105),az=-52,el=24)
 # Use fixed stereo camera image crop across all frames to preserve image motion.
 pix=tr['pixels'][0,:,ids].reshape(-1,2);lo=pix.min(0);hi=pix.max(0);center=(lo+hi)/2
 crop=np.rint([center[0]-145,center[1]-128,center[0]+145,center[1]+128]).astype(int)
 hcolors=colormaps['turbo']((hp['seed'][:,2]-hp['seed'][:,2].min())/np.ptp(hp['seed'][:,2]))[:,:3]*255
 return dict(hp=hp,kin=kin,tr=tr,force=force,camera=camera,ids=ids,hsel=hsel,hproj=hproj,sproj=sproj,crop=crop,hcolors=hcolors)

def lerp_frame(arr,times,t):
 j=int(np.clip(np.searchsorted(times,t)-1,0,len(times)-2));a=float(np.clip((t-times[j])/(times[j+1]-times[j]),0,1));return arr[j]*(1-a)+arr[j+1]*a,j,a

def profile(points):
 lo=points.min(0);hi=points.max(0);cx,cy=(lo[:2]+hi[:2])/2
 rad=np.hypot(points[:,0]-cx,points[:,1]-cy);edges=np.linspace(lo[2],hi[2],18);rs=[]
 for z in edges:
  use=abs(points[:,2]-z)<(hi[2]-lo[2])/20;rs.append(np.max(rad[use]))
 r=gaussian_filter1d(rs,.7);z=np.linspace(lo[2],hi[2],33);r=PchipInterpolator(edges,r)(z)
 th=np.linspace(0,2*np.pi,65)
 mesh=np.stack([cx+r[:,None]*np.cos(th),cy+r[:,None]*np.sin(th),np.broadcast_to(z[:,None],(len(z),len(th)))],-1)
 return mesh

def draw_shell(c,mesh,proj,filled=False,tint=TEAL):
 from scipy.spatial import ConvexHull
 flat=mesh.reshape(-1,3);xy=proj(flat);outline=xy[ConvexHull(xy).vertices]
 # Thin surface tint: a soft mask and rim, with a deliberately unfilled left cap.
 c.poly(outline,'#e0eeea' if filled else BG)
 angles=np.linspace(0,2*np.pi,mesh.shape[1]);front=np.cos(angles)*proj.view[0]+np.sin(angles)*proj.view[1]>0
 for idx in [0,8,16,24,32]:
  q=proj(mesh[idx])
  for j in range(len(angles)-1):
   if front[j]:c.line(q[j:j+2],'#79aba1',.75)
   elif j%6<3:c.line(q[j:j+2],'#c2d9d1',.5)
 for j in range(0,len(angles)-1,8):
  q=proj(mesh[:,j]);c.line(q,'#92bdb2' if front[j] else '#c7ddd5',.6)
 cap=proj(mesh[-1]);c.poly(cap,'#beded2' if filled else BG);c.line(cap,'#8cb7a9',.75)
 center=mesh[-1].mean(0)
 for j in range(0,len(angles)-1,8):c.line(proj(np.vstack([center,mesh[-1,j]])),'#a3c7b9',.6)
 rim=mesh[-1]*.5+center*.5;c.line(proj(rim),'#b0cdbf',.5)
 c.line(np.vstack([outline,outline[0]]),'#6da295',.9)
 if filled:
  # A thin highlight traces the front surface; never an opaque pasted shape.
  j=int(np.argmax(np.cos(angles)*proj.view[0]+np.sin(angles)*proj.view[1]));j=(j+8)%(len(angles)-1)
  c.line(proj(mesh[4:27,j]),'#f8fcfa',1.5)

@lru_cache(None)
def static_frame():
 im=Image.open(P/'method01_perception.png').convert('RGB');c=Canvas(im)
 c.d.rectangle((342*S,135*S,2560,1440),fill=BG)
 c.text(654,149,'Reconstruct motion',24,INK,True,'mt')
 c.text(370,208,'RGB-D',22,TEAL,True)
 c.arrow((549,329),(574,329));c.arrow((722,329),(747,329))
 for x,title,sub in [(455,'Surface reconstruction','Axisymmetry'),(650,'Same volume','∇ · v = 0'),(843,'Particle advection','ẋ = v(x,t)')]:
  c.text(x,413,title,16.5,INK,True,'mt');c.text(x,440,sub,15.5,TEAL if x!=455 else MUTED,anchor='mt')
 c.text(650,236,'Flow prior',12,MUTED,anchor='mt')
 c.line([(369,480),(941,480)],LINE,1)
 c.text(370,494,'Stereo texture',22,BLUE,True)
 c.arrow((550,575),(576,575),BLUE);c.arrow((722,575),(747,575),BLUE)
 for x,title,sub in [(455,'Track texture','One camera view'),(650,'Triangulate','Calibrated stereo views'),(843,'Extend motion inside','Square-symmetric field')]:
  c.text(x,642,title,16.5,INK,True,'mt');c.text(x,668,sub,13.5,MUTED,anchor='mt')
 c.text(843,521,'u(X,t) = Σⱼ cⱼ(t)φⱼ(X)',12.5,BLUE,anchor='mt')
 # Hardware example of the common motion representation, with separate logged force.
 c.line([(968,329),(986,329),(986,449),(1001,449)],TEAL,1.5)
 c.line([(968,575),(986,575),(986,449)],TEAL,1.5);c.dot((986,449),2.4);c.arrow((989,449),(1006,449),TEAL,1.5)
 c.text(1124,154,'Inputs to',22,INK,True,'mt');c.text(1124,182,'identification',24,INK,True,'mt')
 c.text(1124,232,'Reconstructed motion',18.5,INK,True,'mt')
 c.text(1124,447,'Positions · velocities · deformation',12,MUTED,anchor='mt')
 for j in range(80):c.line([(1083+j,469),(1083+j,474)],tuple((np.array(colormaps['turbo'](j/79)[:3])*255).astype(int)),1)
 c.text(1124,480,'Color: initial height',12,MUTED,anchor='mt')
 c.text(1124,518,'Recorded normal force',18,INK,True,'mt')
 x0,y0,w,h=1039,552,178,70
 c.line([(x0,y0),(x0,y0+h),(x0+w,y0+h)],'#9aabb4',1)
 for val in [0,10]:
  y=y0+h-val/12*h;c.line([(x0,y),(x0+w,y)],'#dce4e8',.6);c.text(x0-8,y,str(val),11,MUTED,anchor='rm')
 for tv in [0,1,2]:c.text(x0+tv/2*w,y0+h+7,str(tv),11,MUTED,anchor='mt')
 c.text(1027,543,'N',11,MUTED,anchor='mt');c.text(1228,629,'s',11,MUTED)
 c.text(1124,650,'Robot log · contact baseline',12,MUTED,anchor='mt')
 c.text(1124,674,'RGB-D example',13,MUTED,anchor='mt')
 c.text(655,701,'Reconstruction sequences slowed; left images are reference views',12,MUTED,anchor='mt')
 return im

def draw_method01(t):
 D=data();im=static_frame().copy();c=Canvas(im)
 # One-way playback with a short final hold; no reversed experimental footage.
 phase=float(np.clip(t/12.5,0,1));source_t=.05+1.95*phase
 points,hj,ha=lerp_frame(D['hp']['pos'],D['hp']['times'],source_t)
 mesh=profile(points)
 draw_shell(c,mesh,D['hproj'][0],False)
 draw_shell(c,mesh,D['hproj'][1],True)
 # Frozen IDs chosen for legible interior motion; trajectories remain recorded handoff output.
 pids=json.loads((P/'rgbd_shape_flow_particles_provenance.json').read_text())['particle_ids']
 future,_,_=lerp_frame(D['hp']['pos'],D['hp']['times'],min(source_t+.55,2.5))
 for fid in pids:
  a=D['hproj'][1](points[[fid]])[0];b=D['hproj'][1](future[[fid]])[0]
  if np.linalg.norm(b-a)>4:
   c.arrow(a,b,BG,3.2,5);c.arrow(a,b,'#155e67',1.5,4)
  c.dot(a,2.7,'#155e67','white')
 # One continuous conceptual compression with determinant-one affine deformation.
 reference=D['hp']['pos'][int(np.argmin(abs(D['hp']['times']-.05)))]
 mm=profile(reference);origin=mm.reshape(-1,3).mean(0)
 compression=1-.40*(3*phase**2-2*phase**3);stretch=np.array([1/np.sqrt(compression),1/np.sqrt(compression),compression])
 assert abs(np.prod(stretch)-1)<1e-12
 mm=(mm-origin)*stretch+origin
 bounds=np.vstack([profile(reference).reshape(-1,3),(profile(reference).reshape(-1,3)-origin)*[1/np.sqrt(.6),1/np.sqrt(.6),.6]+origin])
 mp=Projection(bounds,(578,254,145,142));draw_shell(c,mm,mp,True)
 # Actual stereo image sequence. Displayed single view has tracked feature histories.
 sf=int(np.clip(round((.10+1.80*phase)*100),0,200));tr=D['tr'];crop=D['crop'];raw=D['camera'][sf]
 pic=Image.fromarray(raw).crop(tuple(crop)).convert('RGB');c.paste(pic,(369,535,172,96))
 origin2=np.array([369.,535.]);sc=np.array([172/(crop[2]-crop[0]),96/(crop[3]-crop[1])]);colors=['#28a6c6','#e2b23a','#b278c6']
 for fid,col in zip(D['ids'],colors):
  history=tr['pixels'][0,:sf+1,fid];q=(history-crop[:2])*sc+origin2
  if len(q)>1:c.line(q,col,1.3)
  c.dot(q[0],2,'#f4f6f7',col)
  end=q[-1];c.rect((end[0]-5,end[1]-5,end[0]+5,end[1]+5),None,1,col);c.dot(end,2,col)
 # Triangulation construction: two image planes and the actual observed 3D point.
 lp=np.array([[582,537],[612,542],[612,566],[582,561]]);rp=np.array([[688,542],[718,537],[718,561],[688,566]])
 for p in [lp,rp]:c.poly(p,'#e7eff3');c.line(np.vstack([p,p[0]]),'#95b4c5',1)
 fid=D['ids'][0];traj=tr['world'][:,fid];tp=Projection(traj,(636,587,30,36),az=-55,el=25)
 point=tp(tr['world'][sf,[fid]])[0]
 # Pixel motion within the two schematic planes is driven by actual two-camera tracks.
 av=[]
 for ci,cx in [(0,597),(1,703)]:
  pix=tr['pixels'][ci,:,fid];center=(pix.min(0)+pix.max(0))/2
  q=(pix[sf]-center)*.14+[cx,551];av.append(q)
 surface=np.array([point+[-23,0],point+[0,-10],point+[23,1],point+[0,13]])
 c.poly(surface,'#e3eef3');c.line(np.vstack([surface,surface[0]]),'#adc7d5',.8)
 for q in av:c.line([q,point],BLUE,1.5);c.dot(q,2.7,colors[0],'white')
 c.dot(point,3.6,colors[0],INK)
 # Square-symmetric interior field, from unchanged fitted observational kinematics.
 lattice=D['kin']['X'][sf].reshape(11,11,11,3);sp=D['sproj']
 for axis in range(3):
  for a in [0,5,10]:
   for b in [0,5,10]:
    sl=[];j=0
    for dim in range(3):
     if dim==axis:sl.append(slice(None))
     else:sl.append([a,b][j]);j+=1
    c.line(sp(lattice[tuple(sl)]),'#adc6d2',.9)
 for i in [1,5,9]:
  for j in [1,5,9]:c.dot(sp(lattice[i,0,j][None])[0],2.4,BLUE,'white')
 # Dense particle output, using persistent colors from initial material height.
 proj=D['hproj'][2];xy=proj(points);dep=points@proj.view;brightness=.70+.30*(dep-dep.min())/np.ptp(dep)
 for j in np.argsort(dep):c.dot(xy[j],.8,tuple((D['hcolors'][j]*brightness[j]).astype(int)))
 # Progressive reveal and cursor are tied to the same hardware source time.
 f=D['force'];use=(f['time']>=0)&(f['time']<=source_t);ft=f['time'][use];fv=f['incremental_normal_force'][use]
 path=np.c_[1039+ft/2*178,622-fv/12*70]
 if len(path)>1:c.line(path,TEAL,1.6);c.dot(path[-1],2.4,TEAL,'white')
 xx=1039+source_t/2*178;c.line([(xx,552),(xx,622)],'#b9cfc8',.7)
 return im.resize((1280,720),Image.Resampling.LANCZOS)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--render',action='store_true');args=parser.parse_args()
 if args.preview:
  for t in [0,3.5,7,10.5,13.5]:draw_method01(t).save(P/f'animated_method01_{t:g}s.png')
  draw_method01(7).save(P/'method01_animated_poster.png')
 if args.render:
  out=P/'method01_animated.mp4';proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','17','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
  for i in range(DURATION*FPS):
   proc.stdin.write(draw_method01(i/FPS).tobytes())
   if i%75==0:print(f'Rendered {i}/{DURATION*FPS} frames',flush=True)
  proc.stdin.close();assert proc.wait()==0;print(out,flush=True)
 P.joinpath('animation_provenance.json').write_text(json.dumps({'scope':'Display animation only. No physics simulation or material fit. Existing stereo kinematics are re-evaluated without changing parameters and checked against frozen display frame.','duration_s':DURATION,'fps':FPS,'hardware_source_time_s':[.05,2.0],'stereo_source_time_s':[.10,1.90],'playback':'One-way, slowed independently for explanation; final 1.5 s hold. This is not a synchronized material-response comparison. Left approved observation/camera panel remains a static reference.','rgbd_surface':'Framewise display envelope of supplied inferred particles, not directly observed interior motion.','constant_volume':'One continuously deforming conceptual surface with diag(1/sqrt(c),1/sqrt(c),c), determinant one. Not measured footage.','stereo':'Actual texture pixels, saved valid correspondences and unchanged square-symmetric reconstruction. Triangulation construction is schematic; point and pixel motion are driven by recorded stereo observations.','force':'Recorded incremental normal robot force from ep0001 robot_force.npz. Baseline-subtracted, not a sketch or command; no independent calibration/tool inertia correction. Cursor follows the hardware source time.','assumptions':'Hardware handoff: axisymmetry, conserved volume, minimum-dissipation incompressible flow with no-slip contacts. Stereo: approximate square-symmetric field fitted to surface displacement.','left_panel':'Pixel content/layout preserved as the approved static reference; this revision animates the six middle panels and the right outputs.'},indent=2)+'\n')
if __name__=='__main__':main()
