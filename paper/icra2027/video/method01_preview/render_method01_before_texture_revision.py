"""Animated observation/reconstruction slide, using frozen experimental sources.
Public entry draw_method01(t) returns a 1280x720 PIL frame. Main video uses it.
"""
from pathlib import Path
from functools import lru_cache
import argparse,json,math,subprocess
import numpy as np
import cv2
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from matplotlib import colormaps
P=Path(__file__).resolve().parent;ROOT=P.parents[3];DURATION=14;FPS=25;S=2
BG='#f4f6f7';INK='#21333e';TEAL='#167b76';BLUE='#2375aa';MUTED='#566975';LINE='#d9e1e5'
DISPLACEMENT=LinearSegmentedColormap.from_list('displacement',['#2866b5','#36a7d5','#59c29a','#e9d658','#ed9844','#d23f3d'])
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
 hproj=[Projection(hbound,b) for b in [(370,213.5,171,142),(564.5,213.5,171,142),(757.5,213.5,171,142),(1014.5,213.5,219,142)]]
 assert np.allclose([p.scale for p in hproj[:3]],hproj[0].scale)
 rgb=eager(P/'animation_hardware_rgb.npz')
 X=kin['X'];sproj=Projection(X[::10].reshape(-1,3),(766,503,160,94),az=-52,el=24)
 # Use fixed stereo camera image crop across all frames to preserve image motion.
 pix=tr['pixels'][0,:,ids].reshape(-1,2);lo=pix.min(0);hi=pix.max(0);center=(lo+hi)/2
 crop=np.rint([center[0]-145,center[1]-128,center[0]+145,center[1]+128]).astype(int)
 reference,_,_=lerp_frame(hp['pos'],hp['times'],.05)
 displacement=np.linalg.norm(hp['pos'][hsel]-reference,axis=2)
 disp_max_m=float(np.ceil(displacement.max()/.005)*.005)
 return dict(hp=hp,kin=kin,tr=tr,force=force,camera=camera,ids=ids,hsel=hsel,hproj=hproj,sproj=sproj,crop=crop,rgb=rgb,reference=reference,disp_max_m=disp_max_m)

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

def camera_sprite():
    sprite=Image.new('RGBA',(139*S,108*S));sd=ImageDraw.Draw(sprite)
    def pp(points,color):
        sd.polygon([((x-165)*S,(y-224)*S) for x,y in points],fill=color)
    pp([(186,246),(217,224),(304,241),(276,265)],'#789097')
    pp([(276,265),(304,241),(304,306),(276,332)],'#324a56')
    pp([(186,246),(276,265),(276,332),(186,311)],'#4a6470')
    pp([(198,254),(257,267),(257,309),(198,296)],'#21333e')
    sd.rounded_rectangle(((258-165)*S,(271-224)*S,(268-165)*S,(293-224)*S),radius=2*S,fill='#8fa5ad')
    for cx,cy,r,color in [(290,265,3.5,'#62d0b5')]:
        sd.ellipse(((cx-r-165)*S,(cy-r-224)*S,(cx+r-165)*S,(cy+r-224)*S),fill=color)
    for box,color in [((165,258,230,323),'#21333e'),((171,264,224,317),'#839fa7'),((177,270,218,311),'#132d38'),((183,276,212,305),'#24566a')]:
        sd.ellipse(((box[0]-165)*S,(box[1]-224)*S,(box[2]-165)*S,(box[3]-224)*S),fill=color)
    for cx,cy,r,color in [(194,284,5,'#6acdc4'),(201,295,3,'#1f4658')]:
        sd.ellipse(((cx-r-165)*S,(cy-r-224)*S,(cx+r-165)*S,(cy+r-224)*S),fill=color)
    return sprite

def viewing_fan(c,a,b,lens,color):
 a=np.array(a);b=np.array(b);lens=np.array(lens)
 angles=np.unwrap([np.arctan2(*(a-lens)[::-1]),np.arctan2(*(b-lens)[::-1])]);segment=b-a;points=[]
 for angle in np.linspace(*angles,4):
  ray=np.array([np.cos(angle),np.sin(angle)]);distance,along=np.linalg.solve(np.column_stack((ray,-segment)),a-lens)
  q=lens+distance*ray;points.append(q);c.line([q,lens],color,2)
  c.arrow(q+(lens-q)*.43,q+(lens-q)*.54,color,2,5)
  c.dot(q,4.7,INK);c.dot(q,3.8,'white');c.dot(q,2.35,color)
 return np.array(points)

@lru_cache(None)
def cameras():
 sprite=camera_sprite();size=(round(139*.62*S),round(108*.62*S))
 return [sprite.transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize(size,Image.Resampling.LANCZOS),sprite.resize(size,Image.Resampling.LANCZOS)]

def draw_observations(c,D,t,sf):
 # Fixed source crops and frame clocks. Viewing fans are schematic sampling rays,
 # not claimed RGB-D material-point correspondences.
 j=int(np.argmin(abs(D['rgb']['time']-t)));rgb=D['rgb']['rgb'][j]
 for src,box in [(Image.fromarray(rgb),(44,201,270,167)),(Image.fromarray(D['camera'][sf]).convert('RGB').crop((33,180,1247,1052)),(44,453,270,194))]:
  x,y,w,h=box;pic=src.resize((w*S,h*S),Image.Resampling.LANCZOS);mask=Image.new('L',pic.size);ImageDraw.Draw(mask).rounded_rectangle((0,0,*pic.size),radius=7*S,fill=255);c.im.paste(pic,(x*S,y*S),mask)
 # Track the illustrated texture diagonal with measured surface feature motion.
 targets=np.array([[442.,800.],[1045.,576.]])
 valid=D['tr']['valid'][10:191].all(0)
 pix=D['tr']['pixels'][0]
 ends=[]
 for target in targets:
  dist=np.linalg.norm(pix[172]-target,axis=1);dist[~valid]=np.inf;fid=int(np.argmin(dist));ends.append(pix[sf,fid])
 ends=(np.array(ends)-[33,180])*[270/1214,194/872]+[44,453]
 # Match the direction of the upper diagonal to the lower, using visible RGB bounds.
 hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);mask=(((hsv[...,0]<12)|(hsv[...,0]>170))&(hsv[...,1]>100)&(hsv[...,2]>40)).astype('uint8')
 mask[:35]=0;mask[150:]=0;mask[:,:55]=0;mask[:,210:]=0
 height,width=rgb.shape[:2]
 n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8);index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));mask=labels==index
 yy,xx=np.nonzero(mask);center=np.array([np.median(xx),np.median(yy)])
 slope=-(ends[1,1]-ends[0,1])/(ends[1,0]-ends[0,0]);xs=np.linspace(xx.min(),xx.max(),700);ys=center[1]+slope*(270/width)/(167/height)*(xs-center[0]);ix=np.clip(np.rint(xs).astype(int),0,width-1);iy=np.clip(np.rint(ys).astype(int),0,height-1);inside=mask[iy,ix];which=np.flatnonzero(inside);endpoints=np.c_[xs[which[[0,-1]]],ys[which[[0,-1]]]]*[270/width,167/height]+[44,201]
 # Outline only the visible left/right side, avoiding plate and bottom reflection.
 for side in [0,1]:
  curve=[]
  for y in range(int(np.percentile(yy,8)),int(np.percentile(yy,92))):
   x=np.flatnonzero(mask[y]);
   if len(x):curve.append([x[0] if side==0 else x[-1],y])
  curve=np.array(curve)*[270/width,167/height]+[44,201];c.line(curve,'#123a38',2.5);c.line(curve,'#77ead2',.8)
 upper_lens=np.array([75.,383.])+np.array([32.5,108-66.5])*.62
 lower_lens=np.array([205.,383.])+np.array([32.5,66.5])*.62
 viewing_fan(c,*endpoints,upper_lens,TEAL);viewing_fan(c,*ends,lower_lens,BLUE)
 y=383+108*.62/2;c.line([(44,y),(326,y)],TEAL,4);c.poly([(324,y-7.5),(340,y),(324,y+7.5)],TEAL)
 for sprite,xy in zip(cameras(),[(75,383),(205,383)]):c.im.paste(sprite,(xy[0]*S,xy[1]*S),sprite)
 c.text(179,657,'Textured simulation',20,BLUE,True,'mt')

@lru_cache(None)
def static_frame():
 im=Image.open(P/'method01_perception.png').convert('RGB');c=Canvas(im)
 c.d.rectangle((342*S,135*S,2560,im.height-10),fill=BG)
 c.d.rectangle((44*S,198*S,341*S,691*S),fill=BG)
 c.text(654,149,'Reconstruct motion',24,INK,True,'mt')
 c.text(370,183,'RGB-D',22,TEAL,True)
 c.arrow((543,284.5),(565,284.5));c.arrow((735,284.5),(759,284.5))
 for x,title,sub in [(455,'Surface reconstruction','Axisymmetry'),(650,'Same volume','∇ · v = 0'),(843,'Particle advection','ẋ = v(x,t)')]:
  c.text(x,371,title,16.5,INK,True,'mt');c.text(x,398,sub,15.5,TEAL if x!=455 else MUTED,anchor='mt')
 c.text(650,424,'Flow prior',12,MUTED,anchor='mt')
 c.line([(369,448),(941,448)],LINE,1)
 c.text(370,462,'Stereo texture',22,BLUE,True)
 c.arrow((550,550),(576,550),BLUE);c.arrow((722,550),(747,550),BLUE)
 for x,title,sub in [(455,'Track texture','One camera view'),(650,'Triangulate','Calibrated stereo views'),(843,'Extend motion inside','Square-symmetric field')]:
  c.text(x,610,title,16.5,INK,True,'mt');c.text(x,636,sub,13.5,MUTED,anchor='mt')
 c.text(843,489,'u(X,t) = Σⱼ cⱼ(t)φⱼ(X)',12.5,BLUE,anchor='mt')
 # One enclosing input group: motion is reconstructed, force independently logged.
 c.rect((1007,137,1243,679),'#ffffff',13,'#cbd9de')
 c.text(1125,154,'Identification inputs',21,INK,True,'mt')
 c.line([(1022,183),(1228,183)],LINE,.8)
 c.text(1124,193,'Reconstructed motion',18,INK,True,'mt')
 c.text(1124,370,'Positions · velocities · deformation',11.8,MUTED,anchor='mt')
 for j in range(80):c.line([(1083+j,395),(1083+j,400)],tuple((np.array(DISPLACEMENT(j/79)[:3])*255).astype(int)),1)
 c.text(1124,420,'Displacement (mm)',12,MUTED,anchor='mt')
 c.text(1083,404,'0',10.5,MUTED,anchor='mt');c.text(1163,404,f"{data()['disp_max_m']*1000:g}",10.5,MUTED,anchor='mt')
 c.line([(1022,450),(1228,450)],LINE,.8)
 c.text(1124,477,'Recorded normal force',18,INK,True,'mt')
 x0,y0,w,h=1039,505,178,90
 c.line([(x0,y0),(x0,y0+h),(x0+w,y0+h)],'#9aabb4',1)
 for val in [0,10]:
  y=y0+h-val/12*h;c.line([(x0,y),(x0+w,y)],'#dce4e8',.6);c.text(x0-8,y,str(val),11,MUTED,anchor='rm')
 for tv in [0,1,2]:c.text(x0+tv/2*w,y0+h+7,str(tv),11,MUTED,anchor='mt')
 c.text(1027,497,'N',11,MUTED,anchor='mt');c.text(1228,602,'s',11,MUTED)
 c.text(1124,629,'Robot log · contact baseline',12,MUTED,anchor='mt')
 c.text(1124,655,'RGB-D example',13,MUTED,anchor='mt')
 # Full-height bracket encompasses both routes, with a longer centered connector.
 c.line([(956,205),(973,205),(973,662),(956,662)],TEAL,1.5)
 c.dot((973,433.5),2.4);c.arrow((976,433.5),(1003,433.5),TEAL,1.6,7)
 c.text(655,701,'Sequences slowed for clarity · hardware views synchronized',12,MUTED,anchor='mt')
 return im

def draw_method01(t):
 D=data();im=static_frame().copy();c=Canvas(im)
 # One-way playback with a short final hold; no reversed experimental footage.
 phase=float(np.clip(t/12.5,0,1));source_t=.05+1.95*phase
 points,hj,ha=lerp_frame(D['hp']['pos'],D['hp']['times'],source_t)
 mesh=profile(points)
 # All three upper method diagrams use precisely the same surface and projection
 # scale at precisely the same source time. The middle states the flow prior;
 # it no longer substitutes an independently animated affine cartoon.
 for j in range(3):draw_shell(c,mesh,D['hproj'][j],j>0)
 pids=json.loads((P/'rgbd_shape_flow_particles_provenance.json').read_text())['particle_ids']
 for fid in pids:
  use=(D['hp']['times']>=.05)&(D['hp']['times']<source_t)
  history=np.vstack([D['hp']['pos'][use,fid],points[fid]])
  trail=D['hproj'][2](history)
  if len(trail)>1:c.line(trail,'#f8fcfa',4.0)
  for j in range(len(trail)-1):
   a=.45+.55*(j+1)/max(1,len(trail)-1)
   col=tuple(np.rint(np.array([197,226,216])*(1-a)+np.array([17,88,100])*a).astype(int))
   c.line(trail[j:j+2],col,2.15)
  c.dot(trail[-1],3.1,'#155e67','white')
 c.text(940,187,f't = {source_t:.2f} s',13,MUTED,anchor='rt')
 # Actual stereo image sequence. Displayed single view has tracked feature histories.
 sf=int(np.clip(round((.10+1.80*phase)*100),0,200));tr=D['tr'];crop=D['crop'];raw=D['camera'][sf]
 pic=Image.fromarray(raw).crop(tuple(crop)).convert('RGB');c.paste(pic,(369,502,172,96))
 origin2=np.array([369.,502.]);sc=np.array([172/(crop[2]-crop[0]),96/(crop[3]-crop[1])]);colors=['#28a6c6','#e2b23a','#b278c6']
 for fid,col in zip(D['ids'],colors):
  history=tr['pixels'][0,:sf+1,fid];q=(history-crop[:2])*sc+origin2
  if len(q)>1:c.line(q,col,1.3)
  c.dot(q[0],2,'#f4f6f7',col)
  end=q[-1];c.rect((end[0]-5,end[1]-5,end[0]+5,end[1]+5),None,1,col);c.dot(end,2,col)
 # Triangulation construction: two image planes and the actual observed 3D point.
 lp=np.array([[582,504],[612,509],[612,533],[582,528]]);rp=np.array([[688,509],[718,504],[718,528],[688,533]])
 for p in [lp,rp]:c.poly(p,'#e7eff3');c.line(np.vstack([p,p[0]]),'#95b4c5',1)
 fid=D['ids'][0];traj=tr['world'][:,fid];tp=Projection(traj,(636,554,30,36),az=-55,el=25)
 point=tp(tr['world'][sf,[fid]])[0]
 # Pixel motion within the two schematic planes is driven by actual two-camera tracks.
 av=[]
 for ci,cx in [(0,597),(1,703)]:
  pix=tr['pixels'][ci,:,fid];center=(pix.min(0)+pix.max(0))/2
  q=(pix[sf]-center)*.14+[cx,518];av.append(q)
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
 # Color reports total displacement since the displayed initial frame, in
 # physical millimeters, using one fixed scale over the entire interval.
 proj=D['hproj'][3];xy=proj(points);dep=points@proj.view;brightness=.80+.20*(dep-dep.min())/np.ptp(dep)
 displacement=np.linalg.norm(points-D['reference'],axis=1)
 colors=DISPLACEMENT(np.clip(displacement/D['disp_max_m'],0,1))[:,:3]*255
 for j in np.argsort(dep):c.dot(xy[j],.8,tuple((colors[j]*brightness[j]).astype(int)))
 draw_observations(c,D,source_t,sf)
 # Progressive reveal and cursor are tied to the same hardware source time.
 f=D['force'];use=(f['time']>=0)&(f['time']<=source_t);ft=f['time'][use];fv=f['incremental_normal_force'][use]
 path=np.c_[1039+ft/2*178,595-fv/12*90]
 if len(path)>1:c.line(path,TEAL,1.6);c.dot(path[-1],2.4,TEAL,'white')
 xx=1039+source_t/2*178;c.line([(xx,505),(xx,595)],'#b9cfc8',.7)
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
 P.joinpath('animation_provenance.json').write_text(json.dumps({'scope':'Display animation only. No physics simulation or material fit. Existing stereo kinematics are re-evaluated without changing parameters and checked against frozen display frame.','duration_s':DURATION,'fps':FPS,'hardware_source_time_s':[.05,2.0],'stereo_source_time_s':[.10,1.90],'playback':'One-way, slowed independently for explanation; final 1.5 s hold. This is not a synchronized material-response comparison. Input footage and reconstruction are synchronized within each route.','rgbd_surface':'Framewise display envelope of supplied inferred particles, not directly observed interior motion.','constant_volume':'Middle panel now shares the exact same saved-particle surface, source time and fixed scale as both neighboring diagrams. Same volume labels the supplied incompressible reconstruction prior; no independent affine deformation is used.','stereo':'Actual texture pixels, saved valid correspondences and unchanged square-symmetric reconstruction. Triangulation construction is schematic; point and pixel motion are driven by recorded stereo observations.','force':'Recorded incremental normal robot force from ep0001 robot_force.npz. Baseline-subtracted, not a sketch or command; no independent calibration/tool inertia correction. Cursor follows the hardware source time.','assumptions':'Hardware handoff: axisymmetry, conserved volume, minimum-dissipation incompressible flow with no-slip contacts. Stereo: approximate square-symmetric field fitted to surface displacement.','left_panel':'Approved camera geometry retained. Hardware fixed raw crop [330,125,590,286]; simulation fixed crop [33,180,1247,1052], with taller 270x194 display centered on y=550. Both show the upper plate and entire specimen vertically from the start. Schematic viewing rays move with visible surface bounds and stereo features; no RGB-D particle identity is claimed.',
 'color':'Displacement magnitude from displayed initial positions at source t=0.05 s, fixed scale in mm over the interval.',
 'advection':'Five persistent particle IDs with higher-contrast fading history trails and a light underlay, from the displayed initial time; no future-position arrows.',
 'alignment':'Top diagrams and output motion centered at y=284.5. Lower method row, taller input image and force plot centered at y=550. Three top method diagrams have identical projection scales. Right input group enclosed by a titled box; bracket spans y=205..662 at x=973.'},indent=2)+'\n')
if __name__=='__main__':main()
