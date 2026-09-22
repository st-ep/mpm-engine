"""Integrated static Method 1 proposal. Frozen data; no video/manuscript edits."""
from pathlib import Path
import json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from matplotlib import colormaps
P=Path(__file__).resolve().parent;ROOT=P.parents[3];S=2
BG='#f4f6f7';INK='#21333e';TEAL='#167b76';BLUE='#2375aa';MUTED='#566975';LINE='#d9e1e5'
base=Image.open(P/'method01_perception.png').convert('RGB');before=np.asarray(base).copy();im=base.copy();d=ImageDraw.Draw(im);boxes=[]
def text(x,y,s,size=20,color=INK,bold=False,anchor=None):
 font='/usr/share/fonts/truetype/lato/Lato-'+('Bold' if bold else 'Regular')+'.ttf'
 if any(c in s for c in '∇ẋΣφⱼ'):font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
 f=ImageFont.truetype(font,round(size*S));bb=d.textbbox((x*S,y*S),s,font=f,anchor=anchor)
 assert bb[0]>=0 and bb[2]<=im.width and bb[1]>=0 and bb[3]<=im.height,(s,bb)
 d.text((x*S,y*S),s,font=f,fill=color,anchor=anchor);boxes.append((s,bb))
def line(pts,c=TEAL,w=1):d.line([tuple(np.array(p)*S) for p in pts],fill=c,width=max(1,round(w*S)),joint='curve')
def poly(pts,c):d.polygon([tuple(np.array(p)*S) for p in pts],fill=c)
def dot(p,r,c=TEAL,edge=None):
 x,y=p;d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=c,outline=edge,width=S)
def rect(box,c,r=8,edge=None):d.rounded_rectangle(tuple(v*S for v in box),radius=r*S,fill=c,outline=edge,width=S)
def arrow(a,b,c=TEAL,w=1.6,head=7):
 a=np.asarray(a);b=np.asarray(b);v=(b-a)/np.linalg.norm(b-a);n=np.array([-v[1],v[0]])
 line([a,b-v*head/2],c,w);poly([b,b-head*v+head*.45*n,b-head*v-head*.45*n],c)
def paste_crop(src,crop,box):
 x,y,w,h=box;pic=src.crop(tuple(round(v) for v in crop)).resize((round(w*S),round(h*S)),Image.Resampling.LANCZOS)
 im.paste(pic,(round(x*S),round(y*S)))
def project_for(pts,box,az=-60,el=23):
 az,el=np.deg2rad([az,el]);right=np.array([-np.sin(az),np.cos(az),0]);up=np.array([-np.cos(az)*np.sin(el),-np.sin(az)*np.sin(el),np.cos(el)]);view=np.cross(right,up)
 origin=pts.mean(0);xy=np.c_[(pts-origin)@right,-(pts-origin)@up];mid=(xy.min(0)+xy.max(0))/2
 x,y,w,h=box;scale=min(w/np.ptp(xy[:,0]),h/np.ptp(xy[:,1]))
 def project(p):
  p=np.asarray(p)-origin;return (np.c_[p@right,-p@up]-mid)*scale+[x+w/2,y+h/2]
 return project,view
# Preserve every pixel of the approved left column. Rebuild only center/output.
d.rectangle((342*S,135*S,1280*S,720*S),fill=BG)
rect((342,174,970,671),'#e4e9ec',20)
rect((342,170,970,667),'white',20,LINE)
text(369,190,'Reconstruct material motion',29,INK,True)
line([(369,236),(941,236)],LINE)
text(370,250,'RGB-D',25,TEAL,True)
text(940,254,'hardware example',16,MUTED,anchor='rt')
# Approved RGB-D diagrams, used without captions to fit the integrated design.
rgb=Image.open(P/'rgbd_shape_flow_particles.png').copy();rs=3
# Exclude fragments of the standalone figure's inter-panel arrows.
rdraw=ImageDraw.Draw(rgb)
for box in [(199,55,216,78),(384,55,403,78)]:rdraw.rectangle(tuple(v*rs for v in box),fill='white')
paste_crop(rgb,(0,0,216*rs,137*rs),(366,288,176,112))
paste_crop(rgb,(245*rs,0,357*rs,154*rs),(594,281,107,147))
paste_crop(rgb,(384*rs,0,600*rs,137*rs),(754,288,176,112))
arrow((546,342),(579,342),TEAL)
arrow((709,342),(742,342),TEAL)
text(454,403,'Reconstruct surface',17,INK,True,'mt')
text(454,426,'Axisymmetry',15,MUTED,anchor='mt')
# Middle labels are already present in the approved diagram (Same volume, div v).
text(842,403,'Advect particles',17,INK,True,'mt')
text(842,426,'ẋ = v(x,t)',16,TEAL,anchor='mt')
text(648,435,'Flow prior',11.5,MUTED,anchor='mt')
line([(369,452),(941,452)],LINE)
text(370,466,'Stereo texture',25,BLUE,True)
text(940,470,'simulation example',16,MUTED,anchor='rt')
# Stereo panel 1: simultaneous camera patches and the same three texture IDs.
trpath=ROOT/'out/press_separated_20260913/monotonic320/fit_A/tracks.npz';tr=np.load(trpath);frame=172
ids=json.loads((P/'middle_provenance.json').read_text())['stereo']['feature_ids'];colors=['#28a6c6','#d29b27','#ab71bd']
crop_records=[]
for cam,x in [(0,371),(1,461)]:
 raw=np.load(f'/dev/shm/press_separated_20260913/monotonic320/inputs_A/camera_{cam}.npy',mmap_mode='r')[frame]
 center=tr['pixels'][cam,frame,ids].mean(0);crop=np.rint([center[0]-95,center[1]-105,center[0]+95,center[1]+105]).astype(int);crop_records.append(crop.tolist())
 src=Image.fromarray(raw);paste_crop(src,crop,(x,511,77,85))
 for fid,c in zip(ids,colors):
  q=(tr['pixels'][cam,frame,fid]-crop[:2])*[77/190,85/210]+[x,511]
  rect((q[0]-5,q[1]-5,q[0]+5,q[1]+5),None,1,c);dot(q,2.2,c)
 rect((x+4,515,x+19,532),INK,2);text(x+11.5,516,'L' if cam==0 else 'R',12,'white',True,'mt')
for j,c in enumerate(colors):line([(450,536+17*j),(459,536+17*j)],c,1.5)
arrow((547,551),(568,551),BLUE)
# Stereo panel 2: schematic simultaneous viewing rays triangulate one surface point.
# Image planes, not earlier/later frames. Geometry is explicitly illustrative.
leftplane=np.array([[576,506],[611,511],[611,540],[576,535]])
rightplane=np.array([[691,511],[726,506],[726,535],[691,540]])
for plane in [leftplane,rightplane]:
 poly(plane,'#eaf2f6');line(np.vstack([plane,plane[0]]),'#9ab9ca',1)
 for f in [.33,.66]:line([plane[0]*(1-f)+plane[1]*f,plane[3]*(1-f)+plane[2]*f],'#c8dbe5',.6)
patch=np.array([[621,577],[651,565],[683,579],[653,598]])
poly(patch,'#e6f2f6');line(np.vstack([patch,patch[0]]),'#a8c6d5',1)
point=np.array([651,579]);a=np.array([594,523]);b=np.array([708,523])
line([a,point],BLUE,1.7);line([b,point],BLUE,1.7)
for q in [a,b]:dot(q,3.3,colors[0],'white')
dot(point,4.3,colors[0],INK)
text(651,506,'Two views',12,MUTED,anchor='mt')
arrow((731,551),(751,551),BLUE)
# Stereo panel 3: actual saved reconstructed lattice, observed side emphasized.
rdpath=ROOT/'out/method_shaping_separated_20260913/reconstructed_display.npz';rd=np.load(rdpath);X=rd['x'];Q=rd['reference'];project,view=project_for(X,(772,509,143,91),az=-52,el=24)
shape=tuple(len(np.unique(Q[:,j])) for j in range(3));assert shape==(11,11,11),shape
lattice=X.reshape(*shape,3)
for axis in range(3):
 for a in [0,5,10]:
  for b in [0,5,10]:
   index=[a,b];sl=[];j=0
   for dim in range(3):
    if dim==axis:sl.append(slice(None))
    else:sl.append(index[j]);j+=1
   pts=lattice[tuple(sl)]
   line(project(pts),'#b9ced8',.85)
# y-min is the observed front side of this setup. This shows only a few anchors.
for i in [1,5,9]:
 for j in [1,5,9]:dot(project(lattice[i,0,j][None])[0],2.4,BLUE,'white')
for cx,title,sub in [(454,'Match texture','Same features in both views'),(651,'Triangulate','3D surface positions'),(842,'Extend inside','Smooth field + square symmetry')]:
 text(cx,612,title,17,INK,True,'mt');text(cx,637,sub,12.5,MUTED,anchor='mt')
# Compact mathematical expression tied to the reconstructed spatial lattice.
text(844,491,'u(X,t) = Σⱼ cⱼ(t)φⱼ(X)',13.5,BLUE,anchor='mt')
# Two methods feed one common representation, illustrated by the hardware episode.
line([(970,351),(988,351),(988,444),(1007,444)],TEAL,1.6)
line([(970,552),(988,552),(988,444)],TEAL,1.6)
arrow((991,444),(1008,444),TEAL,1.6);dot((988,444),2.6,TEAL)
text(1124,154,'Inputs to',22,INK,True,'mt');text(1124,182,'identification',24,INK,True,'mt')
text(1124,231,'Reconstructed motion',19,INK,True,'mt')
hp=np.load(ROOT/'press_real_data/handoff_ep0-7/ep0001/particles.npz');k=int(np.argmin(abs(hp['times']-1.52)))
points=hp['pos'][k];project,view=project_for(points,(1012,272,218,155),az=-48,el=21)
xy=project(points);depth=points@view;dep=(depth-depth.min())/np.ptp(depth)
z=(hp['seed'][:,2]-hp['seed'][:,2].min())/np.ptp(hp['seed'][:,2]);cs=(colormaps['turbo'](z)[:,:3]*255)
for j in np.argsort(depth):dot(xy[j],.8,tuple((cs[j]*(.7+.3*dep[j])).astype(int)))
text(1124,443,'Positions · velocities · deformation',12.2,MUTED,anchor='mt')
# Color meaning, rather than unexplained rainbow coding.
for j in range(80):
 c=tuple((np.array(colormaps['turbo'](j/79)[:3])*255).astype(int));line([(1083+j,466),(1083+j,471)],c,1)
text(1124,477,'Color: initial height',12,MUTED,anchor='mt')
text(1124,518,'Measured normal force',18,INK,True,'mt')
fp=ROOT/'out/press_observation_assessment_20260913/ep0001/robot_force.npz';force=np.load(fp);use=(force['time']>=0)&(force['time']<=10.5)
t=force['time'][use];f=force['incremental_normal_force'][use]
x0,y0,w,h=1039,552,178,69
line([(x0,y0),(x0,y0+h),(x0+w,y0+h)],'#9aabb4',1)
for yval in [0,10]:
 yy=y0+h-yval/12*h;line([(x0-3,yy),(x0+w,yy)],'#dce4e8',.6);text(x0-8,yy,str(yval),11,MUTED,anchor='rm')
for tv in [0,5,10]:
 xx=x0+tv/10.5*w;line([(xx,y0+h),(xx,y0+h+3)],'#9aabb4',1);text(xx,y0+h+6,str(tv),11,MUTED,anchor='mt')
line(np.c_[x0+t/10.5*w,y0+h-f/12*h],TEAL,1.6)
text(1027,543,'N',11,MUTED,anchor='mt');text(1228,628,'s',11,MUTED)
text(1124,649,'Relative to contact baseline',12,MUTED,anchor='mt')
text(1124,673,'RGB-D example',13,MUTED,anchor='mt')
text(654,693,'Interior motion requires geometric and kinematic assumptions.',17,MUTED,anchor='mt')
# Scope check: the approved observation/camera column stays pixel-identical.
assert np.array_equal(np.asarray(im)[:,:342*S],before[:,:342*S])
# All text bounding boxes should be separated (including actual font extents).
collisions=[]
for i,(name,a) in enumerate(boxes):
 for other,b in boxes[i+1:]:
  if max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3]):collisions.append([name,other])
assert not collisions,collisions
im.save(P/'method01_full_reconstruction.png');im.resize((1280,720),Image.Resampling.LANCZOS).save(P/'method01_full_reconstruction_720p.png')
(P/'full_reconstruction_provenance.json').write_text(json.dumps({'scope':'Full static Method 1 proposal only. Main video and manuscript unchanged. Approved left column pixel-identical.','rgbd':'Approved standalone conceptual reconstruction assets; xdot=v identifies particle advection. See rgbd_shape_flow_particles_provenance.json.','stereo':{'feature_ids':ids,'frame':frame,'time_s':float(tr['time'][frame]),'camera_crops':crop_records,'triangulation':'Illustrative calibrated stereo construction, not a plotted pair of actual camera extrinsics.','interior':'Actual saved reconstructed_display.npz positions, sparse lattice. Approximate square-symmetric spatial field; not an incompressibility constraint.'},'output':{'episode':'ep0001','time_s':float(hp['times'][k]),'color':'initial particle height','force':'Recorded incremental normal force relative to contact baseline; no independent sensor calibration or inertia correction; not recovered from video.','force_time_range_s':[float(t[0]),float(t[-1])]},'sources':[str(p.relative_to(ROOT)) for p in [trpath,rdpath,fp]],'new_simulations':0,'new_fits':0},indent=2)+'\n')
print(P/'method01_full_reconstruction.png')
