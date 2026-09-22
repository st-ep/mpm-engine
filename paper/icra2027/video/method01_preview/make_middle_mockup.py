"""Middle-panel proposal, composed over the unchanged approved outer layout."""
from pathlib import Path
import math,json,hashlib
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import ConvexHull
from matplotlib import colormaps
P=Path(__file__).resolve().parent;ROOT=P.parents[3];S=2
im=Image.open(P/'method01_perception.png').convert('RGB');before=np.asarray(im).copy();d=ImageDraw.Draw(im)
INK='#21333e';TEAL='#167b76';BLUE='#2375aa';MUTED='#566975';LINE='#d9e1e5';texts=[]
def text(x,y,s,size=21,color=INK,bold=False,anchor=None):
 f=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-'+('Bold' if bold else 'Regular')+'.ttf',size*S)
 bb=d.textbbox((x*S,y*S),s,font=f,anchor=anchor)
 assert 0<=bb[0]<bb[2]<=im.width and 0<=bb[1]<bb[3]<=im.height,(s,bb)
 d.text((x*S,y*S),s,font=f,fill=color,anchor=anchor);texts.append((s,bb))
def line(points,color=TEAL,width=2):d.line([(round(x*S),round(y*S)) for x,y in points],fill=color,width=width*S,joint='curve')
def circle(p,r,color,outline=None):
 x,y=p;d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=color,outline=outline,width=S)
def rect(box,color,r=6,outline=None,width=1):d.rounded_rectangle(tuple(v*S for v in box),radius=r*S,fill=color,outline=outline,width=width*S)
def arrow(a,b,color=TEAL):
 line([a,b],color,2);q=math.atan2(b[1]-a[1],b[0]-a[0]);l=10
 d.polygon([(b[0]*S,b[1]*S),((b[0]-l*math.cos(q-.45))*S,(b[1]-l*math.sin(q-.45))*S),((b[0]-l*math.cos(q+.45))*S,(b[1]-l*math.sin(q+.45))*S)],fill=color)
def photo(src,xy,size):
 pic=src.convert('RGB').resize(tuple(round(v*S) for v in size),Image.Resampling.LANCZOS)
 mask=Image.new('L',pic.size);ImageDraw.Draw(mask).rounded_rectangle((0,0,*pic.size),radius=6*S,fill=255)
 im.paste(pic,tuple(round(v*S) for v in xy),mask)
def projector(points,box,az=-55,el=23):
 az,el=np.deg2rad([az,el]);right=np.array([-np.sin(az),np.cos(az),0]);up=np.array([-np.cos(az)*np.sin(el),-np.sin(az)*np.sin(el),np.cos(el)]);view=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
 center=points.mean(0);q=points-center;xy=np.c_[q@right,-q@up]
 x,y,w,h=box;mid=(xy.min(0)+xy.max(0))/2;scale=min((w-10)/max(np.ptp(xy[:,0]),1e-9),(h-10)/max(np.ptp(xy[:,1]),1e-9))
 def project(p):
  q=np.asarray(p)-center;return (np.c_[q@right,-q@up]-mid)*scale+[x+w/2,y+h/2]
 return project,view

def dashed_path(p,color,width=1):
 for j in range(0,len(p)-1,5):line(p[j:min(j+3,len(p))],color,width)

# Replace only the interior content, retaining the approved title and white box.
d.rectangle((357*S,247*S,956*S,604*S),fill='white')
line([(369,429),(941,429)],LINE,1)
text(370,253,'RGB-D',26,TEAL,True)
text(940,258,'hardware example',18,MUTED,False,'rt')
text(370,448,'Stereo texture',26,BLUE,True)
text(940,453,'simulation example',18,MUTED,False,'rt')

# RGB-D observations: a plain RGB silhouette with a few actual registered depth
# samples. No original RGB photograph or dense unexplained cloud is repeated.
hwpath=ROOT/'out/method_hardware_overlay_20260915/hardware_inputs.npz';hw=np.load(hwpath);rgb=hw['observed']
depthpath=ROOT/'press_real_data/ep0001/hand_depth/002097.png';depth=cv2.imread(str(depthpath),-1)
yy,xx=np.mgrid[0:145,0:235];u=200+(xx+195)/1.2;v=44+(yy+155)/1.2
z=depth[np.rint(v).astype(int),np.rint(u).astype(int)]*.001
hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
mask=(((hsv[...,0]<12)|(hsv[...,0]>170))&(hsv[...,1]>100)&(hsv[...,2]>40)).astype('uint8')
mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contour=max(contours,key=cv2.contourArea)[:,0,:]
lo=contour.min(0);hi=contour.max(0);sc=min(145/np.ptp(contour[:,0]),98/np.ptp(contour[:,1]));offset=np.array([473.,345.])-(lo+hi)/2*sc
p=contour*sc+offset
# The fill denotes the visible RGB silhouette, not a filled or measured volume.
d.polygon([tuple(q*S) for q in p],fill='#e6f3ee');line(np.vstack([p,p[0]]),TEAL,2)
uvs=np.array([[93,64],[136,58],[76,83],[153,87],[109,103],[138,104]])
measurements=[]
for uv in uvs:
 x,y=uv;assert mask[y,x] and .18<z[y,x]<.4
 color=tuple((colormaps['viridis'](np.clip((z[y,x]-.278)/.025,0,1))[:3]*np.array([255]*3)).astype(int))
 circle(uv*sc+offset,4.3,color,'white');measurements.append(float(z[y,x]))
# A tiny depth-color key connects the colored samples to what is measured.
for j in range(60):
 color=tuple((np.array(colormaps['viridis'](j/59)[:3])*255).astype(int))
 line([(574,305+j),(581,305+j)],color,1)
text(589,306,'near',16,MUTED)
text(589,352,'far',16,MUTED)
arrow((642,344),(693,344),TEAL)
text(473,400,'Silhouette + depth',21,MUTED,False,'mt')

# Existing measured silhouette profile interpreted axisymmetrically. This is a
# surface illustration under that stated assumption, not a new motion solve.
obspath=ROOT/'out/press_observation_assessment_20260913/ep0001/hand_observations.npz';obs=np.load(obspath)
k=int(np.argmin(abs(obs['time']-1.52)));radius=obs['silhouette_radius_m'][k];height=obs['silhouette_z_base_m'][k]
valid=np.isfinite(radius)&np.isfinite(height);radius=radius[valid];height=height[valid]
height=height-height.min();theta=np.linspace(0,2*np.pi,100)
surface=np.stack([radius[:,None]*np.cos(theta),radius[:,None]*np.sin(theta),np.broadcast_to(height[:,None],(len(height),len(theta)))],-1)
project,view=projector(surface.reshape(-1,3),(725,290,194,105))
# Sparse rings reveal the axisymmetric construction; far-side arcs are dashed.
for idx in np.linspace(3,len(height)-3,6).astype(int):
 ring=surface[idx];xy=project(ring)
 front=np.cos(theta)*view[0]+np.sin(theta)*view[1]>0
 for j in range(len(theta)-1):
  if front[j]:line(xy[j:j+2],'#438f85',1)
  elif j%5<2:line(xy[j:j+2],'#b6cdc8',1)
for angle in np.linspace(0,2*np.pi,10,endpoint=False):
 meridian=np.c_[radius*np.cos(angle),radius*np.sin(angle),height]
 if np.cos(angle)*view[0]+np.sin(angle)*view[1]>0:line(project(meridian),'#75a99f',1)
 else:dashed_path(project(meridian),'#c8d9d5',1)
text(822,400,'Axisymmetric surface',21,MUTED,False,'mt')

# Stereo: simultaneous magnified patches, not an earlier/later comparison.
trackpath=ROOT/'out/press_separated_20260913/monotonic320/fit_A/tracks.npz';tr=np.load(trackpath);frame=172
valid=tr['valid'][145:frame+1].all(0)
center=tr['pixels'][0,frame,31]
dist=np.linalg.norm(tr['pixels'][0,frame]-center,axis=1)
# Three features span the close-up instead of illustrating hundreds of tracks.
ids=[31]
for target in [center+[-45,35],center+[45,45]]:
 dd=np.linalg.norm(tr['pixels'][0,frame]-target,axis=1);dd[~valid]=np.inf;dd[ids]=np.inf
 ids.append(int(np.argmin(dd)))
colors=['#28b3d1','#dba437','#b276cd'];crop_records=[]
for cam,x0 in [(0,382),(1,520)]:
 raw=np.load(f'/dev/shm/press_separated_20260913/monotonic320/inputs_A/camera_{cam}.npy',mmap_mode='r')[frame]
 c=tr['pixels'][cam,frame,ids].mean(0);crop=np.rint([c[0]-105,c[1]-82,c[0]+105,c[1]+82]).astype(int)
 crop_records.append(crop.tolist())
 size=np.array([119.,93.]);origin=np.array([x0,482.])
 photo(Image.fromarray(raw).crop(tuple(crop)),origin,size)
 for j,fid in enumerate(ids):
  q=origin+(tr['pixels'][cam,frame,fid]-crop[:2])*size/(crop[2:]-crop[:2])
  rect((q[0]-7,q[1]-7,q[0]+7,q[1]+7),None,r=2,outline=colors[j],width=2)
  circle(q,2.7,colors[j],'#21333e')
 rect((x0+5,557,x0+22,573),'#21333e',2)
 text(x0+13,558,'L' if cam==0 else 'R',13,'white',True,'mt')
# Color-matched bridges indicate cross-view correspondence, not elapsed time.
for j,color in enumerate(colors):
 line([(503,510+j*16),(517,510+j*16)],color,2)
arrow((657,531),(695,531),BLUE)
text(510,585,'Match texture in two views',20,MUTED,False,'mt')

# A small observed surface patch in 3D, with true measured history trails.
patch_ids=np.flatnonzero(valid&(dist<110))
pts=tr['world'][frame,patch_ids]
history=tr['world'][145:frame+1,patch_ids]
allpoints=np.vstack([pts,history.reshape(-1,3)])
project,view=projector(allpoints,(708,481,224,94),az=-75,el=20)
xy=project(pts);hull=ConvexHull(xy);boundary=xy[hull.vertices]
d.polygon([tuple(q*S) for q in boundary],fill='#eff5f8')
line(np.vstack([boundary,boundary[0]]),'#bbcfdb',1)
for q in xy[::max(1,len(xy)//20)]:circle(q,1.2,'#a3bfce')
for color,fid in zip(colors,ids):
 trail=project(tr['world'][145:frame+1,fid]);line(trail,color,2)
 circle(trail[0],3.1,'white',color);circle(trail[-1],4.0,color,INK)
text(822,585,'3D surface tracks',20,MUTED,False,'mt')
# Layout and scope checks: only the center-panel interior may change.
for i,(s,a) in enumerate(texts):
 for t,b in texts[i+1:]:
  assert not(max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])),(s,t)
after=np.asarray(im);changed=np.any(before!=after,axis=2);ys,xs=np.where(changed)
assert xs.min()>=357*S and xs.max()<=956*S and ys.min()>=247*S and ys.max()<=610*S
im.save(P/'method01_middle_mockup.png');im.resize((1280,720),Image.Resampling.LANCZOS).save(P/'method01_middle_mockup_720p.png')
im.crop((342*S,170*S,970*S,616*S)).save(P/'middle_panel_detail.png')
(P/'middle_provenance.json').write_text(json.dumps({
 'scope':'Static center-panel mockup. Approved left and existing right panel are pixel-identical. Main video and manuscript unchanged.',
 'rgbd':{'silhouette':'Largest color-mask component from saved observed RGB crop; depth samples are registered values at selected specimen pixels.','pixel_samples':uvs.tolist(),'depth_m':measurements,'surface':'Existing silhouette radius profile rendered as a sparse surface of revolution. Axisymmetry is explicitly labeled; far-side arcs are dashed.','profile_frame':k,'profile_time_s':float(obs['time'][k]),'depth_color_range_m':[.278,.303]},
 'stereo':{'frame':frame,'time_s':float(tr['time'][frame]),'feature_ids':ids,'camera_crops':crop_records,'scope':'Two synchronized camera patches at one time. Matching windows are display guides, not exported affine-warp estimates. 3D positions and trails are saved valid tracked observations.','trail_interval_s':[float(tr['time'][145]),float(tr['time'][frame])]},
 'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [hwpath,depthpath,obspath,trackpath]],
 'new_simulations':0,'new_parameter_fits':0,'new_motion_reconstructions':0},indent=2)+'\n')
