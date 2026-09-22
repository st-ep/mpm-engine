"""Static perception explanation with explicit point/ray correspondence."""
from pathlib import Path
import math,json,hashlib
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from matplotlib import colormaps
P=Path(__file__).resolve().parent;ROOT=P.parents[3];S=2
BG='#f4f6f7';INK='#21333e';MUTED='#566975';TEAL='#167b76';BLUE='#2375aa';LINE='#d9e1e5';ACCENT='#d68020'
im=Image.new('RGB',(2560,1440),BG);d=ImageDraw.Draw(im);texts=[]
def txt(x,y,s,size=24,color=INK,bold=False,anchor=None):
 f=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-'+('Bold' if bold else 'Regular')+'.ttf',size*S)
 bb=d.textbbox((x*S,y*S),s,font=f,anchor=anchor)
 assert 0<=bb[0]<bb[2]<=2560 and 0<=bb[1]<bb[3]<=1440,(s,bb)
 d.text((x*S,y*S),s,font=f,fill=color,anchor=anchor);texts.append((s,bb))
def line(p,color=TEAL,width=2):d.line([(x*S,y*S) for x,y in p],fill=color,width=width*S,joint='curve')
def poly(p,c):d.polygon([(x*S,y*S) for x,y in p],fill=c)
def dot(x,y,r,c,outline=None):d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=c,outline=outline,width=S)
def rect(b,c,r=8,outline=None):d.rounded_rectangle(tuple(v*S for v in b),radius=r*S,fill=c,outline=outline,width=S)
def arrow(a,b,color=TEAL,width=2):
 line([a,b],color,width);q=math.atan2(b[1]-a[1],b[0]-a[0]);l=10
 poly([b,(b[0]-l*math.cos(q-.48),b[1]-l*math.sin(q-.48)),(b[0]-l*math.cos(q+.48),b[1]-l*math.sin(q+.48))],color)
def photo(src,xy,size):
 pic=src.convert('RGB').resize(tuple(round(v*S) for v in size),Image.Resampling.LANCZOS)
 mask=Image.new('L',pic.size);ImageDraw.Draw(mask).rounded_rectangle((0,0,*pic.size),radius=7*S,fill=255)
 im.paste(pic,(round(xy[0]*S),round(xy[1]*S)),mask)
def target(p,r=5):
 dot(*p,r+4,'#ffffff');dot(*p,r,ACCENT,INK)
def camera(x,y):
 # Lens center is exactly (x,y): all observation rays terminate here.
 poly([(x-24,y-23),(x-13,y-32),(x+30,y-26),(x+20,y-17)],'#81989f')
 poly([(x+20,y-17),(x+30,y-26),(x+30,y+13),(x+20,y+22)],'#2c4653')
 rect((x-24,y-23,x+21,y+23),'#486571',5)
 for r,c in [(19,'#1c3440'),(15,'#97b2b8'),(12,'#153341'),(8,'#287189')]:dot(x,y,r,c)
 dot(x-3,y-4,3.2,'#85e1cf');dot(x+14,y-14,2,'#68d5b5')

def project(points,box):
 az,el=np.deg2rad([-50,20]);rr=np.array([-np.sin(az),np.cos(az),0]);uu=np.array([-np.cos(az)*np.sin(el),-np.sin(az)*np.sin(el),np.cos(el)]);vv=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
 center=points.mean(0);q=points-center;xy=np.c_[q@rr,-q@uu];dep=q@vv
 x,y,w,h=box;scale=min((w-12)/np.ptp(xy[:,0]),(h-12)/np.ptp(xy[:,1]));mid=(xy.min(0)+xy.max(0))/2
 def transform(p):
  q=p-center;return (np.c_[q@rr,-q@uu]-mid)*scale+[x+w/2,y+h/2]
 return transform(points),dep,transform

def cloud(points,box,ref=None,r=1.5):
 xy,dep,fn=project(points,box);z=points[:,2] if ref is None else ref[:,2];z=(z-z.min())/max(np.ptp(z),1e-9)
 cols=(colormaps['turbo'](z)[:,:3]*255).astype(int);dd=(dep-dep.min())/max(np.ptp(dep),1e-9)
 for k in np.argsort(dep):dot(*xy[k],r,tuple((cols[k]*(.7+.3*dd[k])).astype(int)) if ref is not None else TEAL)
 return fn

tracks_path=ROOT/'out/press_separated_20260913/monotonic320/fit_A/tracks.npz'
tracks=np.load(tracks_path);frame=172;fid=31
raw=[np.load(f'/dev/shm/press_separated_20260913/monotonic320/inputs_A/camera_{i}.npy',mmap_mode='r')[frame] for i in range(2)]
hw_path=ROOT/'out/method_hardware_overlay_20260915/hardware_inputs.npz';hw=np.load(hw_path)
meta=json.loads((ROOT/'press_real_data/ep0001/meta.json').read_text());cal=meta['cameras']['hand'];intr=cal['intrinsics']
depth_path=ROOT/'press_real_data/ep0001/hand_depth/002097.png';depth=cv2.imread(str(depth_path),-1);rgb=hw['observed']
# Exact crop coordinate mapping documented in hardware_source.json.
yy,xx=np.mgrid[0:145,0:235];u=200+(xx+195)/1.2;v=44+(yy+155)/1.2
z=depth[np.rint(v).astype(int),np.rint(u).astype(int)]*cal['depth_scale']
hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);mask=((hsv[...,0]<12)|(hsv[...,0]>170))&(hsv[...,1]>100)&(hsv[...,2]>40)&(z>.18)&(z<.4)
mask=cv2.erode(mask.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
camxyz=np.stack(((u-intr['ppx'])/intr['fx']*z,(v-intr['ppy'])/intr['fy']*z,z),-1)
T=np.array(meta['extrinsics']['cameras']['hand']['T_base_cam']['matrix'])
base=camxyz@T[:3,:3].T+T[:3,3]
validpoints=base[mask][::9]
selected=(95,70);selected3d=base[selected[1],selected[0]];selected_depth=float(z[selected[1],selected[0]])
assert mask[selected[1],selected[0]]
# Depth colors display the actual registered depth, not a generated surface.
lo,hi=.27,.32
color=(colormaps['viridis'](np.clip((z-lo)/(hi-lo),0,1))[...,:3]*255).astype('uint8')
color[(z<=.18)|(z>=.4)]=[235,240,242]

# Strong title, two compact observation explanations, one common handoff.
txt(44,22,'METHOD 01 / OBSERVE',20,TEAL,True)
txt(44,56,'Locate the surface. Reconstruct its motion.',40,INK,True)
line([(44,114),(1236,114)],LINE,1)
line([(640,150),(640,503)],LINE,1)
txt(44,145,'Stereo texture',29,BLUE,True)
txt(604,153,'simulation example',18,MUTED,False,'rt')
txt(677,145,'RGB-D',29,TEAL,True)
txt(1236,153,'hardware example',18,MUTED,False,'rt')

# STEREO: a feature is visible on the specimen, in both camera image patches,
# and at the intersection of two rays. Geometry is a pedagogical schematic;
# feature pixels in all images are actual saved measured correspondences.
xy=tracks['pixels'][:,frame,fid]
crop=(170,400,1160,964);body=(235,239);body_size=(178,101)
photo(Image.fromarray(raw[0]).crop(crop),body,body_size)
p=tuple(np.array(body)+(xy[0]-crop[:2])*[body_size[0]/990,body_size[1]/564])
# One feature, two explicit sight lines: no arbitrary rays into the background.
left_lens=(119,368);right_lens=(524,368)
line([left_lens,p],ACCENT,2);line([right_lens,p],ACCENT,2)
# Thin secondary rays frame a small surface patch around the selected feature.
for lens in [left_lens,right_lens]:
 for q in [(p[0]-9,p[1]-7),(p[0]+9,p[1]+7)]:line([lens,q],'#e9c397',1)
for i,(loc,lens) in enumerate([((57,222),left_lens),((462,222),right_lens)]):
 x,y=xy[i];patch=Image.fromarray(raw[i]).crop((round(x)-90,round(y)-67,round(x)+90,round(y)+67))
 photo(patch,loc,(125,93))
 pc=(loc[0]+62.5,loc[1]+46.5);target(pc,4)
 line([(lens[0],lens[1]-25),(pc[0],315)],'#a9bcc4',1)
 txt(loc[0]+62.5,199,'Left view' if i==0 else 'Right view',20,MUTED,False,'mt')
 camera(*lens)
target(p,5)
# Locate the common feature without covering the texture itself.
txt(330,200,'Same feature',22,ACCENT,True,'mt')
line([(330,228),(p[0],p[1]-10)],ACCENT,1)
txt(44,420,'Match the feature in both views.',24,INK,True)
txt(44,458,'Two calibrated rays locate its 3D position.',23,MUTED)

# RGB-D: highlight precisely the same registered image/depth pixel, then its
# deprojected position in the measured visible surface cloud.
photo(Image.fromarray(rgb),(677,222),(172,106))
photo(Image.fromarray(color),(899,222),(172,106))
for origin in [(677,222),(899,222)]:
 pp=np.array(origin)+np.array(selected)*[172/235,106/145];target(pp,4)
txt(763,198,'RGB pixel',20,MUTED,False,'mt')
txt(985,198,'Aligned depth',20,MUTED,False,'mt')
arrow((866,275),(884,275),TEAL)
rect((1087,247,1227,301),'#fff0da',8)
txt(1157,260,f'{selected_depth*1000:.0f} mm',27,ACCENT,True,'mt')
txt(1157,308,'at this pixel',18,MUTED,False,'mt')
# The cloud is obtained directly from these depth pixels. A ray ends on the
# exact selected point, rather than an arbitrary location in the body.
box=(1031,348,191,126);fn=cloud(validpoints,box,r=1.5)
q=tuple(fn(selected3d[None])[0]);lens=(719,386)
line([lens,q],ACCENT,2)
target(q,4.5);camera(*lens)
txt(861,348,'Viewing direction',20,MUTED,False,'mt')
txt(861,415,'+ measured depth',20,ACCENT,True,'mt')
txt(677,463,'Depth places each pixel in 3D.',24,INK,True)

# Separation between measured surface data and assumed interior trajectories.
line([(44,515),(1236,515)],LINE,1)
txt(44,540,'Across frames',22,TEAL,True)
txt(44,577,'Surface observations',24,INK,True)
txt(44,615,'change over time',23,MUTED)
arrow((301,601),(344,601))
txt(371,554,'Infer interior motion',25,INK,True)
txt(371,595,'Geometric + kinematic',22,MUTED)
txt(371,625,'assumptions',22,MUTED)
ppath=ROOT/'press_real_data/handoff_ep0-7/ep0001/particles.npz';hp=np.load(ppath);k=int(np.argmin(abs(hp['times']-1.52)))
cloud(hp['pos'][k],(674,548,158,121),ref=hp['seed'],r=.72)
arrow((851,604),(903,604))
txt(930,552,'3D motion',28,INK,True)
txt(930,593,'+ contact force',25,INK)
txt(44,681,'NEXT  →  Identify the material law from these observations',23,TEAL,True)
for i,(s,a) in enumerate(texts):
 for t,b in texts[i+1:]:
  assert not(max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])),(s,t)
im.save(P/'method01_insight.png');im.resize((1280,720),Image.Resampling.LANCZOS).save(P/'method01_insight_720p.png')
(P/'insight_provenance.json').write_text(json.dumps({
 'scope':'Static method design. No manuscript or video changes.',
 'stereo':{'source':str(tracks_path.relative_to(ROOT)),'frame':frame,'feature_id':fid,'time_s':float(tracks['time'][frame]),'pixels_in_two_views':xy.tolist(),'body_crop':crop,'diagram':'Ray/camera layout is schematic. All highlighted image points are the same saved feature; patches are from actual left/right observations.'},
 'rgbd':{'episode':'ep0001','frame':2097,'selected_display_pixel':selected,'selected_depth_m':selected_depth,'depth_aligned_to_color':cal['depth_aligned_to_color'],'depth_display_range_m':[lo,hi],'cloud':'Actual depth pixels from the visible red specimen, deprojected using saved intrinsics and extrinsics. No inferred point correspondences.','diagram':'Camera ray positions are schematic; highlighted cloud point is the deprojection of the selected image/depth pixel.'},
 'interior':'Supplied ep0001 reconstruction: axisymmetry, volume conservation, no-slip contacts, minimum-dissipation interior flow. Illustrative particle representation; not ground truth.',
 'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [tracks_path,hw_path,depth_path,ppath,ROOT/'press_real_data/ep0001/meta.json']],
 'new_simulations':0,'new_parameter_fits':0,'new_motion_reconstructions':0},indent=2)+'\n')
