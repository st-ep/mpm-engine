"""Static RGB-D design: observed side curves, display shell, supplied inferred paths.
No geometry fitting for identification, new flow solve, or manuscript/video changes.
"""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw, ImageFont
P=Path(__file__).resolve().parent; ROOT=P.parents[3]; S=3
W,H=600,145
im=Image.new('RGB',(W*S,H*S),'white'); d=ImageDraw.Draw(im)
TEAL='#167b76'; INK='#21333e'; MUTED='#566975'
def line(p,c=TEAL,w=1): d.line([tuple(np.array(q)*S) for q in p],fill=c,width=max(1,round(w*S)),joint='curve')
def dot(p,r,c=TEAL,edge='white'):
 x,y=p;d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=c,outline=edge,width=S)
def text(x,y,t,size=17,c=INK,bold=False):
 f=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-'+('Bold' if bold else 'Regular')+'.ttf',round(size*S))
 d.text((x*S,y*S),t,font=f,fill=c,anchor='mt')
def arrow(a,b,c=TEAL,w=1.5,head=5):
 a=np.array(a);b=np.array(b);v=(b-a)/np.linalg.norm(b-a);n=np.array([-v[1],v[0]])
 line([a,b-v*head*.5],c,w);d.polygon([tuple(q*S) for q in [b,b-head*v+head*.45*n,b-head*v-head*.45*n]],fill=c)
def dashed(p,c='#86aaa5',w=1):
 for i in range(0,len(p)-1,5):line(p[i:i+3],c,w)
hw=np.load(ROOT/'out/method_hardware_overlay_20260915/hardware_inputs.npz')
pdata=np.load(ROOT/'press_real_data/handoff_ep0-7/ep0001/particles.npz')
k=int(np.argmin(abs(pdata['times']-1.52)));pts=pdata['pos'][k].astype(float)
center=(pts.max(0)+pts.min(0))/2; xyz=pts-center
# Smooth radial envelope of supplied reconstructed particles: only a display mesh.
zlo,zhi=xyz[:,2].min(),xyz[:,2].max();zb=np.linspace(zlo,zhi,18);rad=np.hypot(xyz[:,0],xyz[:,1]);rr=[]
for z in zb:
 hit=abs(xyz[:,2]-z)<(zhi-zlo)/20
 rr.append(float(np.max(rad[hit])))
rr=gaussian_filter1d(rr,.7); zs=np.linspace(zlo,zhi,65); rs=PchipInterpolator(zb,rr)(zs)
theta=np.linspace(0,2*np.pi,121)
mesh=np.stack([rs[:,None]*np.cos(theta),rs[:,None]*np.sin(theta),np.broadcast_to(zs[:,None],(65,121))],-1)
az,el=np.deg2rad([-60,23]);right=np.array([-np.sin(az),np.cos(az),0]);up=np.array([-np.cos(az)*np.sin(el),-np.sin(az)*np.sin(el),np.cos(el)]);view=np.cross(right,up)
flat=mesh.reshape(-1,3);proj=np.c_[flat@right,-flat@up];scale=min(148/np.ptp(proj[:,0]),92/np.ptp(proj[:,1]));mid=(proj.min(0)+proj.max(0))/2

def project(p,cx):
 p=np.asarray(p);return (np.c_[p@right,-p@up]-mid)*scale+[cx,59]

def shell(cx,particles=False):
 # Translucent jade shell: continuous shading, sparse construction rings.
 xy=project(flat,cx); boundary=xy[ConvexHull(xy).vertices]
 mask=Image.new('L',im.size);md=ImageDraw.Draw(mask);md.polygon([tuple(q*S) for q in boundary],fill=255)
 yy,xx=np.mgrid[0:H*S,0:W*S]; nx=(xx/S-cx)/74;ny=(yy/S-59)/46
 edge=np.clip(np.abs(nx),0,1)**3
 strength=.10+.20*edge+.05*np.clip(ny,0,1)
 shine=np.exp(-((nx+.48)/.14)**2)*np.clip(1-np.abs(ny)*.55,0,1)
 strength=np.clip(strength-.11*shine,.025,.4)
 col=np.array([75,163,154]);pixels=(255*(1-strength[...,None])+col*strength[...,None]).astype('uint8')
 im.paste(Image.fromarray(pixels),(0,0),mask)
 # hidden rings and center axis, then reconstructed trajectories, then front rings
 front=np.cos(theta)*view[0]+np.sin(theta)*view[1]>0
 for idx in [7,23,40,57]:
  q=project(mesh[idx],cx)
  for j in range(len(theta)-1):
   if not front[j] and j%6<3:line(q[j:j+2],'#aacdc6',.65)
 axis=project(np.array([[0,0,zlo-.003],[0,0,zhi+.003]]),cx)
 dashed(np.linspace(*axis,45),'#a8bab7',.7)
 if particles:
  # Select a few well-separated interior material identities. Paths are exactly
  # saved handoff trajectories, never fabricated radial arrows.
  candidates=np.flatnonzero((rad<.017)&(xyz[:,2]>zlo+.003)&(xyz[:,2]<zhi-.003))
  q=project(xyz[candidates],cx);chosen=[]
  for target in [[cx-41,49],[cx,37],[cx+39,49],[cx-43,72],[cx,65],[cx+39,74],[cx-15,86],[cx+15,51]]:
   for j in np.argsort(np.linalg.norm(q-np.array(target),axis=1)):
    fid=int(candidates[j]);end=project(xyz[[fid]],cx)[0]
    if all(np.linalg.norm(end-project(xyz[[i]],cx)[0])>13 for i in chosen):chosen.append(fid);break
  for fid in chosen:
   trail=project(pdata['pos'][max(0,k-24):k+1,fid]-center,cx)
   line(trail,'#167b76',1.25);dot(trail[0],1.6,'white',TEAL)
   if np.linalg.norm(trail[-1]-trail[-5])>1:arrow(trail[-5],trail[-1],TEAL,1.25,3.5)
   dot(trail[-1],2.5,TEAL)
  ids.extend(chosen)
 for idx in [7,23,40,57]:
  q=project(mesh[idx],cx)
  for j in range(len(theta)-1):
   if front[j]:line(q[j:j+2],'#72ada2',.8)
 line(np.vstack([boundary,boundary[0]]),'#4d998e',1)
 # Specular stroke on the left flank communicates a transparent surface.
 j=int(np.argmax(np.cos(theta)*view[0]+np.sin(theta)*view[1]))
 jj=(j+17)%120
 line(project(mesh[10:53,jj],cx),'#ffffff',1.7)
ids=[]
# Observed sides only, with dashed closure: not a new segmented silhouette.
curves=hw['contour_pixels']; q=curves.copy();q=(q-[120,80])*[1.18,1.18]+[88,59]
poly=np.vstack([q[0],q[1][::-1]])
d.polygon([tuple(a*S) for a in poly],fill='#eef7f3')
for curve in q:line(curve,TEAL,1.5)
for a,b in [(q[0,0],q[1,0]),(q[0,-1],q[1,-1])]:dashed(np.linspace(a,b,45),'#a4b9b4',.8)
# Real registered-depth samples, without an uninformative depth colormap.
depthpath=ROOT/'press_real_data/ep0001/hand_depth/002097.png'
import cv2
depth=cv2.imread(str(depthpath),-1);uv=np.array([[93,64],[136,58],[76,83],[153,87],[109,103]])
zvals=[]
for x,y in uv:
 u=round(200+(x+195)/1.2);v=round(44+(y+155)/1.2);z=float(depth[v,u])*.001;assert .18<z<.4;zvals.append(z)
 dot((np.array([x,y])-[120,80])*[1.18,1.18]+[88,59],2.6,TEAL)
# Two small contact guides; not a drawn material boundary.
line([(38,18),(142,18)],'#adb9bd',2);line([(30,101),(149,101)],'#adb9bd',2)
arrow((169,59),(199,59));shell(298)
arrow((391,59),(421,59));shell(510,True)
for cx,title,sub in [(88,'Observed boundary','RGB outline + depth'),(298,'Reconstruct 3D shape','Axisymmetry + fixed volume'),(510,'Infer interior motion','Incompressible flow prior')]:
 text(cx,109,title,16.5,INK,True);text(cx,130,sub,12.5,MUTED)
im.save(P/'rgbd_three_steps_detail.png')
# Composite only RGB-D row and central title. The approved left/right and existing
# stereo mockup remain exactly unchanged.
base=Image.open(P/'method01_middle_mockup.png').convert('RGB');before=np.asarray(base).copy();bd=ImageDraw.Draw(base)
bd.rectangle((357*2,182*2,956*2,232*2),fill='white')
f=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Bold.ttf',29*2)
bd.text((369*2,190*2),'Reconstruct material motion',font=f,fill=INK)
bd.rectangle((357*2,284*2,956*2,426*2),fill='white')
# Keep the 3 illustrations readable at slide size; fitting preserves aspect.
row=im.resize((590*2,round(H*590/600)*2),Image.Resampling.LANCZOS)
base.paste(row,(361*2,285*2))
base.save(P/'method01_rgbd_three_steps.png');base.resize((1280,720),Image.Resampling.LANCZOS).save(P/'method01_rgbd_three_steps_720p.png')
after=np.asarray(base);assert np.array_equal(before[:,:342*2],after[:,:342*2]);assert np.array_equal(before[:,970*2:],after[:,970*2:]);assert np.array_equal(before[435*2:],after[435*2:])
(P/'rgbd_three_steps_provenance.json').write_text(json.dumps({'scope':'Static RGB-D mockup; left cameras, right output and stereo row unchanged. No video or manuscript edits.', 'observed':'Saved RGB side contours and five registered depth samples. Dashed closures are schematic, not measured contacts. Grey bars indicate contacts schematically.','depth_samples_m':zvals,'shell':'Smoothed axisymmetric radial envelope of supplied ep0001 reconstructed particles, display only; not a measured surface or a new fit.','shell_time_s':float(pdata['times'][k]),'particle_ids':ids,'paths_interval_s':[float(pdata['times'][k-24]),float(pdata['times'][k])],'motion':'Saved handoff inferred trajectories under axisymmetry, conserved volume, minimum-dissipation incompressible flow and no-slip contacts. No newly solved motion.','sources':['out/method_hardware_overlay_20260915/hardware_inputs.npz','press_real_data/ep0001/hand_depth/002097.png','press_real_data/handoff_ep0-7/ep0001/particles.npz'],'qualification':'Illustrates supplied handoff reconstruction; does not assert identical priors to the final material fit.'},indent=2)+'\n')
print(P/'rgbd_three_steps_detail.png')
