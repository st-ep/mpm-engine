"""Static RGB-D design: observed side curves, display shell, supplied inferred paths.
No geometry fitting for identification, new flow solve, or manuscript/video changes.
"""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
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

def glass_render(cx):
 # Orthographic rendering of the same display envelope. Fresnel tint and soft
 # studio reflections are presentational only; they do not alter its geometry.
 xa,ya,ww,hh=round(cx-82),5,164,103
 yy,xx=np.mgrid[0:hh*S,0:ww*S]
 u=(xx/S+xa-cx)/scale+mid[0];v=-(yy/S+ya-59)/scale-mid[1]
 origin=u[...,None]*right+v[...,None]*up
 hit=np.zeros(u.shape,bool);front=np.zeros(u.shape);back=np.zeros(u.shape)
 for dep in np.linspace(.055,-.055,240):
  q=origin+dep*view; r=np.interp(q[...,2],zs,rs)
  inside=(np.hypot(q[...,0],q[...,1])<=r)&(q[...,2]>=zlo)&(q[...,2]<=zhi)
  fresh=inside&~hit;front[fresh]=dep;back[inside]=dep;hit|=inside
 q=origin+front[...,None]*view
 radius=np.maximum(np.hypot(q[...,0],q[...,1]),1e-9)
 dr=np.interp(q[...,2],zs,np.gradient(rs,zs))
 n=np.stack([q[...,0]/radius,q[...,1]/radius,-dr],-1)
 cap=(q[...,2]>zhi-.00065)|(q[...,2]<zlo+.00065)
 n[cap]=np.array([0,0,1]);n/=np.maximum(np.linalg.norm(n,axis=-1)[...,None],1e-9)
 facing=np.abs(n@view);fresnel=(1-facing)**3
 thickness=np.maximum(front-back,0)/.05
 tint=.09+.36*fresnel+.16*thickness
 base=np.array([46.,139.,134.]);rgb=255*(1-tint[...,None])+base*tint[...,None]
 # Two differently sized rectangular light reflections bend with surface normals.
 nr=n@right;nu=n@up
 soft=np.exp(-((nr+.49)/.13)**6-((nu-.14)/.65)**6)*.88
 slim=np.exp(-((nr-.68)/.055)**4-((nu-.05)/.72)**6)*.74
 reflection=np.maximum(soft,slim)*(1-.35*fresnel)
 rgb=rgb*(1-reflection[...,None])+255*reflection[...,None]
 rgba=np.dstack([np.clip(rgb,0,255).astype('uint8'),hit.astype('uint8')*255])
 # Soft ground shadow anchors the shell while keeping the interior transparent.
 shadow=Image.new('RGBA',im.size);sd=ImageDraw.Draw(shadow)
 sd.ellipse(((cx-46)*S,99*S,(cx+46)*S,104*S),fill=(33,83,79,23))
 shadow=shadow.filter(ImageFilter.GaussianBlur(3*S));im.paste(shadow,(0,0),shadow)
 layer=Image.fromarray(rgba);im.paste(layer,(xa*S,ya*S),layer)

def shell(cx,particles=False):
 glass_render(cx)
 # A sparse curved grid explains the surface; back curves remain faint.
 front=np.cos(theta)*view[0]+np.sin(theta)*view[1]>0
 for idx in [8,24,41,57]:
  q=project(mesh[idx],cx)
  for j in range(len(theta)-1):
   if not front[j] and j%6<3:line(q[j:j+2],'#bad4ce',.45)
 for angle in np.linspace(0,2*np.pi,12,endpoint=False):
  if np.cos(angle)*view[0]+np.sin(angle)*view[1]>0:
   meridian=np.c_[rs*np.cos(angle),rs*np.sin(angle),zs]
   line(project(meridian,cx),'#a0c8c0',.55)
 for idx in [8,24,41,57]:
  q=project(mesh[idx],cx)
  for j in range(len(theta)-1):
   if front[j]:line(q[j:j+2],'#91bcb4',.65)
 if particles:
  # Straight arrows show net displacement over a stated interval, not a
  # smoothed/fabricated trajectory or an instantaneous velocity vector.
  candidates=np.flatnonzero((rad<.017)&(xyz[:,2]>zlo+.003)&(xyz[:,2]<zhi-.003))
  endpoints=project(xyz[candidates],cx)
  starts=project(pdata['pos'][k-24,candidates]-center,cx)
  lengths=np.linalg.norm(endpoints-starts,axis=1)
  chosen=[]
  for target in [[cx-22,45],[cx+18,43],[cx-24,74],[cx+24,74],[cx,64]]:
   scores=np.linalg.norm(endpoints-np.array(target),axis=1)
   scores[(lengths<7)|(lengths>24)|(starts[:,1]<21)]=np.inf
   for j in np.argsort(scores):
    if not np.isfinite(scores[j]):break
    fid=int(candidates[j]);end=endpoints[j]
    if all(np.linalg.norm(end-project(xyz[[i]],cx)[0])>17 for i in chosen):chosen.append(fid);break
  for fid in chosen:
   a=project(pdata['pos'][[k-24],fid]-center,cx)[0];b=project(xyz[[fid]],cx)[0]
   # White underlay keeps vectors legible through the surface pattern.
   arrow(a,b,'white',3.6,6.8);arrow(a,b,'#115f64',1.9,5.3)
   dot(a,2.5,'#115f64','white')
  ids.extend(chosen)
ids=[]
# A subdued real image makes the highlighted contour self-explanatory.
# No anonymous depth points or artificial surface identities are displayed.
source=Image.fromarray(hw['observed']).crop((49,40,187,118))
source=ImageEnhance.Color(source).enhance(.4)
source=Image.blend(source,Image.new('RGB',source.size,'white'),.35)
source=source.resize((154*S,87*S),Image.Resampling.LANCZOS)
mask=Image.new('L',source.size);ImageDraw.Draw(mask).rounded_rectangle((0,0,*source.size),radius=6*S,fill=255)
im.paste(source,(11*S,15*S),mask)
for curve in hw['contour_pixels']:
 q=(curve-[49,40])*[154/138,87/78]+[11,15]
 line(q,'white',3.7);line(q,TEAL,1.9)
arrow((169,59),(199,59));shell(298)
arrow((391,59),(421,59));shell(510,True)
for cx,title,sub in [(88,'Observed boundary','Contour from RGB frames'),(298,'Reconstruct 3D shape','Axisymmetry + fixed volume'),(510,'Infer interior motion','Incompressible flow prior')]:
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
(P/'rgbd_three_steps_provenance.json').write_text(json.dumps({'scope':'Static RGB-D mockup; left cameras, right output and stereo row unchanged. No video or manuscript edits.', 'observed':'Saved RGB side contours over the matching real RGB crop; image desaturated and lightened for readability. No anonymous depth samples are shown.','shell':'Smoothed axisymmetric radial envelope of supplied ep0001 reconstructed particles, display only; not a measured surface or a new fit.','shell_time_s':float(pdata['times'][k]),'particle_ids':ids,'displacement_interval_s':[float(pdata['times'][k-24]),float(pdata['times'][k])],'arrows':'True endpoint displacements over the specified interval, without length amplification. Straight arrows are not instantaneous velocities or full paths.','motion':'Saved handoff inferred trajectories under axisymmetry, conserved volume, minimum-dissipation incompressible flow and no-slip contacts. No newly solved motion.','sources':['out/method_hardware_overlay_20260915/hardware_inputs.npz','press_real_data/ep0001/hand_depth/002097.png','press_real_data/handoff_ep0-7/ep0001/particles.npz'],'qualification':'Illustrates supplied handoff reconstruction; does not assert identical priors to the final material fit.'},indent=2)+'\n')
print(P/'rgbd_three_steps_detail.png')
