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
W,H=600,193
im=Image.new('RGB',(W*S,H*S),'white'); d=ImageDraw.Draw(im)
TEAL='#167b76'; INK='#21333e'; MUTED='#566975'
def line(p,c=TEAL,w=1): d.line([tuple(np.array(q)*S) for q in p],fill=c,width=max(1,round(w*S)),joint='curve')
def dot(p,r,c=TEAL,edge='white'):
 x,y=p;d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=c,outline=edge,width=S)
def text(x,y,t,size=17,c=INK,bold=False):
 f=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-'+('Bold' if bold else 'Regular')+'.ttf',round(size*S))
 if '∇' in t: f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',round(size*S))
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
# Independent proposal: only the three RGB-D pictures, not the full slide.
ids=[]
shell(91)
axis=project(np.array([[0,0,zlo-.001],[0,0,zhi+.001]]),91)
dashed(np.linspace(*axis,50),'#6fa69b',.85)
arrow((172,59),(201,59),TEAL,1.5)

# Cutaway: a radial/vertical plane reveals a schematic divergence-free flow.
# This is an explanatory prior, not a freshly estimated experimental flow field.
shell(298)
ytop,ybottom=19,100
height=zhi-zlo
zcut=np.linspace(zlo,zhi,80)
radius=PchipInterpolator(zs,rs)(zcut)
# The viewing plane is deliberately face-on so the flow constraint is readable.
cutscale=46/max(rs)
outline=np.vstack([np.c_[298-radius*cutscale,ybottom-(zcut-zlo)/height*(ybottom-ytop)],np.c_[298+radius[::-1]*cutscale,ybottom-(zcut[::-1]-zlo)/height*(ybottom-ytop)]])
d.polygon([tuple(q*S) for q in outline],fill='#edf6f2')
line(np.vstack([outline,outline[0]]),'#7ea99f',.8)
# Surface motion arrows use a contrasting teal and a simple consistent pattern.
for x in [280,298,316]:arrow((x,5),(x,20),'#137b77',1.7,4.4)
arrow((253,56),(241,56),'#137b77',1.7,4.4)
arrow((343,56),(355,56),'#137b77',1.7,4.4)
# Analytic axisymmetric incompressible schematic:
# vr = r f'(z)/2, vz = -f(z), f(z)=H(3s²-2s³).
# Zero tangential velocity at both contacts, bottom stationary, top descending.
for s0 in [.27,.52,.78]:
 for r0 in [-.7,0,.7]:
  x=298+r0*35;y=ybottom-s0*(ybottom-ytop)
  vr=(r0*35)*3*s0*(1-s0);vz=(3*s0*s0-2*s0**3)*(ybottom-ytop)
  velocity=np.array([vr,vz]);delta=velocity*.22
  arrow((x,y),np.array([x,y])+delta,'#458d8a',1.15,3.4)
# A minimal equation states the constraint without implying it uniquely determines flow.
text(298,109,'∇ · v = 0',16,TEAL,True)
text(298,128,'Flow prior · schematic',9.5,MUTED)
arrow((391,59),(420,59),TEAL,1.5)

# Sparse material particles inside the same shell; actual supplied displacements.
shell(508)
candidates=np.flatnonzero((rad<.019)&(xyz[:,2]>zlo+.0025)&(xyz[:,2]<zhi-.0025))
q=project(xyz[candidates],508);chosen=[]
for target in [[481,35],[508,31],[533,38],[474,55],[496,51],[521,55],[540,60],[481,77],[506,74],[530,81],[504,93]]:
 for j in np.argsort(np.linalg.norm(q-target,axis=1)):
  fid=int(candidates[j]);end=q[j]
  if all(np.linalg.norm(end-project(xyz[[i]],508)[0])>11 for i in chosen):chosen.append(fid);break
for j,fid in enumerate(chosen):
 end=project(xyz[[fid]],508)[0]
 start=project(pdata['pos'][[k-18],fid]-center,508)[0]
 if j%2==0 and np.linalg.norm(end-start)>3:
  arrow(start,end,'white',2.7,5);arrow(start,end,'#176c72',1.25,4)
  dot(start,2,'white','#7eada7')
 dot(end,2.8,'#176c72','white')
ids.extend(chosen)
text(91,113,'Reconstructed surface',10.5,MUTED)
text(508,113,'Reconstructed particles',10.5,MUTED)
for cx,title,sub in [(91,'Reconstruct the shape','Silhouettes + axisymmetry'),(298,'Extend motion inside','Volume-preserving flow'),(508,'Move the particles','Positions → motion + deformation')]:
 text(cx,148,title,16,INK,True);text(cx,172,sub,11.5,MUTED)
im.save(P/'rgbd_shape_flow_particles.png')
(P/'rgbd_shape_flow_particles_provenance.json').write_text(json.dumps({
 'scope':'Standalone static RGB-D triptych proposal only. No full slide, video, manuscript or scientific results changed.',
 'shell':'Same display-only radial envelope of supplied ep0001 reconstructed particles as preceding mockup.',
 'middle':'Schematic cross-section with analytic axisymmetric divergence-free field: v_r=r f_prime(z)/2, v_z=-f(z), f(z)=H(3s^2-2s^3), s=z/H. Not fitted to the recording; illustrates the incompressible flow prior, which also needs boundary conditions and a selection principle.',
 'particles':'Saved handoff reconstructed positions and true finite displacements, not observed RGB-D material identities.',
 'particle_ids':ids,'frame':int(k),'time_s':float(pdata['times'][k]),'displacement_interval_s':[float(pdata['times'][k-18]),float(pdata['times'][k])],
 'source':'press_real_data/handoff_ep0-7/ep0001/particles.npz',
 'underlying_handoff_assumptions':['axisymmetry','conserved volume','minimum-dissipation incompressible flow','no-slip contacts']
},indent=2)+'\n')
print(P/'rgbd_shape_flow_particles.png')
