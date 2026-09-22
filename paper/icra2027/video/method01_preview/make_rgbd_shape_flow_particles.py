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
W,H=600,201
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
def hollow_shell(cx,filled=False):
 # Thin-film surface illustration. The center is intentionally unpopulated;
 # top-cap grid lines establish a closed surface rather than an open container.
 glass_render(cx)
 box=(int(cx-83)*S,3*S,int(cx+83)*S,108*S)
 patch=im.crop(box);patch=Image.blend(patch,Image.new('RGB',patch.size,'white'),.30 if filled else .88)
 im.paste(patch,(box[0],box[1]))
 front=np.cos(theta)*view[0]+np.sin(theta)*view[1]>0
 for idx in [0,15,32,49,64]:
  q=project(mesh[idx],cx)
  for j in range(len(theta)-1):
   if front[j]:line(q[j:j+2],'#78aaa2',.75)
   elif idx==64:line(q[j:j+2],'#abc9c1',.65)
   elif j%6<3:line(q[j:j+2],'#d1e3dc',.5)
 for angle in np.linspace(0,2*np.pi,12,endpoint=False):
  meridian=np.c_[rs*np.cos(angle),rs*np.sin(angle),zs]
  isfront=np.cos(angle)*view[0]+np.sin(angle)*view[1]>0
  line(project(meridian,cx),'#91bcb2' if isfront else '#d2e4dd',.65 if isfront else .45)
 # No fill on the left cap: white background shows through the surface grid.
 # On the right the same cap is deliberately tinted to distinguish volume motion.
 capedge=project(mesh[-1],cx)
 d.polygon([tuple(q*S) for q in capedge],fill='#c3e0d8' if filled else 'white')
 line(capedge,'#84b4a7',.75)
 for angle in np.linspace(0,2*np.pi,8,endpoint=False):
  rr=np.linspace(0,rs[-1],35)
  cap=np.c_[rr*np.cos(angle),rr*np.sin(angle),np.full(len(rr),zhi)]
  line(project(cap,cx),'#a1c6b9',.5)
 for fraction in [.5]:
  cap=np.c_[rs[-1]*fraction*np.cos(theta),rs[-1]*fraction*np.sin(theta),np.full(len(theta),zhi)]
  line(project(cap,cx),'#b2d0c6',.5)
 # Fine double edge adds a film-like rim without filling the interior.
 xy=project(flat,cx);boundary=xy[ConvexHull(xy).vertices]
 line(np.vstack([boundary,boundary[0]]),'#75a99f',.85)

# Independent proposal: only the three RGB-D pictures, not the full slide.
ids=[]
hollow_shell(91)
# Sparse material particles inside the same shell; actual supplied displacements.
hollow_shell(508,filled=True)
candidates=np.flatnonzero((rad<.019)&(xyz[:,2]>zlo+.0025)&(xyz[:,2]<zhi-.0025))
ends=project(xyz[candidates],508)
starts=project(pdata['pos'][k-30,candidates]-center,508)
lengths=np.linalg.norm(ends-starts,axis=1)
chosen=[]
for target in [[483,44],[526,45],[482,70],[523,69],[503,84]]:
 scores=np.linalg.norm(ends-target,axis=1)
 scores[(lengths<12)|(lengths>24)|(starts[:,1]<27)|(ends[:,1]>96)]=np.inf
 for j in np.argsort(scores):
  if not np.isfinite(scores[j]):break
  fid=int(candidates[j]);end=ends[j];start=starts[j]
  if all(min(np.linalg.norm(start-project(pdata['pos'][[k-30],i]-center,508)[0]),np.linalg.norm(end-project(xyz[[i]],508)[0]))>16 for i in chosen):
   chosen.append(fid);break
assert len(chosen)==5, len(chosen)
for fid in chosen:
 end=project(xyz[[fid]],508)[0]
 start=project(pdata['pos'][[k-30],fid]-center,508)[0]
 unit=(end-start)/np.linalg.norm(end-start)
 # One filled particle at the tail; one unobscured triangular head at the end.
 # No second marker is placed on top of the arrowhead.
 tail=start+unit*3.1
 arrow(tail,end,'white',3.3,6)
 arrow(tail,end,'#155e67',1.8,4.6)
 dot(start,2.7,'#155e67','white')
ids.extend(chosen)
# Increase the space given to the actual reconstructions. The middle bridge
# uses roughly half a picture's width, not a full third of the layout.
left=im.crop((8*S,3*S,174*S,108*S));rightpic=im.crop((425*S,3*S,591*S,108*S))
im=Image.new('RGB',(W*S,H*S),'white');d=ImageDraw.Draw(im)
for src,x in [(left,0),(rightpic,384)]:
 im.paste(src.resize((216*S,137*S),Image.Resampling.LANCZOS),(x*S,0))
arrow((207,66),(238,66),TEAL,1.4)
arrow((362,66),(393,66),TEAL,1.4)

# Vertically stacked versions of the same reconstructed display shape.
# The lower shape is an illustrative volume-preserving affine compression,
# not another measured frame or a new physical prediction.
icon_scale=70/np.ptp(proj[:,0])
def shape_icon(cx,cy,compression):
 stretch=np.array([1/np.sqrt(compression),1/np.sqrt(compression),compression])
 assert abs(np.prod(stretch)-1)<1e-12
 points=flat*stretch
 coords=np.c_[points@right,-points@up]
 middle=(coords.min(0)+coords.max(0))/2
 def ip(points):
  points=np.asarray(points)*stretch
  return (np.c_[points@right,-points@up]-middle)*icon_scale+[cx,cy]
 xy=ip(flat);boundary=xy[ConvexHull(xy).vertices]
 d.polygon([tuple(q*S) for q in boundary],fill='#e5f2ec')
 front=np.cos(theta)*view[0]+np.sin(theta)*view[1]>0
 for idx in [0,16,32,48,64]:
  q=ip(mesh[idx])
  for j in range(len(theta)-1):
   if front[j]:line(q[j:j+2],'#83b5a6',.65)
   elif j%6<3:line(q[j:j+2],'#bad9cd',.45)
 for angle in np.linspace(0,2*np.pi,10,endpoint=False):
  meridian=np.c_[rs*np.cos(angle),rs*np.sin(angle),zs]
  isfront=np.cos(angle)*view[0]+np.sin(angle)*view[1]>0
  line(ip(meridian),'#a0c6b8' if isfront else '#c8e1d7',.55)
 cap=ip(mesh[-1]);d.polygon([tuple(q*S) for q in cap],fill='#d7eadf')
 line(cap,'#9ec7b7',.65)
 for angle in np.linspace(0,2*np.pi,8,endpoint=False):
  rr=np.linspace(0,rs[-1],25)
  cap=np.c_[rr*np.cos(angle),rr*np.sin(angle),np.full(len(rr),zhi)]
  line(ip(cap),'#b0d0c1',.45)
 line(np.vstack([boundary,boundary[0]]),'#659e8d',.9)
 return xy.min(0),xy.max(0)
upper=shape_icon(301,32,1.)
lower=shape_icon(301,92,.60)
assert upper[1][1]<lower[0][1]
arrow((301,upper[1][1]+2),(301,lower[0][1]-2),'#689e92',1.2,3.5)
text(301,120,'Same volume',11.5,MUTED)
text(301,138,'∇ · v = 0',15,TEAL)
text(108,140,'Surface only',10.5,MUTED)
text(492,140,'Reconstructed particles',10.5,MUTED)
for cx,title,sub in [(108,'Reconstruct the shape','Silhouettes + axisymmetry'),(301,'Infer flow','Boundary + flow prior'),(492,'Move the particles','Positions → motion + deformation')]:
 text(cx,161,title,16,INK,True);text(cx,184,sub,11.2,MUTED)
im.save(P/'rgbd_shape_flow_particles.png')
(P/'rgbd_shape_flow_particles_provenance.json').write_text(json.dumps({
 'scope':'Standalone static RGB-D triptych proposal only. No full slide, video, manuscript or scientific results changed.',
 'shell':'Same display-only radial envelope of supplied ep0001 reconstructed particles. First panel: white/unfilled top cap with fine surface grid. Third panel: tinted top cap and stronger volume tint. Both enlarged equally for display. Hollow appearance communicates missing interior kinematics, not a physically hollow specimen.',
 'middle':'Vertically stacked versions of the same display shell. Lower shape is a schematic affine compression diag(1/sqrt(0.6),1/sqrt(0.6),0.6), determinant 1, so volume is exactly preserved. Neither pair nor compression is a new measurement or physical prediction. Boundary + flow prior refers to the supplied minimum-dissipation reconstruction with its contact assumptions.',
 'particles':'Five saved handoff particle displacements. Each filled dot marks the earlier position and its arrow ends at the later position. True endpoint displacements, no length amplification, no dot covering the arrowhead; not observed RGB-D material identities.',
 'particle_ids':ids,'frame':int(k),'time_s':float(pdata['times'][k]),'displacement_interval_s':[float(pdata['times'][k-30]),float(pdata['times'][k])],
 'source':'press_real_data/handoff_ep0-7/ep0001/particles.npz',
 'underlying_handoff_assumptions':['axisymmetry','conserved volume','minimum-dissipation incompressible flow','no-slip contacts']
},indent=2)+'\n')
print(P/'rgbd_shape_flow_particles.png')
