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
# Increase the space given to the actual reconstructions. The middle bridge
# uses roughly half a picture's width, not a full third of the layout.
left=im.crop((8*S,3*S,174*S,108*S));rightpic=im.crop((425*S,3*S,591*S,108*S))
im=Image.new('RGB',(W*S,H*S),'white');d=ImageDraw.Draw(im)
for src,x in [(left,0),(rightpic,384)]:
 im.paste(src.resize((216*S,137*S),Image.Resampling.LANCZOS),(x*S,0))
arrow((207,66),(238,66),TEAL,1.4)
arrow((362,66),(393,66),TEAL,1.4)

# A compact, explicitly conceptual constant-volume deformation icon.
# Ellipsoids have equal a*b*c, not equal projected area. It is not a measured pair.
def ellipsoid_icon(cx,cy,a,c):
 t=np.linspace(0,2*np.pi,140)
 elev=np.deg2rad(18);height=np.sqrt((a*np.sin(elev))**2+(c*np.cos(elev))**2)
 border=np.c_[cx+a*np.cos(t),cy+height*np.sin(t)]
 d.polygon([tuple(q*S) for q in border],fill='#e3f1eb')
 line(border,'#5c9f91',.9)
 for h in [-.45,0,.45]:
  r=a*np.sqrt(1-h*h)
  q=np.c_[cx+r*np.cos(t),cy+r*np.sin(t)*np.sin(elev)-c*h*np.cos(elev)]
  line(q,'#accfc2',.55)
 # One meridian helps distinguish a volume from a flat silhouette.
 q=np.c_[cx+a*.5*np.cos(t),cy+height*np.sin(t)]
 line(q,'#accfc2',.55)
ellipsoid_icon(272,49,15,19)
a=(15*15*19/9)**.5
ellipsoid_icon(330,49,a,9)
assert abs(15*15*19-a*a*9)<1e-8
arrow((292,49),(303,49),'#689e92',1.1,3.3)
text(301,76,'Same volume',11.5,MUTED)
text(301,96,'∇ · v = 0',15,TEAL)
text(108,140,'Surface only',10.5,MUTED)
text(492,140,'Reconstructed particles',10.5,MUTED)
for cx,title,sub in [(108,'Reconstruct the shape','Silhouettes + axisymmetry'),(301,'Infer flow','Boundary + flow prior'),(492,'Move the particles','Positions → motion + deformation')]:
 text(cx,161,title,16,INK,True);text(cx,184,sub,11.2,MUTED)
im.save(P/'rgbd_shape_flow_particles.png')
(P/'rgbd_shape_flow_particles_provenance.json').write_text(json.dumps({
 'scope':'Standalone static RGB-D triptych proposal only. No full slide, video, manuscript or scientific results changed.',
 'shell':'Same display-only radial envelope of supplied ep0001 reconstructed particles. First panel: white/unfilled top cap with fine surface grid. Third panel: tinted top cap and stronger volume tint. Both enlarged equally for display. Hollow appearance communicates missing interior kinematics, not a physically hollow specimen.',
 'middle':'Compact conceptual constant-volume icon: two ellipsoids with identical a*b*c, followed by incompressibility equation. This icon is not measured specimen motion. Boundary + flow prior refers to the supplied minimum-dissipation reconstruction with its contact assumptions.',
 'particles':'Saved handoff reconstructed positions and true finite displacements, not observed RGB-D material identities.',
 'particle_ids':ids,'frame':int(k),'time_s':float(pdata['times'][k]),'displacement_interval_s':[float(pdata['times'][k-18]),float(pdata['times'][k])],
 'source':'press_real_data/handoff_ep0-7/ep0001/particles.npz',
 'underlying_handoff_assumptions':['axisymmetry','conserved volume','minimum-dissipation incompressible flow','no-slip contacts']
},indent=2)+'\n')
print(P/'rgbd_shape_flow_particles.png')
