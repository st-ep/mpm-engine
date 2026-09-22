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
P=Path(__file__).resolve().parent;ROOT=P.parents[3];DURATION=16;FPS=25;S=2
BG='#f4f6f7';INK='#21333e';TEAL='#167b76';BLUE='#2375aa';ORANGE='#c96932';MUTED='#566975';LINE='#d9e1e5'
RAY_GREEN='#63c86b';RAY_BLUE='#55b9ec'
DIVIDER_Y=(137+691)/2  # 414 px: balanced visible image gaps and outline-centered bracket.
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
 camera=np.load(P.parent/'sources/stereo_press_A/camera_0.npy',mmap_mode='r')
 # Three persistent features spread across the observed texture surface.
 ids=[828,814,361]
 camera_right=np.load(P.parent/'sources/stereo_press_A/camera_1.npy',mmap_mode='r')
 valid=tr['valid'][0:201,ids].all();assert valid
 # Static framing across time: deformation is never hidden by per-frame resizing.
 hsel=np.flatnonzero((hp['times']>=0)&(hp['times']<=2.05))
 reference,_,_=lerp_frame(hp['pos'],hp['times'],.05)
 display_pos=centered_positions(hp['pos'],reference)
 hbound=display_pos[hsel[::3],::3].reshape(-1,3)
 hproj=[Projection(hbound,b) for b in [(370,213.5,171,142),(564.5,213.5,171,142),(757.5,213.5,171,142),(1014.5,213.5,219,142)]]
 assert np.allclose([p.scale for p in hproj[:3]],hproj[0].scale)
 rgb=eager(P/'animation_hardware_rgb.npz')
 X=kin['X'];grid=X.reshape(-1,11,11,11,3)
 # Fixed magnification of a material subvolume, never amplified displacement.
 local=grid[10:191,3:8,3:8,3:9]
 sproj=Projection(local.reshape(-1,3),(769,473,159,154),az=-52,el=24)
 # Use fixed stereo camera image crop across all frames to preserve image motion.
 pix=tr['pixels'][0,:,ids].reshape(-1,2);lo=pix.min(0);hi=pix.max(0);center=(lo+hi)/2
 crop=np.array([660,630,930,807])  # historical close-up, retained for provenance
 stereo_crops=[]
 for ci in [0,1]:
  q=tr['pixels'][ci,10:191][:,ids].reshape(-1,2);lo=q.min(0);hi=q.max(0);mid=(lo+hi)/2
  ch=max(hi[1]-lo[1]+32,(hi[0]-lo[0]+40)/(172/56));cw=ch*172/56
  stereo_crops.append(np.rint([mid[0]-cw/2,mid[1]-ch/2,mid[0]+cw/2,mid[1]+ch/2]).astype(int))
 # Magnify the texture region around the three highlighted material features.
 q=tr['pixels'][0,10:191][:,ids].reshape(-1,2);lo=q.min(0);hi=q.max(0);mid=(lo+hi)/2
 ch=max(hi[1]-lo[1]+34,(hi[0]-lo[0]+40)/(184/154));cw=ch*184/154
 texture_crop=np.rint([mid[0]-cw/2,mid[1]-ch/2,mid[0]+cw/2,mid[1]+ch/2]).astype(int)
 # Local material patch: interpolate only among observed stereo surface tracks.
 # A fixed material-domain grid is advected by these saved surface observations.
 from scipy.spatial import Delaunay
 valid_ids=np.flatnonzero(tr['valid'][10:191].all(0));ref=tr['world'][10,valid_ids]
 xz=tr['world'][10,ids][:,[0,2]];patch_lo=xz.min(0)-.0015;patch_hi=xz.max(0)+.0015
 gx,gz=np.meshgrid(np.linspace(patch_lo[0],patch_hi[0],17),np.linspace(patch_lo[1],patch_hi[1],17))
 query=np.c_[gx.ravel(),gz.ravel()];tri=Delaunay(ref[:,[0,2]])
 simplex=tri.find_simplex(query);assert (simplex>=0).all(), 'No surface extrapolation allowed'
 transform=tri.transform[simplex];bary=np.einsum('nij,nj->ni',transform[:,:2],query-transform[:,2]);weights=np.c_[bary,1-bary.sum(1)]
 vertices=valid_ids[tri.simplices[simplex]]
 patch=np.einsum('qn,tqnc->tqc',weights,tr['world'][:,vertices]).reshape(len(tr['time']),17,17,3)
 # Shallow constant extrusion is a schematic depth cue, not recovered interior.
 patch_depth=.002;offset=np.array([0.,patch_depth,0.])
 bound=np.vstack([patch[10:191:5].reshape(-1,3),(patch[10:191:5]+offset).reshape(-1,3)])
 surface_proj=Projection(bound,(570,473,159,154),az=-58,el=23)


 reference,_,_=lerp_frame(hp['pos'],hp['times'],.05)
 displacement=np.linalg.norm(hp['pos'][hsel]-reference,axis=2)
 disp_max_m=float(np.ceil(displacement.max()/.005)*.005)
 return dict(patch=patch,patch_depth=patch_depth,patch_vertices=vertices,patch_weights=weights,patch_reference_bounds=np.stack([patch_lo,patch_hi]),texture_crop=texture_crop,surface_proj=surface_proj,hp=hp,display_pos=display_pos,kin=kin,tr=tr,force=force,camera=camera,camera_right=camera_right,ids=ids,hsel=hsel,hproj=hproj,sproj=sproj,crop=crop,rgb=rgb,reference=reference,disp_max_m=disp_max_m)

def surface_view(D,t):
 # Fixed camera for the deforming local observed surface patch.
 return D['surface_proj']

def draw_surface_patch(c,D,proj,frame):
 front=D['patch'][frame];back=front+np.array([0.,D['patch_depth'],0.])
 f=proj(front.reshape(-1,3)).reshape(17,17,2);b=proj(back.reshape(-1,3)).reshape(17,17,2)
 def boundary(grid):return np.vstack([grid[0],grid[1:,-1],grid[-1,-2::-1],grid[-2:0:-1,0]])
 rear=boundary(b)
 c.poly(rear,'#ecf4f6');c.line(np.vstack([rear,rear[0]]),'#c5dce4',.7)
 # The top and side are shallow glass-like schematic extrusion faces.
 top=np.vstack([f[-1],b[-1,::-1]]);side=np.vstack([f[:,-1],b[::-1,-1]])
 c.poly(top,'#deedf2');c.line(np.vstack([top,top[0]]),'#a5c6d5',.8)
 c.poly(side,'#d2e6ee');c.line(np.vstack([side,side[0]]),'#a0c3d3',.8)
 edge=boundary(f);c.poly(edge,'#e0eef3')
 # Sparse surface lines, no dense point cloud or tiny triangle texture.
 for i in [4,8,12]:
  c.line(f[i],'#b0ccd9',.75);c.line(f[:,i],'#abc9d8',.75)
 c.line(np.vstack([edge,edge[0]]),'#78a8bf',1.1)
 c.line(f[2:15,2],'#f8fcfd',1.8)
 c.line(f[-1,1:-1],'#f7fcfd',1.3)


# Persistent strictly interior grid nodes; their paths are from the saved field.
INTERIOR_NODES=[(4,4,7),(4,6,4),(6,4,7),(6,6,4)]
def draw_stereo_interior(c,D,frame):
 lattice=D['kin']['X'][frame].reshape(11,11,11,3);sp=D['sproj']
 local=lattice[3:8,3:8,3:9]
 def boundary(grid):return np.vstack([grid[0],grid[1:,-1],grid[-1,-2::-1],grid[-2:0:-1,0]])
 # Pale open cutaway, with no front fill covering the interior paths.
 for grid,fill,edge in [(local[:,-1,:],'#e9f1f5','#c3d8e4'),
                         (local[-1,:,:],'#dcebf2','#a5c5d6'),
                         (local[:,:,0],'#e0edf3','#b2cddd')]:
  rim=sp(boundary(grid));c.poly(rim,fill);c.line(np.vstack([rim,rim[0]]),edge,.85)
 rim=sp(boundary(local[:,0,:]));c.line(np.vstack([rim,rim[0]]),'#a1bfd0',.8)
 for node in INTERIOR_NODES:
  fid=np.ravel_multi_index(node,(11,11,11))
  path=sp(D['kin']['X'][10:frame+1,fid])
  if len(path)>1:
   c.line(path,'#ffffff',5.5);c.line(path,'#205f89',2.8)
  c.dot(path[0],3.4,'#f4f8fa','#205f89')
  c.dot(path[-1],4.4,'#174e75','white')


def lerp_frame(arr,times,t):
 j=int(np.clip(np.searchsorted(times,t)-1,0,len(times)-2));a=float(np.clip((t-times[j])/(times[j+1]-times[j]),0,1));return arr[j]*(1-a)+arr[j+1]*a,j,a

def centered_positions(points,reference):
 # Display-only moving origin: remove common XY translation, preserve world Z.
 # No temporal filtering, scaling, rotation or alteration of source arrays.
 q=np.asarray(points,dtype=np.float64).copy()
 q[...,:2]+=np.asarray(reference,dtype=np.float64)[:,:2].mean(0)-q[...,:2].mean(axis=-2,keepdims=True)
 return q

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
def camera_asset():
 # Approved standalone design; fit without aspect-ratio distortion.
 asset=Image.open(P/'camera_designs/camera_A_refined.png').convert('RGBA')
 metadata=json.loads((P/'camera_designs/camera_A_refined_geometry.json').read_text())
 bbox=asset.getbbox();asset=asset.crop(bbox)
 target=(139*S,108*S);scale=min(target[0]/asset.width,target[1]/asset.height)
 size=(round(asset.width*scale),round(asset.height*scale))
 offset=np.array([(target[0]-size[0])//2,(target[1]-size[1])//2])
 sprite=Image.new('RGBA',target);sprite.alpha_composite(asset.resize(size,Image.Resampling.LANCZOS),tuple(offset))
 lens=(np.array(metadata['lens_center_px'])-bbox[:2])*np.array(size)/np.array(asset.size)+offset
 return sprite,lens/S

def viewing_fan(c,a,b,lens,color,clock=0.):
 a=np.array(a);b=np.array(b);lens=np.array(lens)
 angles=np.unwrap([np.arctan2(*(a-lens)[::-1]),np.arctan2(*(b-lens)[::-1])]);segment=b-a;points=[]
 for angle in np.linspace(*angles,4):
  ray=np.array([np.cos(angle),np.sin(angle)]);distance,along=np.linalg.solve(np.column_stack((ray,-segment)),a-lens)
  q=lens+distance*ray;points.append(q);c.line([q,lens],color,2.5)
  # Traveling chevron indicates visual information going toward the lens.
  progress=.12+.76*(round(clock%1.,9))
  start=q+(lens-q)*(progress-.055);end=q+(lens-q)*(progress+.055)
  c.arrow(start,end,'#f6faf8',4.5,11)
  c.arrow(start,end,color,2.8,9)
  c.dot(q,4.7,INK);c.dot(q,3.8,'white');c.dot(q,2.35,color)
 return np.array(points)

@lru_cache(None)
def cameras():
 # Lenses stay on the left: upper camera is the vertical mirror of the lower.
 # Apply the identical affine map to each icon and its exact lens coordinate.
 source,lens=camera_asset();placements=[]
 for upper,cx in [(True,118.),(False,248.)]:
  icon=source.transpose(Image.Transpose.FLIP_TOP_BOTTOM) if upper else source
  point=lens*S
  if upper:point=np.array([point[0],source.height-point[1]])
  w,h=icon.size;M=cv2.getRotationMatrix2D((w/2,h/2),-25 if upper else 25,1.)
  corners=np.c_[np.array([[0,0],[w,0],[w,h],[0,h]]),np.ones(4)]@M.T
  lo=corners.min(0);hi=corners.max(0);M[:,2]-=lo
  rotated=cv2.warpAffine(np.asarray(icon),M,tuple(np.ceil(hi-lo).astype(int)),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
  tile=Image.fromarray(rotated);point=M@np.r_[point,1.]
  bbox=tile.getbbox();tile=tile.crop(bbox);point-=bbox[:2]
  factor=min(94*S/tile.width,72*S/tile.height)
  size=(round(tile.width*factor),round(tile.height*factor))
  point*=np.array(size)/np.array(tile.size)
  tile=tile.resize(size,Image.Resampling.LANCZOS)
  xy=np.rint([cx*S-size[0]/2,DIVIDER_Y*S-size[1]/2]).astype(int)
  placements.append((tile,xy/S,(xy+point)/S))
 return placements

def draw_observations(c,D,t,sf,clock=0.,arrow_alpha=1.):
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
 placements=cameras();upper_lens=placements[0][2];lower_lens=placements[1][2]
 if arrow_alpha>0:
  y=DIVIDER_Y
  col=tuple(round(a*(1-arrow_alpha)+b*arrow_alpha) for a,b in zip((244,246,247),(22,123,118)))
  c.line([(44,y),(340,y)],col,2.8);c.poly([(338,y-5.5),(350,y),(338,y+5.5)],col)
 for sprite,xy,_ in placements:c.im.paste(sprite,tuple(np.rint(xy*S).astype(int)),sprite)
 # Draw viewing rays in front of the icons so they visibly reach the apertures.
 viewing_fan(c,*endpoints,upper_lens,RAY_GREEN,clock);viewing_fan(c,*ends,lower_lens,RAY_BLUE,clock)
 c.text(179,657,'Textured simulation',20,BLUE,True,'mt')

@lru_cache(None)
def static_frame():
 im=Image.new('RGB',(1280*S,720*S),BG);c=Canvas(im)
 c.d.rectangle((0,0,2560,114*S),fill=BG)
 c.text(44,22,'FORM · From Observed Response to Material laws',23,TEAL,True)
 c.text(1236,22,'01 / OBSERVE',20,TEAL,True,anchor='rt')
 c.text(44,56,'From visual observations to 3D motion',41,INK,True)
 c.line([(44,114),(1236,114)],LINE,1)
 # Three equally styled stage boundaries; content retains a common background.
 for box in [(32,137,326,691),(350,137,949,691),(1007,137,1243,691)]:
  c.rect(box,BG,12,'#ccd9e0')
 c.text(179,149,'Observe deformation',24,INK,True,'mt')
 c.text(179,177,'Real experiment',20,INK,True,'mt')
 c.text(654,149,'Reconstruct motion',24,INK,True,'mt')
 c.text(654,183,'RGB-D',22,TEAL,True,'mt')
 c.arrow((543,284.5),(565,284.5),TEAL,2.4,8);c.arrow((735,284.5),(759,284.5),TEAL,2.4,8)
 for x,title,sub in [(455,'Surface reconstruction','Axisymmetry'),(650,'Same volume','∇ · v = 0'),(843,'Particle advection','ẋ = v(x,t)')]:
  c.text(x,365,title,16.5,INK,True,'mt');c.text(x,388,sub,15.5,TEAL if x!=455 else MUTED,anchor='mt')
 c.line([(350,DIVIDER_Y),(949,DIVIDER_Y)],LINE,1)
 c.text(654,427,'Stereo texture',22,BLUE,True,'mt')
 c.arrow((735,550),(759,550),BLUE,2.4,8)
 for x,title,sub in [(455,'Track texture','Stereo · one view shown'),(650,'Triangulate','3D surface points'),(843,'Infer interior motion','')]:
  c.text(x,637,title,16.5,INK,True,'mt');c.text(x,660,sub,13.5,MUTED,anchor='mt')
 c.text(843,660,'u(X,t) = Σⱼ cⱼ(t)φⱼ(X)',15.5,BLUE,anchor='mt')
 # One enclosing input group: motion is reconstructed, force independently logged.
 c.text(1125,149,'Identification inputs',21,INK,True,'mt')
 c.line([(1022,183),(1228,183)],LINE,.8)
 c.text(1124,193,'Reconstructed motion',18,INK,True,'mt')
 c.text(1124,363,'Positions · velocities · deformation',11.8,MUTED,anchor='mt')
 for j in range(100):c.line([(1074+j,382),(1074+j,387)],tuple((np.array(DISPLACEMENT(j/99)[:3])*255).astype(int)),1)
 c.text(1124,391,'Displacement (mm)',12,MUTED,anchor='mt')
 c.text(1065,384.5,'0',10.5,MUTED,anchor='rm');c.text(1183,384.5,f"{data()['disp_max_m']*1000:g}",10.5,MUTED,anchor='lm')
 c.line([(1007,DIVIDER_Y),(1243,DIVIDER_Y)],LINE,.8)
 c.text(1124,486,'Measured normal force',18,INK,True,'mt')
 x0,y0,w,h=1039,511,178,126
 c.line([(x0,y0),(x0,y0+h),(x0+w,y0+h)],'#9aabb4',1)
 for val in [0,10]:
  y=y0+h-val/12*h;c.line([(x0,y),(x0+w,y)],'#dce4e8',.6);c.text(x0-8,y,str(val),11,MUTED,anchor='rm')
 for tv in [0,1,2]:c.text(x0+tv/2*w,y0+h+7,str(tv),11,MUTED,anchor='mt')
 c.text(1027,507,'N',11,MUTED,anchor='mt');c.text(1228,644,'s',11,MUTED)
 # Full-height bracket encompasses both routes, with a longer centered connector.
 c.line([(956,137),(973,137),(973,691),(956,691)],TEAL,1.5)
 c.dot((973,DIVIDER_Y),2.4);c.arrow((973,DIVIDER_Y),(1007,DIVIDER_Y),TEAL,2.4,8)
 return im

LOOP_SECONDS=4.0
PLAYBACK_RATE=.5

def playback_times(t):
 # Replay the saved press forward; a common clock resets all trails and plots.
 cycle=round(float(t)%LOOP_SECONDS,9)
 return min(.05+PLAYBACK_RATE*cycle,2.0), min(.10+PLAYBACK_RATE*cycle,1.90)

REVEAL_SECONDS=.6
FAINT_OPACITY=.07

def reveal_progress(t,start):
 a=float(np.clip((t-start)/REVEAL_SECONDS,0.,1.))
 return a*a*(3-2*a)

def draw_method01(t,reveal=True):
 D=data();im=static_frame().copy();c=Canvas(im)
 # Four-second loops at half source speed, with no reverse playback.
 source_t,stereo_t=playback_times(t)
 source_points,hj,ha=lerp_frame(D['hp']['pos'],D['hp']['times'],source_t)
 points=centered_positions(source_points,D['reference'])
 mesh=profile(points)
 # All three upper method diagrams use precisely the same surface and projection
 # scale at precisely the same source time. The middle states the flow prior;
 # it no longer substitutes an independently animated affine cartoon.
 for j in range(3):draw_shell(c,mesh,D['hproj'][j],j>0)
 pids=json.loads((P/'rgbd_shape_flow_particles_provenance.json').read_text())['particle_ids']
 for fid in pids:
  use=(D['hp']['times']>=.05)&(D['hp']['times']<source_t)
  history=np.vstack([D['reference'][fid],D['display_pos'][use,fid],points[fid]])
  trail=D['hproj'][2](history)
  if len(trail)>1:c.line(trail,'#f8fcfa',4.0)
  for j in range(len(trail)-1):
   a=.45+.55*(j+1)/max(1,len(trail)-1)
   col=tuple(np.rint(np.array([197,226,216])*(1-a)+np.array([17,88,100])*a).astype(int))
   c.line(trail[j:j+2],col,2.15)
  c.dot(trail[-1],3.1,'#155e67','white')
 c.text(940,187,f't = {source_t:.2f} s',18,MUTED,True,anchor='rt')
 # One enlarged stereo view, retaining both-camera triangulation data.
 sf=int(np.clip(round(stereo_t*100),0,200));tr=D['tr'];colors=['#ffb321','#00d8ed','#ff65d4']
 crop=D['texture_crop'];c.paste(Image.fromarray(D['camera'][sf]).crop(tuple(crop)).convert('RGB'),(365,473,184,154))
 scale=np.array([184/(crop[2]-crop[0]),154/(crop[3]-crop[1])]);trails=[]
 for fid in D['ids']:trails.append((tr['pixels'][0,10:sf+1,fid]-crop[:2])*scale+[365,473])
 proj=surface_view(D,t);selected=proj(tr['world'][sf,D['ids']])
 draw_surface_patch(c,D,proj,sf)
 for trail,dest,col in zip(trails,selected,colors):
  c.line([trail[-1],dest],col,2.5)
 # Large, high-contrast patch markers and visible histories in the texture crop.
 for trail,col in zip(trails,colors):
  if len(trail)>1:c.line(trail,'#17323d',4.2);c.line(trail,col,2.1)
  c.dot(trail[0],3.0,'#17323d','white');q=trail[-1]
  for sx in [-1,1]:
   for sy in [-1,1]:
    corner=q+[sx*8,sy*8];pts=[corner-[sx*5,0],corner,corner-[0,sy*5]]
    c.line(pts,'#17323d',4.2);c.line(pts,col,2.3)
  c.dot(q,2.1,col,'#17323d')
 for fid,q,col in zip(D['ids'],selected,colors):
  history=proj(tr['world'][10:sf+1,fid])
  if len(history)>1:c.line(history,'#f4f6f7',3.8);c.line(history,col,2)
  c.dot(q,4.6,col,INK);c.dot(q,1.3,'white')
 # Unchanged inferred field, shown through persistent interior trajectories.
 draw_stereo_interior(c,D,sf)
 # Color reports total displacement since the displayed initial frame, in
 # physical millimeters, using one fixed scale over the entire interval.
 proj=D['hproj'][3];xy=proj(points);dep=points@proj.view;brightness=.80+.20*(dep-dep.min())/np.ptp(dep)
 displacement=np.linalg.norm(source_points-D['reference'],axis=1)
 colors=DISPLACEMENT(np.clip(displacement/D['disp_max_m'],0,1))[:,:3]*255
 for j in np.argsort(dep):c.dot(xy[j],.8,tuple((colors[j]*brightness[j]).astype(int)))
 draw_observations(c,D,source_t,sf,round(t%LOOP_SECONDS,9),reveal_progress(t,4) if reveal else 1.)
 # Progressive reveal and cursor are tied to the same hardware source time.
 f=D['force'];use=(f['time']>=0)&(f['time']<=source_t);ft=f['time'][use];fv=f['incremental_normal_force'][use]
 path=np.c_[1039+ft/2*178,637-fv/12*126]
 if len(path)>1:c.line(path,ORANGE,1.6);c.dot(path[-1],2.4,ORANGE,'white')
 xx=1039+source_t/2*178;c.line([(xx,511),(xx,637)],'#e5c5ae',.7)
 if reveal:
  # Fade entire stages as groups; observation stays fully visible throughout.
  for box,start,minimum in [((350,137,951,693),4,FAINT_OPACITY),
                             ((1006,137,1245,693),8,FAINT_OPACITY),
                             ((952,134,1006,694),8,0.)]:
   opacity=minimum+(1-minimum)*reveal_progress(t,start)
   bounds=tuple(v*S for v in box);tile=im.crop(bounds)
   im.paste(Image.blend(Image.new('RGB',tile.size,BG),tile,opacity),bounds[:2])
 return im.resize((1280,720),Image.Resampling.LANCZOS)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--render',action='store_true');args=parser.parse_args()
 if args.preview:
  for t in [0,3.5,7,10.5,13.5]:draw_method01(t).save(P/f'animated_method01_{t:g}s.png')
  draw_method01(10.5).save(P/'method01_animated_poster.png')
 if args.render:
  out=P/'method01_animated.mp4';proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','17','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
  for i in range(DURATION*FPS):
   proc.stdin.write(draw_method01(i/FPS).tobytes())
   if i%75==0:print(f'Rendered {i}/{DURATION*FPS} frames',flush=True)
  proc.stdin.close();assert proc.wait()==0;print(out,flush=True)
 P.joinpath('animation_provenance.json').write_text(json.dumps({'scope':'Display animation only. No physics simulation or material fit. Existing stereo kinematics are re-evaluated without changing parameters and checked against frozen display frame.','duration_s':DURATION,'fps':FPS,'hardware_source_time_s':[.05,2.0],'stereo_source_time_s':[.10,1.90],'stage_reveal':'Middle stage and camera-through arrow fade in at 4 s; input stage and gathering bracket/arrow at 8 s. Smooth 0.6 s fades from 7 percent stage opacity; connectors start invisible. Both final loops have all stages visible. Flow prior label removed; Same volume and the divergence-free equation remain. Camera rays use traveling chevrons toward the lens, presentation-only.', 'playback':'Four-second forward loops at 0.5x source speed; all panels, trails, force curve and cursors reset together. Hardware starts at source 0.05 s, stereo at 0.10 s. Hardware holds its 2.0 s endpoint for the final 0.1 s of each loop; stereo holds its 1.90 s endpoint for the final 0.4 s. No reverse footage or additional trials. The 16-second section contains four complete loops. This is not a synchronized material-response comparison; input footage and reconstruction are synchronized within each route.','rgbd_surface':'Framewise display envelope of supplied inferred particles, not directly observed interior motion.','constant_volume':'Middle panel now shares the exact same saved-particle surface, source time and fixed scale as both neighboring diagrams. Same volume labels the supplied incompressible reconstruction prior; no independent affine deformation is used.','stereo':'One enlarged fixed camera crop; both calibrated cameras underlie saved triangulation. Three feature IDs [828,814,361] appear on a local glass-like surface patch. Its fixed material-domain grid interpolates saved valid surface tracks using reference-XZ Delaunay barycentric weights, with no extrapolation. A 2 mm constant offset behind that surface supplies schematic depth only and follows the front; no interior dynamics is inferred here. The schematic-depth label is omitted in the presentation; its construction remains documented here. No dense cloud, tiny triangle texture, or full specimen block is shown. Camera and scale remain fixed. Correspondence links use full feature colors at 2.5 px, matching camera-ray thickness. The separate square-symmetric interior field is unchanged. Its fixed magnified cutaway shows four persistent strictly interior nodes with dark blue history trails, hollow initial positions and filled current positions. The tiny locator is omitted for readability; the local subvolume and fixed magnification are documented in provenance. No colored observed features appear in this panel. Position and geometry share the same magnification; displacement is not amplified. The field formula sits below the diagram.','force':'Plot height 126 px, translated downward with X-axis at y=637 aligned to the top of the lower method captions. Orange curve (#c96932, matching the video force palette). Recorded incremental normal robot force from ep0001 robot_force.npz. Baseline-subtracted, not a sketch or command; no independent calibration/tool inertia correction. Cursor follows the hardware source time.','assumptions':'Hardware handoff: axisymmetry, conserved volume, minimum-dissipation incompressible flow with no-slip contacts. Stereo: approximate square-symmetric field fitted to surface displacement.','left_panel':'Approved refined compact camera A installed in both positions. The lower camera retains its approved 25-degree tilt. The upper camera mirrors it vertically: both lenses stay left, with opposite body diagonals. The same affine transform places both the sprite and lens endpoint. Viewing rays are 2.5 px wide with 8 px arrowheads, light grass green (#63c86b) above and sky blue (#55b9ec) below. Camera generator, sprite, geometry metadata and stereo source images are saved on persistent storage. Hardware fixed raw crop [330,125,590,286]; simulation fixed crop [33,180,1247,1052], with taller 270x194 display centered on y=550. Both show the upper plate and entire specimen vertically from the start. Schematic viewing rays move with visible surface bounds and stereo features; no RGB-D particle identity is claimed.',
 'color':'Displacement magnitude from displayed initial positions at source t=0.05 s, fixed scale in mm over the interval.',
 'display_frame':'RGB-D diagrams use a laterally translating display origin to center the particle centroid in XY. World Z, shape, scale and source timing are unchanged. Source positions, force and the original robot-frame displacement colors remain unchanged. Trails use the same centered display frame. This display convention is documented here; the small slide footer was removed at user request. Initial source motion audit is in initial_motion_audit.json.',
 'advection':'Five persistent particle IDs with higher-contrast fading history trails and a light underlay, from the displayed initial time; no future-position arrows.',
 'alignment':'Top diagrams and output motion centered at y=284.5. Lower method row and taller input image centered at y=550; force X-axis aligned to lower captions at y=637. Three top method diagrams have identical projection scales. All three stages use equal-height rounded outlines and a common background; Camera centers, camera-through arrow, stage dividers and bracket arrow align at y=414, balancing visible image gaps in the initial frame (lower plate begins at y=461, after 8 px of empty image background). Bracket spans y=137..691 symmetrically, matching the block outline top and bottom. Stereo display boxes grow vertically while retaining fixed aspect-correct projections and center y=550; force X-axis stays y=637. Connector arrow tips meet the next stage borders at x=350 and x=1007.'},indent=2)+'\n')
if __name__=='__main__':main()
