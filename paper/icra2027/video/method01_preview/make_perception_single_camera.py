"""Perception-focused static method proposal. Frozen science assets only."""
from pathlib import Path
import json, math, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps

P=Path(__file__).resolve().parent;ROOT=P.parents[3];S=2
BG='#f4f6f7';INK='#21333e';TEAL='#167b76';MUTED='#566975';LINE='#d9e1e5';BLUE='#2375aa'
im=Image.new('RGB',(1280*S,720*S),BG);d=ImageDraw.Draw(im);boxes=[]
def txt(x,y,s,size=24,color=INK,bold=False,anchor=None):
 f=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-'+('Bold' if bold else 'Regular')+'.ttf',size*S)
 bb=d.textbbox((x*S,y*S),s,font=f,anchor=anchor)
 assert bb[0]>=0 and bb[1]>=0 and bb[2]<=im.width and bb[3]<=im.height,(s,bb)
 d.text((x*S,y*S),s,font=f,fill=color,anchor=anchor);boxes.append((s,bb))
def path(points,fill=TEAL,width=2):d.line([(round(x*S),round(y*S)) for x,y in points],fill=fill,width=width*S,joint='curve')
def dot(x,y,r,fill,outline=None):d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=fill,outline=outline,width=S)
def poly(points,fill,outline=None):d.polygon([(x*S,y*S) for x,y in points],fill=fill,outline=outline)
def rect(box,fill,r=10,outline=None,width=1):d.rounded_rectangle(tuple(v*S for v in box),radius=r*S,fill=fill,outline=outline,width=width*S)
def arrow(a,b,color=TEAL,width=2):
 path([a,b],color,width);angle=math.atan2(b[1]-a[1],b[0]-a[0]);l=10
 poly([b,(b[0]-l*math.cos(angle-.48),b[1]-l*math.sin(angle-.48)),(b[0]-l*math.cos(angle+.48),b[1]-l*math.sin(angle+.48))],color)
def bezier(a,b,c,e,fill=TEAL,width=2):
 t=np.linspace(0,1,55)[:,None];p=(1-t)**3*np.array(a)+3*(1-t)**2*t*np.array(b)+3*(1-t)*t*t*np.array(c)+t**3*np.array(e);path(p,fill,width)
def photo(src,xy,size):
 pic=src.convert('RGB').resize(tuple(round(v*S) for v in size),Image.Resampling.LANCZOS)
 mask=Image.new('L',pic.size);ImageDraw.Draw(mask).rounded_rectangle((0,0,*pic.size),radius=7*S,fill=255)
 im.paste(pic,(round(xy[0]*S),round(xy[1]*S)),mask)
def pointview(points,reference,box,r=1.4,solid=False):
 valid=np.isfinite(points).all(-1);points=points[valid];reference=reference[valid]
 az,el=np.deg2rad([-48,21]);right=np.array([-np.sin(az),np.cos(az),0]);up=np.array([-np.cos(az)*np.sin(el),-np.sin(az)*np.sin(el),np.cos(el)]);view=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
 q=points-points.mean(0);xy=np.c_[q@right,-q@up];dep=q@view
 x,y,w,h=box;scale=min((w-12)/np.ptp(xy[:,0]),(h-12)/np.ptp(xy[:,1]));xy=(xy-(xy.min(0)+xy.max(0))/2)*scale+[x+w/2,y+h/2]
 z=(reference[:,2]-reference[:,2].min())/max(np.ptp(reference[:,2]),1e-9)
 colors=(colormaps['turbo'](z)[:,:3]*255).astype(int)
 depth=(dep-dep.min())/max(np.ptp(dep),1e-9)
 for k in np.argsort(dep):
  color=TEAL if solid else tuple((colors[k]*(.68+.32*depth[k])).astype(int))
  dot(*xy[k],r,color)

hwpath=ROOT/'out/method_hardware_overlay_20260915/hardware_inputs.npz'
trackpath=ROOT/'out/method_hardware_overlay_20260915/tracking_inputs.npz'
depthpath=ROOT/'out/press_observation_assessment_20260913/ep0001/hand_observations.npz'
ppath=ROOT/'press_real_data/handoff_ep0-7/ep0001/particles.npz'
hw=np.load(hwpath);st=np.load(trackpath);depth=np.load(depthpath);hp=np.load(ppath)
k=int(np.argmin(abs(hp['times']-1.52)));dk=int(np.argmin(abs(depth['time']-hp['times'][k])))

# Header, consistent with the accepted opening.
txt(44,22,'METHOD 01 / OBSERVE',20,TEAL,True)
txt(44,56,'Turn visual observations into 3D motion',41,INK,True)
path([(44,114),(1236,114)],LINE,1)

# Central workspace dominates the slide; the video and output are subordinate.
rect((342,174,970,616),'#e4e9ec',20)
rect((342,170,970,612),'#ffffff',20,LINE)
txt(369,190,'Recover the surface',31,INK,True)
# The lines inside the workspace encode the two alternative observation routes.
path([(369,238),(941,238)],LINE,1)
txt(370,253,'Stereo texture',26,BLUE,True)
txt(940,258,'simulation example',18,MUTED,False,'rt')
txt(370,448,'RGB-D',26,TEAL,True)
txt(940,453,'hardware example',18,MUTED,False,'rt')
path([(369,429),(941,429)],LINE,1)

# Actual earlier/later stereo crops and measured correspondences.
left,right=(388,294),(743,294);w,h=170,124
for origin,index in [(left,0),(right,1)]:photo(Image.fromarray(st['images'][index]),origin,(w,h))
colors=['#43c9e4','#f6c84f','#ca89dd']
for j,color in enumerate(colors):
 a=np.array(left)+st['pixels'][0,j]*[w/670,h/490]
 b=np.array(right)+st['pixels'][-1,j]*[w/670,h/490]
 bezier(a,(a[0]+90,a[1]-35),(b[0]-90,b[1]-35),b,color,2)
 dot(*a,4.3,color,INK);dot(*b,4.3,color,INK)
# Time-step hints are intentionally small; the correspondence graphic is primary.
txt(473,399,'earlier',17,'#ffffff',True,'mt')
txt(828,399,'later',17,'#ffffff',True,'mt')

# Measured visible boundary + actual RGB-D surface sample, rather than fabricated
# correspondences. The two sources are complementary observations of one episode.
# A close crop puts the boundary itself at center stage.
src=Image.fromarray(hw['observed']);photo(src,(388,488),(172,106))
for curve in hw['contour_pixels']:
 pts=curve*[172/235,106/145]+[388,488]
 path(pts,TEAL,2)
 for q in pts[::10]:dot(*q,2.7,'#5de0c8')
arrow((583,541),(673,541),TEAL)
# Dots moving along this link will later make deprojection legible in animation.
for x in [602,624,646]:dot(x,541,2.5,TEAL)
points=depth['surface_points_base'][dk]
pointview(points,points,(734,486,188,109),r=2.0,solid=True)

# A prominent perspective camera receives a viewing frustum from the specimen.
# This is a schematic visual explanation, not calibrated camera geometry.
photo(src,(44,402),(238,147))
# Four rays terminate at two visible-boundary points and two visible-face samples.
# No material-point identity is implied: these are surface observations in one frame.
surface_curves=hw['contour_pixels']*[238/235,147/145]+[44,402]
lens=np.array([197.5,290.5])
# Boundary endpoints define a diagonal across the visible face. Intersect this
# line with four equally spaced viewing directions; the middle two markers are
# schematic surface samples, not claimed measured temporal correspondences.
a,b=surface_curves[0,42],surface_curves[1,8]
angles=np.linspace(np.arctan2(*(a-lens)[::-1]),np.arctan2(*(b-lens)[::-1]),4)
segment=b-a
selected_surface=[]
for theta in angles:
    ray=np.array([np.cos(theta),np.sin(theta)])
    distance,along=np.linalg.solve(np.column_stack((ray,-segment)),a-lens)
    assert distance>0 and -1e-8<=along<=1+1e-8
    selected_surface.append(lens+distance*ray)
selected_surface=np.array(selected_surface)
np.testing.assert_allclose(selected_surface[[0,-1]],np.array([a,b]),atol=1e-10)
# All four positions must remain on the observed red specimen, not the scenery.
for q in selected_surface:
    pixel=np.rint((q-[44,402])/[238/235,147/145]).astype(int)
    red,green,blue=hw['observed'][pixel[1],pixel[0]].astype(float)
    assert red>1.3*green and red>1.3*blue,(pixel,[red,green,blue])
over=Image.new('RGBA',im.size);od=ImageDraw.Draw(over)
cone=[lens,selected_surface[0],selected_surface[-1]]
od.polygon([(x*S,y*S) for x,y in cone],fill=(41,162,152,16))
for q in selected_surface:
    od.line([(lens[0]*S,lens[1]*S),(q[0]*S,q[1]*S)],fill=(20,137,127,180),width=2*S)
im.paste(over,(0,0),over);d=ImageDraw.Draw(im)
# Arrowheads point from the surface toward the camera; their endpoints stay on
# the same straight viewing rays, rather than adding decorative signal lines.
for q in selected_surface:
    a=q+(lens-q)*.43
    b=q+(lens-q)*.52
    arrow(tuple(a),tuple(b),TEAL,width=2)
# Camera body, side, top, and concentric glass lens.
poly([(186,246),(217,224),(304,241),(276,265)],'#789097')
poly([(276,265),(304,241),(304,306),(276,332)],'#324a56')
poly([(186,246),(276,265),(276,332),(186,311)],'#4a6470')
poly([(198,254),(257,267),(257,309),(198,296)],'#21333e')
rect((258,271,268,293),'#8fa5ad',2)
dot(290,265,3.5,'#62d0b5')
for box,color in [((165,258,230,323),'#21333e'),((171,264,224,317),'#839fa7'),((177,270,218,311),'#132d38'),((183,276,212,305),'#24566a')]:
 d.ellipse(tuple(x*S for x in box),fill=color)
dot(194,284,5,'#6acdc4');dot(201,295,3,'#1f4658')
# Trace only the two observed side boundaries; avoid the plate and background.
for curve in surface_curves:
    path(curve,'#123a38',4)
    path(curve,'#77ead2',2)
# Clear endpoint markers make it possible to follow each ray to the surface.
for q in selected_surface:
    dot(*q,7.0,'#123a38')
    dot(*q,5.5,'#ffffff')
    dot(*q,3.5,TEAL)
# An optical route branches into the two illustrative recovery options.
path([(304,280),(324,280),(324,354),(342,354)],TEAL,2)
path([(324,354),(324,541),(342,541)],TEAL,2)
for y in [354,541]:arrow((328,y),(341,y),TEAL)
dot(324,354,3,TEAL)
txt(163,195,'Observe deformation',24,INK,True,'mt')
txt(163,560,'Visible surface',22,TEAL,True,'mt')

# Both routes share a motion representation. A compact RGB-D example illustrates
# the output; no claim that this sphere was reconstructed from the stereo block.
path([(970,354),(987,354),(987,447),(1009,447)],TEAL,2)
path([(970,541),(987,541),(987,447)],TEAL,2)
arrow((991,447),(1011,447),TEAL)
dot(987,447,3,TEAL)
txt(1125,270,'3D motion',29,INK,True,'mt')
pointview(hp['pos'][k],hp['seed'],(1004,326,234,186),r=1.0)
txt(1125,531,'Particle trajectories',21,MUTED,False,'mt')
txt(1125,558,'RGB-D example',18,MUTED,False,'mt')

# Minimal qualification and the explicit handoff to the next method section.
txt(641,628,'Interior motion requires geometric and kinematic assumptions.',21,MUTED,False,'mt')
rect((44,665,1236,710),'#e6efed',10)
txt(67,677,'NEXT',17,TEAL,True)
txt(175,672,'3D motion  +  contact force',26,INK,True)
arrow((532,687),(591,687),TEAL)
txt(622,672,'Weak-form identification',26,INK,True)
for i,(s,a) in enumerate(boxes):
 for t,b in boxes[i+1:]:
  assert not(max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])),(s,t)
im.save(P/'method01_perception.png');im.resize((1280,720),Image.Resampling.LANCZOS).save(P/'method01_perception_720p.png')
(P/'perception_provenance.json').write_text(json.dumps({
 'scope':'Static alternative design only; no changes to the main video or manuscript.',
 'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [hwpath,trackpath,depthpath,ppath]],
 'stereo':'Saved simulation example at 1.00 and 1.72 s. Connecting curves are correspondence links, not physical trajectories; dots are measured saved feature positions.',
 'rgbd':{'depth_frame':dk,'depth_time_s':float(depth['time'][dk]),'particle_frame':k,'particle_time_s':float(hp['times'][k]),'input_image':'Saved raw hand frame 2097, matched previously to the supplied overlay.','measured_surface':'Finite surface_points_base entries from saved hand-camera depth; no identities inferred between clouds.','particle_output':'Supplied reconstruction with axisymmetry, volume conservation, no-slip contacts and minimum-dissipation interior flow; illustrative example, not stereo-block output.'},
 'camera':{'scope':'Perspective camera and rays are schematic, not a calibration visualization. Two outer rays end on measured visible-side-boundary samples; the two inner rays end on schematic visible-face samples. Four markers are collinear, and ray directions are equally spaced. They do not indicate tracked material identities.','lens_center_display_px':lens.tolist(),'surface_endpoints_display_px':selected_surface.tolist(),'boundary_source':'hardware_inputs.npz:contour_pixels','flow':'Arrowheads indicate observations traveling toward the camera.'},
 'new_simulations':0,'new_fits':0,'new_reconstructions':0},indent=2)+'\n')
