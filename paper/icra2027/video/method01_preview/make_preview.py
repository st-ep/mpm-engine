"""Static method-01 layout using frozen observations and reconstructed particles."""
from pathlib import Path
import json, hashlib, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps

P=Path(__file__).resolve().parent
ROOT=P.parents[3]
S=2
BG='#f4f6f7';INK='#21333e';TEAL='#167b76';MUTED='#566975';BLUE='#2375aa';LINE='#d9e1e5'
im=Image.new('RGB',(1280*S,720*S),BG);d=ImageDraw.Draw(im);text_boxes=[]
def text(x,y,s,size=24,color=INK,bold=False,anchor=None):
    f=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-'+('Bold' if bold else 'Regular')+'.ttf',size*S)
    bb=d.textbbox((x*S,y*S),s,font=f,anchor=anchor)
    assert bb[0]>=0 and bb[1]>=0 and bb[2]<=im.width and bb[3]<=im.height,(s,bb)
    d.text((x*S,y*S),s,font=f,fill=color,anchor=anchor);text_boxes.append((s,bb))
def line(coords,fill=LINE,width=2):d.line([(int(x*S),int(y*S)) for x,y in coords],fill=fill,width=width*S)
def circle(x,y,r,fill,outline=None):d.ellipse(((x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S),fill=fill,outline=outline,width=S)
def arrow(x0,y0,x1,y1,color=TEAL,width=2):
    line([(x0,y0),(x1,y1)],color,width)
    a=math.atan2(y1-y0,x1-x0);l=9
    d.polygon([(x1*S,y1*S),((x1-l*math.cos(a-.5))*S,(y1-l*math.sin(a-.5))*S),((x1-l*math.cos(a+.5))*S,(y1-l*math.sin(a+.5))*S)],fill=color)
def photo(src,box):
    x,y,w,h=box
    src=src.resize((w*S,h*S),Image.Resampling.LANCZOS)
    mask=Image.new('L',src.size);ImageDraw.Draw(mask).rounded_rectangle((0,0,src.width,src.height),radius=8*S,fill=255)
    im.paste(src,(x*S,y*S),mask)
def camera(x,y):
    d.rounded_rectangle((x*S,y*S,(x+28)*S,(y+19)*S),radius=3*S,outline=TEAL,width=2*S)
    circle(x+15,y+9.5,5,BG,TEAL)
    line([(x+4,y),(x+7,y-4),(x+15,y-4),(x+18,y)],TEAL,2)

def cloud(points,reference,box,trails=None):
    # Orthographic 3D view; one scale in all three axes. Positions unchanged.
    az,el=np.deg2rad([-48,21])
    right=np.array([-np.sin(az),np.cos(az),0.])
    up=np.array([-np.cos(az)*np.sin(el),-np.sin(az)*np.sin(el),np.cos(el)])
    view=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
    q=points-points.mean(0)
    xy=np.c_[q@right,-q@up];depth=q@view
    x,y,w,h=box
    scale=min((w-22)/np.ptp(xy[:,0]),(h-20)/np.ptp(xy[:,1]))
    center=(xy.max(0)+xy.min(0))/2
    proj=lambda pts: (np.c_[(pts-points.mean(0))@right,-(pts-points.mean(0))@up]-center)*scale+np.array([x+w/2,y+h/2])
    xy=proj(points)
    # A small, fixed subset of true saved histories indicates particle motion.
    if trails is not None:
        ids=np.arange(0,len(points),max(1,len(points)//28))
        for k in ids:
            path=proj(trails[:,k])
            line(path.tolist(),'#9faeb9',1)
    z=(reference[:,2]-reference[:,2].min())/np.ptp(reference[:,2])
    colors=(colormaps['turbo'](z)[:,:3]*255).astype(int)
    nd=(depth-depth.min())/max(np.ptp(depth),1e-10)
    for k in np.argsort(depth):
        color=tuple((colors[k]*(.62+.38*nd[k])).astype(int))
        circle(*xy[k],1.25 if len(points)>2000 else 1.9,color)

text(44,22,'METHOD 01 / OBSERVE',20,TEAL,True)
text(44,56,'From video to 3D motion',43,INK,True)
line([(44,114),(1236,114)],LINE,1)
# Three columns with two observation routes.
for x,num,label in [(44,'1','Record the interaction'),(445,'2','Recover the surface'),(866,'3','Infer particle motion')]:
    text(x,134,num,24,TEAL,True)
    text(x+30,134,label,26,INK,True)

stereo_path=ROOT/'out/press_separated_20260913/monotonic320/fit_A/tracks.npz'
recon_path=ROOT/'out/method_shaping_separated_20260913/reconstructed_display.npz'
hw_path=ROOT/'out/method_hardware_overlay_20260915/hardware_inputs.npz'
particles_path=ROOT/'press_real_data/handoff_ep0-7/ep0001/particles.npz'
tracks=np.load(stereo_path);recon=np.load(recon_path);hw=np.load(hw_path);hp=np.load(particles_path)

text(44,176,'Textured stereo · simulation',22,BLUE,True)
text(44,402,'RGB-D · hardware',22,TEAL,True)
raw=Image.open(P/'stereo_source.png').convert('RGB')
crop=(170,400,1160,964)
photo(raw.crop(crop),(44,208,325,185))
photo(Image.fromarray(hw['observed']).crop((0,5,235,139)),(44,434,325,185))
# Distinct cameras emphasize that each route uses its own observations.
for y in [281,512]:
    camera(385,y)
    arrow(419,y+9,445,y+9)
    arrow(795,y+9,838,y+9)

text(621,202,'Track surface features',25,BLUE,True,'mt')
ids=np.array([1,94,31,562,295,820,460,704,211,536,306,669])
pix=tracks['pixels'][0,20:173][:,ids]
# Display measured trajectories with a uniform scale and no path amplification.
lo=pix.min(axis=(0,1));hi=pix.max(axis=(0,1));scale=min(270/(hi[0]-lo[0]),116/(hi[1]-lo[1]))
projected=(pix-(lo+hi)/2)*scale+np.array([621,299])
for j in range(len(ids)):
    color=['#2375aa','#167b76','#ce813d'][j//4]
    path=projected[:,j]
    line(path.tolist(),color,2)
    circle(*path[0],3.0,BG,color)
    circle(*path[-1],3.4,color)
text(621,366,'Correspondences across frames',21,MUTED,False,'mt')
text(621,431,'Recover surface shape',25,TEAL,True,'mt')
contours=hw['contour_pixels']
lo=contours.min(axis=(0,1));hi=contours.max(axis=(0,1));scale=min(200/(hi[0]-lo[0]),107/(hi[1]-lo[1]))
for curve in contours:
    pts=(curve-(lo+hi)/2)*scale+np.array([621,527])
    line(pts.tolist(),TEAL,2)
    for q in pts[::6]:circle(*q,2.4,TEAL)
text(621,596,'Depth + silhouette',22,MUTED,False,'mt')

cloud(recon['x'],recon['reference'],(851,199,385,173))
k=int(np.argmin(abs(hp['times']-1.52)))
cloud(hp['pos'][k],hp['seed'],(851,425,385,173),hp['pos'][max(0,k-15):k+1:3])
text(1043,376,'Square-symmetric reconstruction',21,MUTED,False,'mt')
text(1043,602,'Axisymmetric reconstruction',21,MUTED,False,'mt')
text(857,174,'Geometry + kinematic assumptions',20,MUTED)
# Compact handoff: force is introduced here, not as another explanatory chart.
d.rounded_rectangle((44*S,657*S,1236*S,709*S),radius=10*S,fill='#e6efed')
text(63,671,'OUTPUT',17,TEAL,True)
text(163,667,'3D motion',26,INK,True)
text(300,667,'+',26,TEAL,True)
text(337,667,'Contact force',26,INK,True)
arrow(522,684,569,684)
text(596,667,'Identify the material law',26,INK,True)
text(1209,672,'NEXT',17,TEAL,True,'rt')
for i,(s,a) in enumerate(text_boxes):
    for t,b in text_boxes[i+1:]:
        assert not(max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])),(s,t)
im.save(P/'method01_proposal.png');im.resize((1280,720),Image.Resampling.LANCZOS).save(P/'method01_proposal_720p.png')
paths=[stereo_path,recon_path,hw_path,particles_path]
(P/'provenance.json').write_text(json.dumps({
 'scope':'Static method-slide layout only; main video and manuscript unchanged.',
 'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
 'stereo':{'frame':172,'time_s':float(tracks['time'][172]),'crop_xyxy':crop,'feature_ids':ids.tolist(),'reconstruction':'Existing square-symmetric spatial reconstruction; saved 1331-point display snapshot.'},
 'hardware':{'episode':'ep0001','particle_frame':k,'time_s':float(hp['times'][k]),'image':'Previously matched raw hand frame 2097 in hardware_inputs.npz','additional_display_crop_xyxy':[0,5,235,139],'surface':'Two visible RGB side boundaries, not a new depth measurement.','particles':'Philip supplied reconstruction; axisymmetry, volume conservation, no-slip contacts, minimum-dissipation interior flow. Illustrates observation route, not a new identification result.'},
 'rendering':'Orthographic projection with equal spatial scale in all axes; colors encode initial height; depth shading is for display. Stereo trails are measured pixel tracks; hardware trails are supplied inferred paths.',
 'new_simulations':0,'new_parameter_fits':0,'new_motion_reconstructions':0},indent=2)+'\n')
