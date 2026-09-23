"""Private ICRA supplement. Deterministic editing of frozen footage, no physics runs.

Run with the repository .venv Python. --preview writes section review frames;
--render [section ...] renders sections; --assemble makes master/submission MP4s.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, subprocess, sys, shutil
from functools import lru_cache
from opening_preview.render_opening import draw_opening, DURATION as OPENING_DURATION, approved_frame
from method01_preview.render_method01 import draw_method01, DURATION as OBSERVE_DURATION
sys.path.insert(0,str(Path(__file__).resolve().parent/'method02_preview'))
from render_method02_animated import draw_frame as draw_method02, DURATION as IDENTIFY_PLAN_DURATION
from narrative_captions import ALL_SECTIONS,CUES,caption_at,paint_caption,export_captions
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS, SECTIONS, REVIEW = [HERE / p for p in ('assets','sections','review')]
for p in (ASSETS, SECTIONS, REVIEW): p.mkdir(exist_ok=True)
W, H, FPS = 1280, 720, 25
BG, WHITE, INK = '#f4f6f7', '#ffffff', '#21333e'
MUTED, LINE, BLUE, ORANGE, TEAL = '#566975', '#d9e1e5', '#2375aa', '#c96932', '#167b76'
FONT = '/usr/share/fonts/truetype/lato/'
SOURCES = {k: ROOT/'paper/videos'/v for k,v in {
 'insert':'01_insertion_matched_swapped.mp4', 'golf':'02_golf_matched_swapped.mp4',
 'pour':'03_pouring_glycerol_real_vs_simulation.mp4',
 'real':'04_pressing_real_red_yellow_gray.mp4', 'sim':'05_pressing_simulation_red_yellow_gray.mp4',
 'shape':'06_shaping_simulation_matched_swapped.mp4'}.items()}
SOURCES['bend']=ROOT/'out/strip_texture_20260912/media/texture_bending.mp4'
TIMELINE = [
 ('opening',OPENING_DURATION,'Identify material laws. Plan robot actions.'),
 ('observe',OBSERVE_DURATION,'Observe one interaction'),
 ('balance',IDENTIFY_PLAN_DURATION,'Identify the material law and plan robot actions'),
 ('insertion',15,'Elastic rod insertion'),
 ('golf',15,'Putting with a flexible club'),
 ('simshape',23,'Plan plastic shaping'),
 ('hardware',30,'Hardware pressing and shaping'),
 ('pouring',36,'Identify and plan target-volume pouring'),
 ('takeaway',10,'From an interaction to a reusable material model'),
]
TOTAL = sum(d for _,d,_ in TIMELINE)
assert [(name,duration) for name,duration,_ in TIMELINE]==ALL_SECTIONS
TEXT_LOG = []

@lru_cache(None)
def font(size, bold=False):
    return ImageFont.truetype(FONT+('Lato-Bold.ttf' if bold else 'Lato-Regular.ttf'),size)

def text(im, xy, s, size=26, color=INK, bold=False, width=None, anchor=None):
    d=ImageDraw.Draw(im); f=font(size,bold)
    if width is not None:
        assert d.textlength(s,font=f)<=width, (s, size, width, d.textlength(s,font=f))
    bounds=d.textbbox(xy,s,font=f,anchor=anchor)
    assert bounds[0]>=0 and bounds[1]>=0 and bounds[2]<=W and bounds[3]<=H,(s,bounds)
    d.text(xy,s,font=f,fill=color,anchor=anchor)
    TEXT_LOG.append({'text':s,'size':size,'bbox':bounds})

def lines(im,xy,ss,size=26,color=INK,bold=False,step=None,width=None):
    for i,s in enumerate(ss): text(im,(xy[0],xy[1]+i*(step or size+9)),s,size,color,bold,width)

def box(im,bounds,fill=WHITE,radius=16,outline=None):
    ImageDraw.Draw(im).rounded_rectangle(bounds,radius,fill=fill,outline=outline,width=2)

def arrow(im,a,b,color=TEAL,width=4):
    d=ImageDraw.Draw(im); d.line([a,b],fill=color,width=width)
    angle=math.atan2(b[1]-a[1],b[0]-a[0]); l=13
    d.polygon([b,(b[0]-l*math.cos(angle-.5),b[1]-l*math.sin(angle-.5)),
                 (b[0]-l*math.cos(angle+.5),b[1]-l*math.sin(angle+.5))],fill=color)

@lru_cache(None)
def still(name):
    source=Image.open(ASSETS/(name+'.png')).convert('RGBA')
    out=Image.new('RGBA',source.size,WHITE);out.alpha_composite(source)
    return out.convert('RGB')

def fit(im,source,bounds):
    x,y,w,h=bounds; scale=min(w/source.width,h/source.height)
    new=source.resize((round(source.width*scale),round(source.height*scale)),Image.Resampling.LANCZOS)
    im.paste(new,(round(x+(w-new.width)/2),round(y+(h-new.height)/2)))

class Clip:
    def __init__(self,path):
        self.cap=cv2.VideoCapture(str(path)); assert self.cap.isOpened(),path
        self.fps=self.cap.get(cv2.CAP_PROP_FPS); self.n=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.index=-1; self.last=None
    def get(self,t):
        idx=min(self.n-1,max(0,int(t*self.fps+1e-6)))
        if idx==self.index: return self.last
        if idx!=self.index+1: self.cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ok,f=self.cap.read(); assert ok,(idx,self.n)
        self.index=idx; self.last=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));return self.last

CLIPS={}
@lru_cache(None)
def insertion_probe_data():
    with np.load(ASSETS/'insertion_probe.npz') as data:
        return {key:data[key] for key in data.files}

def insertion_probe(t):
    """A then B: each actual loading episode, followed by a neutral B hold."""
    data=insertion_probe_data()
    material='A' if t<2 else 'B'
    local=t if t<2 else t-2
    index=round(min(1,max(0,local/1.6))*40)
    rgb=data[material][index].astype(float)
    mask=data[material+'_mask'][index].astype(float)/255
    tint=max(0,min(1,(4-t)/.2))
    color=np.array([35,117,170] if material=='A' else [201,105,50])
    shade=rgb.mean(axis=2,keepdims=True)/180
    colored=np.clip(shade*color,0,255)
    alpha=(.72*tint*mask)[...,None]
    result=np.clip(rgb*(1-alpha)+colored*alpha,0,255).astype('uint8')
    return Image.fromarray(result),material

@lru_cache(None)
def insertion_math_token(source,size,color):
    """Common-size math raster with its baseline retained across color spans."""
    from matplotlib.mathtext import MathTextParser
    from matplotlib.font_manager import FontProperties
    from PIL import ImageColor
    if color in (INK,'#000000'):
        color='#000000'
    raster=MathTextParser('agg').parse('$'+source+'$',dpi=144,prop=FontProperties(size=size))
    mask=np.asarray(raster.image).copy()
    rgba=np.empty((*mask.shape,4),dtype=np.uint8)
    rgba[:,:,:3]=ImageColor.getrgb(color);rgba[:,:,3]=mask
    tile=Image.fromarray(rgba).resize((round(mask.shape[1]/2),round(mask.shape[0]/2)),Image.Resampling.LANCZOS)
    return tile,raster.depth/2

def insertion_math(im,parts,center,baseline,size=22,max_width=224):
    # For the paper's fixed-corotated law with known nu=.45, sigma=E*T_nu(F),
    # T_nu(F)=(F-R)F^T/[J(1+nu)] + nu*(J-1)*I/[(1+nu)(1-2nu)].
    # This response is dimensionless; displayed E values include kPa units.
    tokens=[insertion_math_token(source,size,color) for source,color in parts]
    width=sum(tile.width for tile,_ in tokens)
    assert width<=max_width,(parts,width,max_width)
    x=round(center-width/2)
    for tile,depth in tokens:
        im.paste(tile,(x,round(baseline-tile.height+depth)),tile)
        x+=tile.width

def frame(key,t):
    if key not in CLIPS: CLIPS[key]=Clip(SOURCES[key])
    return CLIPS[key].get(t)

@lru_cache(None)
def shaping_identification():
    path=ASSETS/'shaping_identification.npz'
    if not path.exists():
        from prepare_shaping_identification import prepare
        prepare()
    with np.load(path) as data:
        frames={k:data[k] for k in data.files}
    laws={m:json.loads((ROOT/f'out/press_separated_20260913/monotonic320/separated_{m}/identification.json').read_text()) for m in 'AB'}
    results=json.loads((ROOT/'out/press_paper_update_20260913/execution_summary.json').read_text())['results']
    errors={(r['material'],r['planned_for']):r['surface_mm'] for r in results}
    return frames,laws,errors


def shaping_law_graph(im,laws,t):
    """Exact scalar reduction of the saved Hencky/J2 law under monotone
    isochoric coaxial loading: ||dev tau|| = min(2 mu ||dev log F||, Y).
    The engine defines Y as the deviatoric Kirchhoff-stress norm, not the
    sqrt(3/2)-scaled equivalent stress. This is a constitutive response,
    not a force/strain measurement from the nonuniform pressing experiment.
    """
    box(im,(44,411,540,648),fill=WHITE,radius=12,outline=LINE)
    text(im,(292,422),'Identified material laws',24,TEAL,True,anchor='mt')
    text(im,(66,458),'Deviatoric stress (kPa)',19,INK)
    text(im,(324,458),'E: stiffness · Y: yield',19,INK)
    d=ImageDraw.Draw(im)
    x0,x1,y0,y1=91,302,593,491
    def xy(strain,stress):return (x0+(x1-x0)*strain/.22,y0-(y0-y1)*stress/12)
    for stress in (0,6,12):
        y=xy(0,stress)[1]
        d.line((x0,y,x1,y),fill=LINE,width=1)
        text(im,(82,y),str(stress),18,MUTED,anchor='rm')
    d.line((x0,y1,x0,y0,x1,y0),fill=MUTED,width=2)
    for strain in (0,.1,.2):
        x=xy(strain,0)[0]
        text(im,(x,599),f'{strain:g}',18,MUTED,anchor='mt')
    text(im,(195,623),'Deviatoric log strain',18,INK,anchor='mt')
    for m,y,start,color in [('A',490,1.86,BLUE),('B',568,3.86,ORANGE)]:
        layer=im.copy();ld=ImageDraw.Draw(layer)
        E=laws[m]['E_pa']/1000;Y=laws[m]['yield_pa']/1000
        slope=E/(1+.3)
        # Include the exact elastic/plastic corner, avoiding sampled rounding.
        ld.line([xy(0,0),xy(Y/slope,Y),xy(.22,Y)],fill=color,width=4)
        text(layer,(329,y),m,21,color,True)
        text(layer,(361,y),f'E = {E:.2f} kPa',20,color)
        text(layer,(361,y+25),f'Y = {Y:.2f} kPa',20,color)
        im=Image.blend(im,layer,max(0,min(1,(t-start)/.14)))
    return im


def base(section,title,subtitle,elapsed):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    if section.startswith('FORM ·'):
        text(im,(44,22),'FORM',20,TEAL,True)
        text(im,(1236,22),section.split(' · ',1)[1],20,TEAL,True,anchor='rt')
        text(im,(44,56),title,41,INK,True,width=1190)
    else:
        text(im,(44,26),section.upper(),19,TEAL,True)
        text(im,(44,57),title,38,INK,True,width=1190)
    if subtitle: text(im,(44,105),subtitle,23,MUTED,width=1190)
    d.line((44,145,1236,145),fill=LINE,width=2)
    d.rectangle((0,716,W,719),fill=LINE)
    d.rectangle((0,716,int(W*elapsed/TOTAL),719),fill=TEAL)
    return im

def footer(im,s,second=None):
    # Retain scientific/playback qualifications above the shared subtitle band.
    # The former summary line is now conveyed by the timed spoken captions.
    text(im,(44,643),second or s,15,MUTED,width=1192)

def comparison(im,key,t,bounds=(44,158,832,470)):
    pic=frame(key,t)
    # Redraw margin labels at video-readable size; preserve the full four scene panels.
    for col,label in enumerate(['Matched ID','Swapped ID']):
        text(im,(270+col*405,160),label,24,INK,True,anchor='mt')
    if key in ('shape','insert'):
        rects=[(56+784*c,56+540*r,816+784*c,576+540*r) for r in range(2) for c in range(2)]
    else:
        rects=[(32,36,812,464),(820,36,1610,464),(32,496,812,944),(820,496,1610,944)]
    for i,rect in enumerate(rects):
        row,col=divmod(i,2)
        fit(im,pic.crop(rect),(70+col*405,198+row*222,398,213))
    text(im,(45,287),'A',24,BLUE,True)
    text(im,(45,509),'B',24,ORANGE,True)

def side(im,title,body,accent=BLUE,y=185):
    text(im,(912,y),title,28,accent,True,width=320)
    lines(im,(912,y+49),body,25,INK,step=35,width=320)

def prepare_assets():
    method=Image.open(ROOT/'paper/icra2027/figs/method_overview.png').convert('RGB')
    # Exact crops of the current figure, not a re-created measurement.
    method.crop((20,105,738,492)).save(ASSETS/'observations.png')
    method.crop((77,610,590,795)).save(ASSETS/'force.png')
    md=ROOT/'out/method_optimize_motion_20260915'
    Image.open(md/'test_field.png').save(ASSETS/'field.png')
    Image.open(md/'target.png').save(ASSETS/'target.png')
    Image.open(ROOT/'out/press_shaping_strip_20260915/renders/target.png').save(ASSETS/'sim_target.png')
    method.crop((1580,505,1979,752)).save(ASSETS/'shape_action.png')
    records=json.loads((ROOT/'out/hardware_press_shape_figure_20260915/provenance.json').read_text())['records']
    for r in records:
        pic=Image.open(r['shaping_source']).convert('RGB').transpose(Image.Transpose.ROTATE_180)
        pic=pic.crop(tuple(r['shape_crop_xyxy'])).transpose(Image.Transpose.ROTATE_270)
        pic.save(ASSETS/(r['material']+'.png'))
    # Crops mirror the installed strip's final four panels, including target outlines.
    strip=Image.open(ROOT/'paper/icra2027/figs/identification_plastic_shaping.png')
    strip.save(ASSETS/'sim_shape_strip.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sensor=np.genfromtxt(ROOT/'out/press_separated_20260913/monotonic320/inputs_A/force.csv',names=True,delimiter=',')
    fig,ax=plt.subplots(figsize=(4.4,2.1),dpi=180);fig.patch.set_facecolor(BG);ax.set_facecolor(BG)
    ax.plot(sensor['time'],sensor['Fz'],color=ORANGE,lw=2.5)
    ax.set(xlim=(0,2),xlabel='Time (s)',ylabel='Force (N)',xticks=[0,1,2])
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=12);ax.xaxis.label.set_size(13);ax.yaxis.label.set_size(13)
    fig.tight_layout(pad=.45);fig.savefig(ASSETS/'force.png');plt.close(fig)
    rows=list(csv.DictReader((ROOT/'out/pour_hardware_receiver_remap_review_20260911/summary.csv').open()))
    water=list(csv.DictReader((ROOT/'out/pour_water_figure_20260915/water_receiver_readings.csv').open()))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED})
    fig,ax=plt.subplots(figsize=(8.4,4.9),dpi=150);fig.patch.set_facecolor(BG);ax.set_facecolor(BG)
    xs=[float(r['target_ml']) for r in rows]
    ax.plot([50,170],[50,170],color='#8a959e',ls='--',lw=1.5,label='Target')
    ax.plot(xs,[float(r['mpm_receiver_ml']) for r in rows],'s--',color='#549fc4',mfc='white',lw=2,label='MPM (glycerol)')
    ax.errorbar(xs,[float(r['mean_ml']) for r in rows],yerr=[float(r['sample_sd_ml']) for r in rows],
                color=ORANGE,fmt='o-',capsize=5,lw=2,label='Glycerol: mean ± SD')
    ax.plot(xs,[float(r['receiver_ml']) for r in water],'^-',color='#17365d',mfc='white',lw=2,label='Water (glycerol plan)')
    ax.set(xlim=(50,170),ylim=(45,195),xticks=xs,yticks=[60,100,140,180],xlabel='Target volume (mL)',ylabel='Receiver volume (mL)')
    ax.grid(axis='y',color=LINE);ax.set_axisbelow(True);ax.legend(loc='upper left',fontsize=12,frameon=False)
    fig.tight_layout(pad=.7);fig.savefig(ASSETS/'volumes.png');plt.close(fig)
    # Math is explanatory; measured values are never inferred from this drawing.
    for name,formula in [('ls',r'$A\,\theta\ \simeq\ b$'),('solve',r'$\hat{\theta}=\arg\min_{\theta}\,\|A\theta-b\|_2^2$')]:
        fig=plt.figure(figsize=(6.2,1.1),dpi=180,facecolor=WHITE)
        fig.text(.5,.5,formula,ha='center',va='center',fontsize=34,color=INK)
        fig.savefig(ASSETS/(name+'.png'),facecolor=WHITE);plt.close(fig)

def draw_section_content(name,t):
    start=sum(d for n,d,_ in TIMELINE[:[n for n,_,_ in TIMELINE].index(name)])
    elapsed=start+t
    if name=='opening':
        return draw_opening(t)
    if name=='observe':
        return draw_method01(t)
    if name=='balance':
        return draw_method02(t)
    if name=='insertion':
        im=Image.new('RGB',(W,H),BG)
        d=ImageDraw.Draw(im)
        prefix='Results · Simulation / '
        text(im,(44,22),prefix,34,TEAL,True)
        prefix_width=d.textlength(prefix,font=font(34,True))
        text(im,(44+prefix_width,22),'Elastic rod insertion',34,INK,True)
        text(im,(44,79),'Identify the material law from bending. Use it to insert the rod through the hole by controlling gripper height and tilt.',23,MUTED,width=1192)
        d.line((44,119,1236,119),fill=LINE,width=2)
        d.rectangle((0,716,W,719),fill=LINE)
        d.rectangle((0,716,int(W*elapsed/TOTAL),719),fill=TEAL)
        # Actual A then actual B loading, each physical interval 0.30–0.90 s
        # shown in 1.6 s. Each estimate appears only after its observed bend.
        probe,active_material=insertion_probe(t)
        text(im,(44,137),'Identify material law',25,INK,True)
        text(im,(44,172),'Same bending probe for A and B',22,MUTED)
        fit(im,probe,(44,250,174,265))
        if t<4:
            text(im,(131,515),f'Material {active_material}',22,
                 BLUE if active_material=='A' else ORANGE,True,anchor='mt')
        text(im,(131,550),'Stereo motion',25,INK,True,anchor='mt')
        text(im,(131,581),'+ force',25,INK,True,anchor='mt')
        # Connect the actual observed bend to the fitted parameters rather
        # than scattering inputs and outputs into disconnected text islands.
        arrow(im,(203,428),(252,428),TEAL,3)
        box(im,(260,238,495,604),fill=WHITE,radius=12,outline=LINE)
        text(im,(377,257),'Identified',23,TEAL,True,anchor='mt')
        text(im,(377,287),'material laws',23,TEAL,True,anchor='mt')
        for material,value,y,color,start in [('A','80.6',352,BLUE,1.6),('B','248.1',403,ORANGE,3.6)]:
            values=im.copy()
            insertion_math(values,[(rf'\sigma_{material}=',INK),(value+r'\,\mathrm{kPa}',color),(r'\,T_\nu(F)',INK)],377,y)
            im=Image.blend(im,values,max(0,min(1,(t-start)/.16)))
        ImageDraw.Draw(im).line((278,428,477,428),fill=LINE,width=1)
        text(im,(377,446),'Colored values:',22,'#000000',anchor='mt')
        # Keep E in the same regular math face as the equation symbols, while
        # all three explanatory labels share the same regular 22 px font.
        prefix='identified stiffness '
        prefix_width=ImageDraw.Draw(im).textlength(prefix,font=font(22))
        symbol,depth=insertion_math_token('E',22,INK)
        left=377-(prefix_width+symbol.width)/2
        text(im,(left,495),prefix,22,'#000000',anchor='ls')
        im.paste(symbol,(round(left+prefix_width),round(495-symbol.height+depth)),symbol)
        insertion_math(im,[(r'T_\nu(F)',INK)],377,541,size=22,max_width=100)
        text(im,(377,558),'Deformation response',22,'#000000',anchor='mt')

        # Identification comes first; the two insertion columns appear together.
        layer=im.copy()
        arrow(layer,(495,428),(544,428),TEAL,3)
        pic=frame('insert',max(0,t-4))
        for col,label in enumerate(['Matched ID','Swapped ID']):
            text(layer,(713+354*col,137),label,25,INK,True,anchor='mt')
            text(layer,(713+354*col,169),['Plan with own material model','Plan with other material’s model'][col],21,MUTED,anchor='mt')
        for row,material in enumerate('AB'):
            text(layer,(523,312+229*row),material,25,BLUE if row==0 else ORANGE,True,anchor='mm')
            for col in range(2):
                rect=(56+784*col,56+540*row,816+784*col,576+540*row)
                fit(layer,pic.crop(rect),(544+354*col,202+229*row,338,222))
        opacity=max(0,min(1,(t-4)/.5))
        opacity=.07+.93*opacity*opacity*(3-2*opacity)
        im=Image.blend(im,layer,opacity)
        return im
    if name=='golf':
        im=Image.new('RGB',(W,H),BG)
        d=ImageDraw.Draw(im)
        prefix='Results · Simulation / '
        text(im,(44,22),prefix,34,TEAL,True)
        prefix_width=d.textlength(prefix,font=font(34,True))
        text(im,(44+prefix_width,22),'Putting with a flexible club',34,INK,True)
        text(im,(44,79),'Reuse the previously identified material laws. Plan forward-stroke duration and aim angle to stop the ball in the target.',23,MUTED,width=1192)
        d.line((44,119,1236,119),fill=LINE,width=2)
        d.rectangle((0,716,W,719),fill=LINE)
        d.rectangle((0,716,int(W*elapsed/TOTAL),719),fill=TEAL)
        # Three synchronized five-second loops: 4.5 s playback + 0.5 s final hold. The
        # same fixed vertical crop removes sky/foreground from all four views;
        # clubs, ball paths and targets share one uniform display scale.
        if 'golf' not in CLIPS:
            CLIPS['golf']=Clip(SOURCES['golf'])
        clip=CLIPS['golf']
        source_t=min((t % 5)/4.5,1)*(clip.n-1)/clip.fps
        pic=clip.get(source_t)
        for col,label in enumerate(['Matched ID','Swapped ID']):
            text(im,(368+584*col,137),label,25,INK,True,anchor='mt')
            text(im,(368+584*col,169),['Plan with own material model','Plan with other material’s model'][col],21,MUTED,anchor='mt')
        for row,material in enumerate('AB'):
            text(im,(53,311+230*row),material,25,BLUE if row==0 else ORANGE,True,anchor='mm')
            for col in range(2):
                rect=(34+790*col,113+466*row,818+790*col,425+466*row)
                fit(im,pic.crop(rect),(84+584*col,198+230*row,568,226))
        return im
    if name=='simshape':
        im=Image.new('RGB',(W,H),BG)
        d=ImageDraw.Draw(im)
        prefix='Results · Simulation / '
        text(im,(44,22),prefix,34,TEAL,True)
        text(im,(44+d.textlength(prefix,font=font(34,True)),22),'Plastic shaping',34,INK,True)
        text(im,(44,79),'Identify stiffness and yield stress from pressing. Plan six pinches to shape a larger block into an X.',23,MUTED,width=1192)
        d.line((44,119,1236,119),fill=LINE,width=2)
        d.rectangle((0,716,W,719),fill=LINE)
        d.rectangle((0,716,int(W*elapsed/TOTAL),719),fill=TEAL)
        frames,laws,errors=shaping_identification()
        material='A' if t<2 else 'B'
        source_t=min(2,t if t<2 else t-2)
        index=min(50,round(source_t/.04))
        text(im,(44,137),'Identify material laws',25,INK,True)
        if t<4:text(im,(404,137),material,25,BLUE if material=='A' else ORANGE,True,anchor='rt')
        rgb=frames[material][index].astype(float)
        color=np.array([35,117,170] if material=='A' else [201,105,50])
        tint=np.clip(rgb.mean(2)[...,None]/185*color,0,255)
        amount=1-max(0,min(1,(t-4)/.25))
        alpha=frames[material+'_mask'][index][...,None]/255*amount
        rgb=rgb*(1-alpha)+tint*alpha
        fit(im,Image.fromarray(np.rint(rgb).astype('uint8')),(44,174,496,208))
        arrow(im,(292,384),(292,405),TEAL,3)
        im=shaping_law_graph(im,laws,t)
        layer=im.copy()
        arrow(layer,(545,411),(640,411),TEAL,3)
        for col,label in enumerate(['Matched ID','Swapped ID']):
            text(layer,(799+280*col,137),label,25,INK,True,anchor='mt')
        for row,m in enumerate('AB'):
            for col,model in enumerate([m,'B' if m=='A' else 'A']):
                key=f'shaping_hand_{m}_{model}'
                if key not in SOURCES:SOURCES[key]=ASSETS/'shaping_hand'/f'{m}_plan_{model}.mp4'
                pic=frame(key,max(0,min(15.96,t-4)))
                fit(layer,pic.crop((64,0,576,440)),(678+280*col,178+246*row,242,208))
                if t>=20:
                    prefix='Surface error: '
                    value=f"{errors[m,model]:.3f} mm"
                    ld=ImageDraw.Draw(layer)
                    prefix_width=ld.textlength(prefix,font=font(21))
                    label_width=prefix_width+ld.textlength(value,font=font(21,True))
                    assert label_width<=242
                    left=799+280*col-label_width/2
                    y=407+246*row
                    text(layer,(left,y),prefix,21,'#000000',anchor='ls')
                    text(layer,(left+prefix_width,y),value,21,INK,True,anchor='ls')
            text(layer,(656,280+246*row),m,25,BLUE if m=='A' else ORANGE,True,anchor='mm')
        opacity=max(0,min(1,(t-4)/.5))
        opacity=.07+.93*opacity*opacity*(3-2*opacity)
        im=Image.blend(im,layer,opacity)
        return im
    if name=='hardware':
        import hardware_slide
        return hardware_slide.draw(sys.modules[__name__],t,elapsed)
    if name=='pouring':
        import pouring_slide
        return pouring_slide.draw(sys.modules[__name__],t,elapsed)
    if name=='takeaway':
        im=Image.new('RGB',(W,H),BG)
        text(im,(48,43),'FORM',24,TEAL,True)
        lines(im,(48,101),['An interaction provides a material model.','That model guides a new manipulation task.'],39,INK,True,step=55,width=1190)
        items=[('Observe',BLUE),('Identify',TEAL),('Plan',BLUE),('Execute',TEAL)]
        for j,(label,col) in enumerate(items):
            x=48+j*309;box(im,(x,282,x+263,384),WHITE,outline=LINE)
            text(im,(x+132,312),label,32,col,True,anchor='mt')
            if j<3: arrow(im,(x+273,333),(x+298,333),TEAL)
        lines(im,(48,447),['No repeated trial simulations during identification.','Simulation-based planning with the recovered model.'],28,INK,step=48,width=1185)
        text(im,(48,602),'Demonstrated in elastic manipulation, plastic shaping, and pouring.',26,MUTED,width=1185)
        return im
    raise ValueError(name)

def draw_section(name,t):
    im=draw_section_content(name,t)
    if name not in ('opening','observe','balance'):
        paint_caption(im,caption_at(name,t))
    return im

def render(name):
    if name=='insertion' and not (ASSETS/'insertion_probe.npz').exists():
        subprocess.run([sys.executable,str(HERE/'prepare_insertion_probe.py')],check=True)
    if name in ('observe','balance'):
        subfolder,script,output=(('method01_preview','render_method01.py','method01_animated.mp4') if name=='observe' else ('method02_preview','render_method02_animated.py','method02_block1_animated.mp4'))
        subprocess.run([sys.executable,str(HERE/subfolder/script),'--render'],check=True)
        shutil.copy2(HERE/subfolder/output,SECTIONS/f'{name}.mp4')
        return
    if name == 'opening':
        subprocess.run([sys.executable, str(HERE/'opening_preview/make_preview.py')], check=True)
        approved_frame.cache_clear()
    dur=next(d for n,d,_ in TIMELINE if n==name)
    cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
         '-an','-c:v','libx264','-preset','fast','-crf','17','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(SECTIONS/(name+'.mp4'))]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for i in range(dur*FPS):
        im=draw_section(name,i/FPS)
        p.stdin.write(im.tobytes())
    p.stdin.close(); assert p.wait()==0
    print('Rendered',name,dur,'seconds',flush=True)

def preview():
    for name,dur,_ in TIMELINE:
        for j,t in enumerate([.2,dur*.5,dur-.25]):
            draw_section(name,t).save(REVIEW/f'{name}_{j}.png')
    imgs=[]
    for name,_,_ in TIMELINE:
        im=Image.open(REVIEW/f'{name}_1.png');im.thumbnail((640,360));imgs.append(im)
    sheet=Image.new('RGB',(1280,360*math.ceil(len(imgs)/2)),WHITE)
    for i,im in enumerate(imgs): sheet.paste(im,((i%2)*640,(i//2)*360))
    sheet.save(REVIEW/'storyboard.jpg',quality=92)
    (REVIEW/'text_layout.json').write_text(json.dumps(TEXT_LOG,indent=2))

def assemble():
    assert (HERE/'hardware_clip.json').exists(),'Hardware footage is still missing; do not publish a placeholder.'
    # Validate every installed section, including the latest reviewed methods.
    for name,duration,_ in TIMELINE:
        cap=cv2.VideoCapture(str(SECTIONS/f'{name}.mp4'))
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==duration*FPS,(name,duration)
        cap.release()
    export_captions(HERE,ALL_SECTIONS,'ICRA2027_video')
    listing=HERE/'sections.txt';listing.write_text(''.join(f"file 'sections/{n}.mp4'\n" for n,_,_ in TIMELINE))
    joined=HERE/'assembly_concat.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-map_metadata','-1',
                    '-movflags','+faststart',str(joined)],check=True)
    master=HERE/'ICRA2027_video_master.mp4'
    # Cached sections can have been rendered before a timing revision. Repaint
    # only the four-pixel global progress bar against the assembled timeline.
    filters=(f'[0:v]drawbox=x=0:y=716:w=iw:h=4:color=0xd9e1e5:t=fill[base];'
             f'[base][1:v]overlay=x=-w+W*t/{TOTAL}:y=716:shortest=1[out]')
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(joined),'-f','lavfi','-i',
                    f'color=c=0x167b76:s=1280x4:r={FPS}:d={TOTAL}',
                    '-filter_complex_threads','2','-filter_complex',filters,'-map','[out]',
                    '-an','-c:v','libx264','-crf','17','-preset','fast','-threads','6',
                    '-pix_fmt','yuv420p','-map_metadata','-1','-movflags','+faststart',str(master)],check=True)
    joined.unlink()
    if master.stat().st_size<19_500_000:
        shutil.copy2(master,HERE/'ICRA2027_video.mp4')
        print('Master already fits the submission limit; copied without another encoding generation.',flush=True)
        return
    # Working cuts can exceed three minutes; derive the compact-copy bitrate
    # from actual duration while retaining the original high-quality master.
    bitrate=min(825_000,int(19_000_000*8/TOTAL))
    for p in [1,2]:
        cmd=['ffmpeg','-y','-v','error','-i',str(HERE/'ICRA2027_video_master.mp4'),'-an','-c:v','libx264','-preset','slow',
             '-b:v',str(bitrate),'-threads','6','-pix_fmt','yuv420p','-pass',str(p),'-passlogfile',str(HERE/'encode_pass'),'-map_metadata','-1']
        cmd+=['-f','null','/dev/null'] if p==1 else ['-movflags','+faststart',str(HERE/'ICRA2027_video.mp4')]
        subprocess.run(cmd,check=True)
    out=HERE/'ICRA2027_video.mp4'
    assert out.stat().st_size<20_000_000
    print('Submission file:',out,out.stat().st_size,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--preview',action='store_true')
    parser.add_argument('--render',nargs='*');parser.add_argument('--assemble',action='store_true');args=parser.parse_args()
    if args.prepare: prepare_assets()
    if args.preview: preview()
    if args.render is not None:
        for name in args.render or [n for n,_,_ in TIMELINE]: render(name)
    if args.assemble: assemble()
