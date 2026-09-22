"""Private ICRA supplement. Deterministic editing of frozen footage, no physics runs.

Run with the repository .venv Python. --preview writes section review frames;
--render [section ...] renders sections; --assemble makes master/submission MP4s.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, subprocess, sys, shutil
from functools import lru_cache
from opening_preview.render_opening import draw_opening, DURATION as OPENING_DURATION, approved_frame
from method01_preview.render_method01 import draw_method01, DURATION as OBSERVE_DURATION
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
TIMELINE = [
 ('opening',OPENING_DURATION,'Identify material laws. Plan robot actions.'),
 ('observe',OBSERVE_DURATION,'Observe one interaction'),
 ('balance',18,'Identify material response'),
 ('plan',10,'Use the model to plan a new task'),
 ('insertion',15,'Elastic rod insertion'),
 ('golf',17,'Putting with a flexible club'),
 ('simshape',23,'Plan plastic shaping'),
 ('pressing',16,'From real pressing to a predictive model'),
 ('hardware',14,'Execute the planned pinches'),
 ('scans',10,'Evaluate the final hardware shapes'),
 ('pouring',13,'Identify from one glycerol pour'),
 ('volumes',10,'Plan for new target volumes'),
 ('takeaway',10,'From an interaction to a reusable material model'),
]
TOTAL = sum(d for _,d,_ in TIMELINE)
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
def frame(key,t):
    if key not in CLIPS: CLIPS[key]=Clip(SOURCES[key])
    return CLIPS[key].get(t)

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
    text(im,(44,654),s,23,INK,width=1192)
    if second: text(im,(44,684),second,19,MUTED,width=1192)

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

def draw_section(name,t):
    start=sum(d for n,d,_ in TIMELINE[:[n for n,_,_ in TIMELINE].index(name)])
    elapsed=start+t
    if name=='opening':
        return draw_opening(t)
    if name=='observe':
        return draw_method01(t)
    if name=='balance':
        im=base('FORM · 02 / IDENTIFY','Recover an explicit material law','Hold the observed motion fixed; impose weak-form momentum balance.',elapsed)
        box(im,(44,165,526,623));fit(im,still('field'),(63,186,444,350))
        text(im,(76,550),'Choose spatial test fields',27,TEAL,True)
        text(im,(76,590),'Arrows show weights, not motion.',22,MUTED)
        box(im,(550,165,1236,623))
        stage=0 if t<4.5 else 1 if t<10 else 2
        text(im,(580,184),'Motion + loads  ≈  weighted internal stress',26,INK,True,width=632)
        fit(im,still('ls'),(595,235,586,95))
        lines(im,(584,341),['Motion defines stress responses in A.','Motion and loads define b; θ holds the coefficients.'],24,INK,step=34,width=614)
        if stage>=1:
            fit(im,still('solve'),(590,412,607,101))
            text(im,(584,513),'Least squares when coefficients enter linearly.',24,TEAL,True,width=614)
        if stage>=2:
            text(im,(584,565),'Pressing: separate elastic and yielded intervals.',23,MUTED,width=614)
        if stage==0: footer(im,'Integrating against chosen test fields gives equations for material parameters.',
                             'Divergence-free fields remove the unknown pressure contribution where applicable.')
        elif stage==1: footer(im,'Fit the coefficients directly from these equations.',
                                'The constitutive family and required reconstruction assumptions are specified beforehand.')
        else: footer(im,'No repeated forward simulations during identification.',
                        'For the plastic press, the interval assumptions give separate fits for stiffness E and yield scale Y.')
        return im
    if name=='plan':
        im=base('FORM · 03 / PLAN','Plan robot actions with the identified law','Freeze material parameters; optimize the robot motion in MPM.',elapsed)
        for x in [44,455,866]: box(im,(x,178,x+370,595))
        text(im,(69,201),'Identified model',28,BLUE,True)
        text(im,(478,201),'Motion optimization',28,TEAL,True)
        text(im,(891,201),'Open-loop execution',27,INK,True)
        text(im,(76,252),'Stiffness E · Yield scale Y',23,BLUE,True,width=310)
        fit(im,still('target'),(75,296,306,130))
        lines(im,(76,441),['Material law + task goal','Parameters stay fixed'],25,INK,step=48,width=310)
        fit(im,still('shape_action'),(475,263,330,192))
        lines(im,(481,477),['Simulate candidate motions','Select the best evaluated plan'],22,INK,step=35,width=332)
        fit(im,frame('shape',t).crop((50,48,810,568)),(888,264,324,216))
        text(im,(894,515),'Execute the frozen motion',23,INK,width=320)
        arrow(im,(419,383),(447,383));arrow(im,(832,383),(858,383))
        footer(im,'Simulation is used for planning, after identification.',
                  'Deployment uses new actions or geometries; no online refitting or replanning.')
        return im
    if name=='insertion':
        im=base('Results · simulation','Elastic rod insertion','A bending probe identifies stiffness; the model transfers to a different geometry.',elapsed)
        comparison(im,'insert',max(0,t-2))
        side(im,'Matched ID',['Plan uses the model','of the executed material.'],BLUE)
        side(im,'Swapped ID',['Plan uses the model','of the other material.'],ORANGE,y=324)
        if t>=10.5:
            side(im,'Both matched insert',['Both swaps collide','with the wall.'],TEAL,y=476)
        footer(im,'Materials A and B: the same task, different identified elastic stiffnesses.',
                  'Recorded simulation speed; final state held.')
        return im
    if name=='golf':
        im=base('Results · simulation','Putting with a flexible club','The same identified elastic models transfer to a second manipulation task.',elapsed)
        # Play once at real simulation speed, then hold. No unmarked repeated trials.
        comparison(im,'golf',max(0,t-1)*.5,(44,180,832,448))
        side(im,'Plan the forward stroke',['Duration and aim change;','backswing stays prescribed.'],BLUE)
        side(im,'Matched ID',['Both balls stop','inside the target.'],TEAL,y=347)
        if t>12: side(im,'Swapped ID',['Both balls miss','the target.'],ORANGE,y=496)
        footer(im,'Club flexibility matters even with the same initial backswing.',
                  '0.5× playback; final state held. A/B denote elastic materials in this experiment.')
        return im
    if name=='simshape':
        im=base('Results · simulation','Plan plastic shaping','A flat-plate press identifies a model for shaping a larger block.',elapsed)
        comparison(im,'shape',t*1.25)
        side(im,'Hencky / von Mises',['Similar fitted stiffness;','different yield scales.'],BLUE)
        fit(im,still('sim_target'),(984,319,146,166))
        text(im,(980,490),'Target shape',23,MUTED,True)
        if t>=18.5:
            lines(im,(912,522),['Surface error (mm)','A: 0.883 / 1.226','B: 1.256 / 1.739'],23,INK,step=31,width=320)
        footer(im,'Six planned pinches; compare matched and swapped models on the same material.',
                  '1.25× playback. Plastic A/B; errors: matched / swapped, symmetric area-weighted surface distance.')
        return im
    if name=='pressing':
        im=base('Results · hardware','From real pressing to a predictive model','Models identified from separate presses are evaluated with a lower applied force.',elapsed)
        # Source strips contain material labels; replace only their label band at larger size.
        for j,label in enumerate(['Red Play-Doh','Yellow butter slime','Gray plasticine']):
            text(im,(252+j*371,164),label,25,INK,True,anchor='mt')
        text(im,(44,253),'Real',22,INK,True)
        text(im,(44,466),'MPM',22,INK,True)
        fit(im,frame('real',t).crop((0,50,1820,410)),(120,206,1114,211))
        fit(im,frame('sim',t).crop((0,50,1820,410)),(120,427,1114,211))
        footer(im,'Prediction discrepancies remain, especially in the soft materials’ spreading and recovery.',
                  '1× playback; common force-baseline reference. Simulation views are mirrored for display.')
        return im
    if name=='hardware':
        im=base('Results · hardware','Execute the planned pinches','The identified model is held fixed while planning four pinches on a fresh specimen.',elapsed)
        conf=HERE/'hardware_clip.json'
        if conf.exists():
            spec=json.loads(conf.read_text());SOURCES['hardware']=Path(spec['path'])
            pic=frame('hardware',spec.get('start',0)+t*spec.get('speed',1))
            if spec.get('crop'): pic=pic.crop(tuple(spec['crop']))
            if spec.get('rotate'): pic=pic.rotate(spec['rotate'],expand=True)
            fit(im,pic,(44,164,835,465))
            side(im,'Plan, then execute',['Finger motions are','specified in advance.'],BLUE)
            side(im,'No online correction',['No material refitting','during execution.'],TEAL,y=386)
            footer(im,spec['caption'],spec['qualification'])
        else:
            fit(im,still('butter_slime'),(80,193,700,400))
            side(im,'Footage pending',['Hardware execution clip','has not been supplied yet.'],ORANGE)
            footer(im,'Assembly placeholder, not for submission.','Replace with the supplied hardware execution recording.')
        return im
    if name=='scans':
        im=base('Results · hardware','Evaluate the final hardware shapes','Intersection over union of the reconstructed XY footprint and the target.',elapsed)
        for j,(key,label,val) in enumerate([('play_doh','Red Play-Doh','75.7%'),('butter_slime','Yellow butter slime','72.4%'),('plasticine','Gray plasticine','77.8%')]):
            x=44+j*403
            text(im,(x+192,180),label,26,INK,True,anchor='mt')
            fit(im,still(key),(x,228,385,317))
            text(im,(x+192,566),'IoU '+val,32,TEAL,True,anchor='mt')
        footer(im,'One scan per material; translation and rotation aligned to the target, without scaling.',
                  'Photo-textured reconstructions; missing surfaces interpolated. Release-to-scan delay was not recorded.')
        return im
    if name=='pouring':
        im=base('Results · hardware + simulation','Identify from one glycerol pour','A reduced, time-integrated weak balance estimates effective viscosity.',elapsed)
        fit(im,frame('pour',t),(44,161,825,473))
        side(im,'One 60° pour',['Effective viscosity','3.44 Pa·s'],BLUE)
        side(im,'Contact calibration',['Same pour; viscosity fixed.','Effective coefficient: 0.272'],TEAL,y=365)
        footer(im,'This is the identification / calibration recording.',
                  '1× playback. Receiver volume, cup motion, and geometry supply the reduced model; blue liquid is simulated.')
        return im
    if name=='volumes':
        im=base('Results · hardware','Plan for new target volumes','Freeze viscosity and contact, then optimize the pouring angle for each target.',elapsed)
        fit(im,still('volumes'),(34,163,852,470))
        text(im,(923,205),'60–160 mL',37,BLUE,True)
        text(im,(923,253),'Six target volumes',25,INK)
        text(im,(923,334),'3.8 mL',41,TEAL,True)
        lines(im,(923,395),['Largest absolute','mean target error','for glycerol'],25,INK,step=34,width=303)
        footer(im,'Glycerol: five trials, mean ± SD. Water: the same commands, one trial per target.',
                  'Volumes read from receiver photographs; cup accuracy and camera parallax remain uncalibrated.')
        return im
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

def render(name):
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
    listing=HERE/'sections.txt';listing.write_text(''.join(f"file 'sections/{n}.mp4'\n" for n,_,_ in TIMELINE))
    subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-map_metadata','-1',
                    '-movflags','+faststart',str(HERE/'ICRA2027_video_master.mp4')],check=True)
    master=HERE/'ICRA2027_video_master.mp4'
    if master.stat().st_size<19_500_000:
        shutil.copy2(master,HERE/'ICRA2027_video.mp4')
        print('Master already fits the submission limit; copied without another encoding generation.',flush=True)
        return
    # 825 kbit/s leaves margin beneath a decimal 20 MB limit for up to 180 seconds, with no audio.
    for p in [1,2]:
        cmd=['ffmpeg','-y','-v','error','-i',str(HERE/'ICRA2027_video_master.mp4'),'-an','-c:v','libx264','-preset','slow',
             '-b:v','825k','-threads','6','-pix_fmt','yuv420p','-pass',str(p),'-passlogfile',str(HERE/'encode_pass'),'-map_metadata','-1']
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
