"""Separate review prototype: spatial expression -> dt contributions -> integral.

Symbolic explanation only. Token size/count are not computed stresses, physical
volumes, or a replacement quadrature rule. Leaves the main film and the existing
method preview unchanged.
"""
from pathlib import Path
import json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageColor
import render_method02_animated as a

P=Path(__file__).resolve().parent
CLOCK=0.
EVENTS=(.55,1.75,2.95)
LOOP=6.;FPS=25
original_operators=a.operator_scene


def smooth(x):
    x=float(np.clip(x,0,1));return x*x*(3-2*x)


def tint_outline(c,box,opacity,color,width=2):
    if opacity<=0:return
    overlay=Image.new('RGBA',c.im.size,(0,0,0,0));d=ImageDraw.Draw(overlay)
    d.rounded_rectangle(tuple(int(v*a.S) for v in box),radius=5*a.S,
        outline=(*ImageColor.getrgb(color),round(255*opacity)),width=round(width*a.S))
    c.im.paste(Image.alpha_composite(c.im.convert('RGBA'),overlay).convert('RGB'))


def tile(c,x,y,opacity,label='sum × Δt'):
    if opacity<=0:return
    bounds=(round(x*a.S),round(y*a.S),round((x+66)*a.S),round((y+18)*a.S))
    layer=c.im.copy();d=a.flow.Canvas(layer)
    d.rect((x,y,x+65,y+17),'#e0ece9',3,edge=a.TEAL)
    d.text(x+32.5,y+1,label,13,a.INK,True,'mt')
    c.im.paste(Image.blend(c.im.crop(bounds),layer.crop(bounds),float(opacity)),bounds[:2])


def integration(c,anchors,source):
    clock=CLOCK%LOOP
    merge=smooth((clock-4.15)/.95)
    c.text(112,601,'Time contributions',12.5,a.MUTED,anchor='mt')
    c.text(112,682,'Integrate over time',14,a.TIME,True,'mt')
    ix,iy=anchors['time']['bottom']
    path=[(112,598),(112,591),(ix,591),(ix,iy+6)]
    a.operator_arrow(c,path,a.TIME,source,1.5)
    # Pair each symbolic time contribution with a highlight of the actual
    # weighted spatial expression; no unrelated plot or specimen icon.
    for i,start in enumerate(EVENTS):
        born=smooth((clock-start-.4)/.35)
        tile(c,79,620+19*i,born*(1-merge))
    # A final pulse enters the integral along its existing orthogonal connector.
    if 4.15<=clock<=5.15:
        distance=smooth((clock-4.15)/1.)
        pts=np.asarray(path,float);lens=np.linalg.norm(np.diff(pts,axis=0),axis=1)
        remain=distance*lens.sum()
        for i,l in enumerate(lens):
            if remain<=l or i==len(lens)-1:
                p=pts[i]+min(1,remain/l)*(pts[i+1]-pts[i]);c.dot(p,4.,a.TEAL,'white');break
            remain-=l
    active=max([smooth((clock-start)/.15)*(1-smooth((clock-start-.55)/.2)) for start in EVENTS])
    # Change only background pixels: no new outline crosses mathematical glyphs
    # or the existing operator connectors.
    box=(220*a.S,489*a.S,510*a.S,576*a.S)
    region=np.asarray(c.im.crop(box)).copy()
    background=np.array(ImageColor.getrgb('#eaf0f3'))
    mask=np.max(np.abs(region.astype(float)-background),axis=2)<3
    shade=np.array(ImageColor.getrgb('#cde6df'))
    region[mask]=np.rint(region[mask]*(1-active)+shade*active).astype(np.uint8)
    c.im.paste(Image.fromarray(region),box[:2])
    # Badge sits in the empty baseline space below the stress/test-field terms.
    tile(c,329,550,active,'× Δt')
    if clock>=4.9:
        strength=smooth((clock-4.9)/.3)
        box=(round((ix-24)*a.S),478*a.S,round((ix+24)*a.S),559*a.S)
        region=np.asarray(c.im.crop(box)).copy()
        old=np.array(ImageColor.getrgb(a.TIME));new=np.array(ImageColor.getrgb(a.TEAL))
        mask=np.max(np.abs(region.astype(float)-old),axis=2)<15
        region[mask]=np.rint(region[mask]*(1-strength)+new*strength).astype(np.uint8)
        c.im.paste(Image.fromarray(region),box[:2])
    if clock>=5.25:
        c.text(112,634,'Whole press',14,a.TEAL,True,'mt')
        c.text(112,652,'combined',14,a.TEAL,True,'mt')


def operators(c,anchors,source):
    original_operators(c,anchors,source,integration_draw=integration)


a.operator_scene=operators


def frame(t,full=False):
    global CLOCK
    CLOCK=round(t%LOOP,8)
    # Existing source clock at half speed; hold the final specimen state while
    # the final symbolic accumulation is explained, rather than slowing motion.
    source_clock=min(CLOCK,3.9)
    return a.draw_frame(source_clock,full=full)


def main():
    for t in [.85,2.65,3.85,5.5]:frame(t,True).save(P/f'integration_concept_{t:g}s.png')
    frame(2.65,True).save(P/'integration_concept_poster.png')
    out=P/'integration_concept.mp4'
    proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','17','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    frames=[]
    for i in range(2*int(LOOP*FPS)):
        if i<LOOP*FPS:frames.append(frame(i/FPS).tobytes())
        proc.stdin.write(frames[i%int(LOOP*FPS)])
        if i%50==0:print(f'Rendered {i}/300 frames',flush=True)
    proc.stdin.close();assert proc.wait()==0
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(out),'-vf',
      'crop=548:574:44:138,scale=768:804:flags=lanczos','-an','-c:v','libx264',
      '-crf','17','-preset','fast','-threads','4','-pix_fmt','yuv420p',
      '-movflags','+faststart',str(P/'integration_concept_closeup.mp4')],check=True)
    (P/'integration_concept.json').write_text(json.dumps({'duration_s':12,'loop_s':6,
      'idea':'Highlight weighted spatial expression; introduce symbolic dt contributions; combine them into the integral over the recorded press.',
      'symbolic_only':'Three visible tokens are explanatory samples, not actual stresses, numerical quadrature, or positive-only accumulations. Unknown material coefficients remain unknown.',
      'motion':'Existing 0.05–2.0 s reconstruction/force at half speed, followed by a two-second endpoint hold for the accumulation explanation.',
      'scope':'Separate design prototype. Only block one changes; inter-block arrows and blocks two and three remain static.'},indent=2)+'\n')
    print(out,flush=True)

if __name__=='__main__':main()
