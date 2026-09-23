"""LaTeX-first still revision. Keeps the installed animation unchanged for review.

The adjacent equations.tex is compiled by Tectonic and rasterized by pdftocairo.
Complete equations retain Computer Modern spacing, baselines, and limits.
"""
from pathlib import Path
from functools import lru_cache
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from render_method02 import (
    Canvas, ReviewCanvas, S, BG, INK, BLUE, TEAL, ORANGE, MUTED, LINE,
    force, motion, asset, TEXT_RECORDS,
)

P = Path(__file__).resolve().parent
LATEX = P/'revision_latex'
VOLUME = '#92733f'
BASIS = '#796399'
PARTICLES = '#5a788b'
CONNECTORS = []
MATH_BOUNDS = []


@lru_cache(None)
def latex_raster(index, width=None, height=None):
    # pdftocairo pads page numbers once the PDF has ten or more pages.
    path = LATEX/f'equation-{index:02d}.png'
    if not path.exists(): path = LATEX/f'equation-{index}.png'
    im = Image.open(path).convert('RGBA')
    im = im.crop(im.getbbox())
    scale = width*S/im.width if width else height*S/im.height
    im = im.resize((round(im.width*scale), round(im.height*scale)), Image.Resampling.LANCZOS)
    return im


def tex(c, index, center, width=None, height=None):
    im = latex_raster(index,width,height)
    x, y = round(center[0]*S-im.width/2), round(center[1]*S-im.height/2)
    c.im.paste(im, (x,y), im)
    MATH_BOUNDS.append({'equation':index,'bbox':[x/S,y/S,(x+im.width)/S,(y+im.height)/S]})
    anchors = {}
    a = np.asarray(im)
    for name, col in [('load',ORANGE),('volume',VOLUME),('basis',BASIS),('stress',BLUE),('weight',TEAL),('sum',PARTICLES),('time','#485862')]:
        mask = (abs(a[:,:,:3].astype(int)-ImageColor.getrgb(col)).max(2)<5)&(a[:,:,3]>180)
        yy, xx = np.where(mask)
        if len(xx):
            anchors[name] = {'top': ((x+(xx.min()+xx.max())/2)/S,(y+yy.min())/S),
                             'bottom': ((x+(xx.min()+xx.max())/2)/S,(y+yy.max())/S)}
    return anchors


def arrow(c, start, end, color, width=1.9, head=7):
    CONNECTORS.append({'from':start,'to':end,'color':color})
    c.arrow(start,end,color,width,head)


def small_inputs(c):
    # Both repeated observations occupy precisely the same 164 x 104 px footprint.
    src = Image.new('RGB',(1280*S,720*S),BG)
    d = Canvas(src)
    force(d)
    motion(d)
    for bounds,box in [((47,191,268,337),(49,180,164,104)),
                       ((320,192,538,335),(431,180,164,104))]:
        crop = src.crop(tuple(v*S for v in bounds))
        x,y,w,h=box
        scale=min(w*S/crop.width,h*S/crop.height)
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
        c.im.paste(crop,(round((x+w/2)*S-crop.width/2),round((y+h/2)*S-crop.height/2)))
    c.text(131,148,'Measured force',18,ORANGE,True,'mt')
    c.text(319,148,'Material particles',18,PARTICLES,True,'mt')
    c.text(513,148,'Reconstructed motion',18,BLUE,True,'mt')
    c.text(744,148,'Chosen test weights',18,TEAL,True,'mt')
    asset(c,'field',(662,175,164,110))
    tex(c,2,(744,309),height=22)
    c.text(735,331,'Remove unknown pressure',17,TEAL,anchor='mt')
    c.text(735,355,'from this equation',17,TEAL,anchor='mt')
    c.text(513,306,'Material state',17,BLUE,True,'mt')


def particle_diagram(c):
    """Schematic quadrature particles and one represented volume, not new data."""
    def project(q):
        x,y,z=np.asarray(q)
        return np.array([310+18*x-13*y,250+7*x+6*y-23*z])
    def cube(center,half,projector,colors,edge):
        v=np.array([[a,b,d] for a in [-half,half] for b in [-half,half] for d in [-half,half]])+center
        xy=np.array([projector(q) for q in v])
        for indices,col in [([1,5,7,3],colors[0]),([0,1,5,4],colors[1]),([4,5,7,6],colors[2])]:
            pp=xy[indices];c.poly(pp,col);c.line(np.vstack([pp,pp[0]]),edge,.8)
        return xy
    # A sparse glass envelope makes the spatial sum explicit without implying
    # that this synthetic lattice is the measured hardware particle arrangement.
    points=np.array([[x,y,z] for x in range(4) for y in range(3) for z in range(3)])
    corners=np.array([[x,y,z] for x in [-.45,3.45] for y in [-.45,2.45] for z in [-.45,2.45]])
    xy=np.array([project(q) for q in corners])
    for ids,col in [([1,5,7,3],'#e2edf1'),([0,1,5,4],'#edf3f6'),([4,5,7,6],'#d5e4eb')]:
        p=xy[ids];c.poly(p,col);c.line(np.vstack([p,p[0]]),'#b0c5ce',.8)
    for point in points[np.argsort(points[:,1])[::-1]]:
        c.dot(project(point),2.6,PARTICLES,'#f4f6f7')
    chosen=np.array([2.,0.,1.])
    cube(chosen,.42,project,('#e2d9ef','#d1c1e2','#b69ecf'),VOLUME)
    p=project(chosen);c.dot(p,3.1,VOLUME,'white')
    # Thin leader to the enlarged particle-volume cell; semantic arrow to Vp
    # starts below its caption and never crosses the dot or the cube.
    c.line([p+[8,8],(382,302)],VOLUME,1)
    def zoom(q):
        x,y,z=q
        return np.array([382+23*x-16*y,324+8*x+7*y-26*z])
    cube(np.zeros(3),.5,zoom,('#e2d9ef','#d1c1e2','#b69ecf'),VOLUME)
    c.dot(zoom([0,0,0]),3.1,VOLUME,'white')
    c.text(386,348,'Particle volume',17,VOLUME,True,'mt')


def curve(c):
    c.text(719,551,'Identified law',23,BLUE,True,'mt')
    c.line([(637,590),(637,676),(818,676)],'#9cadb7',1.2)
    c.line([(645,670),(713,608),(811,608)],BLUE,2.8)
    c.line([(713,608),(713,676)],'#bed3df',1)
    c.text(628,581,'Stress',16,MUTED,anchor='rt')
    c.text(677,642,'E',19,BLUE,True)
    c.text(805,581,'Y',19,BLUE,True,'mt')
    c.text(817,680,'Strain',16,MUTED,anchor='rt')
    c.text(715,704,'Schematic stiffness / yield response',15,MUTED,anchor='ms')


def render():
    TEXT_RECORDS.clear()
    CONNECTORS.clear()
    MATH_BOUNDS.clear()
    im = Image.new('RGB',(1280*S,720*S),BG)
    c = ReviewCanvas(im)
    c.text(44,22,'FORM · From Observed Response to Material laws',23,TEAL,True)
    c.text(1236,22,'02 / IDENTIFY & PLAN',20,TEAL,True,'rt')
    c.text(44,56,'Recover the material law. Plan the robot action.',41,INK,True)
    c.line([(44,114),(1236,114)],LINE,1)
    small_inputs(c)
    particle_diagram(c)

    # The balance is now the largest visual element on the identification side.
    c.rect((44,388,846,536),'#eaf0f3',12)
    anchors = tex(c,1,(442,449),width=686)
    # Connectors stop outside the glyph bounds, never on top of the mathematics.
    load = anchors['load']['top']
    stress = anchors['stress']['top']
    weight = anchors['weight']['top']
    arrow(c,(131,334),(load[0],load[1]-13),ORANGE)
    arrow(c,(513,334),(stress[0],stress[1]-13),BLUE)
    arrow(c,(735,382),(weight[0],weight[1]-13),TEAL)
    c.text(131,306,'Motion + loads',17,ORANGE,True,'mt')
    vp = anchors['volume']['top']
    arrow(c,(382,376),(vp[0],vp[1]-13),VOLUME,1.8,6)
    total = anchors['sum']['top']
    arrow(c,(300,302),(total[0],total[1]-13),PARTICLES,1.8,6)
    c.text(284,350,'All particles',17,PARTICLES,True,'rt')

    c.text(62,551,'One linear least-squares solve',23,INK,True)
    c.text(62,581,'Known responses; unknown material coefficients',17,MUTED)
    tex(c,3,(281,617),width=245)
    tex(c,4,(281,662),width=268)
    c.text(281,714,'No forward simulation during identification',16,MUTED,anchor='ms')
    arrow(c,(518,648),(566,648),BLUE,2,7)
    curve(c)

    c.line([(866,145),(866,547)],LINE,1.1)
    c.line([(866,613),(866,700)],LINE,1.1)
    c.text(1062,149,'Plan with MPM',25,INK,True,'mt')
    asset(c,'target',(950,185,224,126),white_to_bg=True)
    c.text(1218,236,'Target',18,MUTED,anchor='rt')
    arrow(c,(1062,319),(1062,337),TEAL,1.9,6)
    c.text(1062,343,'Simulate candidate actions',19,INK,True,'mt')
    asset(c,'shape_action',(899,375,337,201),white_to_bg=True)
    tex(c,5,(1062,607),width=314)
    arrow(c,(833,580),(896,580),BLUE,2.1,7)
    c.text(1062,647,'Predicted shape vs. target',18,MUTED,anchor='mt')
    c.text(1062,676,'Material law stays fixed',21,BLUE,True,'mt')

    im.save(P/'method02_latex_revision.png')
    im.resize((1280,720),Image.Resampling.LANCZOS).save(P/'method02_latex_revision_720p.png')
    (P/'revision_latex/review.json').write_text(json.dumps({
        'status':'Static revision only; installed video unchanged.',
        'math':'Complete LaTeX equations compiled with Tectonic / Computer Modern.',
        'volume':'Vp denotes current particle volume, not velocity times pressure.',
        'emphasis':'Equal 164x104 observation panels; central balance enlarged.',
        'particle_diagram':'Schematic material-point collection. A highlighted represented volume is enlarged and connected to Vp; the collection connects to the sum over p. This is not a measured discretization or a literal recovered cell partition.',
        'text':TEXT_RECORDS,'connectors':CONNECTORS,'math_bounds':MATH_BOUNDS,'term_anchors':anchors},indent=2)+'\n')
    print(P/'method02_latex_revision.png')


if __name__=='__main__':
    subprocess.run(['/home/stepan/miniconda3/bin/tectonic','--keep-logs','equations.tex'],cwd=LATEX,check=True)
    subprocess.run(['pdftocairo','-png','-transp','-r','240',str(LATEX/'equations.pdf'),str(LATEX/'equation')],check=True)
    render()
