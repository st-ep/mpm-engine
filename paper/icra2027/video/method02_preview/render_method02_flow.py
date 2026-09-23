"""Three-stage static layout: weak balance -> identify law -> plan actions.

Reuses compiled LaTeX and frozen illustration sources. Does not replace the
installed video while this layout is being reviewed.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
from render_method02_revision import (
    Canvas, ReviewCanvas, S, BG, INK, BLUE, TEAL, ORANGE, MUTED, LINE,
    VOLUME, BASIS, PARTICLES, force, motion, asset, particle_diagram, tex, arrow,
    TEXT_RECORDS, CONNECTORS, MATH_BOUNDS,
)

P = Path(__file__).resolve().parent
HARDWARE_CROP = (700, 210, 1440, 660)


def hardware_execution(c, bounds):
    spec=json.loads((P.parent/'hardware_clip.json').read_text())
    cap=cv2.VideoCapture(spec['path'])
    cap.set(cv2.CAP_PROP_POS_MSEC,35000)
    ok,bgr=cap.read();cap.release()
    assert ok,'Cannot read confirmed hardware execution recording'
    pic=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop(HARDWARE_CROP)
    fitted(c,pic,bounds)


def fitted(c, source, bounds):
    x,y,w,h = bounds
    scale = min(w*S/source.width,h*S/source.height)
    pic = source.resize((round(source.width*scale),round(source.height*scale)),Image.Resampling.LANCZOS)
    c.im.paste(pic,(round((x+w/2)*S-pic.width/2),round((y+h/2)*S-pic.height/2)))


def ingredients(c):
    # Sources remain independent, with aspect-preserving placement.
    for draw,crop,box in [(force,(47,191,268,337),(64,223,112,84)),
                          (motion,(320,192,538,335),(337,223,112,84)),
                          (particle_diagram,(265,180,409,346),(205,219,111,128))]:
        src = Image.new('RGB',(1280*S,720*S),BG)
        draw(Canvas(src))
        fitted(c,src.crop(tuple(v*S for v in crop)),box)
    c.text(120,193,'Measured force',16.5,ORANGE,True,'mt')
    c.text(258,193,'Particles',17,PARTICLES,True,'mt')
    c.text(393,182,'Reconstructed',16.5,BLUE,True,'mt')
    c.text(393,204,'motion',16.5,BLUE,True,'mt')
    c.text(524,193,'Test weights',17,TEAL,True,'mt')
    asset(c,'field',(470,219,108,92))
    tex(c,2,(524,335),height=17)
    c.text(524,355,'Remove unknown',15.5,TEAL,anchor='mt')
    c.text(524,376,'pressure term',15.5,TEAL,anchor='mt')
    c.text(120,336,'Motion + loads',16,ORANGE,True,'mt')
    c.text(393,336,'Material state',16,BLUE,True,'mt')
    c.text(229,351,'All particles',15.5,PARTICLES,True,'rt')
    c.text(305,367,'Particle volume',15.5,VOLUME,True,'mt')


def route(c, points, color, width=1.4, arrowhead=False):
    """Explicit orthogonal routes and short term leaders, all audited for text."""
    if arrowhead:
        c.line(points[:-1],color,width)
    else:
        c.line(points,color,width)
    for a,b in zip(points,points[1:]):
        CONNECTORS.append({'from':a,'to':b,'color':color,'kind':'process' if arrowhead else 'leader'})
    if arrowhead:
        c.arrow(points[-2],points[-1],color,width,6)


def construction(c, source_time=None, operators=None, compact=False):
    if compact:
        # Move the complete upper diagram as a group, preserving its internal
        # alignment, and replace the two-line introduction with one sentence.
        text_start=len(TEXT_RECORDS); math_start=len(MATH_BOUNDS); line_start=len(CONNECTORS)
        scratch=Image.new('RGB',c.im.size,BG)
        anchors=construction(ReviewCanvas(scratch),source_time,operators=lambda *_: None)
        region=scratch.crop((67*S,239*S,579*S,586*S))
        c.im.paste(region,(67*S,215*S))
        TEXT_RECORDS[text_start:]=[r for r in TEXT_RECORDS[text_start:] if r['bbox'][1]>=239]
        for record in TEXT_RECORDS[text_start:]+MATH_BOUNDS[math_start:]:
            record['bbox'][1]-=24;record['bbox'][3]-=24
        for record in CONNECTORS[line_start:]:
            for key in ('from','to'):
                x,y=record[key];record[key]=(x,y-24)
        anchors={name:{side:(xy[0],xy[1]-24) for side,xy in ends.items()} for name,ends in anchors.items()}
        c.text(64,187,'by weighting Newton’s law and integrating over space and time.',17,INK)
        c.text(566,455,'Weak balance',15,TEAL,True,'rt')
        if operators is not None:operators(c,anchors,source_time)
        return anchors

    c.text(64,189,'Motion and contact forces constrain internal stress.',18,INK)
    c.text(64,216,'Chosen test functions eliminate the pressure term.',17.5,MUTED)

    # A single observation group supplies the measured side of the balance.
    # Its motion also branches horizontally into the material state.
    c.rect((67,239,225,464),BG,7,edge='#dcb69d')
    c.text(146,248,'Reconstructed',17,BLUE,True,'mt')
    c.text(146,268,'motion',17,BLUE,True,'mt')
    c.text(146,365,'Contact force',17,ORANGE,True,'mt')
    for draw,crop,box in [(motion,(320,192,538,335),(79,289,134,74)),
                          (force,(47,191,268,337),(79,387,134,58))]:
        src=Image.new('RGB',(1280*S,720*S),BG)
        if source_time is None:draw(Canvas(src))
        else:draw(Canvas(src),source_time=source_time)
        fitted(c,src.crop(tuple(v*S for v in crop)),box)
    c.text(146,447,'Motion + contact loads',13.7,ORANGE,True,'mt')

    # Shared top and bottom edges give both inputs the same visual level.
    # The state shares the stress model's left edge; basis shares its right edge.
    c.rect((250,239,366,373),'#eaf0f3',6)
    c.text(308,248,'Material state',16,BLUE,True,'mt')
    tex(c,7,(308,280),width=102)
    arrow(c,(225,326),(250,326),BLUE,1.8,5)
    for yy,symbol,first,second in [(299,'F:','deformation','gradient'),
                                  (338,'D:','strain-rate','tensor')]:
        c.text(260,yy,symbol,13.5,MUTED,True)
        c.text(280,yy,first,13.5,MUTED)
        c.text(280,yy+16,second,13.5,MUTED)

    # Two alternative basis sources, with a visibly layered neural-network icon.
    c.rect((378,239,576,373),'#efedf3',6)
    c.text(477,248,'Chosen stress basis',16,BASIS,True,'mt')
    c.text(421,272,'Analytic',14.5,INK,anchor='mt')
    c.text(421,289,'material laws',14.5,INK,anchor='mt')
    # A labeled schematic stress-strain law; no marker implying tracked data.
    c.text(390,307,'Stress',12.5,MUTED)
    c.line([(390,323),(390,353),(459,353)],'#a1adbb',1)
    c.line([(395,350),(421,328),(456,328)],BASIS,2.2)
    c.text(459,357,'Strain',12.5,MUTED,anchor='rt')
    c.text(471,325,'or',13,MUTED,anchor='mt')
    c.text(530,272,'Function',14.5,INK,anchor='mt')
    c.text(530,289,'encoder',14.5,INK,anchor='mt')
    layers=[[(504,318),(504,333),(504,348)],
            [(528,314),(528,325),(528,337),(528,352)],
            [(552,324),(552,342)]]
    for left,right in zip(layers,layers[1:]):
        for a in left:
            for b in right:c.line([a,b],'#b5a2c8',.8)
    for layer in layers:
        for point in layer:c.dot(point,3.3,BASIS,'#fcfbfe')
    c.text(530,357,'Pretrained offline',11.5,MUTED,anchor='mt')

    # Blue follows the motion -> state -> stress path, including its output.
    c.rect((250,377,576,464),'#eaf0f3',7)
    c.text(413,383,'Stress model',18,BLUE,True,'mt')
    stress_anchors=tex(c,9,(413,421),width=250)
    c.text(413,443,'θ: unknown material coefficients',15.5,INK,anchor='mt')
    # Definitions are inside the state card; a straight connector leaves its
    # bottom edge without crossing either definition or the stress title.
    for key,start_y,color in [('stress',373,BLUE),('basis',373,BASIS)]:
        x,y=stress_anchors[key]['top']
        arrow(c,(x,start_y),(x,y-7),color,1.8,6)

    # A smaller, raised balance frees space for three legible operator cues.
    c.rect((67,473,578,585),'#eaf0f3',9,edge='#91bab3')
    anchors=tex(c,1,(330,525),width=420)
    for key,col,start_y in [('load',ORANGE,464),('stress',BLUE,464)]:
        x,y=anchors[key]['top']
        arrow(c,(x,start_y),(x,y-8),col,1.8,6)

    if operators is not None:
        operators(c,anchors,source_time)
        return anchors

    # Schematic quadrature: highlight one represented particle volume, then
    # enlarge it. This is a volume cue, not a measured voxel partition.
    def small_cube(center,half,colors,edge):
        def project(q):
            x,y,z=q
            return np.array([center[0]+x-y*.64,center[1]+x*.35+y*.35-z])
        vertices=np.array([[a,b,d] for a in [-half,half] for b in [-half,half] for d in [-half,half]])
        xy=np.array([project(q) for q in vertices])
        for ids,color in [([1,5,7,3],colors[0]),([2,3,7,6],colors[1]),([4,5,7,6],colors[2])]:
            pts=xy[ids];c.poly(pts,color);c.line(np.vstack([pts,pts[0]]),edge,.8)
        return project
    def glass_volume(center,half,particle_radius):
        """Transparent represented volume, centered on a visible material point."""
        vertices=np.array([[a,b,d] for a in [-half,half] for b in [-half,half] for d in [-half,half]])
        xy=np.array([[center[0]+a-.64*b,center[1]+.35*a+.35*b-d] for a,b,d in vertices])
        layer=Image.new('RGBA',c.im.size,(0,0,0,0))
        draw=ImageDraw.Draw(layer)
        for ids,alpha in [([1,5,7,3],38),([2,3,7,6],23),([4,5,7,6],48)]:
            draw.polygon([tuple(p*S) for p in xy[ids]],fill=(181,143,66,alpha))
        c.im.paste(Image.alpha_composite(c.im.convert('RGBA'),layer).convert('RGB'))
        for i in range(8):
            for j in range(i+1,8):
                if np.count_nonzero(vertices[i]!=vertices[j])!=1:continue
                if i==0:
                    # Rear edges remain visible through the translucent faces.
                    for t in np.arange(0,1,.24):
                        c.line([xy[i]+t*(xy[j]-xy[i]),xy[i]+min(1,t+.12)*(xy[j]-xy[i])],'#bdad8d',.65)
                else:c.line([xy[i],xy[j]],VOLUME,.85)
        c.dot(center,particle_radius,PARTICLES,'#fffdf6')
        return xy

    project=small_cube((177,639),23,('#e2edf1','#edf3f6','#d5e4eb'),'#b0c5ce')
    selected_position=np.array([-5.,0.,0.])
    for y in [15,0,-15]:
        for z in [-15,0,15]:
            for x in [-15,-5,5,15]:
                c.dot(project((x,y,z)),2.2,PARTICLES,'#f4f6f7')
    x,y=anchors['sum']['bottom']
    route(c,[(219,619),(x,619),(x,y+7)],PARTICLES,1.7,arrowhead=True)
    c.text(177,682,'Sum over particles',14.5,PARTICLES,True,'mt')

    x,y=anchors['volume']['bottom']
    selected=project(selected_position)
    glass_volume(selected,7,2.8)
    # Foreground particles retain their depth order over the interior highlight.
    view=np.array([.64,1.,.574])
    for py in [15,0,-15]:
        for pz in [-15,0,15]:
            for px in [-15,-5,5,15]:
                point=np.array([px,py,pz])
                if point@view>selected_position@view:
                    c.dot(project(point),2.2,PARTICLES,'#f4f6f7')
    # The callout joins an interior cell to its enlarged transparent counterpart.
    # It is schematic quadrature volume, not a recovered physical cell boundary.
    route(c,[(selected[0]+11.48,selected[1]),(x-27.88,selected[1])],VOLUME,1)
    glass_volume((x,639),17,4)
    arrow(c,(x,609),(x,y+7),VOLUME,1.7,6)
    c.text(309,682,'Particle volume',14.5,VOLUME,True,'mt')

    x,y=anchors['weight']['bottom']
    asset(c,'field',(x-60,602,120,67))
    arrow(c,(x,601),(x,y+7),TEAL,1.8,6)
    c.text(x,671,'Chosen test functions',16,TEAL,True,'mt')
    tex(c,2,(x,699),height=16)

    return anchors


def recovered_law(c):
    c.rect((641,185,868,338),'#eaf0f3',7)
    c.text(754,195,'Identified law',22,BLUE,True,'mt')
    c.line([(666,229),(666,313),(847,313)],'#98aab5',1.2)
    c.text(669,223,'Stress',14.5,MUTED)
    c.text(849,319,'Strain',15.5,MUTED,anchor='rt')
    c.line([(673,306),(742,246),(840,246)],BLUE,2.8)
    c.line([(742,246),(742,313)],'#bfd1dc',1)
    c.text(711,278,'E',19,BLUE,True)
    c.text(832,221,'Y',19,BLUE,True,'mt')
    c.text(660,319,'Schematic',13.5,MUTED)


def identification_solve(c):
    """A single shared panel, read bottom-up from assembly to the solve."""
    c.rect((641,369,868,703),BG,7,edge='#ccd9e0')
    c.text(754,389,'Linear',22,INK,True,'mt')
    c.text(754,416,'least-squares solve',22,INK,True,'mt')
    tex(c,4,(754,471),width=220)
    arrow(c,(754,535),(754,510),BLUE,2.2,7)
    c.text(754,551,'Assemble equations',21,INK,True,'mt')
    tex(c,6,(754,601),width=130)
    # A contains integrated basis responses evaluated on the observed motion;
    # b is the known weighted balance, not just a force measurement.
    c.text(654,627,'A:',15.5,BLUE,True)
    c.text(677,627,'integrated stress bases',14.5,MUTED)
    c.text(677,645,'on observed motion',14.5,MUTED)
    c.text(654,664,'b:',15.5,ORANGE,True)
    c.text(677,664,'motion + contact loads',14.5,MUTED)
    c.text(654,683,'θ:',15.5,INK,True)
    c.text(677,683,'material coefficients',14.5,MUTED)
    arrow(c,(754,369),(754,338),BLUE,2.2,7)


def render():
    TEXT_RECORDS.clear();CONNECTORS.clear();MATH_BOUNDS.clear()
    im = Image.new('RGB',(1280*S,720*S),BG)
    c = ReviewCanvas(im)
    c.text(44,22,'FORM · From Observed Response to Material laws',23,TEAL,True)
    c.text(1236,22,'02 / IDENTIFY & PLAN',20,TEAL,True,'rt')
    c.text(44,56,'Recover the material law. Plan the robot action.',41,INK,True)
    c.line([(44,114),(1236,114)],LINE,1)

    # Three quiet enclosures, matching the approved Observe slide.
    for x0,x1 in [(44,591),(626,882),(917,1236)]:
        c.rect((x0,138,x1,711),BG,10,edge='#ccd9e0')
    c.text(64,150,'1  Build the weak balance',23,INK,True)
    c.text(645,150,'2  Identify the law',23,INK,True)
    c.text(936,150,'3  Plan and execute',23,INK,True)

    anchors=construction(c)

    # One shared panel makes assembly and the linear solve a single stage.
    identification_solve(c)
    arrow(c,(578,525),(641,525),TEAL,2.2,7)
    recovered_law(c)

    # Planning is followed by actual hardware execution. The small target remains
    # an input to planning, rather than occupying an entire workflow stage.
    c.rect((933,185,1220,439),'#eaf0f3',7)
    c.text(1076,200,'Plan with MPM',22,INK,True,'mt')
    asset(c,'shape_action',(944,239,191,123),white_to_bg=True)
    asset(c,'target',(1147,246,62,47),white_to_bg=True)
    c.text(1178,303,'Target',15.5,MUTED,anchor='mt')
    tex(c,5,(1076,386),width=253)
    c.text(1076,414,'Keep the identified law fixed',16.5,BLUE,True,'mt')

    # Recovered law and MPM now align: a single horizontal model-transfer arrow.
    arrow(c,(868,264),(933,264),BLUE,2.1,7)
    arrow(c,(1076,450),(1076,482),TEAL,2.2,7)
    c.rect((933,492,1220,703),'#eaf0f3',7)
    c.text(1076,501,'Execute on real specimen',20,INK,True,'mt')
    hardware_execution(c,(943,535,267,162))

    # Check connectors against text / equations before saving the review image.
    overlaps=[]
    for link in CONNECTORS:
        a,b=np.asarray(link['from']),np.asarray(link['to'])
        pts=a+np.linspace(0,1,1000)[:,None]*(b-a)
        for item in TEXT_RECORDS:
            x0,y0,x1,y1=item['bbox']
            hit=(pts[:,0]>=x0-1)&(pts[:,0]<=x1+1)&(pts[:,1]>=y0-1)&(pts[:,1]<=y1+1)
            if hit.any():overlaps.append({'connector':link,'text':item['text']})
    for math in MATH_BOUNDS:
        x0,y0,x1,y1=math['bbox']
        for item in TEXT_RECORDS:
            a,b,d,e=item['bbox']
            if min(x1,d)>max(x0,a) and min(y1,e)>max(y0,b):overlaps.append({'math':math['equation'],'text':item['text']})
    im.save(P/'method02_flow_layout.png')
    im.resize((1280,720),Image.Resampling.LANCZOS).save(P/'method02_flow_layout_720p.png')
    (P/'method02_flow_layout_review.json').write_text(json.dumps({
        'status':'Static layout preview; installed video unchanged.',
        'input_cards':{'shared_top':239,'state_and_basis_bottom':373,'observations_bounds':[67,239,225,464],'state_bounds':[250,239,366,373],'basis_bounds':[378,239,576,373],'stress_model_bounds':[250,377,576,464]},
        'palette':{'motion_state_stress_path':BLUE,'contact_loads':ORANGE,'stress_basis':BASIS,'particle_volume':VOLUME,'test_functions':TEAL,'stress_model_housing':'pale blue'},
        'flow':'Balance -> least-squares solve -> identified law -> MPM action search -> hardware execution.',
        'alignment':{'balance_output_y':525,'assembled_equations_y':601,'law_output_y':264,'mpm_input_y':264,'middle_column_direction':'bottom to top'},
        'stress_basis_sources':'Analytic material-law basis or an offline-pretrained function encoder. Both supply known Tk responses evaluated at observed state; online identification solves the coefficients, not network weights.',
        'construction':'Motion and contact force join into the measured balance. Motion also provides deformation/rate to a parameterized stress model. The stress is not directly observed. Sum and particle volume share one schematic illustration. Test weights apply to the full balance. Matrix assembly precedes the upward least-squares solve.',
        'hardware_execution':{'source':json.loads((P.parent/'hardware_clip.json').read_text()),'source_time_s':35,'display_crop_xyxy':HARDWARE_CROP,'crop_note':'Trimmed mainly from the bottom and left; smaller trim on right and top. Aspect-preserving resize, no image synthesis.'},
        'illustration_scope':'The preexisting blue simulation image is an illustrative MPM shaping example; the hardware still is the user-confirmed red matched-ID execution. This slide illustrates the workflow, not a frame-matched simulation/hardware comparison.',
        'main_equation_width_px':420,'previous_width_px':468,
        'text':TEXT_RECORDS,'connectors':CONNECTORS,'math_bounds':MATH_BOUNDS,
        'overlaps':overlaps},indent=2)+'\n')
    print('Overlap audit:',json.dumps(overlaps))
    print(P/'method02_flow_layout.png')


if __name__=='__main__':render()
