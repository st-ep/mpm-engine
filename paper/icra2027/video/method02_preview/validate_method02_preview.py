import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import render_method02_animated as a
from render_method02_revision import latex_raster
fail=[]
base=np.asarray(a.draw_frame(0,True,reveal=False,presentation=False))
def overlap(x,y):
    return min(x[2],y[2])-max(x[0],y[0])>1 and min(x[3],y[3])-max(x[1],y[1])>1
def math_ink_overlap(text,math):
    # Tall integral glyphs create a large transparent rectangle around shorter
    # terms. Check actual LaTeX ink rather than treating that empty area as ink.
    if not overlap(text['bbox'],math['bbox']):return False
    x0,y0,x1,y1=math['bbox']
    raster=latex_raster(math['equation'],width=x1-x0)
    l,t,r,b=text['bbox']
    bounds=(max(0,int((l-x0)*a.S)-2),max(0,int((t-y0)*a.S)-2),min(raster.width,int((r-x0)*a.S)+2),min(raster.height,int((b-y0)*a.S)+2))
    return np.asarray(raster.crop(bounds))[:,:,3].max(initial=0)>32
for t in np.linspace(0,a.DURATION-.04,33):
    im=np.asarray(a.draw_frame(float(t),True,reveal=False,presentation=False))
    if not np.array_equal(im[:,591*a.S:917*a.S],base[:,591*a.S:917*a.S]):fail.append(['static panels',float(t)])
    for c in a.CONNECTORS:
        p,q=np.array(c['from']),np.array(c['to'])
        if abs(p[0]-q[0])>1e-5 and abs(p[1]-q[1])>1e-5:fail.append(['non-orthogonal',float(t),str(c)])
        # Test annotation intersections, excluding endpoints via one-pixel inset.
        bounds=[min(p[0],q[0])-.5,min(p[1],q[1])-.5,max(p[0],q[0])+.5,max(p[1],q[1])+.5]
        for text in a.TEXT_RECORDS:
            b=text['bbox']
            if min(bounds[2],b[2])-max(bounds[0],b[0])>.1 and min(bounds[3],b[3])-max(bounds[1],b[1])>.1:
                fail.append(['connector text',float(t),text['text']])
    for i,text in enumerate(a.TEXT_RECORDS):
        for other in a.TEXT_RECORDS[i+1:]:
            if overlap(text['bbox'],other['bbox']):fail.append(['text overlap',float(t),text['text'],other['text']])
        for math in a.MATH_BOUNDS:
            if math_ink_overlap(text,math):fail.append(['math text',float(t),text['text'],math['equation']])
for t in [0,.8,2.8,3.99]:
    if not np.array_equal(np.array(a.draw_frame(t,True,reveal=False,presentation=False))[:,:917*a.S],np.array(a.draw_frame(t+4,True,reveal=False,presentation=False))[:,:917*a.S]):fail.append(['loop mismatch',t])
result={'sampled_frames':33,'static_middle_panel_and_interblock_arrows':not any(f[0]=='static panels' for f in fail),'loop_length_s':4,'failures':fail}
(a.P/'method02_block1_animation_checks.json').write_text(json.dumps(result,indent=2)+'\n')
# Both matched-plan image streams must actually move over the full section.
first=np.asarray(a.draw_frame(20,True,reveal=False,presentation=False));later=np.asarray(a.draw_frame(26,True,reveal=False,presentation=False))
for name,box in [('simulation',(938,225,1168,394)),('hardware',(938,529,1215,697))]:
 x0,y0,x1,y1=[int(v*a.S) for v in box]
 if np.array_equal(first[y0:y1,x0:x1],later[y0:y1,x0:x1]):fail.append([name+' is static'])
result['action_views_change']=not any('is static' in f[0] for f in fail)
# Reveal timing and the incoming arrows must agree, independently of animation.
for start in [a.SECOND_START,a.THIRD_START]:
 assert a.reveal_progress(start-.04,start)==0
 assert a.reveal_progress(start,start)==0
 assert abs(a.reveal_progress(start+.3,start)-.5)<1e-10
 assert a.reveal_progress(start+.6,start)>1-1e-10
assert a.action_time(19.96)==0 and a.action_time(20)==0
assert a.action_time(24)==8 and a.action_time(28)==16
for t in [0,11.96,12.3,12.6,19.96,20.3,20.6,27.96]:
    shown=np.asarray(a.draw_frame(t,True,presentation=False))
    raw=np.asarray(a.draw_frame(t,True,reveal=False,presentation=False))
    for box,start in [((645,390,862,702),a.SECOND_START),((939,530,1210,695),a.THIRD_START)]:
        x0,y0,x1,y1=[v*a.S for v in box]
        tile=raw[y0:y1,x0:x1]
        alpha=a.FAINT_OPACITY+(1-a.FAINT_OPACITY)*a.reveal_progress(t,start)
        expected=np.asarray(a.Image.blend(a.Image.new('RGB',(x1-x0,y1-y0),a.BG),a.Image.fromarray(tile),alpha))
        if not np.array_equal(shown[y0:y1,x0:x1],expected):fail.append(['reveal opacity',t,start])
# Reframing translates the body by 60 px without shrinking or modifying it.
for t in [0,12.6,20.6,27.96]:
    raw=np.asarray(a.draw_frame(t,True,presentation=False))
    framed=np.asarray(a.draw_frame(t,True))
    assert np.array_equal(raw[135*a.S:715*a.S],framed[75*a.S:655*a.S])
result['reveal_seconds']=[a.SECOND_START,a.THIRD_START]
result['duration_seconds']=a.DURATION
result['failures']=fail
(a.P/'method02_block1_animation_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert not fail
