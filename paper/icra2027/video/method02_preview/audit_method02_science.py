"""Read-only scientific audit of method slide 2; outputs stay in science_audit.

Run from the repository root with .venv/bin/python. This evaluates mathematical
identities and saved linear systems; it does not refit physics or render movies.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

P = Path(__file__).resolve().parent
ROOT = P.parents[3]
sys.path[:0] = [str(P), str(P.parent), str(ROOT), str(ROOT / 'src')]
import render_method02_animated as slide
from narrative_captions import CUES, subtitle_font
from experiments.robotics.plate_separated_identify import solve_columns, hencky_columns
from experiments.robotics.press_hardware_moving_tests import fields
from ident.weakform.elastic_grid import _frame_nodal_terms


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = P / 'science_audit'
    out.mkdir(exist_ok=True)
    sources = [P / name for name in ['render_method02_animated.py',
        'render_method02_flow.py', 'render_method02_revision.py', 'render_method02.py',
        'revision_latex/equations.tex', 'revision_latex/equations.pdf']]
    sources += [P.parent / name for name in ['narrative_captions.py',
        'method_slide_frame.py', 'sections/balance.mp4', 'references/2027_ICRA_FORM.pdf']]
    sources += [ROOT / name for name in [
        'experiments/robotics/plate_separated_identify.py',
        'experiments/robotics/press_hardware_linear_identify.py',
        'experiments/robotics/press_hardware_moving_tests.py',
        'src/ident/weakform/elastic_grid.py']]
    before = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    result = {}
    rng = np.random.default_rng(20260923)

    # The animated example is an actual curl field, not arbitrary arrows.
    q = rng.uniform(-.5, .5, (200, 3))
    psi, grad = slide.field(q)
    eps = 1e-6
    fd = np.stack([(slide.field(q + eps*np.eye(3)[j])[0] -
                    slide.field(q - eps*np.eye(3)[j])[0])/(2*eps)
                   for j in range(3)], axis=-1)
    gradient_error = float(abs(fd-grad).max())
    div_error = float(abs(np.trace(grad, axis1=-2, axis2=-1)).max())
    assert gradient_error < 1e-8 and div_error < 1e-12
    boundary = np.vstack([np.eye(3), -np.eye(3), 1.2*np.eye(3)])
    assert np.allclose(slide.field(boundary)[0], 0)
    Dw = (grad + grad.swapaxes(-1, -2))/2
    stress = rng.normal(size=(len(q), 3, 3))
    stress = (stress + stress.swapaxes(-1, -2))/2
    pressure = rng.normal(size=len(q))*100
    full = stress - pressure[:, None, None]*np.eye(3)
    cancellation_error = float(abs(np.einsum('nij,nij->n', full-stress, Dw)).max())
    assert cancellation_error < 1e-10
    result['illustrated_test_field'] = dict(gradient_fd_error=gradient_error,
        divergence_max=div_error, pressure_contraction_error=cancellation_error,
        scope='Valid illustrative curl field; not the exact contact-flat pressing field.')

    # Independently check time integration by parts, including the convective term.
    t = np.linspace(0, 1, 4001)
    x0=np.array([.05,-.1,.03]); v0=np.array([.02,.03,.01]); acc=np.array([.04,-.02,.03])
    x=x0+t[:,None]*v0+.5*t[:,None]**2*acc; v=v0+t[:,None]*acc
    psi,jac=slide.field(x); chi=np.sin(np.pi*t)**2; dchi=np.pi*np.sin(2*np.pi*t)
    w=chi[:,None]*psi
    along=dchi[:,None]*psi+chi[:,None]*np.einsum('tij,tj->ti',jac,v)
    gravity=np.array([0,0,-9.81])
    b_acc=np.trapezoid(w@(gravity-acc),t)
    b_tw=np.trapezoid(w@gravity+np.einsum('ti,ti->t',v,along),t)
    assert abs(b_acc-b_tw)<1e-7
    result['time_weak_sign_and_material_derivative'] = dict(error=float(abs(b_acc-b_tw)))

    # Actual pressing fields remove pressure and have known net-force boundary work.
    x=np.array([[[.012,.007,.013],[.007,.003,0],[.01,.008,.04]]])
    h=np.array([.04]); hd=np.array([-.004])
    w,g,wt=fields(x,h,hd)
    gt=np.stack([(fields(x+eps*np.eye(3)[j],h,hd)[0]-
                  fields(x-eps*np.eye(3)[j],h,hd)[0])/(2*eps) for j in range(3)],axis=-1)
    time_fd=(fields(x,h+eps*hd,hd)[0]-fields(x,h-eps*hd,hd)[0])/(2*eps)
    assert abs(gt-g).max()<1e-5
    assert abs(time_fd-wt).max()<1e-8
    assert abs(np.trace(g,axis1=-2,axis2=-1)).max()<1e-12
    assert np.allclose(w[:,:,1],0) and np.allclose(w[:,:,2],[0,0,1])
    result['actual_pressing_tests'] = dict(spatial_gradient_fd_error=float(abs(gt-g).max()),
        partial_time_fd_error=float(abs(time_fd-wt).max()), bottom_weight='zero',
        top_weight='unit vertical translation; measured net force provides boundary work')

    # Reproduce both saved scalar fits and their equivalent block-diagonal solve.
    result['saved_pressing_linear_systems']={}
    for material in ['A','B']:
        folder=ROOT/f'out/press_separated_20260913/monotonic320/separated_{material}'
        saved=json.loads((folder/'identification.json').read_text())
        data=dict(np.load(folder/'weak_system.npz'))
        values,_=solve_columns(**data,config=saved['config'])
        blocks=[]; loads=[]
        for k,(key,column,bounds) in enumerate([
            ('E_pa','elastic',saved['config']['elastic_centers']),
            ('yield_pa','plastic',saved['config']['plastic_centers'])]):
            use=(data['centers']>=bounds[0]-1e-10)&(data['centers']<=bounds[1]+1e-10)
            block=np.zeros((int(use.sum()),2));block[:,k]=data[column][use]
            blocks.append(block);loads.append(data['b'][use])
            assert np.isclose(values[key],saved[key],rtol=1e-12)
        A=np.vstack(blocks);b=np.concatenate(loads)
        theta=np.linalg.lstsq(A,b,rcond=None)[0]
        assert np.allclose(theta,[values['E_pa'],values['yield_pa']],rtol=1e-12)
        result['saved_pressing_linear_systems'][material]=dict(coefficients_Pa=theta.tolist(),
            rank=int(np.linalg.matrix_rank(A)), rows=len(b),
            scope='Separate early-elastic and late-yielded intervals; equivalent block-diagonal LS.')

    # The production momentum operator is linear in the supplied stress columns.
    x=rng.uniform(.25,.75,(25,3)); mass=rng.uniform(.8,1.2,25)
    columns=rng.normal(size=(25,3,3,3)); columns=(columns+columns.swapaxes(-1,-2))/2
    theta=np.array([.7,-.3,1.2]);valid=np.ones(25,dtype=bool)
    assembled=_frame_nodal_terms(x,columns,mass,valid,12,1.)['A']
    combined=np.einsum('pkij,k->pij',columns,theta)[:,None]
    direct=_frame_nodal_terms(x,combined,mass,valid,12,1.)['A'][...,0]
    linear_error=float(abs(np.einsum('dik,k->di',assembled,theta)-direct).max())
    assert linear_error<1e-10
    result['production_stress_assembly_linearity']=dict(max_error=linear_error)

    # Hencky basis uses Kirchhoff stress with reference-volume quadrature:
    # V_current * sigma_Cauchy = V_reference * tau_Kirchhoff.
    F=np.array([np.diag([1.1,1.04,.89]),np.diag([1.05,1.05,1/1.05**2])])
    E,Y=hencky_columns(F,.3)
    assert np.allclose(E,E.swapaxes(-1,-2)) and np.allclose(np.trace(E,axis1=-2,axis2=-1),0)
    J=np.linalg.det(F);tau=80000*E
    assert np.allclose(J[:,None,None]*(tau/J[:,None,None]),tau)
    result['stress_measure']='Slide: Cauchy stress with current volume. Pressing columns: equivalent Kirchhoff stress with reference-volume quadrature.'

    # The gold volume is illustrative, but its advertised constant volume is exact.
    worst=0.
    reference=np.array([[a,b,c] for a in [-1,1] for b in [-1,1] for c in [-1,1]])*.0018
    for t in np.linspace(slide.T0,slide.T1,17):
        _,points,cells=slide.subvolume_at(t)
        normalized=slide.constant_volume_cells(points,cells)
        maps=np.einsum('ij,njk->nik',np.linalg.pinv(reference),normalized)
        worst=max(worst,float(abs(np.linalg.det(maps)-1).max()))
    assert worst<1e-10
    result['illustrated_particle_volume']=dict(max_relative_volume_error=worst,
        scope='Display-only affine cell, normalized to constant volume for this incompressible example.')

    # Save complete frames and all active labels/equation/connector bounds for review.
    frame=slide.draw_frame(18,full=True,reveal=False)
    frame.save(out/'all_blocks.png')
    (out/'layout.json').write_text(json.dumps(dict(text=slide.TEXT_RECORDS,
        equations=slide.MATH_BOUNDS,connectors=slide.CONNECTORS),indent=2))
    cues=CUES['balance'];font=subtitle_font(24)
    assert cues[0][0]==0 and cues[-1][1]==slide.DURATION
    for i,(start,end,text) in enumerate(cues):
        assert i==0 or start==cues[i-1][1]
        assert font.getlength(text)<=1192 and len(text)/(end-start)<=20
    clip=P.parent/'sections/balance.mp4'
    cap=cv2.VideoCapture(str(clip));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==700
    tiles=[]
    for i,(start,end,_) in enumerate(cues):
        cap.set(cv2.CAP_PROP_POS_MSEC,1000*(start+end)/2);ok,bgr=cap.read();assert ok
        pic=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));pic.save(out/f'caption_{i}.png')
        tiles.append(pic.resize((640,360)))
    cap.release()
    sheet=Image.new('RGB',(1280,1440),'white')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*640,(i//2)*360))
    sheet.save(out/'encoded_review.jpg',quality=93)
    subprocess.run(['ffmpeg','-v','error','-i',str(clip),'-f','null','-'],check=True)
    result['presentation']=dict(captions=cues,duration_s=28,frames=700,full_decode='PASS')
    result['source_sha256']=before
    assert all(digest(ROOT/name)==value for name,value in before.items()),'Source changed during audit'
    result['checks_passed']=True
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':
    main()
