"""Evaluation and independent repeat, kept separate from the image/force fit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from experiments.elastic.strip_camera_observe import save


def isolate(root):
    repo=Path(__file__).resolve().parents[2];work=root/'isolated_check';work.mkdir(exist_ok=False)
    for k in 'AB':
        dest=work/f'inputs_{k}';dest.mkdir()
        for name in ['camera_0.npy','camera_1.npy','time.npy','known.json','calibration.json','force.csv']:
            os.link(root/f'inputs_{k}'/name,dest/name)
    if (root/'identification_protocol.json').exists():
        (work/'identification_protocol.json').write_bytes((root/'identification_protocol.json').read_bytes())
    script='''import sys
from pathlib import Path
repo,work=map(Path,sys.argv[1:])
def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes)):return
    p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
    if p.is_relative_to(repo/'out') and not p.is_relative_to(work):
        raise PermissionError('Forbidden experiment input: '+str(p))
sys.addaudithook(guard)
try:
    open(repo/'out/forbidden_simulator_state.npy','rb')
    raise AssertionError('Guard failed')
except PermissionError:pass
import json
from experiments.robotics.plate_observable_identify import track
from experiments.robotics.plate_observable_field import select
from experiments.robotics.plate_observable_divfree import fit
cfg=json.loads((work/'identification_protocol.json').read_text()) if (work/'identification_protocol.json').exists() else None
for k in 'AB':track(work/f'inputs_{k}',work/f'fit_{k}',**(cfg['tracking'] if cfg else {}))
if cfg:
    from experiments.robotics.plate_observable_refinement import fit_converged
    fit_converged(work)
else:
    select(work)
    for k in 'AB':fit(work,k)
'''
    with (work/'repeat.log').open('w') as log:
        subprocess.run([sys.executable,'-c',script,str(repo),str(work.resolve())],cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    results={}
    for k in 'AB':
        a=np.load(root/f'fit_{k}/tracks.npz');b=np.load(work/f'fit_{k}/tracks.npz')
        np.testing.assert_allclose(a['world'],b['world'],rtol=0,atol=0,equal_nan=True)
        a=json.loads((root/f'divfree_{k}/identification.json').read_text());b=json.loads((work/f'divfree_{k}/identification.json').read_text())
        for name in ['E_pa','yield_pa']:np.testing.assert_allclose(a[name],b[name],rtol=1e-12)
        results[k]=dict(exact_tracks_reproduced=True,parameters_reproduced=True,forbidden_data_guard_active=True)
    save(root/'observation_isolation.json',results)


def ideal(root):
    from experiments.robotics.plate_observable_divfree import fit
    work=root/'ideal_surface_check';work.mkdir(exist_ok=False)
    (work/'field_selection.json').write_bytes((root/'field_selection.json').read_bytes())
    config=json.loads((root/'identification_protocol.json').read_text()) if (root/'identification_protocol.json').exists() else {}
    order=json.loads((root/'integration_selection.json').read_text())['selected_order'] if (root/'integration_selection.json').exists() else None
    if config.get('test_fields')=='interior':
        from experiments.robotics.plate_observable_interior import fit
    errors={}
    for k in 'AB':
        p=json.loads((root/f'inputs_{k}/known.json').read_text());d=np.load(root/f'fit_{k}/tracks.npz')
        x=np.load(root/f'truth_{k}/x.npy',mmap_mode='r');axes=[np.unique(x[0,:,i]) for i in range(3)];shape=tuple(map(len,axes))+(3,)
        ref=d['reference'].copy();ref[:,1]=p['surface_y']
        true=np.array([RegularGridInterpolator(axes,x[round(t/.01)].reshape(shape),bounds_error=False,fill_value=None)(ref) for t in d['time']])
        motion=(d['world']-d['world'][0])-(true-true[0]);errors[k]=float(np.sqrt(np.mean(np.sum(motion[d['valid']]**2,axis=1)))*1000)
        dest=work/f'inputs_{k}';dest.mkdir()
        for name in ['known.json','force.csv']:os.link(root/f'inputs_{k}'/name,dest/name)
        dest=work/f'fit_{k}';dest.mkdir();np.savez(dest/'tracks.npz',reference=true[0],world=true,time=d['time'],valid=d['valid'])
        fit(work,k,quadrature_order=order,tag='divfree')
    save(root/'tracking_accuracy.json',dict(surface_motion_rmse_mm=errors,scope='Evaluation only; exact simulated surface positions do not enter reported estimates.'))


def evaluate(root):
    p=json.loads((root/'protocol.json').read_text());results={}
    for k in 'AB':
        fit=json.loads((root/f'divfree_{k}/identification.json').read_text());truth=np.load(root/f'truth_{k}/x.npy',mmap_mode='r');pred=np.load(root/f'prediction_{k}/x.npy',mmap_mode='r')
        t=np.load(root/f'truth_{k}/time.npy');validation_start=p.get('validation_start',p['fit_end']+.1);use=t>=validation_start-1e-9
        rms=np.sqrt(np.mean(np.sum((pred-truth)**2,axis=-1),axis=-1))*1000
        ft=np.genfromtxt(root/f'truth_{k}/force.csv',names=True,delimiter=',');fp=np.genfromtxt(root/f'prediction_{k}/force.csv',names=True,delimiter=',');fs=ft['time']>=validation_start-1e-9
        fine=np.genfromtxt(root/f'fine_{k}/force.csv',names=True,delimiter=',')
        # Relative L2 includes the complete held-out interval, including zero contact.
        results[k]=dict(validation_start=validation_start,E_pa=fit['E_pa'],yield_pa=fit['yield_pa'],E_error_percent=100*(fit['E_pa']/p['E_pa'][k]-1),
            yield_error_percent=100*(fit['yield_pa']/p['yield_pa'][k]-1),
            held_out_particle_rmse_mm=float(np.sqrt(np.mean(rms[use]**2))),final_particle_rmse_mm=float(rms[-1]),
            held_out_force_relative_l2=float(np.linalg.norm((fp['Fz']-ft['Fz'])[fs])/np.linalg.norm(ft['Fz'][fs])),
            grid_force_relative_l2=float(np.linalg.norm((fine['Fz']-ft['Fz'])[fs])/np.linalg.norm(fine['Fz'][fs])),
            truth_peak_force_N=float(ft['Fz'].max()))
        np.save(root/f'prediction_particle_rmse_{k}.npy',rms)
        if (root/f'fine_prediction_{k}/completion.json').exists():
            x=np.load(root/f'fine_{k}/x.npy',mmap_mode='r');y=np.load(root/f'fine_prediction_{k}/x.npy',mmap_mode='r')
            rf=np.sqrt(np.mean(np.sum((x-y)**2,axis=-1),axis=-1))*1000
            f=np.genfromtxt(root/f'fine_prediction_{k}/force.csv',names=True,delimiter=',')
            results[k].update(fine_held_out_particle_rmse_mm=float(np.sqrt(np.mean(rf[use]**2))),fine_final_particle_rmse_mm=float(rf[-1]),
                fine_held_out_force_relative_l2=float(np.linalg.norm((f['Fz']-fine['Fz'])[fs])/np.linalg.norm(fine['Fz'][fs])))
    save(root/'evaluation.json',results);print(json.dumps(results,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['isolate','ideal','evaluate']);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();globals()[a.stage](a.out)
