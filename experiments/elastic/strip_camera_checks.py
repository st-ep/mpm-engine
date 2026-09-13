"""Reproducible selection and observation-isolation checks for strip cameras."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments.elastic.strip_camera_identify import spline_basis,save


def select_basis(root):
    scores=[]
    for degree in [2,3]:
        for spacing in [.006,.008,.010,.012,.016]:
            errors=[]; ranks=[]
            for k in ['A','B']:
                p=json.loads((root/f'inputs_{k}/known.json').read_text())
                p.update(spline_degree_x=degree,spline_spacing_z=spacing)
                d=np.load(root/f'fit_{k}/tracks.npz')
                reference=d['reference'].copy(); reference[~d['valid'][0]]=p['center']
                M=spline_basis(reference,p)[0]
                grip=np.stack(np.meshgrid(np.linspace(.114,.126,5),[.110],
                              np.linspace(.171,.180,3),indexing='ij'),axis=-1).reshape(-1,3)
                C=spline_basis(grip,p)[0]*3
                for f in [30,40,50,60,70,80]:
                    valid=d['valid'][f]&d['valid'][0]
                    target=(d['world'][f]-reference)[:,[0,2]]
                    for fold in range(4):
                        hold=(np.arange(len(reference))%4==fold)&valid; use=valid&~hold
                        mat=np.vstack([M[use],C]); y=np.vstack([target[use],np.zeros((len(C),2))])
                        c,_,rank,_=np.linalg.lstsq(mat,y,rcond=1e-9)
                        ranks.append(rank==mat.shape[1]); errors.extend((M[hold]@c-target[hold]).ravel())
            score=dict(spline_degree_x=degree,spline_spacing_z=spacing,
                       marker_cv_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))*1000),
                       all_full_rank=all(ranks))
            scores.append(score)
    selected=min((s for s in scores if s['all_full_rank']),key=lambda s:s['marker_cv_rmse_mm'])
    result=dict(criterion='Four-fold held-marker displacement prediction, loading frames only, pooled A/B; no E or simulator states used',
                candidates=scores,selected=selected)
    save(root/'reconstruction_selection_recheck.json',result)
    return result


def isolated(root):
    repo=Path(__file__).resolve().parents[2]
    records={}; workspace=root/'isolated_check'; workspace.mkdir(exist_ok=False)
    for k in ['A','B']:
        bundle=workspace/f'inputs_{k}'; bundle.mkdir()
        for name in ['camera_0.npy','camera_1.npy','time.npy','force.csv','known.json','calibration.json']:
            os.link(root/f'inputs_{k}'/name,bundle/name)
        dest=workspace/f'fit_{k}'
        script='''import sys,runpy
from pathlib import Path
repo,bundle,dest=map(Path,sys.argv[1:])
opened=set()
def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes)): return
    p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
    if p.is_relative_to(repo/'out'):
        if not (p.is_relative_to(bundle) or p.is_relative_to(dest)):
            raise PermissionError('Forbidden experiment data: '+str(p))
        opened.add(str(p))
sys.addaudithook(guard)
try:
    open(repo/'out/forbidden_simulator_state.npy','rb')
    raise AssertionError('Guard failed')
except PermissionError:
    pass
sys.argv=['strip_camera_identify','--inputs',str(bundle),'--out',str(dest)]
runpy.run_module('experiments.elastic.strip_camera_identify',run_name='__main__')
print('ALLOWED_DATA_FILES',sorted(opened))
'''
        run=subprocess.run([sys.executable,'-c',script,str(repo),str(bundle.resolve()),str(dest.resolve())],
                           cwd=repo,text=True,capture_output=True,check=True)
        (workspace/f'{k}.log').write_text(run.stdout+run.stderr)
        a=json.loads((root/f'fit_{k}/identification.json').read_text())
        b=json.loads((dest/'identification.json').read_text())
        np.testing.assert_allclose(a['E_pa'],b['E_pa'],rtol=1e-12)
        original=np.load(root/f'fit_{k}/tracks.npz'); repeated=np.load(dest/'tracks.npz')
        np.testing.assert_allclose(original['world'],repeated['world'],rtol=0,atol=0,equal_nan=True)
        records[k]=dict(E_pa=b['E_pa'],exact_tracks_reproduced=True,fit_reproduced=True,
                        forbidden_data_access_guard_active=True)
    save(root/'observation_isolation.json',records)
    return records


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['select','isolate']); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); print(select_basis(a.out) if a.stage=='select' else isolated(a.out))
