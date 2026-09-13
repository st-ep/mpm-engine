"""Choose a shared numerical integration rule without using true parameters."""
import argparse
import json
from pathlib import Path
import shutil
from experiments.robotics.plate_observable_field import select
from experiments.robotics.plate_observable_divfree import fit
from experiments.elastic.strip_camera_observe import save

CONFIG=dict(tracking=dict(features=750,margin_mm=1.,min_distance=12),
            admissibility_quadrature_order=8,quadrature_orders=[6,8,10,12],relative_tolerance=.005)


def fit_converged(root):
    config=json.loads((root/'identification_protocol.json').read_text())
    method=fit
    if config.get('test_fields')=='interior':
        from experiments.robotics.plate_observable_interior import fit as interior_fit
        method=interior_fit
    if not (root/'field_selection.json').exists():select(root,config['admissibility_quadrature_order'],config.get('validate_volume_folds',False))
    previous=None;history=[]
    for order in config['quadrature_orders']:
        current={}
        for k in 'AB':
            dest=root/f'gauss{order}_{k}'
            if not dest.exists():method(root,k,tag=f'gauss{order}',quadrature_order=order)
            current[k]=json.loads((dest/'identification.json').read_text())
        delta=None if previous is None else max(abs(current[k][n]/previous[k][n]-1) for k in 'AB' for n in ['E_pa','yield_pa'])
        history.append(dict(order=order,max_relative_parameter_change=delta));print(history[-1],flush=True)
        if delta is not None and delta<=config['relative_tolerance']:
            for k in 'AB':shutil.copytree(root/f'gauss{order}_{k}',root/f'divfree_{k}')
            save(root/'integration_selection.json',dict(history=history,selected_order=order,converged=True,
                criterion='Both inferred parameters of both materials change less than tolerance under quadrature refinement; no true parameters read.'))
            return
        previous=current
    save(root/'integration_selection.json',dict(history=history,converged=False))
    raise RuntimeError('Integration did not converge; no final fit selected')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();fit_converged(a.out)
