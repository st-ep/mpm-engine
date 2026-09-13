"""Reproduce a frozen refined plate protocol in a fresh output directory.

The generation protocol includes true material parameters. The separate
identification protocol does not. Observation isolation checks this boundary.
"""
import argparse
import json
import os
from pathlib import Path


def repeat_finer_observations(root):
    """Regenerate camera inputs from finer reference data; report even if worse."""
    from experiments.robotics.plate_observable_camera import render
    from experiments.robotics.plate_observable_identify import track
    from experiments.robotics.plate_observable_refinement import fit_converged
    p=json.loads((root/'protocol.json').read_text())
    cfg=json.loads((root/'identification_protocol.json').read_text())
    check=root/'fine_observation_check';check.mkdir(exist_ok=False)
    fine=dict(p,grid=p['fine_grid'])
    fine['scope']='Numerical observation-repeat check; no further grid refinement is scheduled here.'
    (check/'protocol.json').write_text(json.dumps(fine,indent=2)+'\n')
    (check/'identification_protocol.json').write_bytes((root/'identification_protocol.json').read_bytes())
    for k in 'AB':
        (check/f'truth_{k}').symlink_to((root/f'fine_{k}').resolve())
        render(check,k);track(check/f'inputs_{k}',check/f'fit_{k}',**cfg['tracking'])
    fit_converged(check)
    # Ground truth enters only this post-fit evaluation, never the estimator.
    errors={}
    for k in 'AB':
        f=json.loads((check/f'divfree_{k}/identification.json').read_text())
        errors[k]=dict(E_percent=100*(f['E_pa']/p['E_pa'][k]-1),
                       yield_percent=100*(f['yield_pa']/p['yield_pa'][k]-1))
    result=dict(source=str(check.resolve()),signed_errors=errors,
                all_four_within_five_percent=all(abs(v)<=5 for e in errors.values() for v in e.values()))
    (root/'numerical_robustness.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--identification-protocol',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'1')
    from experiments.robotics.plate_observable_probe import record
    from experiments.robotics.plate_observable_camera import render
    from experiments.robotics.plate_observable_identify import track
    from experiments.robotics.plate_observable_refinement import fit_converged
    from experiments.robotics.plate_observable_sensitivity import sensitivity
    from experiments.robotics.plate_observable_audit import isolate,evaluate
    from experiments.robotics.plate_observable_report import provenance
    from experiments.robotics.plate_observable_summary import report
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'protocol.json').write_bytes(a.protocol.read_bytes())
    (a.out/'identification_protocol.json').write_bytes(a.identification_protocol.read_bytes())
    for name in ['design.md','validation_scope.md']:
        if (a.protocol.parent/name).exists():
            (a.out/name).write_bytes((a.protocol.parent/name).read_bytes())
    cfg=json.loads(a.identification_protocol.read_text())
    for k in 'AB':
        record(a.out,k,'truth',a.device);render(a.out,k)
        track(a.out/f'inputs_{k}',a.out/f'fit_{k}',**cfg['tracking'])
    fit_converged(a.out)
    for k in 'AB':
        sensitivity(a.out,k)
        for kind in ['prediction','fine']:record(a.out,k,kind,a.device)
    isolate(a.out);evaluate(a.out);repeat_finer_observations(a.out)
    report(a.out);provenance(a.out)


if __name__=='__main__':main()
