"""Test constant-velocity transport through the actual MPM transfers.

An isolated fluid blob has uniform velocity, no external force or contact, and
zero initial stress. Its analytical solution is rigid translation with constant
momentum. This diagnostic reads no liquid recording or experimental endpoint.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant', choices=['original','stable_volume','compensated_position'], required=True)
parser.add_argument('--duration', type=float, default=.2)
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
source = ROOT / 'src' if args.variant == 'original' else ROOT / 'out/pour_physics_audit' / (args.variant+'_20260907') / 'isolated_src'
sys.path.insert(0,str(source))

import hashlib
import json
import time
import numpy as np
from warpmpm import Solver, GridConfig, newtonian
from warpmpm.kernels import mpm_utils


def main():
    assert Path(mpm_utils.__file__).resolve().is_relative_to(source)
    out = ROOT / 'out/pour_physics_audit/uniform_transport_20260907' / args.variant
    out.mkdir(parents=True,exist_ok=False)
    rng = np.random.default_rng(1809)
    x0 = rng.uniform([.1,.18,.27],[.13,.21,.30],(64,3)).astype(np.float32)
    v0 = np.tile(np.array([.001,-.001,.001],np.float32),(len(x0),1))
    protocol = dict(purpose=__doc__,variant=args.variant,duration_s=args.duration,
        velocity_m_s=v0[0].tolist(),seed=1809,particle_count=len(x0),
        no_external_force=True,no_colliders=True,dt_scales=[1.,.5,.25],
        relative_momentum_error_threshold=.001,mean_position_error_threshold_m=3e-7,
        actual_kernel_module=str(Path(mpm_utils.__file__).resolve()),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__),*sorted((source/'warpmpm').rglob('*.py'))]},
        liquid_data_used=[],identification_changed=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    for scale in protocol['dt_scales']:
        solver = Solver(grid=GridConfig(n_grid=16,grid_lim=.4),device=args.device).load_particles(
            x0,np.full(len(x0),1e-8,np.float32))
        solver.set_material(newtonian(eta=3.4392377844275503,density=1260.,bulk_modulus=9e5),g=[0.,0.,0.])
        solver.set_v(v0)
        steps=int(round(args.duration/((1/60)/763*scale)))
        dt=args.duration/steps
        start=time.monotonic()
        for done in range(0,steps,1000):
            solver.step(dt,min(1000,steps-done))
        x,v=solver.x().astype(float),solver.v().astype(float)
        assert np.isfinite(x).all() and np.isfinite(v).all()
        expected=x0.astype(float)+steps*float(np.float32(dt))*v0.astype(float)
        mean_velocity_error=v.mean(0)-v0.astype(float).mean(0)
        relative_momentum_error=float(np.linalg.norm(mean_velocity_error)/np.linalg.norm(v0[0]))
        mean_position_error=float(np.linalg.norm((x-expected).mean(0)))
        row=dict(dt_scale=scale,dt_s=dt,steps=steps,elapsed_s=time.monotonic()-start,
            mean_velocity_m_s=v.mean(0).tolist(),mean_velocity_error_m_s=mean_velocity_error.tolist(),
            relative_momentum_error=relative_momentum_error,
            mean_position_error_m=mean_position_error,max_position_error_m=float(np.max(abs(x-expected))),
            mean_displacement_m=(x-x0.astype(float)).mean(0).tolist(),
            momentum_check_passed=relative_momentum_error<=protocol['relative_momentum_error_threshold'],
            mean_position_check_passed=mean_position_error<=protocol['mean_position_error_threshold_m'])
        rows.append(row)
        (out/'results.json').write_text(json.dumps(dict(protocol=protocol,cases=rows),indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
