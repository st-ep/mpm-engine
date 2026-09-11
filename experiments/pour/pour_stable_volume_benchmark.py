"""Run analytical checks against the isolated stable-volume prototype."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'out/pour_physics_audit/stable_volume_20260907'
sys.path.insert(0,str(OUT/'isolated_src'))

import json
import numpy as np
import warp as wp
from warpmpm.kernels.warp_utils import MPMStateStruct, MPMModelStruct
from warpmpm.kernels.mpm_utils import stress_update_particle, fluid_logdet_increment
from warpmpm.kernels import mpm_utils
from experiments.pour import pour_float_precision_audit as baseline


@wp.kernel
def stable_compression(state: MPMStateStruct, model: MPMModelStruct,
                       rates: wp.array(dtype=float), dt: float, steps: int):
    p=wp.tid()
    rate=rates[p]/3.0
    L=wp.mat33(rate,0.0,0.0,0.0,rate,0.0,0.0,0.0,rate)
    state.particle_L[p]=L
    for _ in range(steps):
        state.particle_logJ[p]=state.particle_logJ[p]+fluid_logdet_increment(L,dt)
        state.particle_F_trial[p]=(wp.identity(n=3,dtype=float)+dt*L)*state.particle_F[p]
        stress_update_particle(state,model,dt,p)


@wp.kernel
def general_increment(L: wp.array(dtype=wp.mat33),dt: float,out: wp.array(dtype=float)):
    p=wp.tid()
    out[p]=fluid_logdet_increment(L[p],dt)


def main():
    if not str(Path(mpm_utils.__file__).resolve()).startswith(str(OUT/'isolated_src')):
        raise RuntimeError('Wrong package imported')
    baseline.OUT=OUT/'analytic'
    baseline.prescribed_compression=stable_compression
    baseline.main()
    results=json.loads((baseline.OUT/'results.json').read_text())
    max_relative=max(abs(r['relative_pressure_error']) for r in results['compression'])
    # Accuracy criteria based on the analytical solution, before any full pour.
    assert max_relative<.002, f'Pressure error exceeds0.2%: {max_relative}'
    rng=np.random.default_rng(125)
    L=rng.normal(size=(300,3,3))*np.geomspace(.001,1e4,300)[:,None,None]
    L=L.astype(np.float32);dt=float(np.float32((1/60)/763))
    out=wp.zeros(len(L),dtype=float,device='cuda:0')
    wp.launch(general_increment,dim=len(L),inputs=[wp.array(L,dtype=wp.mat33,device='cuda:0'),dt,out],device='cuda:0')
    sign,expected=np.linalg.slogdet(np.eye(3)[None]+dt*L.astype(float))
    assert np.all(sign>0)
    observed=out.numpy()
    error=np.max(abs(observed-expected))
    assert error<2e-7, f'General logdet increment error: {error}'
    review=dict(max_relative_pressure_error=max_relative,pressure_error_limit=.002,
        general_gradient_increment_max_absolute_error=float(error),increment_error_limit=2e-7,
        all_analytic_checks_passed=True,liquid_data_used=[],physical_parameters_changed=False,
        actual_kernel_module=str(Path(mpm_utils.__file__).resolve()),
        prototype_provenance=str(OUT/'prototype_provenance.json'))
    (OUT/'analytic_review.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(review,indent=2))


if __name__=='__main__':main()
