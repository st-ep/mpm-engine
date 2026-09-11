"""Analytical checks of the unmodified fluid update at pouring timesteps.

No liquid recordings or measured endpoints are read. Prescribed uniform
compression has a known volume ratio and pressure, so it isolates arithmetic
from boundaries, transfers, fitting and spatial discretization.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import warp as wp

from warpmpm import Solver, GridConfig, newtonian
from warpmpm.kernels.warp_utils import MPMStateStruct, MPMModelStruct
from warpmpm.kernels.mpm_utils import stress_update_particle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/pour_physics_audit/float_precision_20260907"


@wp.kernel
def prescribed_compression(state: MPMStateStruct, model: MPMModelStruct,
                           rates: wp.array(dtype=float), dt: float, steps: int):
    p = wp.tid()
    rate = rates[p]/3.0
    L = wp.mat33(rate,0.0,0.0,0.0,rate,0.0,0.0,0.0,rate)
    ident = wp.identity(n=3,dtype=float)
    state.particle_L[p] = L
    for _ in range(steps):
        state.particle_F_trial[p] = (ident+dt*L)*state.particle_F[p]
        stress_update_particle(state,model,dt,p)


@wp.kernel
def prescribed_translation(x: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3),
                           dt: float, steps: int):
    p = wp.tid()
    value = x[p]
    for _ in range(steps):
        value = value+dt*velocities[p]
    x[p] = value


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    rates = np.array([-.1,-.03,-.01,-.003,-.001,-.0003,.001,.01],np.float32)
    rows, translation = [], []
    for scale in [1.,.5,.25]:
        steps = int(763*12/scale)
        dt = float(np.float32((1/60)/(763/scale)))
        duration = dt*steps
        pos = np.tile(np.array([.15,.15,.15],np.float32),(len(rates),1))
        solver = Solver(grid=GridConfig(n_grid=8,grid_lim=.4),device="cuda:0").load_particles(
            pos,np.full(len(rates),1e-9,np.float32))
        solver.set_material(newtonian(eta=3.4392377844275503,density=1260.,bulk_modulus=9e5))
        st,model=solver._sim.mpm_state,solver._sim.mpm_model
        wp.launch(prescribed_compression,dim=len(rates),inputs=[st,model,
            wp.array(rates,dtype=float,device="cuda:0"),dt,steps],device="cuda:0")
        F=st.particle_F.numpy().astype(float)
        J=np.linalg.det(F)
        pressure=-np.trace(st.particle_stress.numpy().astype(float),axis1=1,axis2=2)/(3*J)
        # Compare with the same explicit-Euler kinematics in high precision.
        # The exact continuum solution is also retained to separate truncation.
        logJ=3*steps*np.log1p(dt*rates.astype(float)/3)
        expectedJ=np.exp(logJ)
        expectedP=9e5*np.expm1(-1.1*logJ)
        for i,rate in enumerate(rates):
            rows.append(dict(dt_scale=scale,dt_s=dt,steps=steps,duration_s=duration,
                divergence_per_s=float(rate),actual_J=float(J[i]),expected_discrete_J=float(expectedJ[i]),
                expected_continuum_J=float(np.exp(duration*float(rate))),
                actual_pressure_pa=float(pressure[i]),expected_discrete_pressure_pa=float(expectedP[i]),
                relative_pressure_error=float((pressure[i]-expectedP[i])/expectedP[i])))
        speeds=np.array([.0001,.0003,.001,.003,.01,.03],np.float32)
        v=np.repeat(speeds[:,None],3,axis=1)
        xp=np.tile(np.array([.1,.2,.3],np.float32),(len(v),1))
        x=wp.array(xp,dtype=wp.vec3,device="cuda:0")
        wp.launch(prescribed_translation,dim=len(v),inputs=[x,wp.array(v,dtype=wp.vec3,device="cuda:0"),dt,steps],device="cuda:0")
        actual=x.numpy().astype(float)-xp.astype(float)
        expected=v.astype(float)*duration
        translation.append(dict(dt_scale=scale,speeds_m_s=speeds.tolist(),
            expected_displacement_m=expected.tolist(),actual_displacement_m=actual.tolist(),
            relative_displacement_error=((actual-expected)/expected).tolist()))
    paths=[Path(__file__),ROOT/'src/warpmpm/kernels/mpm_utils.py',ROOT/'src/warpmpm/kernels/warp_utils.py']
    result=dict(purpose=__doc__,compression=rows,translation=translation,
        constitutive_identification_changed=False,liquid_data_used=[],
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        print(f"dt{r['dt_scale']:g} divergence{r['divergence_per_s']:+.4f}/s: pressure {r['actual_pressure_pa']:.3f} vs {r['expected_discrete_pressure_pa']:.3f} Pa ({100*r['relative_pressure_error']:+.1f}%)")


if __name__=='__main__':
    main()
