"""Check the isolated prototype's GPU pipelines at an acoustic-stable timestep."""
from pathlib import Path
import sys,json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'out/pour_physics_audit/stable_volume_20260907'
sys.path.insert(0,str(OUT/'isolated_src'))

import numpy as np
from warpmpm import Solver,GridConfig,newtonian


def run(graphs):
    rng=np.random.default_rng(0)
    x=rng.random((2000,3),dtype=np.float32)*.1+.15
    s=Solver(grid=GridConfig(n_grid=48,grid_lim=.4),device='cuda:0').load_particles(
        x,np.full(len(x),1e-7,np.float32))
    s.set_material(newtonian(eta=5.,density=1000.,bulk_modulus=9e5))
    s._sim.use_cuda_graph=graphs
    s.add_domain_walls()
    dt=5e-5;substeps=12;ticks=16
    for _ in range(ticks):s.step(dt,substeps)
    st=s._sim.mpm_state
    return dict(x0=x,x=s.x(),v=s.v(),logJ=st.particle_logJ.numpy(),
                duration=dt*substeps*ticks,dt=dt,
                gravity=np.array(list(s._sim.mpm_model.gravitational_accelaration),float))


def main():
    a,b=run(True),run(False)
    for r in [a,b]:
        assert np.isfinite(r['x']).all() and np.isfinite(r['v']).all() and np.isfinite(r['logJ']).all()
        assert np.max(abs(r['logJ']))<1e-5
    position_difference=float(np.max(abs(a['x']-b['x'])))
    velocity_difference=float(np.max(abs(a['v']-b['v'])))
    assert position_difference<2e-6 and velocity_difference<1e-5
    expected_v=a['gravity']*a['duration']
    velocity_error=float(np.max(abs(a['v']-expected_v)))
    assert velocity_error<1e-5
    result=dict(gpu_pipeline_stable=True,dt_s=a['dt'],duration_s=a['duration'],
        graph_vs_live_max_position_difference_m=position_difference,
        graph_vs_live_max_velocity_difference_m_s=velocity_difference,
        maximum_freefall_velocity_error_m_s=velocity_error,
        maximum_abs_logJ=float(np.max(abs(a['logJ']))),
        note='The existing CUDA bitwise test also fails in the original solver. Its dt=2e-4 exceeds the acoustic restriction; this test uses5e-5. Original numerical differences and prototype instability are retained, not reclassified as passes.',
        liquid_data_used=[])
    (OUT/'gpu_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
