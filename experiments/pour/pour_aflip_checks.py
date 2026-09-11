"""Analytical/state and stable-timestep GPU checks for the isolated AFLIP copy."""
from pathlib import Path
import sys,json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'out/pour_physics_audit/aflip_20260907'
sys.path.insert(0,str(OUT/'isolated_src'))
import warpmpm
import numpy as np
from experiments.pour import pour_compensated_position_benchmark as state_check
from experiments.pour import pour_stable_volume_gpu_check as freefall


def main():
    assert Path(warpmpm.__file__).resolve().is_relative_to(OUT/'isolated_src')
    for device in ['cpu','cuda:0']:
        if (OUT/f'benchmark_{device.replace(":", "")}.json').exists():
            raise RuntimeError('Preserve existing checks; do not overwrite')
        state_check.OUT=OUT
        sys.argv=[sys.argv[0],'--device',device]
        state_check.main()
    a,b=freefall.run(True),freefall.run(False)
    for r in [a,b]:
        assert np.isfinite(r['x']).all() and np.isfinite(r['v']).all() and np.isfinite(r['logJ']).all()
        assert np.max(abs(r['logJ']))<1e-5
    dx=float(np.max(abs(a['x']-b['x'])))
    dv=float(np.max(abs(a['v']-b['v'])))
    expected=a['gravity']*a['duration']
    gravity_error=float(np.max(abs(a['v']-expected)))
    assert dx<2e-6 and dv<1e-5 and gravity_error<1e-5
    result=dict(gpu_pipeline_stable=True,dt_s=a['dt'],duration_s=a['duration'],
        graph_live_position_difference_m=dx,graph_live_velocity_difference_m_s=dv,
        maximum_freefall_velocity_error_m_s=gravity_error,
        actual_package=str(Path(warpmpm.__file__).resolve()),
        reused_checks=['pour_compensated_position_benchmark','pour_stable_volume_gpu_check.run'],
        liquid_data_used=[],identification_changed=False)
    (OUT/'gpu_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
