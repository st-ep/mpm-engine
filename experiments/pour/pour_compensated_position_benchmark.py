"""Analytical translation and state checks; no pouring outcomes or parameter fit."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'out/pour_physics_audit/compensated_position_20260907'
sys.path.insert(0, str(OUT / 'isolated_src'))

import numpy as np
import warp as wp
from warpmpm import Solver, GridConfig, newtonian
from warpmpm.kernels import mpm_utils
from warpmpm.kernels.mpm_utils import compensated_position_step
from warpmpm.kernels.warp_utils import MPMStateStruct


@wp.kernel
def translate(state: MPMStateStruct, velocities: wp.array(dtype=wp.vec3),
              dt: float, steps: int):
    p = wp.tid()
    for _ in range(steps):
        state.particle_x[p] = compensated_position_step(state, p, dt, velocities[p])


def scene(x, device):
    return Solver(grid=GridConfig(n_grid=16, grid_lim=.4), device=device).load_particles(
        x, np.full(len(x), 1e-9, np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    assert str(Path(mpm_utils.__file__).resolve()).startswith(str(OUT / 'isolated_src'))
    manifest = json.loads((OUT / 'prototype_provenance.json').read_text())
    for relative, expected in manifest['prototype_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected

    speeds = np.array([.0001,.0003,.001,.003,.01,.03], np.float32)
    speeds = np.concatenate([speeds, -speeds])
    x0 = np.tile(np.array([.1,.2,.3], np.float32), (len(speeds), 1))
    v = np.repeat(speeds[:, None], 3, axis=1)
    velocities = wp.array(v, dtype=wp.vec3, device=args.device)
    rows = []
    for scale in [1., .5, .25]:
        steps = int(763 * 12 / scale)
        dt = float(np.float32((1/60) / (763/scale)))
        s = scene(x0, args.device)
        st = s._sim.mpm_state
        wp.launch(translate, dim=len(x0), inputs=[st, velocities, dt, steps], device=args.device)
        actual = s.x().astype(float)
        expected = x0.astype(float) + dt * steps * v.astype(float)
        error = float(np.max(abs(actual - expected)))
        assert error < 3e-8, (scale, error)
        residual = st.particle_x_comp.numpy().copy()
        ptrs = (st.particle_x.ptr, st.particle_x_comp.ptr)
        s.set_x(s.x())
        np.testing.assert_array_equal(st.particle_x_comp.numpy(), residual)
        # CPU exports can share storage; make an external edit as GPU exports do.
        edited = s.x().copy()
        edited[::2, 1] += np.float32(.001)
        s.set_x(edited)
        reset_expected = residual.copy()
        reset_expected[::2, 1] = 0
        np.testing.assert_array_equal(st.particle_x_comp.numpy(), reset_expected)
        assert (st.particle_x.ptr, st.particle_x_comp.ptr) == ptrs
        rows.append(dict(dt_scale=scale, duration_s=dt*steps,
                         max_position_error_m=error,
                         actual_displacement_m=(actual-x0.astype(float)).tolist(),
                         expected_displacement_m=(expected-x0.astype(float)).tolist()))

    # Repeated no-op projection imports must preserve sub-ULP motion.
    s = scene(x0, args.device)
    dt = float(np.float32((1/60)/(763/.5)))
    substeps = 1526
    for _ in range(12):
        wp.launch(translate, dim=len(x0), inputs=[s._sim.mpm_state, velocities, dt, substeps], device=args.device)
        s.set_x(s.x())
    repeated_error = float(np.max(abs(s.x().astype(float)-x0.astype(float)-v.astype(float)*dt*substeps*12)))
    assert repeated_error < 3e-8

    # Particle reordering must carry both extra fields without replacing buffers.
    st = s._sim.mpm_state
    n = len(x0)
    logj = np.linspace(-.01, .01, n, dtype=np.float32)
    wp.copy(st.particle_logJ, wp.array(logj, dtype=float, device=args.device))
    before = np.column_stack([s.x(), st.particle_x_comp.numpy(), st.particle_logJ.numpy()])
    ptrs = (st.particle_x.ptr, st.particle_x_comp.ptr, st.particle_logJ.ptr)
    perm = np.arange(n-1, -1, -1, dtype=np.int32)
    s._sim.permute_particles(perm, device=args.device)
    after = np.column_stack([s.x(), st.particle_x_comp.numpy(), st.particle_logJ.numpy()])
    np.testing.assert_array_equal(after, before[perm])
    assert ptrs == (st.particle_x.ptr, st.particle_x_comp.ptr, st.particle_logJ.ptr)

    # Exercise the actual G2P and fused/split paths at a stable timestep.
    trajectories = []
    for fused in [False, True]:
        rng = np.random.default_rng(144)
        x = rng.uniform(.13,.21,(128,3)).astype(np.float32)
        s = scene(x, args.device)
        s.set_material(newtonian(eta=3.4392377844275503, density=1260., bulk_modulus=9e5))
        s.fused = fused
        for _ in range(4):
            s.step(2e-5, 12)
        assert np.isfinite(s.x()).all() and np.isfinite(s.v()).all()
        trajectories.append((s.x(), s.v()))
    dx = float(np.max(abs(trajectories[0][0]-trajectories[1][0])))
    dv = float(np.max(abs(trajectories[0][1]-trajectories[1][1])))
    assert dx < 2e-6 and dv < 1e-5
    result = dict(device=args.device, translation=rows, position_error_limit_m=3e-8,
                  no_op_import_max_position_error_m=repeated_error,
                  edits_reset_only_changed_coordinates=True,
                  state_permutation_and_pointer_checks_pass=True,
                  fused_split_position_difference_m=dx,
                  fused_split_velocity_difference_m_s=dv,
                  all_passed=True, liquid_data_used=[],
                  identification_changed=False,
                  prototype_manifest_sha256=hashlib.sha256((OUT/'prototype_provenance.json').read_bytes()).hexdigest())
    destination = OUT / ('benchmark_' + args.device.replace(':','') + '.json')
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'translation'}, indent=2))


if __name__ == '__main__':
    main()
