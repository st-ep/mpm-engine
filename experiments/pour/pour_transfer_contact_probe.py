"""One bounded source-contact probe using only the original 60-degree replay.

Requires independent review of the preceding fixed-contact timestep pair. This
runner consumes no measured receiver endpoint and cannot launch an angle search.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path, required=True)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--source-friction', type=float, required=True)
parser.add_argument('--grid', type=int, choices=[160, 192], default=160)
parser.add_argument('--dt-scale', type=float, choices=[1., .5], default=1.)
args = parser.parse_args()
source = args.source_dir.resolve()
sys.path.insert(0, str(source))
import numpy as np
from warpmpm.kernels import mpm_utils
from experiments.pour import pour_navier_reference as reference


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n')


def main():
    mu = args.source_friction
    if not np.isfinite(mu) or not .05 <= mu <= .2:
        raise ValueError('Contact must remain within the original [0.05, 0.20] interval')
    assert Path(mpm_utils.__file__).resolve().is_relative_to(source)
    candidate = source.parent
    parent = candidate / 'reference_protocol.json'
    assert digest(parent) == parent.with_suffix('.sha256').read_text().strip()
    protocol = json.loads(parent.read_text())
    review_path = candidate / 'independent_review.json'
    review = json.loads(review_path.read_text())
    assert review['timestep_pair_passed'] and review['frozen_input_hashes_passed']
    assert review['initial_settling_and_capture_passed']
    for row in review['cases']:
        r = candidate / 'reference_replays/replays/original-separable' / f'recorded60_bNA_n160_phase0_dt{row["dt_scale"]:g}_mu0.117000/result.json'
        assert digest(r) == row['result_sha256']
    hashes = dict(protocol['input_sha256'])
    for p in [Path(__file__), parent, review_path]:
        hashes[str(p.relative_to(ROOT))] = digest(p)
    for rel, sha in hashes.items():
        assert digest(ROOT / rel) == sha, rel
    work = candidate / f'contact_probe_mu{mu:.6f}_n{args.grid}_dt{args.dt_scale:g}'
    work.mkdir(exist_ok=False)
    write(work / 'protocol.json', dict(purpose=__doc__, input_sha256=hashes,
        eta_pa_s=reference.ETA, source_coulomb_friction=mu, initial_volume_ml=300.,
        grid=args.grid, dt_scale=args.dt_scale, recorded_angle_deg=60., contact_interval=[.05, .2],
        actual_kernel_module=str(Path(mpm_utils.__file__).resolve()),
        measured_endpoints_read=[], validation_outcomes_used=[], identification_changed=False,
        production_accuracy_accepted=False, target_search_authorized=False))
    reference.OUT = work / 'forward'
    old_project = reference.twin.project_out_of_solid
    gate = {}

    def checked_project(x, v, *positional, **keywords):
        if not gate:
            speed = np.linalg.norm(v, axis=1)
            gate.update(mean_speed_m_s=float(speed.mean()),
                        passed=bool(np.isfinite(speed).all() and speed.mean() < reference.twin.SETTLE_SPEED))
            write(work / 'settling_gate.json', gate)
            if not gate['passed']:
                raise RuntimeError('Initial settling failed; no pour will be run')
        return old_project(x, v, *positional, **keywords)

    reference.twin.project_out_of_solid = checked_project
    try:
        reference.run(SimpleNamespace(source_friction=mu, grid=args.grid, phase=0.,
            wall='original-separable', slip_mm=None, angle=None, max_angle=60.,
            dt_scale=args.dt_scale, device=args.device))
    finally:
        reference.twin.project_out_of_solid = old_project
    results = list((work / 'forward').rglob('result.json'))
    assert len(results) == 1 and gate['passed']
    result = json.loads(results[0].read_text())
    assert result['eta_pa_s'] == reference.ETA and result['source_coulomb_friction'] == mu
    assert result['particle_count'] == {160: 229280, 192: 396197}[args.grid]
    assert result['outside_ml'] <= 3. and result['tail_variation_ml'] <= .5
    for rel, sha in hashes.items():
        assert digest(ROOT / rel) == sha, rel
    result.update(actual_kernel_module=str(Path(mpm_utils.__file__).resolve()),
                  numerical_correction_protocol_sha256=digest(work / 'protocol.json'),
                  initial_settling_gate=gate, identification_changed=False)
    write(results[0], result)
    write(work / 'completion.json', dict(receiver_ml=result['receiver_ml'],
        result_sha256=digest(results[0]), receiver_endpoint_fitted=False,
        identification_changed=False, physical_robot_accuracy_established=False))


if __name__ == '__main__':
    main()
