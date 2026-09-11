"""Analytical no-slip check of the existing physical-wall reconstruction.

The Robin length is exactly zero, as in the weak spout-edge assumption. The
existing reconstruction and material parameters are not fitted or modified.
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
parser.add_argument('--device', default='cuda:1')
args = parser.parse_args()
source = args.source_dir.resolve()
sys.path.insert(0, str(source))
from warpmpm.kernels import mpm_utils
from experiments.pour import pour_navier_film as benchmark
from experiments.pour import pour_navier_quadratic_wall as boundary


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert Path(mpm_utils.__file__).resolve().is_relative_to(source)
    manifest = source.parent / 'prototype_provenance.json'
    hashes = dict(json.loads(manifest.read_text())['prototype_sha256'])
    for p in [Path(__file__), Path(benchmark.__file__), Path(boundary.__file__), manifest]:
        hashes[str(p.relative_to(ROOT))] = digest(p)
    for rel, sha in hashes.items():
        assert digest(ROOT / rel) == sha, rel
    work = source.parent / 'quadratic_noslip_film_check'
    work.mkdir(exist_ok=False)
    trials = [(16, 0., 1.), (16, .5, 1.), (32, 0., 1.), (32, .5, 1.), (16, 0., .5), (32, 0., .5)]
    protocol = dict(purpose=__doc__, input_sha256=hashes,
        actual_kernel_module=str(Path(mpm_utils.__file__).resolve()),
        method='Existing quadratic Robin reconstruction at exactly zero slip length',
        slip_length_m=0., eta_pa_s=3.4392377844275503, extent_m=.035,
        height_m=.0109375, angle_deg=60., duration_s=.3, trials=trials,
        flux_relative_error_limit=.05, profile_relative_error_limit=.05,
        tail_change_relative_limit=.01, time_flux_relative_difference_limit=.01,
        robot_outcomes_used=[], parameters_fitted=[])
    (work / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    cases = []
    for grid, phase, scale in trials:
        benchmark.run(SimpleNamespace(extent=.035, height=.0109375, grid=grid,
            phase=phase, angle=60., eta=protocol['eta_pa_s'], method='quadratic',
            slip_mm=0., duration=.3, dt_scale=scale, production_step=True,
            device=args.device, output=work))
        path = work / f'b0_n{grid}_phase{phase:g}_dt{scale:g}/result.json'
        r = json.loads(path.read_text())
        cases.append(dict(grid=grid, phase=phase, dt_scale=scale,
            flux_ratio=r['flux_ratio'], profile_relative_l2=r['profile_relative_l2'],
            tail_change_relative=r['tail_change_relative'],
            flux_check_passed=abs(r['flux_ratio']-1) <= .05,
            profile_check_passed=r['profile_relative_l2'] <= .05,
            stationary_tail_passed=r['tail_change_relative'] <= .01,
            result_sha256=digest(path)))
    time_checks = []
    for grid in [16, 32]:
        pair = [r for r in cases if r['grid'] == grid and r['phase'] == 0.]
        delta = abs(pair[0]['flux_ratio']-pair[1]['flux_ratio'])
        time_checks.append(dict(grid=grid, difference_relative_to_exact_mean=delta, passed=delta <= .01))
    for rel, sha in hashes.items():
        assert digest(ROOT / rel) == sha, rel
    review = dict(cases=cases, timestep_checks=time_checks,
        protocol_sha256=digest(work / 'protocol.json'),
        all_gates_passed=all(r['flux_check_passed'] and r['profile_check_passed'] and r['stationary_tail_passed'] for r in cases) and all(r['passed'] for r in time_checks),
        robot_outcomes_used=[], parameters_fitted=[], identification_changed=False,
        interpretation='Analytical verification only. No full-cup model is selected by this script.')
    (work / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    main()
