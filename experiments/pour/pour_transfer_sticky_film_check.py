"""Check the unchanged sticky boundary with an isolated transfer candidate.

Uses the existing analytical film benchmark unchanged. No cup recording or
receiver endpoint is loaded. Its 2/1 mm spacings and 8 mm film are independent
test cases, not a claim to resolve the real spout at production spacing.
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
from experiments.pour import pour_film_benchmark as benchmark


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert Path(mpm_utils.__file__).resolve().is_relative_to(source)
    manifest = source.parent / 'prototype_provenance.json'
    provenance = json.loads(manifest.read_text())
    hashes = dict(provenance['prototype_sha256'])
    for p in [Path(__file__), Path(benchmark.__file__), manifest]:
        hashes[str(p.relative_to(ROOT))] = digest(p)
    for rel, sha in hashes.items():
        assert digest(ROOT / rel) == sha, rel
    work = source.parent / 'sticky_film_check'
    work.mkdir(exist_ok=False)
    protocol = dict(purpose=__doc__, actual_kernel_module=str(Path(mpm_utils.__file__).resolve()),
        input_sha256=hashes, eta_pa_s=3.4392377844275503, height_m=.008,
        extent_m=.032, angle_deg=60., grids=[16, 32], phases=[0., .5],
        duration_s=.2, surface='sticky', collider='sdf', band_cells=.5,
        flux_relative_error_limit=.05, profile_relative_error_limit=.05,
        tail_change_relative_limit=.01, liquid_data_used=[], parameters_fitted=[])
    (work / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    cases = []
    for grid in protocol['grids']:
        for phase in protocol['phases']:
            benchmark.run(SimpleNamespace(grid=grid, phase=phase, angle=60.,
                eta=protocol['eta_pa_s'], duration=.2, device=args.device,
                surface='sticky', collider='sdf', band_cells=.5, output=work))
            path = work / f'sdf_sticky_n{grid}_phase{phase:g}_band0.5/result.json'
            result = json.loads(path.read_text())
            cases.append(dict(grid=grid, phase=phase,
                flux_ratio=result['flux_ratio_to_noslip'],
                profile_relative_l2=result['profile_relative_l2'],
                tail_change_relative=result['tail_change_relative'],
                flux_check_passed=abs(result['flux_ratio_to_noslip']-1) <= protocol['flux_relative_error_limit'],
                profile_check_passed=result['profile_relative_l2'] <= protocol['profile_relative_error_limit'],
                stationary_tail_passed=result['tail_change_relative'] <= protocol['tail_change_relative_limit'],
                result_sha256=digest(path)))
    for rel, sha in hashes.items():
        assert digest(ROOT / rel) == sha, rel
    review = dict(cases=cases, protocol_sha256=digest(work / 'protocol.json'),
        all_gates_passed=all(r['flux_check_passed'] and r['profile_check_passed'] and r['stationary_tail_passed'] for r in cases),
        parameters_fitted=[], liquid_data_used=[],
        interpretation='Boundary verification only; this test does not select a full-cup boundary or calibrate an endpoint.')
    (work / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    main()
