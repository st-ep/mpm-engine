"""Audit analytical evidence for the two predeclared velocity blends."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'out/pour_physics_audit'
CASES = ['viscous_aflip_fixed_n16', 'viscous_aflip_time_n16',
         'viscous_aflip_time_n32', 'inviscid_aflip_time_n16']


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    hashes, evidence = {}, []
    for name in CASES:
        folder = AUDIT / 'shear_decay_20260907' / name
        data = json.loads((folder / 'results.json').read_text())
        protocol = data['protocol']
        assert len(data['cases']) == 3 and not protocol['liquid_data_used']
        for rel, expected in protocol['source_sha256'].items():
            assert digest(ROOT / rel) == expected, rel
        errors = np.array([r['relative_amplitude_error'] for r in data['cases']])
        ratios = np.array([r['actual_amplitude_ratio'] for r in data['cases']])
        spread = float(np.ptp(ratios) / protocol['expected_amplitude_ratio'])
        evidence.append(dict(case=name, relative_amplitude_errors=errors.tolist(),
            timestep_spread_relative=spread,
            amplitude_check_passed=bool(np.max(abs(errors)) <= protocol['relative_amplitude_error_threshold']),
            timestep_check_passed=spread <= protocol['timestep_amplitude_difference_threshold']))
        for path in [folder / 'protocol.json', folder / 'results.json']:
            hashes[str(path.relative_to(ROOT))] = digest(path)
    by_name = {e['case']: e for e in evidence}
    time_coarse = by_name['viscous_aflip_time_n16']
    time_fine = by_name['viscous_aflip_time_n32']
    inviscid = by_name['inviscid_aflip_time_n16']
    assert time_coarse['timestep_check_passed']
    assert time_fine['timestep_check_passed'] and time_fine['amplitude_check_passed']
    assert inviscid['timestep_check_passed'] and inviscid['amplitude_check_passed']
    work = AUDIT / 'aflip_blend_time_20260907'
    for name in ['benchmark_cpu.json', 'benchmark_cuda0.json', 'gpu_check.json']:
        data = json.loads((work / name).read_text())
        assert data.get('all_passed', data.get('gpu_pipeline_stable', False))
        hashes[str((work / name).relative_to(ROOT))] = digest(work / name)
    result = dict(cases=evidence, input_sha256=hashes,
        fixed_blend_rejected_for_timestep_spread=not by_name['viscous_aflip_fixed_n16']['timestep_check_passed'],
        bounded_settling_check_supported=True,
        exploratory_full_pour_requires_settling_pass=True,
        coarse_amplitude_threshold_passed=time_coarse['amplitude_check_passed'],
        production_accuracy_accepted=False, robot_accuracy_established=False,
        physical_parameters_refitted=[], liquid_data_used=[], identification_changed=False,
        caveat='Coarse analytical amplitude error exceeds 5%. Fine-grid analytical success does not bound coarse full-pour error. Blending adds numerical damping.',
        reviewer_sha256=digest(Path(__file__)))
    with (work / 'analytic_review.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
