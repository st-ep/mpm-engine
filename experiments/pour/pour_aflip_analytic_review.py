"""Review analytical evidence before any AFLIP full-pour trial."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'out/pour_physics_audit/aflip_20260907'
CASES=['compensated_position','viscous_compensated_n32',
       'viscous_aflip_n16','viscous_aflip_n32','inviscid_aflip_n16']


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    evidence=[];hashes={}
    for name in CASES:
        d=ROOT/'out/pour_physics_audit/shear_decay_20260907'/name
        data=json.loads((d/'results.json').read_text())
        protocol=data['protocol']
        for rel,expected in protocol['source_sha256'].items():
            if digest(ROOT/rel)==expected:continue
            assert rel=='experiments/pour/pour_shear_decay_benchmark.py'
            assert digest(d/f'benchmark_{expected}.py')==expected
        assert len(data['cases'])==3 and not protocol['liquid_data_used']
        errors=[r['relative_amplitude_error'] for r in data['cases']]
        ratios=[r['actual_amplitude_ratio'] for r in data['cases']]
        spread=float(np.ptp(ratios)/protocol['expected_amplitude_ratio'])
        evidence.append(dict(case=name,grid=protocol['grid'],relative_amplitude_errors=errors,
                             maximum_timestep_spread_relative=spread,
                             maximum_profile_l2=max(r['relative_profile_l2'] for r in data['cases']),
                             all_within_original_5percent_amplitude_threshold=all(abs(e)<.05 for e in errors)))
        for f in [d/'protocol.json',d/'results.json']:
            hashes[str(f.relative_to(ROOT))]=digest(f)
    indexed={r['case']:r for r in evidence}
    coarse=indexed['viscous_aflip_n16'];fine=indexed['viscous_aflip_n32'];zero=indexed['inviscid_aflip_n16']
    assert coarse['maximum_timestep_spread_relative']<.01
    assert fine['maximum_timestep_spread_relative']<.01
    assert fine['all_within_original_5percent_amplitude_threshold']
    assert max(abs(e) for e in fine['relative_amplitude_errors'])<max(abs(e) for e in coarse['relative_amplitude_errors'])
    assert max(abs(e) for e in zero['relative_amplitude_errors'])<1e-4
    result=dict(cases=evidence,exploratory_recorded_pour_test_supported=True,
        production_accuracy_accepted=False,coarse_amplitude_threshold_passed=False,
        next_step='Bounded original60 replay at fixed viscosity/contact and two timesteps; no angle search',
        alpha=1.,beta=0.,parameters_fitted=[],liquid_data_used=[],
        caveat='Coarse analytical amplitude error remains about7%; finer-grid results are not a bound on coarse full-pour error.',
        input_sha256=hashes,reviewer_sha256=digest(Path(__file__)))
    (WORK/'analytic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
