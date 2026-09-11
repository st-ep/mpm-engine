"""Freeze the revised MPM and propose target checks from simulated volumes only.

The original timestep criterion remains failed; this is experimental target
verification, not a numerical-convergence certificate. No real-error forecasts
or historical liquid validation values are stored or read here.
"""
from pathlib import Path
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'out/pour_physics_audit/aflip_blend_time_20260907/candidate_mu0.272000'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def linear_proposal(points, target):
    points = sorted(points)
    if len(points)<2 or any(b[0]<=a[0] or b[1]<=a[1] for a,b in zip(points[:-1],points[1:])):
        raise ValueError('Simulation curve must increase; preserve and review any reversal')
    if target<=points[0][1]:
        a,b = points[:2]
    elif target>=points[-1][1]:
        a,b = points[-2:]
    else:
        a,b = next((a,b) for a,b in zip(points[:-1],points[1:]) if a[1]<=target<=b[1])
    angle = round(a[0]+(target-a[1])*(b[0]-a[0])/(b[1]-a[1]),2)
    if not 45.<=angle<=68.:
        raise ValueError('Proposed angle outside the frozen range')
    return angle


def main():
    path = WORK/'target_model_protocol.json'
    if path.exists():
        raise FileExistsError('Preserve the existing frozen target protocol')
    decision_path = WORK/'diagnostic_probe_decision.json'
    decision = json.loads(decision_path.read_text())
    assert decision['source_contact']==.272 and decision['eta_pa_s']==3.4392377844275503
    assert not decision['numerical_acceptance_passed'] and not decision['original_time_check_passed']
    points=[];sources=[]
    inputs = {**decision['input_sha256'],str(decision_path.relative_to(ROOT)):digest(decision_path)}
    for angle in [48.37,53.04]:
        case = WORK/f'diagnostic_timing_probe_{angle:.2f}'
        review_path = case/'independent_review.json'
        review = json.loads(review_path.read_text())
        assert review['accounting_finite_state_and_frozen_inputs_passed']
        assert review['eta_pa_s']==decision['eta_pa_s'] and review['source_contact']==.272
        assert not review['fluid_validation_outcomes_used']
        for relative,expected in review['input_sha256'].items():
            assert digest(ROOT/relative)==expected,relative
        inputs.update(review['input_sha256'])
        inputs[str(review_path.relative_to(ROOT))]=digest(review_path)
        points.append((angle,review['receiver_ml']))
        sources.append(dict(angle_deg=angle,receiver_ml=review['receiver_ml'],
            review_path=str(review_path.relative_to(ROOT)),review_sha256=digest(review_path)))
    for source in [Path(__file__),Path(__file__).with_name('pour_transfer_target_case.py'),
                   Path(__file__).with_name('pour_transfer_target_review.py')]:
        inputs[str(source.relative_to(ROOT))]=digest(source)
    for relative,expected in inputs.items():
        assert digest(ROOT/relative)==expected,relative
    model = dict(purpose=__doc__,created_unix=time.time(),eta_pa_s=decision['eta_pa_s'],
        source_contact=.272,cases=decision['cases'],input_sha256=inputs,
        simulation_only_target_verification=True,forecast_reported_in_chat_before_target_runs=True,
        numerical_acceptance_passed=False,original_time_check_passed=False,
        original_time_limit_ml=1.,original_time_difference_ml=decision['original_time_difference_ml'],
        original_criteria_changed=False,parameters_changed_after_time_check=False,
        identification_changed=False,geometry_changed=False,validation_outcomes_used=[],
        angle_bounds_deg=[45.,68.],targets_ml=[60,80,100,120,140,160,180,200],
        grid=160,dt_scale=1.,initial_volume_ml=300.,particle_count=229280,
        target_simulation_tolerance_ml=1.,maximum_new_target_runs=20,wall_budget_s=10800,
        prior_timing_campaign_unchanged=True,prior_timing_case_count=2,
        revised_model_diagnostic_case_count=2,
        maximum_combined_correction_target_cases=24,
        note='New verification campaign for the frozen revised model; earlier campaigns remain stopped and their budgets/results are not reset.',
        calibration_interpretation='One-pour weak viscosity plus same-pour numerical source-contact calibration',
        robot_accuracy_established=False,real_error_forecast_saved=False,
        source_simulations=sources)
    save_new(path,model)
    path.with_suffix('.sha256').write_text(digest(path)+'\n')
    proposals = [dict(target_ml=target,angle_deg=linear_proposal(points,target),
        method='Linear inversion/extrapolation of the two frozen MPM points; requires a direct forward check',
        directly_verified=False) for target in [60,100]]
    save_new(WORK/'initial_target_proposals.json',dict(proposals=proposals,
        source_simulations=sources,model_protocol_sha256=digest(path),
        fluid_validation_outcomes_used=[],real_error_forecast_saved=False))
    print(json.dumps(dict(model_protocol=str(path),proposals=proposals,
        maximum_new_target_runs=20,simulations_launched=0),indent=2))


if __name__=='__main__':
    main()
