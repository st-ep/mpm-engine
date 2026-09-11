"""Audit the two MPM timing probes, without reading real liquid outcomes.

Compare raw simulation results with the previous timing-corrected simulations.
Changes reflect both the numerical transfer and same-pour boundary recalibration.
They are not estimates of real-robot errors; those remain chat-only diagnostics.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'out/pour_physics_audit/aflip_blend_time_20260907'
OLD = ROOT / 'out/pour_navier_calibration/measured_timing_20260907'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contact', type=float, required=True)
    parser.add_argument('--angle', type=float, choices=[48.37, 53.04], required=True)
    args = parser.parse_args()
    case = WORK / f'candidate_mu{args.contact:.6f}/timing_probe_{args.angle:.2f}'
    review_path = case / 'independent_review.json'
    if review_path.exists():
        raise FileExistsError('Preserve the existing review')
    protocol_path = case / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for rel, sha in protocol['input_sha256'].items():
        assert digest(ROOT / rel) == sha, rel
    completion = json.loads((case / 'completion.json').read_text())
    path = ROOT / completion['result_path']
    assert completion['result_sha256'] == digest(path)
    assert completion['protocol_sha256'] == digest(protocol_path)
    assert completion['initial_settling_gate']['passed']
    assert completion['actual_kernel_module'] == protocol['actual_kernel_module']
    assert Path(completion['actual_kernel_module']).resolve().is_relative_to(WORK / 'isolated_src')
    result = json.loads(path.read_text())
    assert result['eta_pa_s'] == 3.4392377844275503
    assert result['source_coulomb_friction'] == args.contact
    assert result['planned_angle_deg'] == args.angle
    assert result['n_grid']==160 and result['dt_scale']==1.
    assert result['initial_volume_ml']==300. and result['particle_count']==229280
    assert result['source_wall']=='original-separable' and result['receiver_wall']=='separable, friction=0.05'
    assert result['identification_episode']=='09-04-60-2s'
    assert not result['measured_endpoints_read_by_forward_runner'] and not result['validation_outcomes_used']
    assert not protocol['fluid_validation_outcomes_used'] and not completion['fluid_validation_outcomes_used']
    assert result['outside_ml']<=3. and result['tail_variation_ml']<=.5
    metrics = path.parent / result['simulation_episode'] / 'metrics.csv'
    with metrics.open() as stream:
        rows = list(csv.DictReader(stream))
    t = np.array([float(r['t']) for r in rows])
    counts = np.array([[int(r[k]) for k in ['n_src','n_rcv','n_air_spill']] for r in rows])
    assert len(rows)>600 and np.all(np.diff(t)>0)
    assert np.all(counts>=0) and np.all(counts.sum(1)==229280)
    v = 300.*counts[:,1]/229280
    assert abs(v[-1]-result['receiver_ml'])<1e-9
    assert abs(np.ptp(v[t>=t[-1]-.5])-result['tail_variation_ml'])<1e-9
    final = np.load(metrics.parent / 'final_n160.npz')
    assert np.isfinite(final['x']).all() and np.isfinite(final['v']).all()
    prior_path = OLD / f'replays/original-separable/planned{args.angle:g}_bNA_n160_phase0_dt1_mu0.117000/result.json'
    prior = json.loads(prior_path.read_text())
    assert prior['eta_pa_s']==result['eta_pa_s'] and prior['planned_angle_deg']==args.angle
    assert prior['timing_characterization_sha256'] == protocol['input_sha256'][str((OLD/'motion_characterization.json').relative_to(ROOT))]
    assert prior['n_grid']==160 and prior['dt_scale']==1. and prior['particle_count']==229280
    output = dict(purpose=__doc__, angle_deg=args.angle, eta_pa_s=result['eta_pa_s'],
        source_contact=args.contact, receiver_ml=result['receiver_ml'],
        previous_timing_simulation_ml=prior['receiver_ml'],
        simulated_model_change_ml=result['receiver_ml']-prior['receiver_ml'],
        outside_ml=result['outside_ml'], tail_variation_ml=result['tail_variation_ml'],
        accounting_finite_state_and_frozen_inputs_passed=True,
        initial_settling_gate=completion['initial_settling_gate'],
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            [path, protocol_path, case/'completion.json', metrics, prior_path, Path(__file__)]},
        fluid_validation_outcomes_used=[], physical_target_accuracy_established=False,
        real_error_forecast_saved=False, bulk_search_queued=False)
    with review_path.open('x') as stream:
        json.dump(output, stream, indent=2);stream.write('\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
