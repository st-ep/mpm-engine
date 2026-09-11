"""Audit frozen-model recorded replays without reading measured liquid amounts."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'out/pour_physics_audit/aflip_blend_time_20260907'
CANDIDATE = BASE/'candidate_mu0.272000'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--angle', type=int, choices=[45, 50], required=True)
    args = parser.parse_args()
    case = CANDIDATE/f'recorded_validation/recorded_{args.angle}'
    review_path = case/'independent_review.json'
    if review_path.exists():
        raise FileExistsError('Preserve the existing review')
    protocol_path = case/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for rel, expected in protocol['input_sha256'].items():
        assert digest(ROOT/rel) == expected, rel
    assert protocol['recorded_joint_track_replayed'] and not protocol['timing_correction_applied']
    assert not protocol['identification_changed'] and not protocol['contact_changed']
    assert not protocol['geometry_changed'] and protocol['command_table_frozen_before_validation']
    assert protocol['receiver_endpoints_read'] == protocol['liquid_outcomes_used_for_calibration'] == []
    assert not protocol['original_time_check_passed'] and not protocol['numerical_acceptance_passed']
    completion_path = case/'completion.json'
    completion = json.loads(completion_path.read_text())
    assert digest(protocol_path) == completion['protocol_sha256']
    assert completion['initial_settling_gate']['passed']
    assert not completion['parameters_refitted'] and not completion['receiver_endpoints_read']
    result_path = ROOT/completion['result_path']
    assert digest(result_path) == completion['result_sha256']
    result = json.loads(result_path.read_text())
    assert result['eta_pa_s'] == 3.4392377844275503 and result['source_coulomb_friction'] == .272
    assert result['n_grid'] == 160 and result['dt_scale'] == 1.
    assert result['particle_count'] == 229280 and result['initial_volume_ml'] == 300.
    assert result['identification_episode'] == '09-04-60-2s'
    assert not result['measured_endpoints_read_by_forward_runner'] and not result['validation_outcomes_used']
    assert result['outside_ml'] <= 3. and result['tail_variation_ml'] <= .5
    assert Path(protocol['actual_kernel_module']).resolve().is_relative_to(BASE/'isolated_src')

    export = result_path.parent/result['simulation_episode']
    source = ROOT/f'pouring_real_data/09-04-{args.angle}-2s'
    meta_path = export/'meta.json'
    meta = json.loads(meta_path.read_text())['planning']
    assert meta['recorded_validation_episode'] == source.name
    assert meta['reference_states_sha256'] == digest(source/'states.jsonl')
    assert not meta['spatial_path_changed'] and not meta['recorded_clock_changed']
    moves = [a for a in jsonl(source/'actions.jsonl')
        if a.get('type') in {'execute_trajectory','go_to_pose'}]
    index = next(i for i,a in enumerate(moves) if a.get('pour_angle_deg') == args.angle)
    pour, ret = moves[index:index+2]
    end = ret['t_ack']+2.5
    if index+2 < len(moves):
        end = min(end,moves[index+2]['t_send']-.05)
    original = [r for r in jsonl(source/'states.jsonl')
        if pour['t_send']-1.5 <= r['t'] <= end+.5]
    t = np.array([r['t']-pour['t_send'] for r in original])
    q = np.array([r['state']['joint_position'] for r in original])
    rows = jsonl(export/'states.jsonl')
    exported_t = np.array([r['t'] for r in rows])
    exported_q = np.array([r['state']['joint_position'] for r in rows])
    expected_q = np.stack([np.interp(exported_t,t,column) for column in q.T],axis=1)
    assert np.all(np.diff(t)>0) and np.all(np.diff(exported_t)>0)
    max_joint_difference = float(np.max(abs(exported_q-expected_q)))
    assert max_joint_difference < 1e-8
    synthetic = jsonl(export/'actions.jsonl')
    np.testing.assert_allclose(
        [synthetic[0]['t_ack'],synthetic[1]['t_send'],synthetic[1]['t_ack']],
        [pour['t_ack']-pour['t_send'],ret['t_send']-pour['t_send'],ret['t_ack']-pour['t_send']],
        atol=1e-12,rtol=0.)

    metrics_path = export/'metrics.csv'
    with metrics_path.open() as stream:
        rows = list(csv.DictReader(stream))
    time = np.array([float(r['t']) for r in rows])
    counts = np.array([[int(r[k]) for k in ['n_src','n_rcv','n_air_spill']] for r in rows])
    assert len(rows)>600 and np.all(np.diff(time)>0)
    assert np.all(counts>=0) and np.all(counts.sum(1)==229280)
    volume = 300.*counts[:,1]/229280
    assert abs(volume[-1]-result['receiver_ml'])<1e-9
    assert abs(np.ptp(volume[time>=time[-1]-.5])-result['tail_variation_ml'])<1e-9
    final_path = export/'final_n160.npz'
    final = np.load(final_path)
    assert np.isfinite(final['x']).all() and np.isfinite(final['v']).all()
    archive = ROOT/'out/pour_navier_calibration/philip_corrected_20260908/philip_corrected_handoff.zip'
    assert digest(archive) == protocol['command_archive_sha256'] == completion['command_archive_sha256']
    output = dict(recorded_angle_deg=args.angle, simulation_source_episode=source.name,
        identification_episode='09-04-60-2s', eta_pa_s=result['eta_pa_s'], source_contact=.272,
        receiver_ml=result['receiver_ml'], outside_ml=result['outside_ml'],
        tail_variation_ml=result['tail_variation_ml'], initial_settling_gate=completion['initial_settling_gate'],
        maximum_joint_export_difference_rad=max_joint_difference,
        original_command_type=pour['type'], recorded_action_clock_preserved=True,
        accounting_finite_state_and_frozen_inputs_passed=True,
        command_table_unchanged=True, measured_liquid_amounts_read=[], parameters_refitted=[],
        real_error_forecast_saved=False, numerical_acceptance_passed=False,
        original_time_check_passed=False, actual_robot_accuracy_claimed=False,
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),protocol_path,
            completion_path,result_path,meta_path,export/'states.jsonl',export/'actions.jsonl',
            metrics_path,final_path,source/'states.jsonl',source/'actions.jsonl',archive]})
    with review_path.open('x') as stream:
        json.dump(output,stream,indent=2)
        stream.write('\n')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
