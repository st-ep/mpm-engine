"""Replay the original 45/50 joint tracks after freezing the MPM command table.

All material, contact and geometry inputs remain from the one-60-pour model.
Other recorded joints are forward inputs only. No receiver endpoint, error
forecast or alternate identification output is read. These are retrospective
validation diagnostics, not unseen experiments and not calibration trials.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'out/pour_physics_audit/aflip_blend_time_20260907'
CANDIDATE = BASE/'candidate_mu0.272000'
PACKAGE = ROOT/'out/pour_navier_calibration/philip_corrected_20260908'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--angle', type=int, choices=[45, 50])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not args.preflight and args.angle is None:
        parser.error('--angle is required for a replay')

    sys.path.insert(0, str(BASE/'isolated_src'))
    import numpy as np
    from warpmpm.kernels import mpm_utils
    from experiments.pour import pour_transfer_recalibration_reference as reference
    from experiments.pour.pour_angle_sweep import build_motion, write_planned_episode, interp_columns

    assert Path(mpm_utils.__file__).resolve().is_relative_to(BASE/'isolated_src')

    def identity_motion(angle):
        source = ROOT/f'pouring_real_data/09-04-{angle}-2s'
        motion, episode = build_motion(source, reference.MESH)
        assert motion.reference_angle_deg == angle
        t = np.unique(np.r_[motion.sample_times(angle),
            np.linspace(motion.reference_t.min(), motion.reference_t.max(), 2001)])
        np.testing.assert_allclose(motion.reference_clock(t, angle), t, atol=1e-12, rtol=0.)
        np.testing.assert_allclose(motion.joints(t, angle),
            interp_columns(t, motion.reference_t, motion.reference_q), atol=1e-12, rtol=0.)
        return source, motion, episode

    if args.preflight:
        checks = []
        for angle in [45, 50]:
            source, motion, episode = identity_motion(angle)
            with tempfile.TemporaryDirectory(prefix='pour-recorded-validation-') as folder:
                export = Path(folder)/'episode'
                write_planned_episode(source, export, motion, episode, angle)
                replay = reference.twin.load_episode(export, reference.twin.PRE_ROLL,
                    reference.twin.HOLD_SECONDS)
                t = replay['ts']-replay['t_pour']
                np.testing.assert_allclose(replay['qs'], motion.joints(t, angle), atol=1e-8, rtol=0.)
                np.testing.assert_allclose(
                    [replay[k]-replay['t_pour'] for k in ['t_hold','t_return_start','t_return_done']],
                    motion.timing(angle), atol=1e-12, rtol=0.)
                checks.append(dict(angle_deg=angle, exact_joint_track_export=True,
                    recorded_action_times_preserved=True, exported_states=len(t)))
        print(json.dumps(dict(checks=checks, simulations_launched=0,
            actual_kernel_module=mpm_utils.__file__, receiver_endpoints_read=[]), indent=2))
        return

    # Require the complete command artifact before any extra validation replay.
    model_path = CANDIDATE/'target_model_protocol.json'
    assert digest(model_path) == model_path.with_suffix('.sha256').read_text().strip()
    model = json.loads(model_path.read_text())
    package_path = PACKAGE/'provenance.json'
    package = json.loads(package_path.read_text())
    archive = PACKAGE/'philip_corrected_handoff.zip'
    assert digest(archive) == package['zip_sha256']
    assert package['model_protocol_sha256'] == digest(model_path)
    assert [r['target_ml'] for r in package['predictions']] == model['targets_ml']
    assert all(abs(r['simulated_target_error_ml']) <= 1. for r in package['predictions'])
    assert package['source_contact'] == model['source_contact'] == .272
    assert package['eta_pa_s'] == model['eta_pa_s'] == reference.ETA == 3.4392377844275503
    assert package['fluid_validation_outcomes_used'] == model['validation_outcomes_used'] == []
    assert not model['original_time_check_passed'] and not package['numerical_acceptance_passed']
    assert time.time() < model['created_unix']+model['wall_budget_s']
    hashes = dict(model['input_sha256'])
    for row in model['cases']:
        path = ROOT/row['protocol_path']
        assert digest(path) == row['protocol_sha256']
        hashes.update(json.loads(path.read_text())['input_sha256'])
    source, motion, episode = identity_motion(args.angle)
    for path in [Path(__file__), Path(__file__).with_name('pour_transfer_recorded_validation_review.py'),
                 model_path, package_path, archive,
                 *[source/name for name in ['actions.jsonl','states.jsonl','meta.json']]]:
        hashes[str(path.relative_to(ROOT))] = digest(path)

    def validate():
        for rel, expected in hashes.items():
            assert digest(ROOT/rel) == expected, rel

    validate()
    validation_root = CANDIDATE/'recorded_validation'
    validation_root.mkdir(exist_ok=True)
    with (CANDIDATE/'target_cases/.allocation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        n_target = len(list((CANDIDATE/'target_cases').glob('angle_*')))
        n_validation = len(list(validation_root.glob('recorded_*')))
        assert n_validation < 2
        assert n_target+n_validation < model['maximum_new_target_runs']
        destination = validation_root/f'recorded_{args.angle}'
        destination.mkdir(exist_ok=False)
    protocol_path = destination/'protocol.json'
    write_new(protocol_path, dict(purpose=__doc__, input_sha256=hashes,
        simulation_source_episode=source.name, identification_episode=reference.REFERENCE.name,
        material_identification_pours=1, source_contact_calibration_episode=reference.REFERENCE.name,
        eta_pa_s=reference.ETA, source_contact=.272, initial_volume_ml=300.,
        n_grid=160, dt_scale=1., particle_count=229280,
        recorded_joint_track_replayed=True, timing_correction_applied=False,
        reason_no_timing_correction='The actual recorded joint clock already contains robot delays.',
        geometry_changed=False, identification_changed=False, contact_changed=False,
        input_motion_used_as_forward_input_only=True, receiver_endpoints_read=[],
        liquid_outcomes_used_for_calibration=[], real_error_forecast_saved=False,
        maximum_recorded_validation_runs=2, shared_case_budget=model['maximum_new_target_runs'],
        shared_deadline_unix=model['created_unix']+model['wall_budget_s'],
        command_table_frozen_before_validation=True, command_archive_sha256=digest(archive),
        numerical_acceptance_passed=False, original_time_check_passed=False,
        original_time_difference_ml=model['original_time_difference_ml'],
        actual_kernel_module=str(Path(mpm_utils.__file__).resolve())))

    old_factory = reference.planned_motion
    old_writer = reference.write_planned_episode
    old_project = reference.twin.project_out_of_solid
    gate = {}

    def recorded_factory(max_angle):
        return motion, episode

    def recorded_writer(reference_dir, output_dir, supplied_motion, supplied_episode, angle):
        assert reference_dir == reference.REFERENCE and angle == args.angle
        assert supplied_motion is motion and supplied_episode is episode
        output = write_planned_episode(source, output_dir, motion, episode, angle)
        meta_path = output_dir/'meta.json'
        meta = json.loads(meta_path.read_text())
        meta['planning'].update(method='Identity export of the original recorded joint track',
            recorded_validation_episode=source.name, spatial_path_changed=False,
            recorded_clock_changed=False, timing_correction_applied=False)
        meta_path.write_text(json.dumps(meta, indent=2)+'\n')
        replay = reference.twin.load_episode(output_dir, reference.twin.PRE_ROLL,
            reference.twin.HOLD_SECONDS)
        np.testing.assert_allclose(replay['qs'],
            motion.joints(replay['ts']-replay['t_pour'], angle), atol=1e-8, rtol=0.)
        return output

    def checked_project(x, v, *positional, **keywords):
        if not gate:
            speed = np.linalg.norm(v, axis=1)
            gate.update(mean_speed_m_s=float(speed.mean()),
                passed=bool(np.isfinite(speed).all() and speed.mean()<reference.twin.SETTLE_SPEED))
            write_new(destination/'settling_gate.json', gate)
            if not gate['passed']:
                raise RuntimeError('Initial settling failed; recorded validation did not start')
        return old_project(x, v, *positional, **keywords)

    reference.OUT = destination/'forward'
    reference.planned_motion = recorded_factory
    reference.write_planned_episode = recorded_writer
    reference.twin.project_out_of_solid = checked_project
    try:
        reference.run(SimpleNamespace(source_friction=.272, grid=160, phase=0.,
            wall='original-separable', slip_mm=None, angle=float(args.angle), max_angle=70.,
            dt_scale=1., device=args.device))
    finally:
        reference.planned_motion = old_factory
        reference.write_planned_episode = old_writer
        reference.twin.project_out_of_solid = old_project
    results = list((destination/'forward').rglob('result.json'))
    assert len(results) == 1 and gate['passed']
    result = json.loads(results[0].read_text())
    assert result['eta_pa_s'] == reference.ETA and result['source_coulomb_friction'] == .272
    assert result['identification_episode'] == '09-04-60-2s'
    assert result['outside_ml'] <= 3. and result['tail_variation_ml'] <= .5
    validate()
    write_new(destination/'completion.json', dict(receiver_ml=result['receiver_ml'],
        actual_recorded_motion=source.name, result_path=str(results[0].relative_to(ROOT)),
        result_sha256=digest(results[0]), protocol_sha256=digest(protocol_path),
        initial_settling_gate=gate, command_archive_sha256=digest(archive),
        identification_changed=False, parameters_refitted=[], receiver_endpoints_read=[],
        real_error_forecast_saved=False, numerical_acceptance_passed=False,
        original_time_check_passed=False))


if __name__ == '__main__':
    main()
