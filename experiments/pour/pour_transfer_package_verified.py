"""Replace interpolated command labels with completed, reviewed MPM checks.

Keep all model parameters frozen; refine only the 180 mL command using the
explicit MPM-only follow-up protocol. Preserve the earlier ZIP and publish the
new three-file handoff after output-integrity reviews. Keep the explicitly
documented 180 mL target-tolerance failure separate from direct-check completion.
This packager never launches a simulation or reads measured robot volumes.
"""
from pathlib import Path
import csv
import fcntl
import json
import os
import subprocess
import time
import zipfile

import numpy as np

from experiments.pour.pour_transfer_target_search import ROOT, BASE, MODEL, Search, digest

PREVIOUS = ROOT / 'out/pour_navier_calibration/philip_corrected_20260908'
DEST = ROOT / 'out/pour_navier_calibration/philip_corrected_verified_20260908'
CHECKS = BASE / 'direct_check_120_180_20260908'
FINAL = ROOT / 'final/philip_corrected_handoff.zip'
TARGETS = [60, 80, 100, 120, 140, 160, 180]


def save_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    with (CHECKS / '.package.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        package()


def package():
    previous_path = PREVIOUS / 'provenance.json'
    previous = json.loads(previous_path.read_text())
    previous_archive = PREVIOUS / 'philip_corrected_handoff.zip'
    assert digest(previous_archive) == previous['zip_sha256']
    assert digest(FINAL) == previous['zip_sha256']
    assert [r['target_ml'] for r in previous['predictions']] == TARGETS
    protocol_path = CHECKS / 'protocol.json'
    check_protocol = json.loads(protocol_path.read_text())
    assert check_protocol['maximum_direct_check_runs'] == 2
    assert check_protocol['previous_handoff_sha256'] == previous['zip_sha256']
    assert check_protocol['model_protocol_sha256'] == digest(MODEL) == previous['model_protocol_sha256']
    assert not check_protocol['parameters_changed'] and not check_protocol['measured_validation_outcomes_used']
    first_refinement_path = CHECKS / 'refine_180_protocol.json'
    first_refinement = json.loads(first_refinement_path.read_text())
    assert first_refinement['initial_direct_check_protocol_sha256'] == digest(protocol_path)
    refinement_path = CHECKS / 'refine_180_second_protocol.json'
    refinement = json.loads(refinement_path.read_text())
    assert refinement['previous_refinement_protocol_sha256'] == digest(first_refinement_path)
    assert refinement['model_protocol_sha256'] == digest(MODEL)
    assert refinement['target_ml'] == 180 and refinement['proposed_angle_deg'] == 62.73
    assert refinement['maximum_additional_runs'] == 1
    assert not refinement['physical_parameters_changed'] and not refinement['measured_validation_outcomes_used']
    release_path = CHECKS / 'experimental_release_decision.json'
    release = json.loads(release_path.read_text())
    assert release['last_refinement_protocol_sha256'] == digest(refinement_path)
    assert release['initial_protocol_sha256'] == digest(protocol_path)
    assert release['experimental_release_exception_targets_ml'] == [180]
    assert release['original_simulation_target_tolerance_ml'] == 1.
    assert not release['original_180_target_check_passed'] and not release['original_target_criterion_changed']
    for unit in ('pour-aflip-verify120-20260908', 'pour-aflip-verify180-20260908',
                 'pour-aflip-refine180-20260908', 'pour-aflip-refine180b-20260908'):
        state = subprocess.check_output(['systemctl', '--user', 'show',
            unit + '.service', '-p', 'MainPID', '-p', 'Result'], text=True)
        assert 'MainPID=0\n' in state and 'Result=success\n' in state, state
    reader = Search.__new__(Search)
    reader.model = json.loads(MODEL.read_text())
    reader.validate_model()
    points = {p['angle_deg']: p for p in previous['simulation_points']}
    for angle in (54.54, 62.50, 62.65, refinement['proposed_angle_deg']):
        points[angle] = reader.read_case(angle)
    for old_check in [*first_refinement['prior_direct_checks'], refinement['previous_direct_check']]:
        assert digest(ROOT / old_check['independent_review_path']) == old_check['independent_review_sha256']
        assert points[old_check['angle_deg']]['receiver_ml'] == old_check['receiver_ml']
    rows = []
    for old in previous['predictions']:
        angle = refinement['proposed_angle_deg'] if old['target_ml'] == 180 else old['command_angle_deg']
        point = reader.read_case(angle)
        error = point['receiver_ml'] - old['target_ml']
        tolerance_passed = abs(error) <= reader.model['target_simulation_tolerance_ml']
        if old['target_ml'] == 180:
            chosen = release['chosen_180_check']
            assert not tolerance_passed
            assert point['receiver_ml'] == chosen['receiver_ml'] and angle == chosen['angle_deg']
            assert point['review_sha256'] == chosen['review_sha256']
        else:
            assert tolerance_passed, (old['target_ml'], error)
        rows.append(dict(target_ml=old['target_ml'], command_angle_deg=angle,
            tilt_duration_s=round(angle / 10, 3), predicted_ml=point['receiver_ml'],
            simulated_target_error_ml=error, directly_verified=True,
            original_simulation_target_tolerance_passed=tolerance_passed,
            prediction_method='Completed direct MPM replay',
            result_path=point['result_path'], result_sha256=point['result_sha256'],
            review_path=point['review_path'], review_sha256=point['review_sha256']))
    points = sorted(points.values(), key=lambda p: p['angle_deg'])
    for point in points:
        assert reader.read_case(point['angle_deg'], point['kind'] == 'diagnostic') == point
    aa = np.array([p['angle_deg'] for p in points])
    vv = np.array([p['receiver_ml'] for p in points])
    assert np.isfinite(vv).all() and np.all(np.diff(aa) > 0) and np.all(np.diff(vv) > 0)
    DEST.mkdir(exist_ok=False)
    with (DEST / 'angle_table.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['target_ml', 'command_angle_deg', 'tilt_duration_s'])
        for row in rows:
            writer.writerow([row['target_ml'], f'{row["command_angle_deg"]:.2f}', f'{row["tilt_duration_s"]:.3f}'])
    for row, old in zip(rows[:-1], previous['predictions'][:-1]):
        assert row['command_angle_deg'] == old['command_angle_deg']
        assert row['tilt_duration_s'] == old['tilt_duration_s']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    lo, hi = rows[0]['command_angle_deg'], rows[-1]['command_angle_deg']
    x = np.unique(np.r_[np.linspace(lo, hi, 600), aa[(aa >= lo) & (aa <= hi)]])
    ax.plot(x, np.interp(x, aa, vv), color='#2166ac', lw=2, label='MPM prediction curve')
    ax.scatter([r['command_angle_deg'] for r in rows], [r['predicted_ml'] for r in rows],
               color='#2166ac', s=40, zorder=3, label='Direct MPM check')
    for row in rows:
        ax.annotate(f'{row["target_ml"]} mL', (row['command_angle_deg'], row['predicted_ml']),
                    xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
    ax.set(xlabel='Commanded pour angle (degrees)', ylabel='Received volume (mL)',
           title='300 mL initial fill · 2 s hold', xlim=(lo - .6, hi + .6), ylim=(48, 194))
    ax.grid(alpha=.2)
    ax.legend(loc='upper left', frameon=False)
    fig.savefig(DEST / 'angle_table.png', dpi=170)
    plt.close(fig)
    (DEST / 'START_HERE.txt').write_text('''POUR COMMANDS — ROBOT VALIDATION

Fill the source to 300 mL and empty the receiver before each run. Use the angle
and tilt-command duration in angle_table.csv. Keep the same robot program:
hold 2.0 seconds after the tilt acknowledgement, then command a 2.0-second return.
Keep the cup mounted throughout the session if possible. Record joints, action
timestamps, side video and the final received amount for every target.

All seven commands now have completed direct MPM checks. The 180 mL row now
uses 62.73 degrees and 6.273 seconds; the other six commands are unchanged.
The 180 mL command predicts 181.3 mL in MPM.
Viscosity comes from one 60-degree video;
source contact uses that same pour. Additional joint/action logs characterize
robot timing. Earlier validation amounts were not fitted to these commands.
Robot accuracy is not yet established.
''')
    names = ['angle_table.csv', 'angle_table.png', 'START_HERE.txt']
    archive = DEST / 'philip_corrected_handoff.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(DEST / name, arcname=name)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == names and z.testzip() is None
        for name in names:
            assert z.read(name) == (DEST / name).read_bytes()
    save_new(DEST / 'provenance.json', dict(
        model_protocol_sha256=digest(MODEL), direct_check_protocol_sha256=digest(protocol_path),
        mpm_only_refinement_protocol_sha256=digest(refinement_path),
        experimental_release_decision_sha256=digest(release_path),
        first_refinement_protocol_sha256=digest(first_refinement_path),
        previous_180_direct_checks=[first_refinement['prior_direct_checks'][1], refinement['previous_direct_check']],
        previous_provenance_sha256=digest(previous_path), predictions=rows, simulation_points=points,
        eta_pa_s=reader.model['eta_pa_s'], source_contact=reader.model['source_contact'],
        calibration_interpretation=reader.model['calibration_interpretation'],
        commands_changed=True, changed_command_targets_ml=[180],
        identification_changed=False, parameters_changed=False,
        original_time_check_passed=False, numerical_acceptance_passed=False,
        original_time_difference_ml=reader.model['original_time_difference_ml'],
        all_command_forward_checks_complete=True, unverified_command_targets_ml=[],
        all_command_simulation_target_checks_passed=False,
        failed_original_simulation_target_checks_ml=[180],
        original_target_criterion_changed=False,
        simulation_target_tolerance_ml=reader.model['target_simulation_tolerance_ml'],
        robot_accuracy_established=False, fluid_validation_outcomes_used=[], real_error_forecast_saved=False,
        release_status='All seven commands directly simulated; 180 mL misses original 1 mL target tolerance; experimental robot validation; original timestep criterion remains failed.',
        previous_zip_sha256=previous['zip_sha256'], zip_sha256=digest(archive),
        package_file_sha256={name: digest(DEST / name) for name in names},
        packager_sha256=digest(Path(__file__)), simulations_launched_by_packager=0,
    ))
    # Publish atomically only after the checks, preserving the earlier artifact.
    assert digest(previous_archive) == previous['zip_sha256']
    assert digest(FINAL) == previous['zip_sha256']
    temporary = FINAL.with_suffix('.zip.tmp')
    with temporary.open('xb') as stream:
        stream.write(archive.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    assert digest(temporary) == digest(archive)
    os.replace(temporary, FINAL)
    save_new(DEST / 'publication.json', dict(published_unix=time.time(), final_path=str(FINAL),
        final_sha256=digest(FINAL), old_archive_preserved=str(previous_archive),
        changed_command_targets_ml=[180], all_seven_commands_directly_checked=True))
    print(json.dumps(dict(final_zip=str(FINAL), bytes=FINAL.stat().st_size,
                         predictions=rows, previous_zip_preserved=True), indent=2))


if __name__ == '__main__':
    main()
