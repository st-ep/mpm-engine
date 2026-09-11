"""Package completed MPM results after the user stopped further refinement.

Keep the original model, search, failed criteria and all outputs intact. The
two unfinished target checks are replaced by explicitly labelled interpolation
between completed simulations. This module never launches a simulation.
"""
from pathlib import Path
import csv
import fcntl
import json
import time
import zipfile

import numpy as np

from experiments.pour.pour_transfer_target_plan import linear_proposal
from experiments.pour.pour_transfer_target_search import (
    ROOT, BASE, WORK, MODEL, Search, digest,
)

TARGETS = [60, 80, 100, 120, 140, 160, 180]
DEST = ROOT / 'out/pour_navier_calibration/philip_corrected_20260908'


def save_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    with (WORK / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        package()


def package():
    # Use the existing result validator without constructing a search/controller.
    reader = Search.__new__(Search)
    reader.model = json.loads(MODEL.read_text())
    reader.validate_model()
    state_path = WORK / 'state.json'
    state = json.loads(state_path.read_text())
    points = sorted(state['points'].values(), key=lambda p: p['angle_deg'])
    for point in points:
        assert reader.read_case(point['angle_deg'], point['kind'] == 'diagnostic') == point
    aa = np.array([p['angle_deg'] for p in points])
    vv = np.array([p['receiver_ml'] for p in points])
    assert np.isfinite(aa).all() and np.isfinite(vv).all()
    assert np.all(np.diff(aa) > 0) and np.all(np.diff(vv) > 0)
    rows = []
    for target in TARGETS:
        if str(target) in state['targets']:
            row = dict(state['targets'][str(target)])
            point = next(p for p in points if p['angle_deg'] == row['command_angle_deg'])
            assert point['receiver_ml'] == row['predicted_ml']
            assert point['result_sha256'] == row['result_sha256']
            assert abs(row['simulated_target_error_ml']) <= reader.model['target_simulation_tolerance_ml']
            row.update(directly_verified=True, prediction_method='Completed direct MPM replay')
        else:
            assert target in (120, 180) and vv[0] < target < vv[-1]
            angle = linear_proposal(list(zip(aa, vv)), target)
            bracket = next((a, b) for a, b in zip(points[:-1], points[1:])
                           if a['angle_deg'] < angle < b['angle_deg'])
            predicted = float(np.interp(angle, aa, vv))
            row = dict(target_ml=target, command_angle_deg=angle,
                       tilt_duration_s=round(angle / 10, 3), predicted_ml=predicted,
                       simulated_target_error_ml=None,
                       interpolated_target_residual_ml=predicted - target,
                       directly_verified=False,
                       prediction_method='Piecewise linear interpolation between completed MPM replays',
                       source_simulations=list(bracket),
                       reason_no_direct_check='User requested no final refinement runs.')
        rows.append(row)
    assert [r['target_ml'] for r in rows if not r['directly_verified']] == [120, 180]
    old = ROOT / 'out/pour_navier_calibration/philip_pilot_20260907_200'
    old_provenance = json.loads((old / 'provenance.json').read_text())
    assert digest(old / 'philip_pilot_handoff.zip') == old_provenance['zip_sha256']
    scope_path = WORK / 'no_refinement_through_180_scope.json'
    save_new(scope_path, dict(
        created_unix=time.time(), requested_targets_ml=TARGETS,
        user_instruction="don't do final refiment, just skip 200 mL",
        scope_interpretation='Stop remaining target refinements; package completed MPM curve through 180 mL.',
        model_protocol_sha256=digest(MODEL), search_state_sha256=digest(state_path),
        physical_parameters_changed=False, identification_changed=False,
        original_acceptance_criteria_changed=False,
        incomplete_refinements_cancelled_deg=[54.54, 62.50],
        queued_200_refinement_not_launched_deg=65.64,
        original_deadline_unix=reader.model['created_unix'] + reader.model['wall_budget_s'],
        simulations_launched=0, liquid_validation_outcomes_used=[],
        unfinished_checks_claimed_as_passed=False,
        selection_rule='Retain previously verified rows; otherwise use the unchanged MPM-only linear inverse.',
    ))
    DEST.mkdir(exist_ok=False)
    with (DEST / 'angle_table.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['target_ml', 'command_angle_deg', 'tilt_duration_s'])
        for row in rows:
            writer.writerow([row['target_ml'], f'{row["command_angle_deg"]:.2f}', f'{row["tilt_duration_s"]:.3f}'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    # Use the same piecewise linear curve for both interpolation and plotting.
    lo, hi = rows[0]['command_angle_deg'], rows[-1]['command_angle_deg']
    x = np.unique(np.r_[np.linspace(lo, hi, 600), aa[(aa >= lo) & (aa <= hi)]])
    ax.plot(x, np.interp(x, aa, vv), color='#2166ac', lw=2, label='MPM prediction curve')
    for direct, marker, label in [(True, 'o', 'Direct MPM check'),
                                  (False, 'D', 'Interpolated MPM command')]:
        subset = [r for r in rows if r['directly_verified'] == direct]
        ax.scatter([r['command_angle_deg'] for r in subset], [r['predicted_ml'] for r in subset],
                   marker=marker, s=40, edgecolors='#2166ac',
                   facecolors='#2166ac' if direct else 'white', zorder=3, label=label)
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

The 120 and 180 mL commands interpolate completed MPM simulations; their direct
checks were not completed. The other five commands have direct MPM checks.
Viscosity comes from one 60-degree video; source contact uses that same pour.
Earlier validation amounts were not used to fit these commands. Robot accuracy
is not yet established.
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
        model_protocol_sha256=digest(MODEL), scope_revision_path=str(scope_path.relative_to(ROOT)),
        scope_revision_sha256=digest(scope_path), predictions=rows, simulation_points=points,
        eta_pa_s=reader.model['eta_pa_s'], source_contact=reader.model['source_contact'],
        calibration_interpretation=reader.model['calibration_interpretation'],
        original_time_check_passed=False, numerical_acceptance_passed=False,
        original_time_difference_ml=reader.model['original_time_difference_ml'],
        all_command_forward_checks_complete=False, unverified_command_targets_ml=[120, 180],
        robot_accuracy_established=False,
        release_status='Experimental commands; two interpolated commands; original timestep criterion remains failed.',
        fluid_validation_outcomes_used=[], real_error_forecast_saved=False,
        previous_zip_sha256=old_provenance['zip_sha256'], zip_sha256=digest(archive),
        package_file_sha256={name: digest(DEST / name) for name in names},
        packager_sha256=digest(Path(__file__)), simulations_launched=0,
    ))
    reader.validate_model()
    assert digest(state_path) == json.loads(scope_path.read_text())['search_state_sha256']
    assert digest(old / 'philip_pilot_handoff.zip') == old_provenance['zip_sha256']
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                         predictions=rows, simulations_launched=0), indent=2))


if __name__ == '__main__':
    main()
