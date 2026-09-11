"""Add readable command labels without changing predictions or the MPM curve.

Build a separate artifact; --publish atomically updates the final ZIP after
visual inspection. Earlier artifacts and all failed acceptance flags survive.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / 'out/pour_navier_calibration/philip_corrected_verified_20260908'
DEST = ROOT / 'out/pour_navier_calibration/philip_corrected_labeled_20260908'
FINAL = ROOT / 'final/philip_corrected_handoff.zip'
NAMES = ['angle_table.csv', 'angle_table.png', 'START_HERE.txt']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def source():
    previous = json.loads((PREVIOUS / 'provenance.json').read_text())
    assert digest(PREVIOUS / FINAL.name) == previous['zip_sha256']
    assert digest(FINAL) == previous['zip_sha256']
    for name in NAMES:
        assert digest(PREVIOUS / name) == previous['package_file_sha256'][name]
    return previous


def build():
    previous = source()
    rows = previous['predictions']
    assert [r['target_ml'] for r in rows] == [60, 80, 100, 120, 140, 160, 180]
    points = sorted(previous['simulation_points'], key=lambda p: p['angle_deg'])
    aa = np.array([p['angle_deg'] for p in points])
    vv = np.array([p['receiver_ml'] for p in points])
    assert np.isfinite(aa).all() and np.isfinite(vv).all()
    assert np.all(np.diff(aa) > 0) and np.all(np.diff(vv) > 0)
    lo, hi = rows[0]['command_angle_deg'], rows[-1]['command_angle_deg']
    visible = (aa >= lo) & (aa <= hi)
    assert aa[visible][0] == lo and aa[visible][-1] == hi
    # All intermediate raw MPM points remain in the piecewise-linear curve.
    x = np.linspace(lo, hi, 1000)
    assert np.array_equal(np.interp(x, aa[visible], vv[visible]), np.interp(x, aa, vv))
    DEST.mkdir(exist_ok=False)
    for name in ('angle_table.csv', 'START_HERE.txt'):
        (DEST / name).write_bytes((PREVIOUS / name).read_bytes())
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    color = '#2166ac'
    ax.plot(aa[visible], vv[visible], color=color, lw=1.8, marker='o',
            markersize=3, label='MPM prediction (joined runs)')
    ax.scatter([r['command_angle_deg'] for r in rows], [r['predicted_ml'] for r in rows],
               color=color, s=44, zorder=3, label='Selected targets · direct MPM')
    labels = []
    for row in rows:
        point = (row['command_angle_deg'], row['predicted_ml'])
        labels.append(ax.annotate(f'{row["command_angle_deg"]:.2f}°', point,
            xytext=(0, 25), textcoords='offset points', ha='center', va='bottom',
            fontsize=11, weight='bold', color=color))
        labels.append(ax.annotate(f'{row["target_ml"]} mL', point,
            xytext=(0, 10), textcoords='offset points', ha='center', va='bottom',
            fontsize=9, color='#333333'))
    ax.set(xlabel='Commanded pour angle (degrees)', ylabel='Received volume (mL)',
           title='300 mL initial fill · 2 s hold', xlim=(lo - .85, hi + .85), ylim=(48, 198))
    ax.set_yticks(np.arange(60, 181, 20))
    ax.grid(alpha=.2)
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    for label in labels:
        bbox = label.get_window_extent(renderer)
        assert bounds.contains(bbox.x0, bbox.y0) and bounds.contains(bbox.x1, bbox.y1)
    fig.savefig(DEST / 'angle_table.png', dpi=170)
    plt.close(fig)
    with zipfile.ZipFile(DEST / FINAL.name, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in NAMES:
            archive.write(DEST / name, arcname=name)
    updated = dict(previous)
    updated.update(previous_provenance_path=str(PREVIOUS / 'provenance.json'),
        previous_provenance_sha256=digest(PREVIOUS / 'provenance.json'),
        previous_zip_sha256=previous['zip_sha256'], zip_sha256=digest(DEST / FINAL.name),
        commands_changed=False, changed_command_targets_ml=[], plot_only_revision=True,
        plot_changes='Command angle and target volume above each selected point; all in-range raw MPM runs marked and joined linearly.',
        raw_curve_changed=False, predictions_changed=False, acceptance_criteria_changed=False,
        package_file_sha256={name: digest(DEST / name) for name in NAMES},
        packager_sha256=digest(Path(__file__)), simulations_launched_by_packager=0)
    save(DEST / 'provenance.json', updated)
    print(json.dumps({'plot': str(DEST / 'angle_table.png'), 'published': False}))


def publish():
    previous = source()
    updated = json.loads((DEST / 'provenance.json').read_text())
    assert updated['packager_sha256'] == digest(Path(__file__))
    assert updated['previous_provenance_sha256'] == digest(PREVIOUS / 'provenance.json')
    for field in ('predictions', 'simulation_points', 'eta_pa_s', 'source_contact',
                  'numerical_acceptance_passed', 'robot_accuracy_established',
                  'failed_original_simulation_target_checks_ml', 'original_time_check_passed'):
        assert updated[field] == previous[field], field
    for name in ('angle_table.csv', 'START_HERE.txt'):
        assert (DEST / name).read_bytes() == (PREVIOUS / name).read_bytes()
    archive_path = DEST / FINAL.name
    assert digest(archive_path) == updated['zip_sha256']
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == NAMES and archive.testzip() is None
        for name in NAMES:
            assert archive.read(name) == (DEST / name).read_bytes()
            assert digest(DEST / name) == updated['package_file_sha256'][name]
    temporary = FINAL.with_suffix('.zip.tmp')
    with temporary.open('xb') as stream:
        stream.write(archive_path.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    assert digest(FINAL) == previous['zip_sha256']
    assert digest(temporary) == updated['zip_sha256']
    os.replace(temporary, FINAL)
    save(DEST / 'publication.json', dict(published_unix=time.time(),
        final_path=str(FINAL), final_sha256=digest(FINAL), bytes=FINAL.stat().st_size,
        previous_archive_preserved=str(PREVIOUS / FINAL.name), commands_changed=False))
    print((DEST / 'publication.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    publish() if args.publish else build()
