"""Audit frozen force-pulse executions and plot commanded/measured forces."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from experiments.robotics import x_force_pulse_plan as planning
from experiments.robotics import x_force_contact_control as control
from experiments.robotics.plastic_shaping_study import save_json
from experiments.robotics.plastic_shaping_figure import surface, COLORS
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.x_force_report import render
from experiments.robotics.x_shaping_franka import audit as robot_audit


def audit(folder):
    planning.stage(folder)
    laws = json.loads((folder / 'inputs/models.json').read_text())
    target = pv.read(folder / 'target.vtp')
    assert target.n_open_edges == 0
    np.testing.assert_allclose(target.volume, .00009, rtol=2e-6)
    rows = []
    for setting, (grid, dt) in planning.study.SETTINGS.items():
        for material in 'AB':
            for planned in 'AB':
                path = folder / f'{setting}_{material}_plan_{planned}.npz'
                d = np.load(path)
                record = json.loads(path.with_suffix('.json').read_text())
                cfg = json.loads(path.with_suffix('.config.json').read_text())
                phases = json.loads(path.with_suffix('.phases.json').read_text())
                selected = json.loads((folder / f'plans/{planned}/selected.json').read_text())
                assert cfg['law'] == laws[f'true_{material}']
                assert cfg['peaks_n'] == selected['peaks_n']
                np.testing.assert_array_equal(cfg['durations_s'], selected['durations_s'])
                assert cfg['n_grid'] == grid and cfg['dt'] == dt
                assert record['data_sha256'] == planning.study.digest(path)
                assert record['material'] == material and record['planned_for'] == planned
                for key in d.files: assert np.all(np.isfinite(d[key])), (path, key)
                assert int(d['inverted_count']) == 0
                np.testing.assert_allclose(d['vol0'].sum(dtype=float), .00009, rtol=1e-6)
                np.testing.assert_allclose(np.diff(d['time']), .004, atol=1e-10)
                normal = d['tool_centers'][:, 1] - d['tool_centers'][:, 0]
                distance = np.linalg.norm(normal, axis=1)
                normal /= distance[:, None]
                force = np.einsum('tfc,tc->tf', d['reaction_force'], normal) * [-1, 1]
                np.testing.assert_allclose(force, d['force_per_finger'], atol=1e-10)
                measured = force.mean(axis=1)
                alpha = 1 - np.exp(-.004 / control.CONTROL['filter_s'])
                filtered = np.empty_like(measured); previous = 0.
                for j, f in enumerate(measured):
                    previous += alpha * (f - previous); filtered[j] = previous
                np.testing.assert_allclose(filtered, d['filtered_force'], atol=1e-10)
                previous_filtered = np.r_[0., filtered[:-1]]
                active = d['pulse_id'] >= 0
                np.testing.assert_allclose(d['force_error'][active],
                    (d['target_force'] - previous_filtered)[active], atol=1e-10)
                speed = np.linalg.norm(d['tool_velocity'][active], axis=-1)
                assert speed.max() <= .050000001
                gaps = (distance - .028) * 1000
                assert gaps.min() >= 12. - 1e-7 and gaps.max() <= 72. + 1e-7
                pulse_audits = []
                for i in range(4):
                    indices = np.flatnonzero(d['pulse_id'] == i)
                    p = next(p for p in phases if p['name'] == f'{i}:force_pulse')
                    times = d['time'][indices] - p['start_s'] - .002
                    targets = np.array([control.pulse(t, d['peaks_n'][i], d['durations_s'][i]) for t in times])
                    np.testing.assert_allclose(targets, d['target_force'][indices], atol=1e-9)
                    if record['stops'][i] == 'duration':
                        assert len(indices) == round(d['durations_s'][i] / .004)
                    else:
                        assert record['stops'][i] == 'travel_guard'
                        assert gaps[indices[-1]] <= 12. + 1e-7
                        assert np.all(gaps[indices[:-1]] > 12. - 1e-8)
                    v = -np.einsum('tfc,tc->tf', d['tool_velocity'][indices], normal[indices]) * [-1, 1]
                    np.testing.assert_allclose(v[:, 0], v[:, 1], atol=1e-9)
                    delta_v = np.abs(np.diff(np.r_[0., v[:, 0]]))
                    bounded = delta_v[:-1] if record['stops'][i] == 'travel_guard' else delta_v
                    assert bounded.max() <= .5 * .004 + 1e-9
                    j = next(k for k, phase in enumerate(phases) if phase['name'] == f'{i}:engage')
                    contact = np.flatnonzero(d['phase_id'] == j)
                    assert len(contact) >= 3 and np.all(filtered[contact[-3:]] >= .05)
                    actual = measured[indices]
                    pulse_audits.append(dict(pinch=i + 1,
                        commanded_peak_n=float(d['peaks_n'][i]), duration_s=float(d['durations_s'][i]),
                        executed_duration_s=len(indices) * .004, min_gap_mm=float(gaps[indices].min()),
                        measured_peak_n=float(actual.max()), measured_min_n=float(actual.min()),
                        force_relative_l2_percent=float(100 * np.linalg.norm(actual - targets) / np.linalg.norm(targets)),
                        per_finger_disagreement_rms_n=float(np.sqrt(np.mean(np.diff(force[indices], axis=1) ** 2))),
                        speed_limited_fraction=float(d['speed_limited'][indices].mean()),
                        final_tick_velocity_change_m_s=float(delta_v[-1]),
                        acceleration_scope='Servo updates bounded at 0.5 m/s²; hard travel stops and transitions between motion phases are kinematic'))
                rotations = [i for i, p in enumerate(phases) if p['name'].endswith('rotate')]
                assert np.max(np.abs(d['reaction_force'][np.isin(d['phase_id'], rotations)])) == 0
                assert np.max(np.abs(d['reaction_force'][d['phase_id'] == len(phases) - 1])) == 0
                mesh = surface(d['x_after_1s'], d['vol0'], h=.00125)
                error = mesh_distance_mm(mesh, target)
                np.testing.assert_allclose(error, record['surface_mm'], rtol=1e-10)
                for top in [False, True]:
                    mask = render(mesh, mask=True, top=top)[..., :3].mean(-1) < 128
                    assert not mask[[0, -1], :].any() and not mask[:, [0, -1]].any()
                connected = mesh.connectivity(extraction_mode='all').compute_cell_sizes(length=False, volume=False)
                areas = np.bincount(connected.cell_data['RegionId'], weights=connected.cell_data['Area'])
                penetration = np.maximum(float(d['floor']) - d['x_after_1s'][:, 2], 0)
                row = dict(**record, pulses_audit=pulse_audits, surface_components=len(areas),
                    largest_surface_area_fraction=float(areas.max() / areas.sum()),
                    max_final_floor_penetration_mm=float(penetration.max() * 1000),
                    reconstruction_voxels_mm=[1., 1.25, 1.5],
                    reconstruction_errors_mm=[mesh_distance_mm(surface(d['x_after_1s'], d['vol0'], h=h), target)
                                              for h in [.001, .00125, .0015]],
                    finer_quadrature_error_mm=mesh_distance_mm(mesh, target.subdivide(1, subfilter='linear')))
                if material == planned:
                    assert record['feasible'] and 'travel_guard' not in record['stops']
                rows.append(row)
    # Commands and physical initial specimens are identical across true materials.
    for setting in planning.study.SETTINGS:
        for planned in 'AB':
            a, b = [np.load(folder / f'{setting}_{n}_plan_{planned}.npz') for n in 'AB']
            for key in ['initial', 'vol0', 'peaks_n', 'durations_s']:
                np.testing.assert_array_equal(a[key], b[key])
    summary = dict(results=rows, executions=len(rows),
        scope='Supplied-state identification and ideal simulated force sensing; force tracking and numerical sensitivity, not hardware validation or convergence')
    save_json(folder / 'summary.json', summary)
    save_json(folder / 'additional_audit.json', summary)
    shutil.copy2(__file__, folder / 'audit_source.py')
    robot_audit(folder, check_speeds=True)
    print(json.dumps([dict(material=r['material'], planned_for=r['planned_for'], setting=r['setting'],
                            surface_mm=r['surface_mm'], stops=r['stops']) for r in rows], indent=2), flush=True)


def diagnostics(folder):
    fig, axes = plt.subplots(4, 4, figsize=(13, 9))
    for row, (material, planned) in enumerate((a, b) for a in 'AB' for b in 'AB'):
        d = np.load(folder / f'baseline_{material}_plan_{planned}.npz')
        phases = json.loads((folder / f'baseline_{material}_plan_{planned}.phases.json').read_text())
        for i, ax in enumerate(axes[row]):
            mask = d['pulse_id'] == i
            start = next(p['start_s'] for p in phases if p['name'] == f'{i}:force_pulse')
            t = d['time'][mask] - start
            ax.plot(t, d['target_force'][mask], '--', color='black', lw=1, label='Command')
            ax.plot(t, d['force_per_finger'][mask].mean(axis=1), color=COLORS[material], lw=1, label='Measured')
            ax.set_title(f'Pinch {i + 1}', fontsize=9)
            if i == 0: ax.set_ylabel(f'True {material}, plan {planned}\nForce per finger (N)')
            if row == 3: ax.set_xlabel('Time after contact (s)')
            ax.grid(alpha=.15)
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(folder / 'force_tracking_diagnostics.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['audit', 'diagnostics'])
    p.add_argument('--out', type=Path, default=planning.study.OUT)
    a = p.parse_args()
    if a.stage == 'audit': audit(a.out)
    else: diagnostics(a.out)
