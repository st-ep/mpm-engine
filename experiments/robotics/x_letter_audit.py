"""Audit selected letter-X rollouts without changing simulation or render data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.plastic_shaping_study import save_json
from experiments.robotics.x_letter_refine import OUT, check, target_mesh
from experiments.robotics.x_letter_report import render


def audit(folder, cases, target):
    check(folder)
    mesh_target = target_mesh(target)
    target_mask = render(mesh_target, top=True, mask=True)[..., :3].mean(-1) < 128
    rows = []
    for case in cases:
        path = folder / case
        record = json.loads(path.with_suffix('.json').read_text())
        data = np.load(path)
        actual = surface(data['x_after_1s'], data['vol0'], h=.00125)
        mask = render(actual, top=True, mask=True)[..., :3].mean(-1) < 128
        connected = actual.connectivity(extraction_mode='all').compute_cell_sizes(length=False, volume=False)
        areas = np.bincount(connected.cell_data['RegionId'], weights=connected.cell_data['Area'])
        penetration = np.maximum(float(data['floor']) - data['x_after_1s'][:, 2], 0.)
        phases = json.loads(path.with_suffix('.phases.json').read_text())
        turns = [i for i, p in enumerate(phases) if p['name'].endswith('rotate')]
        force = data['reaction_force'][np.isin(data['phase_id'], turns)]
        first_approach = next(i for i, p in enumerate(phases) if p['name'] == '0:approach')
        approach_force = data['reaction_force'][data['phase_id'] == first_approach]
        assert np.max(np.abs(force)) == 0.
        assert areas.max() / areas.sum() > .995
        assert int(data['inverted_count']) == 0
        error = mesh_distance_mm(actual, mesh_target)
        np.testing.assert_allclose(error, record['errors_mm'][target], rtol=1e-10)
        pressed_height = max(float(data[f'stage_{i}_pressed'][:, 2].max() - data['floor'])
                             for i in range(len(data['gaps_mm'])))
        assert pressed_height < .0455, 'Material extends above cylinder working height'
        rows.append(dict(
            file=case, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), config=record['config'],
            mean_surface_error_mm=error,
            footprint_iou=float(np.count_nonzero(mask & target_mask) / np.count_nonzero(mask | target_mask)),
            reconstruction_voxels_mm=[1., 1.25, 1.5],
            reconstruction_errors_mm=[mesh_distance_mm(surface(data['x_after_1s'], data['vol0'], h=h), mesh_target)
                                      for h in [.001, .00125, .0015]],
            finer_target_quadrature_error_mm=mesh_distance_mm(actual, mesh_target.subdivide(1, subfilter='linear')),
            connected_surface_area_fraction=float(areas.max() / areas.sum()),
            maximum_floor_penetration_mm=float(penetration.max() * 1000),
            fraction_particles_below_floor=float(np.mean(penetration > 0)),
            volume_weighted_floor_penetration_mm=float(np.average(penetration, weights=data['vol0']) * 1000),
            particle_extents_mm=(np.ptp(data['x_after_1s'], axis=0) * 1000).tolist(),
            max_pressed_height_mm=pressed_height * 1000,
            peak_rotation_contact_force_n=float(np.max(np.abs(force))),
            peak_first_approach_force_per_finger_n=float(np.linalg.norm(approach_force, axis=-1).max()),
            peak_force_per_finger_n=float(np.linalg.norm(data['reaction_force'], axis=-1).max()),
            final_rms_speed_m_s=record['rms_speed_m_s'],
            reconstructed_volume_ml=float(actual.volume * 1e6)))
    save_json(folder / 'selected_audit.json', dict(target=target, cases=rows,
        note='Fixed commands across numerical repeats. No alignment, rescaling, or particle clipping. Numerical sensitivity checks, not a convergence proof.'))
    dest = folder / 'audit_source'
    dest.mkdir(exist_ok=True)
    shutil.copy2(__file__, dest / Path(__file__).name)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--target', default='balanced')
    a = p.parse_args()
    audit(a.out, a.cases, a.target)
