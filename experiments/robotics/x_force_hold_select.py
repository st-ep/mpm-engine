"""Select force plans by identified-model numerical checks before true execution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from experiments.robotics import x_force_hold_plan as planning
from experiments.robotics.plastic_shaping_study import save_json

PROTOCOL = dict(candidates=3,
    candidates_from='Three distinct feasible four-pinch commands with lowest identified-model planning error',
    settings={'baseline': [64, .00005], 'grid80': [80, .00005]},
    selection='Lowest worst-case identified-model surface error over planning and both validation settings, with completed dwells, <=15% raw force RMS error in every dwell, and height <=45.5mm in every setting',
    repeat='One fresh baseline rerun of the selected commands in the identified model, before true execution',
    scope='Numerical model checks only; no true-material shaping results used')


def prepare(folder):
    planning.stage(folder)
    p = folder / 'selection_protocol.json'
    assert not p.exists()
    save_json(p, PROTOCOL)
    shutil.copy2(__file__, folder / 'selection_source.py')


def select(folder, material, device):
    planning.stage(folder)
    assert json.loads((folder / 'selection_protocol.json').read_text()) == PROTOCOL
    assert planning.study.digest(__file__) == planning.study.digest(folder / 'selection_source.py')
    dest = folder / 'plans' / material
    assert not (dest / 'selection_complete.json').exists()
    selected = json.loads((dest / 'selected.json').read_text())
    backup = dest / 'search_selected.json'
    if not backup.exists(): shutil.copy2(dest / 'selected.json', backup)
    rows = json.loads((dest / 'evaluations.json').read_text())
    rows += json.loads((dest / 'initialization_evaluations.json').read_text())
    unique = {}
    for r in rows:
        if r['feasible'] and len(r['peaks_n']) == 4:
            key = (tuple(r['peaks_n']), tuple(r['durations_s']))
            unique[key] = r
    candidates = sorted(unique.values(), key=lambda r: r['surface_mm'])[:PROTOCOL['candidates']]
    law = json.loads((folder / f'inputs/initialization_{material}.json').read_text())['law']
    checked = []
    for i, r in enumerate(candidates):
        checks = []
        for name, (grid, dt) in PROTOCOL['settings'].items():
            print(json.dumps(dict(starting=material, candidate=i + 1, setting=name)), flush=True)
            _, result = planning.study.rollout(folder, folder / 'model_checks' / material / name,
                law, r['peaks_n'], r['durations_s'], device, grid, dt)
            checks.append(dict(**result, numerical_setting=name))
            print(json.dumps(dict(material=material, candidate=i + 1, setting=name,
                                  surface_mm=result['surface_mm'], feasible=result['feasible'])), flush=True)
        checked.append(dict(planning=r, checks=checks,
            feasible=all(c['feasible'] for c in checks),
            worst_error_mm=max([r['surface_mm'], *[c['surface_mm'] for c in checks]])))
        save_json(dest / 'model_checks.json', checked)
    candidates = [r for r in checked if r['feasible']]
    assert candidates, 'No candidate survives identified-model numerical checks'
    best = min(candidates, key=lambda r: r['worst_error_mm'])
    baseline = next(r for r in best['checks'] if r['numerical_setting'] == 'baseline')
    a = np.load(folder / baseline['file'])
    b, repeat = planning.study.rollout(folder, folder / 'model_checks' / material / 'repeat_baseline',
        law, baseline['peaks_n'], baseline['durations_s'], device, 64, .00005)
    repeat_check = dict(surface_mm=repeat['surface_mm'], surface_error_difference_mm=abs(repeat['surface_mm'] - baseline['surface_mm']),
        final_particle_rms_difference_mm=float(np.sqrt(np.mean(np.sum((a['x_after_1s'] - b['x_after_1s']) ** 2, axis=-1))) * 1000),
        repeated_file=repeat['file'], same_stops=repeat['stops'] == baseline['stops'], feasible=repeat['feasible'])
    save_json(dest / 'repeat_check.json', repeat_check)
    assert repeat['feasible'] and repeat_check['same_stops']
    assert repeat_check['surface_error_difference_mm'] < .10, repeat_check
    assert repeat_check['final_particle_rms_difference_mm'] < .5, repeat_check
    result = dict(**best['planning'], model=selected['model'],
        initialization_evaluations=selected['initialization_evaluations'],
        optimization_evaluations=selected['optimization_evaluations'],
        prior_pilot_evaluations=selected['prior_pilot_evaluations'], prior_position_evaluations=32,
        selection_protocol_sha256=planning.study.digest(folder / 'selection_protocol.json'),
        model_validation_evaluations=len(checked) * 2 + 1,
        worst_model_error_mm=best['worst_error_mm'], numerical_checks=best['checks'], repeat_check=repeat_check)
    save_json(dest / 'selected.json', result)
    save_json(dest / 'selection_complete.json', dict(selected_sha256=planning.study.digest(dest / 'selected.json'),
        selection_source_sha256=planning.study.digest(__file__), record=result))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'select'])
    p.add_argument('--out', type=Path, default=planning.study.OUT)
    p.add_argument('--material', choices=['A', 'B'])
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.stage == 'prepare': prepare(a.out)
    else: select(a.out, a.material, a.device)
