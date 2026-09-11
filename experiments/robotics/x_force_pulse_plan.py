"""Plan and validate contact-triggered force pulses, using identified models only.

The pilot and its controller remain frozen in x_force_contact_study. This stage
first finds pulse durations that approximate each identified model's existing
position-plan openings, then optimizes force/time commands against the X target.
Opening distances are initialization objectives only; execution uses force/time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import minimize

from experiments.robotics import x_force_contact_study as study
from experiments.robotics.plastic_shaping_study import save_json

PLAN_PROTOCOL = dict(
    initial_force_factor='Smallest pilot force factor with a trial reaching the model reference opening; otherwise the feasible pilot nearest that opening; identical A/B rule',
    duration_initialization='Seven bisection trials after endpoint trials per pinch, stopping within 0.5 mm of the model reference opening; retain closest feasible prefix',
    reference='The earlier identified-model position plan, not true-material outcomes',
    force_bounds_n=[.05, 14.], duration_bounds_s=[.2, 2.4],
    search='Bounded Nelder-Mead over four forces and four durations, normalized by the initialized values',
    max_evaluations=48, simplex_fraction=.05,
    objective='Symmetric area-weighted 3D surface error against the unchanged X, 1 s after full withdrawal',
    feasibility='No opening-guard activation or sampled height above 45.5 mm',
    selection='Lowest feasible identified-model error among full four-pinch calibration and search evaluations',
    scope='No online shape feedback; no tuning or selection on true-material executions')


def stage(folder):
    study.check(folder)
    path = folder / 'planning_protocol.json'
    if not path.exists():
        save_json(path, PLAN_PROTOCOL)
        shutil.copy2(__file__, folder / 'planning_source.py')
    assert json.loads(path.read_text()) == PLAN_PROTOCOL
    assert study.digest(folder / 'planning_source.py') == study.digest(__file__)


def trial(folder, material, peaks, durations, device, label):
    initial = json.loads((folder / f'inputs/initialization_{material}.json').read_text())
    print(json.dumps(dict(starting=material, label=label, peaks_n=list(peaks), durations_s=list(durations))), flush=True)
    t = time.monotonic()
    data, row = study.rollout(folder, folder / 'plans' / material, initial['law'], peaks, durations, device)
    gap = row['diagnostics']['pulses'][-1]['min_gap_mm']
    print(json.dumps(dict(material=material, label=label, surface_mm=row['surface_mm'],
                          feasible=row['feasible'], gaps_mm=[p['min_gap_mm'] for p in row['diagnostics']['pulses']],
                          elapsed_s=time.monotonic() - t)), flush=True)
    return row, gap


def initialize(folder, material, device):
    stage(folder)
    dest = folder / 'plans' / material; dest.mkdir(parents=True, exist_ok=True)
    completed = dest / 'initialized.json'
    if completed.exists(): return json.loads(completed.read_text())
    initial = json.loads((folder / f'inputs/initialization_{material}.json').read_text())
    pilot_rows = json.loads((folder / f'pilot/{material}/evaluations.json').read_text())
    reaching = [r for r in pilot_rows if r['diagnostics']['pulses'][0]['min_gap_mm'] <= initial['gaps_mm'][0]]
    pilot = (min(reaching, key=lambda r: r['force_factor']) if reaching else
             json.loads((folder / f'pilot/{material}/selected.json').read_text()))
    peaks = np.minimum(np.array(initial['reference_peak_n']) * pilot['force_factor'], 14.)
    durations, all_rows, decisions = [], [], []
    for i in range(4):
        goal = initial['gaps_mm'][i]
        candidates = []
        low, high = .2, 2.4
        for k in range(9):
            duration = low if k == 0 else high if k == 1 else (low + high) / 2
            duration = round(duration / .004) * .004
            row, gap = trial(folder, material, peaks[:i + 1], durations + [duration], device, f'initialize_{i + 1}_{k + 1}')
            all_rows.append(row); candidates.append((abs(gap - goal), row, gap))
            save_json(dest / 'initialization_evaluations.json', all_rows)
            if row['feasible'] and abs(gap - goal) <= .5: break
            if k == 0: continue
            if gap > goal: low = duration
            else: high = duration
            if high - low <= .004: break
        feasible = [r for r in candidates if r[1]['feasible']]
        assert feasible, f'No feasible duration for {material} pinch {i + 1}'
        _, best, gap = min(feasible, key=lambda r: r[0])
        durations.append(best['durations_s'][-1])
        decisions.append(dict(pinch=i + 1, target_model_gap_mm=goal, actual_model_gap_mm=gap,
                              peaks_n=best['peaks_n'], durations_s=best['durations_s'], file=best['file']))
    result = dict(peaks_n=peaks.tolist(), durations_s=durations, decisions=decisions,
                  model=initial['model'], law=initial['law'], evaluations=len(all_rows),
                  source_pilot=pilot['file'], source_force_factor=pilot['force_factor'])
    save_json(completed, result)
    return result


def plan(folder, material, device):
    stage(folder)
    initial = initialize(folder, material, device)
    dest = folder / 'plans' / material
    if (dest / 'selected.json').exists(): return
    scale = np.r_[initial['peaks_n'], initial['durations_s']]
    bounds = np.array([PLAN_PROTOCOL['force_bounds_n']] * 4 + [PLAN_PROTOCOL['duration_bounds_s']] * 4)
    bounds = bounds / scale[:, None]
    rows = []
    def objective(relative):
        values = relative * scale
        try:
            row, _ = trial(folder, material, values[:4], values[4:], device, f'optimize_{len(rows) + 1}')
            penalty = 100 * row['stops'].count('travel_guard')
            penalty += 100 * max(0, row['diagnostics']['max_recorded_height_mm'] / 45.5 - 1)
            row = dict(**row, objective_mm=row['surface_mm'] + penalty)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            row = dict(peaks_n=values[:4].tolist(), durations_s=values[4:].tolist(),
                       feasible=False, objective_mm=1000., failure=str(exc))
        rows.append(row); save_json(dest / 'evaluations.json', rows)
        return row['objective_mm']
    simplex = np.ones((9, 8)); simplex[1:] += .05 * np.eye(8)
    result = minimize(objective, np.ones(8), method='Nelder-Mead', bounds=bounds,
        options=dict(maxfev=PLAN_PROTOCOL['max_evaluations'], xatol=0., fatol=0., initial_simplex=simplex))
    calibration = json.loads((dest / 'initialization_evaluations.json').read_text())
    candidates = [r for r in rows + calibration if r['feasible'] and len(r['peaks_n']) == 4]
    assert candidates, 'No feasible complete force plan'
    best = min(candidates, key=lambda r: r['surface_mm'])
    save_json(dest / 'selected.json', dict(**best, model=initial['model'],
        initialization_evaluations=initial['evaluations'], optimization_evaluations=len(rows),
        prior_pilot_evaluations=9, prior_position_evaluations=32,
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        planning_protocol_sha256=study.digest(folder / 'planning_protocol.json')))
    print(json.dumps(json.loads((dest / 'selected.json').read_text()), indent=2), flush=True)


def evaluate(folder, material, device, settings):
    stage(folder)
    models = json.loads((folder / 'inputs/models.json').read_text())
    for setting in settings:
        grid, dt = study.SETTINGS[setting]
        for planned in 'AB':
            selected = json.loads((folder / f'plans/{planned}/selected.json').read_text())
            key = f'{setting}_{material}_plan_{planned}'
            _, record = study.rollout(folder, folder / 'executions', models[f'true_{material}'],
                                      selected['peaks_n'], selected['durations_s'], device, grid, dt)
            src = folder / record['file']
            for suffix in ['.npz', '.config.json', '.phases.json']:
                shutil.copy2(src.with_suffix(suffix), folder / f'{key}{suffix}')
            record = dict(**record, material=material, planned_for=planned, setting=setting)
            record['file'] = key + '.npz'
            save_json(folder / f'{key}.json', record)
            print(json.dumps(record), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['initialize', 'plan', 'evaluate'])
    p.add_argument('--out', type=Path, default=study.OUT)
    p.add_argument('--material', choices=['A', 'B'], required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--settings', choices=list(study.SETTINGS), nargs='+', default=list(study.SETTINGS))
    a = p.parse_args()
    if a.stage == 'evaluate': evaluate(a.out, a.material, a.device, a.settings)
    elif a.stage == 'initialize': initialize(a.out, a.material, a.device)
    else: plan(a.out, a.material, a.device)


if __name__ == '__main__': main()
