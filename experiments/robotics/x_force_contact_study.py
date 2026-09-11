"""Reproducible contact-triggered force-amplitude/duration planning using the common A/B fits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pyvista as pv
from scipy.optimize import minimize

from experiments.robotics import x_force_contact_control as control
from experiments.robotics import x_shaping as core
from experiments.robotics.plastic_shaping_figure import surface
from experiments.robotics.hex_shaping_surface import mesh_distance_mm
from experiments.robotics.plastic_shaping_study import save_json, save_npz

ROOT = core.ROOT
BASE = ROOT / 'out/x_common_identification_study_20260910'
OUT = ROOT / 'out/x_force_contact_study_20260910'
SETTINGS = {'baseline': (64, .00005), 'half_dt': (64, .000025), 'grid80': (80, .00005)}
PROTOCOL = dict(scene=core.CONFIG, control=control.CONTROL,
    initialization_source=str(BASE.relative_to(ROOT)),
    pilot=dict(force_factors=[1.00, 1.05, 1.15], durations_s=[.4, .7, 1.0],
               score='Absolute distance to first-pinch opening of the identified-model position plan; no true-material outcomes'),
    planning_grid=64, planning_dt=.0001, settings=SETTINGS,
    planning='Eight controls: four mean compressive per-finger force amplitudes and four total pulse durations',
    pulse='Clock starts after 3 consecutive filtered 0.05 N contact samples at 10 mm/s approach; raised-cosine rise for 20%, constant force for 60%, raised-cosine fall for 20%',
    force_bounds_n=[.05, 14.], duration_bounds_s=[.2, 2.4],
    comparison='Identical force/time commands and controller cross-executed in fresh A/B specimens',
    target='Frozen rounded X, 64 x 80 x 24.946587 mm, 90 mL; no geometry or material adjustment',
    selection='Lowest feasible identified-model final 3D surface error; no true-material outcomes used for planning',
    feasibility='No 12 mm opening-guard activation; maximum sampled specimen height <= 45.5 mm',
    scope='Contact simulation with supplied-state identification; requires a force-capable gripper, stock Franka Hand force-pulse capability not established', seed=0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(folder):
    assert not (folder / 'protocol.json').exists(), 'Use a fresh experiment directory'
    folder.mkdir(parents=True, exist_ok=True)
    inputs = folder / 'inputs'; inputs.mkdir()
    for src in (BASE / 'inputs').iterdir():
        if src.is_file(): shutil.copy2(src, inputs / src.name)
    shutil.copytree(BASE / 'validation', folder / 'validation')
    for name in ['target.vtp', 'target.json', 'target_outline_m.npy', 'identification_validation.json', 'observation_isolation.json']:
        shutil.copy2(BASE / name, folder / name)
    shutil.copytree(BASE / 'franka/panda_model_snapshot', folder / 'franka/panda_model_snapshot')
    for material in 'AB':
        prediction = BASE / f'prediction_{material}.npz'
        data = np.load(prediction)
        direction = data['tool_centers'][:, 1] - data['tool_centers'][:, 0]
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        f = (np.einsum('tfc,tc->tf', data['reaction_force'], direction) * [-1, 1]).mean(axis=1)
        peaks = []
        for i in range(4):
            means = np.convolve(f[data['pinch_id'] == i], np.ones(5) / 5, mode='valid')
            peaks.append(float(means.max()))
        selected = json.loads((BASE / f'plans/{material}/selected.json').read_text())
        save_json(inputs / f'initialization_{material}.json', dict(reference_peak_n=peaks,
            gaps_mm=selected['gaps_mm'], model=selected['model'], law=selected['law'],
            source=str(prediction), source_sha256=digest(prediction), force_smoothing_s=.02,
            prior_planning_evaluations=32, scope='Identified-model predictions only, not measured true-material forces'))
    save_json(folder / 'protocol.json', PROTOCOL)
    source_paths = list(json.loads((BASE / 'source_sha256.json').read_text()))
    source_paths += ['experiments/robotics/x_force_contact_control.py', 'experiments/robotics/x_force_contact_study.py']
    hashes = {}
    for rel in sorted(set(source_paths)):
        src = ROOT / rel
        dst = folder / 'source_snapshot' / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst); hashes[rel] = digest(src)
    save_json(folder / 'source_sha256.json', hashes)
    save_json(folder / 'input_sha256.json', {str(p.relative_to(folder)): digest(p)
        for p in [*inputs.iterdir(), *folder.glob('target*'), *folder.joinpath('validation').iterdir(),
                  folder / 'identification_validation.json', folder / 'observation_isolation.json'] if p.is_file()})
    save_json(folder / 'robot_input_sha256.json', {str(p.relative_to(folder)): digest(p)
        for p in folder.joinpath('franka/panda_model_snapshot').rglob('*') if p.is_file()})
    (folder / 'packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    (folder / 'commit.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    (folder / 'gpu_before.txt').write_bytes(subprocess.check_output(['nvidia-smi']))


def check(folder):
    assert json.loads((folder / 'protocol.json').read_text()) == json.loads(json.dumps(PROTOCOL))
    for rel, sha in json.loads((folder / 'source_sha256.json').read_text()).items():
        assert digest(ROOT / rel) == digest(folder / 'source_snapshot' / rel) == sha, rel
    for manifest in ['input_sha256.json', 'robot_input_sha256.json']:
        for rel, sha in json.loads((folder / manifest).read_text()).items():
            assert digest(folder / rel) == sha, rel


def rollout(folder, dest, law, peaks, durations, device, grid=64, dt=.0001):
    peaks = np.round(peaks, 6).tolist()
    durations = (np.round(np.asarray(durations) / core.CONFIG['tick']) * core.CONFIG['tick']).tolist()
    config = dict(law=law, peaks_n=peaks, durations_s=durations, n_grid=grid, dt=dt,
                  device=device, seed=0, protocol_sha256=digest(folder / 'protocol.json'))
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    path = dest / (key + '.npz'); dest.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.with_suffix('.config.json').read_text()) == config
        return np.load(path), json.loads(path.with_suffix('.json').read_text())
    save_json(path.with_suffix('.config.json'), config)
    start = time.monotonic()
    data, phases, stops = control.execute(law, peaks, durations, grid, dt, device)
    save_npz(path, data); save_json(path.with_suffix('.phases.json'), phases)
    diag = control.diagnostics(data, phases)
    error = mesh_distance_mm(surface(data['x_after_1s'], data['vol0'], h=.00125), pv.read(folder / 'target.vtp'))
    feasible = 'travel_guard' not in stops and diag['max_recorded_height_mm'] <= 45.5
    row = dict(file=str(path.relative_to(folder)), peaks_n=peaks, durations_s=durations,
               law=law, surface_mm=error, stops=stops, feasible=feasible, diagnostics=diag,
               data_sha256=digest(path), elapsed_s=time.monotonic() - start)
    save_json(path.with_suffix('.json'), row)
    return data, row


def pilot(folder, material, device):
    check(folder)
    initial = json.loads((folder / f'inputs/initialization_{material}.json').read_text())
    rows = []
    for factor in PROTOCOL['pilot']['force_factors']:
        for duration in PROTOCOL['pilot']['durations_s']:
            print(json.dumps(dict(starting=material, factor=factor, duration_s=duration)), flush=True)
            _, row = rollout(folder, folder / 'pilot' / material, initial['law'],
                             [initial['reference_peak_n'][0] * factor], [duration], device)
            result = dict(**row, force_factor=factor,
                          desired_first_gap_mm=initial['gaps_mm'][0],
                          first_gap_difference_mm=abs(row['diagnostics']['pulses'][0]['min_gap_mm'] - initial['gaps_mm'][0]))
            rows.append(result)
            save_json(folder / f'pilot/{material}/evaluations.json', rows)
            print(json.dumps(result), flush=True)
    feasible = [r for r in rows if r['feasible']]
    if feasible:
        save_json(folder / f'pilot/{material}/selected.json', min(feasible, key=lambda r: r['first_gap_difference_mm']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'pilot'])
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--material', choices=['A', 'B'])
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.stage == 'prepare': prepare(a.out)
    else: pilot(a.out, a.material, a.device)


if __name__ == '__main__': main()
