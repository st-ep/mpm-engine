"""Review preserved settling evidence, including a post-run report-writer error."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    protocol = json.loads((work / 'protocol.json').read_text())
    for rel, sha in protocol['input_sha256'].items():
        assert digest(ROOT / rel) == sha, rel
    trace = json.loads((work / 'progress.json').read_text())
    assert trace['stage'] == 'one_quiet_replay_frame'
    cases = list((work / 'forward').rglob('result.json'))
    assert len(cases) == 1
    forward = json.loads(cases[0].read_text())
    assert forward['particle_count'] == 229280 and forward['eta_pa_s'] == protocol['eta_pa_s']
    assert forward['source_coulomb_friction'] == protocol['source_coulomb_friction']
    cache = cases[0].parent / '09-04-60-2s/settled_n160_v300.npz'
    state = np.load(cache)
    assert np.isfinite(state['x_world']).all() and np.isfinite(state['v']).all()
    assert abs(float(np.linalg.norm(state['v'], axis=1).mean()) - trace['after_projection']['mean_speed_m_s']) < 1e-9
    metrics = cases[0].parent / '09-04-60-2s/metrics.csv'
    with metrics.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and float(rows[0]['t']) < .02
    assert sum(int(rows[0][key]) for key in ['n_src', 'n_rcv', 'n_air_spill']) == 229280
    passed = trace['before_projection']['finite'] and trace['before_projection']['mean_speed_m_s'] < protocol['mean_speed_threshold_m_s']
    result = {**trace, 'stage': 'reviewed',
        'protocol_sha256': digest(work / 'protocol.json'),
        'settling_threshold_passed': passed,
        'outside_ml_after_quiet_frame': forward['outside_ml'],
        'particle_count': forward['particle_count'],
        'simulation_elapsed_s': forward['elapsed_s'],
        'full_pour_performed': False, 'goal_accuracy_established': False,
        'frozen_inputs_and_raw_outputs_checked': True,
        'writer_error': 'Simulation completed; original wrapper failed only while creating the final dictionary because stage was supplied twice. This separate reviewer validates the saved raw outputs.',
        'source_sha256': digest(Path(__file__)),
        'evidence_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [cases[0], cache, metrics, work / 'progress.json']}}
    with (work / 'review.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
