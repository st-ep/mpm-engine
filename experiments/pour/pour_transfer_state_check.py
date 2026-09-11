"""Reuse analytical/state checks for one explicitly selected isolated package."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path, required=True)
args = parser.parse_args()
source = args.source_dir.resolve()
out = source.parent
sys.path.insert(0, str(source))
import warpmpm
import numpy as np
from experiments.pour import pour_compensated_position_benchmark as state_check
from experiments.pour import pour_stable_volume_gpu_check as freefall


def main():
    assert Path(warpmpm.__file__).resolve().is_relative_to(source)
    for device in ['cpu', 'cuda:0']:
        destination = out / f'benchmark_{device.replace(":", "")}.json'
        if destination.exists():
            raise RuntimeError(f'Preserve existing check: {destination}')
        state_check.OUT = out
        sys.argv = [sys.argv[0], '--device', device]
        state_check.main()
    a, b = freefall.run(True), freefall.run(False)
    for row in [a, b]:
        assert all(np.isfinite(row[key]).all() for key in ['x', 'v', 'logJ'])
        assert np.max(abs(row['logJ'])) < 1e-5
    dx = float(np.max(abs(a['x'] - b['x'])))
    dv = float(np.max(abs(a['v'] - b['v'])))
    gravity_error = float(np.max(abs(a['v'] - a['gravity'] * a['duration'])))
    assert dx < 2e-6 and dv < 1e-5 and gravity_error < 1e-5
    files = [Path(__file__), Path(state_check.__file__), Path(freefall.__file__),
             out / 'prototype_provenance.json']
    result = dict(gpu_pipeline_stable=True, dt_s=a['dt'], duration_s=a['duration'],
                  graph_live_position_difference_m=dx,
                  graph_live_velocity_difference_m_s=dv,
                  maximum_freefall_velocity_error_m_s=gravity_error,
                  actual_package=str(Path(warpmpm.__file__).resolve()),
                  input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in files},
                  liquid_data_used=[], identification_changed=False)
    destination = out / 'gpu_check.json'
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
