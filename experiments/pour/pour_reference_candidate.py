"""Isolated reference replay for independently motivated physics diagnostics.

No receiver endpoint is read. The original runner and identified artifacts are
unchanged. The only wall experiment changes the source SDF from separable to
sticky; receiver and all other numerical settings retain their original values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from examples import pour_recorded_twin as twin

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'out/pour_weakform_recovery/identified'
EPISODE = ROOT/'pouring_real_data/09-04-60-2s'


def run(args):
    identity = json.loads((BASE/'identify.json').read_text())
    if (identity['calibration_pour_count'] != 1 or identity['measured_endpoints_used']
            or identity['validation_measurements_used'] or identity['viscosity_fitted_by_simulator']):
        raise ValueError('Only endpoint-free, one-pour weak identification is admissible')
    geometry = json.loads((BASE/'geometry.json').read_text())
    eta_path = BASE/'identify.json'
    eta = identity['eta']
    if args.transport:
        eta_path = ROOT/'out/pour_physics_audit/transport/identify.json'
        candidate = json.loads(eta_path.read_text())
        if (candidate['calibration_pour_count'] != 1 or candidate['other_recordings_used']
                or candidate['measured_endpoints_used'] or candidate['viscosity_fitted_by_simulator']):
            raise ValueError('Invalid transport candidate provenance')
        eta = candidate['eta_pa_s']
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError('Invalid viscosity')
    name = f"{'transport' if args.transport else 'baseline'}_source_{args.source_wall}_n{args.grid}"
    out = ROOT/'out/pour_physics_audit/replay'/name
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(twin.__file__), BASE/'geometry.json', eta_path,
             *sorted((ROOT/'src/warpmpm').rglob('*.py')),
             *[EPISODE/k for k in ['states.jsonl','actions.jsonl','meta.json']]]
    protocol = dict(episode=str(EPISODE.relative_to(ROOT)), initial_volume_ml=300.,
                    eta_pa_s=eta, source_wall=args.source_wall, receiver_wall='separable',
                    grid=args.grid, source_band_cells=.5,
                    purpose='Physical/numerical consistency diagnostic; not a viscosity fit',
                    measured_endpoints_used=[], validation_recordings_used=[],
                    input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in paths})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    original_solver, original_out = twin.Solver, twin.OUT_ROOT

    class AuditSolver(original_solver):
        def add_sdf_collider(self, *positional, **keywords):
            index = getattr(self, '_audit_sdf_count', 0)
            if index == 0 and args.source_wall == 'sticky':
                keywords = dict(keywords, surface='sticky', friction=0.)
            self._audit_sdf_count = index+1
            return super().add_sdf_collider(*positional, **keywords)

    twin.Solver, twin.OUT_ROOT = AuditSolver, out
    started = time.monotonic()
    try:
        result = twin.run(EPISODE, device=args.device, n_grid=args.grid,
                          video=False, side_by_side=False, rebake=True,
                          eta=eta, volume_ml=300., **geometry)
    finally:
        twin.Solver, twin.OUT_ROOT = original_solver, original_out
    rows = result['rows']
    counts = [r['n_src']+r['n_rcv']+r['n_air_spill'] for r in rows]
    if len(set(counts)) != 1:
        raise RuntimeError('Particle ledger changed')
    last, count = rows[-1], counts[0]
    tail = [r['n_rcv'] for r in rows if r['t'] >= last['t']-.5]
    summary = dict(**protocol, receiver_ml=300*last['n_rcv']/count,
                   source_depletion_ml=300*(1-last['n_src']/count),
                   outside_ml=300*last['n_air_spill']/count,
                   tail_variation_ml=300*float(np.ptp(tail))/count,
                   particle_count=count, elapsed_s=time.monotonic()-started,
                   status='Diagnostic result; not released for robot control')
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'input_sha256'}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid',type=int,default=384)
    p.add_argument('--source-wall',choices=['sticky','separable'],required=True)
    p.add_argument('--transport',action='store_true')
    p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
