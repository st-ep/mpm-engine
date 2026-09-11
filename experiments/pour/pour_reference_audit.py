"""Read-only phase and mass-balance audit of the single identification pour."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'out/pour_weakform_recovery'
OUT = ROOT / 'out/pour_physics_audit'


def main():
    OUT.mkdir(exist_ok=True)
    ident = json.loads((BASE/'identified/identify.json').read_text())
    obs = dict(np.load(BASE/'identified/observations.npz'))
    weak = dict(np.load(BASE/'identified/brink_fit.npz'))
    direct = dict(np.load(BASE/'brink_forward_60_0.01.npz'))
    actions = [json.loads(line) for line in
               (ROOT/'pouring_real_data/09-04-60-2s/actions.jsonl').read_text().splitlines()]
    pour = next(a for a in actions if a.get('pour_angle_deg') == 60)
    ret = next(a for a in actions if a.get('trajectory_name') == 'unpour')
    ack, send, done = [float(v - pour['t_send']) for v in
                       [pour['t_ack'], ret['t_send'], ret['t_ack']]]
    metrics = list(csv.DictReader((BASE/'mpm/n384_shift0/angle_060.00/metrics.csv').open()))
    result = json.loads((BASE/'mpm/n384_shift0/angle_060.00/result.json').read_text())
    t = np.array([float(r['t']) - 1. for r in metrics])
    n = result['particle_count']
    vr = np.array([float(r['n_rcv']) * 300/n for r in metrics])
    depletion = np.array([(n-float(r['n_src'])) * 300/n for r in metrics])
    ot, ov = obs['t'], obs['rcv_vol']*1e6
    def at(time, radius=.10):
        vals = ov[(abs(ot-time) <= radius) & np.isfinite(ov)]
        return dict(median_ml=float(np.median(vals)) if len(vals) else None,
                    range_ml=[float(vals.min()), float(vals.max())] if len(vals) else None,
                    n_frames=len(vals))
    boundaries = []
    for name, time in [('initial', 0.), ('tilt_ack', ack), ('return_send', send),
                       ('return_ack', done), ('settled', t[-1]-.25)]:
        volume = at(time, .25 if name == 'settled' else .10)
        if name == 'initial':
            volume = dict(median_ml=0., range_ml=[0., 0.], n_frames=0,
                          source='Protocol: receiver initially empty')
        boundaries.append(dict(name=name, time_s=time, video=volume,
                               mpm_receiver_ml=float(np.interp(time, t, vr)),
                               mpm_source_depletion_ml=float(np.interp(time, t, depletion)),
                               brink_source_depletion_ml=float(np.interp(time, direct['t'], direct['volume_ml']))))
    phases = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        phases.append(dict(start=a['name'], end=b['name'],
                           observed_receiver_gain_ml=b['video']['median_ml']-a['video']['median_ml'],
                           mpm_receiver_gain_ml=b['mpm_receiver_ml']-a['mpm_receiver_ml'],
                           mpm_source_gain_ml=b['mpm_source_depletion_ml']-a['mpm_source_depletion_ml'],
                           brink_source_gain_ml=b['brink_source_depletion_ml']-a['brink_source_depletion_ml']))
    wt = weak['t']
    fall = np.maximum(np.interp(wt, ot, obs['lip'][:, 2])-np.interp(wt, ot, obs['rcv_level']), 0)
    tau = np.sqrt(2*fall/9.81)
    tau_f = tau * weak['Fq'] * 1e6
    flight = tau_f / ident['eta']
    # d(V_receiver + V_flight)/dt = Q_source. For the same quasi-steady
    # flight approximation V_flight=tau*F/eta, eta*dV_receiver=int(F)-d(tau*F).
    # This is an AUDIT with the old forcing frozen, not a released new estimate.
    corrected_rhs = weak['cumF'] - (tau_f-tau_f[0])
    dv = weak['delta_volume_ml']
    corrected_eta = float(corrected_rhs @ corrected_rhs / (corrected_rhs @ dv))
    report = dict(identification_episode='09-04-60-2s', eta_pa_s=ident['eta'],
                  purpose='Diagnosis only; no identification or command artifact changed',
                  boundaries=boundaries, phases=phases,
                  final_mpm_receiver_ml=result['receiver_ml'],
                  final_video_receiver_ml=boundaries[-1]['video']['median_ml'],
                  final_reported_receiver_ml=159., reported_endpoint_used_for_fitting=False,
                  flight_audit=dict(transit_time_start_end_s=[float(tau[0]),float(tau[-1])],
                                    estimated_air_volume_start_end_ml=[float(flight[0]),float(flight[-1])],
                                    estimated_air_volume_change_ml=float(flight[-1]-flight[0]),
                                    measured_receiver_gain_ml=float(dv[-1]),
                                    omission='Existing fit corrects source head for in-flight volume but omits the change of in-flight inventory from its integrated balance',
                                    corrected_eta_with_old_forcing_only_pa_s=corrected_eta,
                                    status='Diagnostic hypothesis: requires manufactured tests and a self-consistent flight model before real-data adoption'),
                  other_pours_or_outcomes_used=[],
                  input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), BASE/'identified/identify.json',
                                          BASE/'identified/observations.npz', BASE/'identified/brink_fit.npz',
                                          BASE/'mpm/n384_shift0/angle_060.00/metrics.csv',
                                          BASE/'brink_forward_60_0.01.npz']})
    (OUT/'reference_phase_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(ot,ov,'.-',ms=2,lw=1,label='Video receiver readout')
    axes[0].plot(t,vr,label='MPM receiver')
    axes[0].set(ylabel='Receiver volume (mL)',ylim=(0,190)); axes[0].legend()
    axes[1].plot(t,depletion,label='MPM source depletion')
    axes[1].plot(direct['t'],direct['volume_ml'],label='Direct brink-law source depletion')
    axes[1].set(ylabel='Source depletion (mL)',ylim=(0,190)); axes[1].legend()
    axes[2].plot(t,depletion-vr,label='MPM outside source/receiver inventory')
    axes[2].plot(wt,flight,label='Quasi-steady flight estimate in weak-fit window')
    axes[2].set(xlabel='Time after tilt command (s)',ylabel='In-flight/outside volume (mL)')
    axes[2].legend()
    for ax in axes:
        for val,label in [(ack,'tilt ACK'),(send,'return send'),(done,'return ACK')]:
            ax.axvline(val,color='gray',ls=':',alpha=.7)
        ax.grid(alpha=.2); ax.set_xlim(0,t[-1])
    axes[0].set_title('Single 60° recording: phase and flight-inventory audit')
    fig.tight_layout(); fig.savefig(OUT/'reference_phase_audit.png',dpi=150); plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__ == '__main__':
    main()
