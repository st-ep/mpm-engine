"""Isolated known-family weak fit: fixed reconstructed motion, no dynamics.

Primary protocol: one 31 N recording per material, divergence-free weak fields,
loading windows only, perfect or linear-overstress Hencky/von Mises response.
The recorded unloading is an additional weak-balance check, not a fit target.
Bulk response is supplied through fixed nu=0.45, never calibrated by rollouts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
from scipy.optimize import minimize


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def unit_history(increments, ratio, relaxation, dt, power=1):
    """Deviatoric Kirchhoff stress at G=1, threshold Y/G=ratio."""
    elastic = np.broadcast_to(np.eye(3), increments.shape[1:]).copy()
    output = []
    active = []
    for inc in increments:
        trial = inc @ elastic
        U, singular, Vh = np.linalg.svd(trial)
        strain = np.log(singular)
        mean = strain.mean(axis=1)
        dev = strain - mean[:, None]
        norm = np.linalg.norm(dev, axis=1)
        excess = np.maximum(norm - ratio / 2, 0)
        if relaxation and power == 2:
            remaining = 2*excess / (1 + np.sqrt(1 + 4*dt/relaxation*excess))
            reduction = excess - remaining
        else:
            reduction = excess * dt / (dt + relaxation)
        dev *= (1 - reduction / np.maximum(norm, 1e-30))[:, None]
        elastic = (U * np.exp(mean[:, None] + dev)[:, None, :]) @ Vh
        output.append((U * (2 * dev)[:, None, :]) @ U.transpose(0, 2, 1))
        active.append(norm > ratio / 2)
    return np.asarray(output), np.asarray(active)


def load_episode(root, episode):
    d = dict(np.load(root / episode / 'motion.npz'))
    geo = json.loads((root / episode / 'geometry.json').read_text())
    d['volume'] = geo['volume_m3']
    F = d['F']
    old = np.concatenate([np.broadcast_to(np.eye(3), F[:1].shape), F[:-1]])
    d['increments'] = F @ np.linalg.inv(old)
    d['dt'] = float(np.median(np.diff(d['time'])))
    d['select'] = (d['window_begin'] >= .4) & (d['window_begin'] + .6 <= 10.400001)
    d['unloading'] = d['window_begin'] >= 10.8
    d['scale'] = float(np.linalg.norm(d['target'][d['select'], :3]))
    return d


def response(d, ratio, relaxation, power=1):
    stress, active = unit_history(d['increments'], ratio, relaxation, d['dt'], power)
    instant = d['volume'] * np.einsum('tnij,tmnij,n->tm', stress, d['test_gradient'][:, :3], d['weights'])
    return d['windows'] @ instant, active


def fit(root, material, episode, include_partial_unloading=False, power=1):
    started = time.monotonic()
    d = load_episode(root, episode)
    if include_partial_unloading:
        d['select'] = (d['window_begin'] >= .4) & (d['window_begin'] + .6 <= 13.400001)
        d['scale'] = float(np.linalg.norm(d['target'][d['select'], :3]))
    records = []

    def evaluate(v, viscous, details=False):
        ratio = float(np.exp(v[0]))
        relaxation = float(np.exp(v[1])) if viscous else 0.
        A, active = response(d, ratio, relaxation, power)
        aa = A[d['select']].ravel()
        b = d['target'][d['select'], :3].ravel()
        G = float(np.clip(aa @ b / max(aa @ aa, 1e-30), 100, 2e6))
        residual = float(np.linalg.norm(G * aa - b) / d['scale'])
        result = dict(G_Pa=G, Y_Pa=G * ratio, ratio_Y_G=ratio, tau_s=relaxation,
                      relative_weak_residual=residual,
                      active_reference_volume_fraction_max=float(np.max(active @ d['weights'])),
                      law='Hencky/von Mises linear overstress' if viscous else 'Hencky/perfect von Mises')
        if viscous and power == 2:
            result.update(flow_power=2., law='Hencky/von Mises quadratic overstress')
        records.append(result)
        return (result, A) if details else residual**2

    candidates = []
    for viscous in [False, True]:
        starts = []
        for ratio in np.logspace(-4, 1, 11):
            for relaxation in ([.01, .1, 1., 10., 100.] if viscous else [None]):
                v = np.log([ratio, relaxation] if viscous else [ratio])
                starts.append((evaluate(v, viscous), v))
        bounds = [(np.log(1e-5), np.log(10))] + ([(np.log(.002), np.log(300))] if viscous else [])
        opts = [minimize(lambda v: evaluate(v, viscous), v, method='Nelder-Mead', bounds=bounds,
                        options=dict(maxiter=100, xatol=.003, fatol=1e-8))
                for _, v in sorted(starts, key=lambda a: a[0])[:2]]
        opt = min(opts, key=lambda a: a.fun)
        result, A = evaluate(opt.x, viscous, True)
        result['optimizer_success'] = bool(opt.success)
        result['optimizer_message'] = str(opt.message)
        result['partial_unloading_weak_residual'] = float(np.linalg.norm(
            result['G_Pa'] * A[d['unloading']] - d['target'][d['unloading'], :3]) /
            max(np.linalg.norm(d['target'][d['unloading'], :3]), 1e-30))
        candidates.append(result)
    best = dict(candidates[1] if candidates[1]['relative_weak_residual'] < .85 * candidates[0]['relative_weak_residual'] else candidates[0])
    G = best['G_Pa']
    best.update(E_Pa=2 * (1 + .45) * G, nu_assumed=.45, K_Pa=2 * G * (1 + .45) / (3 * (1 - 2 * .45)),
                parameter_scope='Weak-balance estimate on one fixed reconstructed motion; no forward fitting. Bulk derived from fixed nu, not identified.',
                training_episodes=[episode])
    # Sensitivity of the actual weak response to each physical parameter;
    # finite differences keep the other physical parameters fixed.
    def physical_prediction(G, Y, T):
        return G * response(d, Y / G, T, power)[0][d['select']].ravel() / d['scale']
    columns = []
    params = ['G_Pa', 'Y_Pa'] + (['tau_s'] if best['tau_s'] else [])
    for key in params:
        plus = dict(best); minus = dict(best)
        plus[key] *= np.exp(.01); minus[key] *= np.exp(-.01)
        columns.append((physical_prediction(plus['G_Pa'], plus['Y_Pa'], plus['tau_s']) -
                        physical_prediction(minus['G_Pa'], minus['Y_Pa'], minus['tau_s'])) / .02)
    singular = np.linalg.svd(np.stack(columns, axis=1), compute_uv=False)
    doubles = {}
    for key in params:
        perturbed = dict(best); perturbed[key] *= 2
        doubles[key] = float(np.linalg.norm(physical_prediction(perturbed['G_Pa'], perturbed['Y_Pa'], perturbed['tau_s']) -
                                          physical_prediction(best['G_Pa'], best['Y_Pa'], best['tau_s'])))
    hashes = {name: hashlib.sha256((root / episode / name).read_bytes()).hexdigest() for name in ['motion.npz', 'geometry.json']}
    result = dict(selected=best, candidates=candidates, search=records, fit_wall_s=time.monotonic()-started,
                  forward_calls_in_identification=0, input_sha256=hashes,
                  selection='One 31 N recording; ' + ('loading and partial-unloading' if include_partial_unloading else 'loading-only') + ' weak residual. Rate term requires at least 15 percent residual improvement. No forward error or baseline fitted parameters read.',
                  partial_unloading_used_in_fit=include_partial_unloading,
                  sensitivity=dict(parameters=params, singular_values=singular.tolist(), double_parameter_response_changes=doubles,
                                   scope='Local sensitivity conditional on reconstructed motion; not confidence intervals.'),
                  status='Frozen weak estimate; identifiability and forward evaluation must be reported separately.')
    save(root / material / 'fit.json', result)
    print(material, json.dumps({k:v for k,v in result.items() if k not in ['search']}), flush=True)


def check(out):
    from experiments.robotics.press_hardware_model import stress_history
    F = np.asarray([np.diag(np.exp([e, -.5*e, -.5*e])) for e in [0, -.001, -.03, -.1, -.09, -.08]])[:, None]
    old = np.concatenate([np.eye(3)[None, None], F[:-1]])
    inc = F @ np.linalg.inv(old)
    errors = []
    for T in [0., .3, 3.]:
        actual = unit_history(inc, .02, T, .05)[0]
        expected = stress_history(F, .02, T, -.5, .05)
        errors.append(float(np.max(abs(actual-expected))))
    assert max(errors) < 1e-12
    identity = np.broadcast_to(np.eye(3), (20, 5, 3, 3))
    zero = unit_history(identity, .01, 1., .05)[0]
    assert np.max(abs(zero)) == 0
    # Independent elastic and perfect-yield checks in uniform log strain.
    elastic = unit_history(inc, 100., 0., .05)[0]
    analytic = np.asarray([np.diag([2*e, -e, -e]) for e in [0, -.001, -.03, -.1, -.09, -.08]])[:, None]
    assert np.max(abs(elastic-analytic)) < 1e-12
    perfect = unit_history(inc, .02, 0., .05)[0]
    assert np.max(np.linalg.norm(perfect, axis=(2,3))) <= .0200000001
    save(out / 'IDENTIFIER_CHECKS.json', dict(status='passed', existing_history_errors=errors,
         identity_zero_stress=True, analytic_elastic_stress=True, perfect_yield_bound=True))


def fit_maxwell(root, material, episode, power=1):
    """Nested Y=0 hypothesis, avoiding a spurious fitted yield at a search floor.

    Compare two two-parameter models by the same complete-probe weak residual:
    perfect plasticity (G,Y) and zero-yield linear flow (G,T). No forward data.
    """
    from scipy.optimize import minimize_scalar
    d = load_episode(root, episode)
    d['select'] = (d['window_begin'] >= .4) & (d['window_begin'] + .6 <= 13.400001)
    d['scale'] = float(np.linalg.norm(d['target'][d['select'], :3]))
    records = []
    def objective(log_T):
        T = float(np.exp(log_T)); A, _ = response(d, 0., T, power)
        a = A[d['select']].ravel(); b = d['target'][d['select'], :3].ravel()
        G = float(np.clip(a @ b / max(a @ a, 1e-30), 100, 2e6))
        residual = float(np.linalg.norm(G*a-b) / d['scale'])
        records.append(dict(G_Pa=G, Y_Pa=0., tau_s=T, relative_weak_residual=residual,
                            E_Pa=2*1.45*G, K_Pa=2*1.45*G/.3, nu_assumed=.45,
                            law='Hencky/zero-yield linear viscous flow',
                            parameter_scope='G and flow time from weak balance. Y=0 is a reduced-model assumption, not a measured absence of yield. K derived from fixed nu=0.45.',
                            training_episodes=[episode]))
        if power == 2:
            records[-1].update(flow_power=2.,law='Hencky/zero-yield quadratic viscous flow',
                              flow_time_scope='T normalizes squared elastic overstress strain; not a linear relaxation time.')
        return residual
    grid = np.linspace(np.log(.002), np.log(300), 51)
    scores = [objective(v) for v in grid]; idx = int(np.argmin(scores))
    opt = minimize_scalar(objective, bounds=(grid[max(idx-1,0)],grid[min(idx+1,len(grid)-1)]),method='bounded',options=dict(xatol=1e-5))
    objective(opt.x); viscous = records[-1]
    existing = json.loads((root/material/'fit.json').read_text())
    perfect = existing['candidates'][0]
    if viscous['relative_weak_residual'] < perfect['relative_weak_residual']:
        selected = viscous
    else:
        selected = dict(perfect); G = selected['G_Pa']
        selected.update(E_Pa=2*1.45*G,K_Pa=2*1.45*G/.3,nu_assumed=.45,
                        parameter_scope='G and Y from weak balance; K derived from fixed nu=0.45.',training_episodes=[episode])
    save(root/material/'reduced_fit.json',dict(selected=selected,zero_yield_candidate=viscous,perfect_candidate=perfect,
        zero_yield_search=records,forward_calls_in_identification=0,
        selection='Same complete-probe weak residual; compare two two-parameter models. Zero-yield limit examined because free-yield rate fits reached their lower search bound. No forward prediction used.',
        original_fit_input_sha256=existing['input_sha256']))
    print(material,'reduced model',json.dumps(selected),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--material', choices=['play_doh','butter_slime','plasticine'])
    parser.add_argument('--episode')
    parser.add_argument('--include-partial-unloading', action='store_true')
    parser.add_argument('--maxwell', action='store_true')
    parser.add_argument('--flow-power', type=int, choices=[1,2], default=1)
    args = parser.parse_args()
    check(args.root)
    if args.material:
        if args.maxwell:
            fit_maxwell(args.root, args.material, args.episode, args.flow_power)
        else:
            fit(args.root, args.material, args.episode, args.include_partial_unloading, args.flow_power)
