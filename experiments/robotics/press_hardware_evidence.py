"""Aggregate real pressing checks and audit supplied reconstructed particle motion.

No constitutive fit is performed. Local particle gradients below are diagnostics
of the supplied reconstruction, not measured deformation gradients.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from experiments.robotics.press_hardware_observe import read, jsonl, save, plot_episode


def particle_check(raw, dest):
    d = np.load(raw / 'particles.npz')
    t, x, reference, stuck = d['times'], d['pos'], d['seed'], d['stuck']
    robot = np.load(dest / 'robot_force.npz')
    z = np.interp(t, robot['time'], robot['ee_pos'][:, 2])
    top = x[:, :, 2].max(1)
    # Extrema of a 2 mm interior lattice do not locate the exact physical surface.
    top_descent = (top[0] - top) * 1000
    robot_descent = (z[0] - z) * 1000
    _, neighbors = cKDTree(reference).query(reference, k=20)
    delta = reference[neighbors] - reference[:, None]
    inverse = np.linalg.pinv(delta)
    frames = np.unique(np.r_[np.arange(0, len(t), 3), len(t) - 1])
    determinant_stats, error_stats, inversion = [], [], []
    for frame in frames:
        current = x[frame, neighbors] - x[frame, :, None]
        transpose_F = inverse @ current
        determinant = np.linalg.det(transpose_F)
        prediction = delta @ transpose_F
        residual = np.sqrt(np.mean(np.sum((prediction - current) ** 2, -1), -1)) * 1000
        determinant_stats.append(np.percentile(determinant, [0, 1, 5, 50, 95, 99, 100]))
        error_stats.append(np.percentile(residual, [50, 95, 100]))
        inversion.append(int((determinant <= 0).sum()))
    determinant_stats, error_stats = np.array(determinant_stats), np.array(error_stats)
    sensitivity = []
    for count in [8, 12, 20, 32]:
        distances, ids = cKDTree(reference).query(reference, k=count)
        dx = reference[ids] - reference[:, None]
        gradient = np.linalg.pinv(dx) @ (x[-1, ids] - x[-1, :, None])
        J = np.linalg.det(gradient)
        sensitivity.append(dict(neighbors=count, reference_support_median_mm=float(np.median(distances[:, -1]) * 1000),
            rank_deficient_neighborhoods=int((np.linalg.matrix_rank(dx) < 3).sum()),
            final_J_p01_p50_p99=np.percentile(J, [1, 50, 99]).tolist(),
            final_nonpositive_count=int((J <= 0).sum())))
    np.savez_compressed(dest / 'particle_reconstruction_checks.npz', time=t,
                        lattice_top_descent_mm=top_descent, robot_descent_mm=robot_descent,
                        gradient_time=t[frames], determinant_percentiles=determinant_stats,
                        affine_residual_percentiles_mm=error_stats, nonpositive_count=inversion)
    result = dict(particles=len(reference), frames=len(t), time_range_s=t[[0, -1]].tolist(),
        finite=bool(np.isfinite(x).all()), monotone_time=bool((np.diff(t) > 0).all()),
        seed_frame0_max_difference_m=float(abs(reference - x[0]).max()),
        final_lattice_top_descent_mm=float(top_descent[-1]), final_robot_descent_mm=float(robot_descent[-1]),
        final_descent_difference_mm=float(top_descent[-1] - robot_descent[-1]),
        gradient_neighbors=20, gradient_sampled_frames=len(frames),
        final_J_percentiles=determinant_stats[-1].tolist(),
        J_percentile_order=[0, 1, 5, 50, 95, 99, 100],
        max_nonpositive_local_determinants=max(inversion),
        neighborhood_sensitivity=sensitivity,
        final_local_affine_residual_p50_p95_max_mm=error_stats[-1].tolist(),
        final_contact_flags={str(i): int((stuck[-1] == i).sum()) for i in range(3)},
        scope='Supplied model-generated trajectories. F estimated with 20-neighbor local affine fits in the 2 mm reference lattice; these are numerical consistency diagnostics, not measured strain or proof of incompressibility. Top lattice extent is not exact plate height.')
    save(dest / 'particle_reconstruction_check.json', result)
    return result


def main(root, out):
    results = []
    for i in range(12):
        name = f'ep{i:04}'
        dest = out / name
        audit = read(dest / 'assessment.json')
        robot = np.load(dest / 'robot_force.npz')
        t, xyz = robot['time'], robot['ee_pos']
        hold = robot['controller_active'] & (robot['phase'] == 'hold') & (t >= 3)
        delta = robot['incremental_normal_force'] - robot['controller_force']
        entry = dict(episode=name, material=audit['material'], force_command_N=audit['force_command_N'],
            baseline_std_N=audit['baseline_std_N'],
            late_hold_force_channel_rms_N=float(np.sqrt(np.mean(delta[hold] ** 2))),
            robot_descent_3_to_10_s_mm=audit['robot_sink_3_to_10_s_mm'],
            mean_force_9_to_10_s_N=audit['mean_force_9_to_10_s_N'],
            initial_center_disagreement_mm=audit.get('independent_initial_center_disagreement_mm'), cameras={})
        camera_times = []
        for cam in ['side', 'hand']:
            obs = np.load(dest / f'{cam}_observations.npz')
            tt, edge = obs['time'], obs['plate_white_edge_y_px']
            select = (tt >= .6) & (tt <= 10.4)
            rz = np.interp(tt[select], t, xyz[:, 2])
            residual = edge[select] - np.polyval(np.polyfit(rz, edge[select], 1), rz)
            denom = np.sum((edge[select] - edge[select].mean()) ** 2)
            r2 = 1 - np.sum(residual ** 2) / denom if denom else None
            frames = jsonl(root / name / f'frames_{cam}.jsonl')
            host = np.array([f['t_host'] for f in frames])
            camera_times.append(host)
            c = audit['cameras'][cam]
            entry['cameras'][cam] = dict(depth_frozen=c['depth_frozen'],
                coverage_median=c['depth_coverage_median'],
                plate_edge_down_3_to_10_s_px=float(np.interp(10, tt, edge) - np.interp(3, tt, edge)),
                plate_edge_vs_robot_linear_R2=r2,
                tracking_3d_10_s=c['tracked_3d_at_s']['10'],
                host_minus_sensor_median_ms=c['host_minus_sensor_median_ms'],
                host_minus_sensor_p01_p99_ms=c['host_minus_sensor_p01_p99_ms'])
        first, second = camera_times
        select = first[(first >= second[0]) & (first <= second[-1])]
        ix = np.searchsorted(second, select).clip(1, len(second) - 1)
        gaps = np.minimum(abs(select - second[ix]), abs(select - second[ix - 1])) * 1000
        entry['nearest_camera_host_time_gap_p50_p99_max_ms'] = np.percentile(gaps, [50, 99, 100]).tolist()
        if i < 8:
            entry['particle_reconstruction'] = particle_check(root / 'handoff_ep0-7' / name, dest)
        # Regenerate plots with the displayed time window also controlling y limits.
        plot_episode(dest, audit, (t, xyz, robot['incremental_normal_force'], robot['controller_active'], robot['phase']))
        results.append(entry)
        print(name, 'evidence checked', flush=True)
    save(out / 'evidence_summary.json', dict(episodes=results,
        contact='User reports slight stickiness; adhesion is unmeasured and cannot be removed or separately measured.',
        limitations=['Force is incremental about a contact baseline, not calibrated total contact force.',
                     'Silhouette contact boundaries are imperfect; volume estimates are diagnostics only.',
                     'Plasticine segmentation uses depth, so valid-depth fraction is conditional on that mask.',
                     'RGB candidate tracks are not independently validated material identities.',
                     'The supplied camera transforms have not passed cross-view geometry checks.']))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for j, material in enumerate(['Play-Doh', 'butter slime', 'plasticine']):
        for e in results[j * 4:j * 4 + 4]:
            d = np.load(out / e['episode'] / 'robot_force.npz')
            keep = (d['time'] >= 3) & (d['time'] <= 10)
            z3 = np.interp(3, d['time'], d['ee_pos'][:, 2])
            axes[j].plot(d['time'][keep], (z3 - d['ee_pos'][keep, 2]) * 1000,
                         label=f"{e['force_command_N']:g} N command")
        axes[j].set(title=material, xlabel='Time from baseline onset (s)', ylim=(0, 7.5))
        axes[j].grid(alpha=.2)
        axes[j].legend(fontsize=8)
    axes[0].set_ylabel('Additional downward robot motion after 3 s (mm)')
    fig.suptitle('Continued press motion during force hold: measured robot motion, not isolated material creep')
    fig.savefig(out / 'hold_motion_comparison.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.out)
