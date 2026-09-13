"""Actual plate-camera inputs, withheld force prediction, and replanned X shaping.

Camera frames retain their grayscale texture. Color identifies materials in
labels, force curves and shaping illustrations, never a fitted surface signal.
"""
from pathlib import Path
import argparse
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from experiments.robotics import x_common_report as common
from experiments.robotics import x_motion_shaping as core
from experiments.robotics import x_motion_search as search
from experiments.robotics.x_shaping_report import render as render_action
from experiments.robotics.plastic_shaping_study import save_json

IDENTIFICATION = core.ROOT / 'out/plate_observable_physical_preload_20260912'
SHAPING = core.ROOT / 'out/plate_to_x_20260912_grid80'
ROBOT = core.ROOT / 'out/x_common_identification_study_20260910/franka/panda_model_snapshot/panda.xml'


def panel_a(ax, images):
    common.panel_frame(ax, '(a) Identification with a plate press')
    times = np.load(IDENTIFICATION / 'inputs_A/time.npy')
    views = [('A', 2.6, 'Press (A)'), ('A', 3.8, 'Released A'), ('B', 3.8, 'Released B')]
    height, width, gap = .85, 1.02, .12
    left = (common.PANEL_WIDTH - 3 * width - 2 * gap) / 2
    records = []
    for index, (k, t, title) in enumerate(views):
        file = IDENTIFICATION / f'inputs_{k}/camera_0.npy'
        frames = np.load(file, mmap_mode='r')
        frame = int(np.argmin(abs(times - t)))
        # Identical sensor, crop and display scale; no per-material recoloring.
        picture = frames[frame, 180:1080, 100:1180]
        ia = common.image_at(ax, np.repeat(picture[..., None], 3, axis=-1),
                             [left, .015, width, height])
        if index == 0:
            # A single motion arrow in the clear space above the upper plate.
            # Compensate for the arrowhead's stroke inset so its visible tip
            # meets the plate's top edge at image y=174.
            ia.add_patch(FancyArrowPatch((540, 12), (540, 186),
                arrowstyle='-|>', mutation_scale=11, lw=1.25,
                color='#45677a', shrinkA=0, shrinkB=0, zorder=5))
        ax.text(left + width / 2, common.TOP_HEIGHT - .30, title,
                ha='center', va='center', fontsize=7.4, color=common.COLORS[k])
        plt.imsave(images / f'plate_view_{index}.png', picture, cmap='gray', vmin=0, vmax=255)
        records.append(dict(material=k, source=str(file), frame=frame, time_s=float(times[frame]),
                            crop=[180, 1080, 100, 1180], grayscale_unchanged=True))
        left += width + gap
    save_json(images / 'plate_views.json', records)


def panel_b(panel):
    common.panel_frame(panel, '(b) Held-out plate-force prediction')
    ax = common.inset_at(panel, [.35, .265, common.PANEL_WIDTH - .39, .705])
    for k in 'AB':
        reference = np.genfromtxt(IDENTIFICATION / f'truth_{k}/force.csv', names=True, delimiter=',')
        prediction = np.genfromtxt(IDENTIFICATION / f'prediction_{k}/force.csv', names=True, delimiter=',')
        np.testing.assert_array_equal(reference['time'], prediction['time'])
        keep = reference['time'] >= 3.9
        t, f = reference['time'][keep] - 3.9, reference['Fz'][keep]
        ax.plot(t, f, color=common.COLORS[k], lw=1.15)
        ax.plot(t, prediction['Fz'][keep], color=common.COLORS[k], lw=0,
                marker='o', markersize=2.3, markerfacecolor='white', markeredgewidth=.65, markevery=32)
        peak = np.argmax(f)
        ax.annotate(k, (t[peak], f[peak]), xytext=(3, 3), textcoords='offset points',
                    color=common.COLORS[k], fontsize=7.4, weight='bold')
    ax.set(xlabel='Time (s)', ylabel='Force (N)', xlim=(0, 1.4), ylim=(-.8, 22))
    ax.set_xticks([0, .4, .8, 1.2]); ax.set_yticks([0, 10, 20])
    ax.tick_params(labelsize=7.4, pad=2, length=3)
    ax.xaxis.label.set_size(7.4); ax.yaxis.label.set_size(7.4)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 2
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(.7)
    ax.grid(axis='y', alpha=.15)
    ax.legend(handles=[Line2D([], [], color='#333333', lw=1.2, label='True'),
        Line2D([], [], color='#333333', marker='o', markerfacecolor='white', lw=0,
               markersize=2.5, label='Identified')], loc='upper left', fontsize=7.4,
               frameon=False, handlelength=1.5, labelspacing=.35, borderaxespad=.2)


def report(output):
    protocol = search.check(SHAPING)
    assert protocol['grid'] == 80 and protocol['spec']['max_evaluations'] == 48
    models = core.read(SHAPING / 'inputs/models.json')
    for name, threshold in [('A', 1000.), ('B', 10000.)]:
        assert models['true_' + name] == dict(E=80000., nu=.3, yield_stress=threshold)
        fit = core.read(IDENTIFICATION / f'divfree_{name}/identification.json')
        assert models['identified_' + name] == dict(E=fit['E_pa'], nu=.3, yield_stress=fit['yield_pa'])
    manifest = core.read(SHAPING / 'checksums.json')
    for rel in ['summary.json', 'protocol.json', 'execution_plan.json',
                'preview/provenance.json', 'inputs/target.vtp']:
        assert core.digest(SHAPING / rel) == manifest[rel]
    records = {r['material'] + '_plan_' + r['planned_for']: r
               for r in core.read(SHAPING / 'summary.json')['results']}
    frozen = core.read(SHAPING / 'execution_plan.json')
    for key, row in records.items():
        path = SHAPING / (key + '.npz')
        assert core.digest(path) == row['data_sha256'] == manifest[path.name]
        selected = frozen['selected'][row['planned_for']]
        with np.load(path) as data, np.load(SHAPING / selected['file']) as planned:
            for channel in ['time', 'tool_centers', 'phase_id', 'pinch_id', 'start_pose', 'gaps_mm']:
                np.testing.assert_array_equal(data[channel], planned[channel])
        assert row['law'] == models['true_' + row['material']]

    output.mkdir(parents=True, exist_ok=False)
    images = output / 'renders'
    images.mkdir()
    xml = ROBOT
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42})
    width, height = common.FIGURE_SIZE
    fig = plt.figure(figsize=(width, height))
    panels = []
    for left, bottom, panel_height in [(x, y, h) for y, h in
            [(2.345, common.TOP_HEIGHT), (.035, common.BOTTOM_HEIGHT)] for x in [.035, 3.72]]:
        ax = fig.add_axes([left / width, bottom / height,
                          common.PANEL_WIDTH / width, panel_height / height])
        ax.set(xlim=(0, common.PANEL_WIDTH), ylim=(0, panel_height))
        panels.append(ax)
    panel_a(panels[0], images)
    panel_b(panels[1])

    ax = panels[2]
    common.panel_frame(ax, '(c) Distance-controlled shaping')
    data = core.arrays(SHAPING / 'A_plan_A.npz')
    image_width = (common.PANEL_WIDTH - .08) / 2
    image_height = image_width * 650 / 700
    action_records = []
    for stage, left, direction in [(4, 0., 'y'), (5, image_width + .08, 'x')]:
        indices = np.flatnonzero(data['pinch_id'] == stage)
        centers = data['tool_centers'][indices[-1]]
        direction_vector = centers[1] - centers[0]
        angle = np.rad2deg(np.arctan2(direction_vector[1], direction_vector[0]))
        opening = (np.linalg.norm(centers[1] - centers[0]) - .028) * 1000
        np.testing.assert_allclose(opening, data['gaps_mm'][stage], atol=1e-7)
        view = dict(vol0=data['vol0'], angles_deg=np.full(6, angle),
                    **{f'stage_{stage}_pressed': data[f'{stage}_close'],
                       f'stage_{stage}_centers': centers})
        picture = render_action(view, stage=stage, color=common.SURFACE_COLORS['A'],
                                path=images / f'action_{stage}.png', robot_xml=xml)
        common.image_at(ax, picture, [left, common.BOTTOM_HEIGHT - .37 - image_height,
                                     image_width, image_height])
        cx = left + image_width / 2
        ax.text(cx, common.BOTTOM_HEIGHT - .30, f'{stage + 1}. Pinch along {direction}',
                ha='center', va='center', fontsize=7.4)
        ax.text(cx, .31, f'Opening: {opening:.1f} mm', ha='center', va='center', fontsize=7.4, zorder=20)
        action_records.append(dict(stage=stage + 1, saved_state=f'{stage}_close',
                                   command_index=int(indices[-1]), opening_mm=float(opening)))

    ax = panels[3]
    common.panel_frame(ax, '(d) Target and executed shapes')
    source_preview = SHAPING / 'preview'
    calibration = core.read(source_preview / 'provenance.json')
    y0, y1, x0, x1 = calibration['crop']
    target = pv.read(SHAPING / 'inputs/target.vtp')
    mask = core.render(target, calibration['common_camera_scale_m'], 'black', mask=True)[y0:y1, x0:x1, :3].mean(-1) < 128
    pictures = {}
    for key in ['target', *records]:
        rel = f'preview/{key}.png'
        assert core.digest(SHAPING / rel) == manifest[rel]
        shutil.copy2(SHAPING / rel, images / (key + '.png'))
        pictures[key] = plt.imread(images / (key + '.png'))
        assert pictures[key].shape[:2] == mask.shape
    columns = [.54, 1.68, 2.84]
    for x, label in zip(columns, ['Target', 'Plan using matched ID', 'Plan using swapped ID']):
        ax.text(x, common.BOTTOM_HEIGHT - .30, label, ha='center', va='center', fontsize=7.)
    image_height = .72
    image_width = image_height * mask.shape[1] / mask.shape[0]
    for row, actual in enumerate('AB'):
        bottom = [common.BOTTOM_HEIGHT - .37 - image_height, .12][row]
        ax.text(.025, bottom + image_height / 2, actual, color=common.COLORS[actual],
                fontsize=8., weight='bold', va='center')
        for col, planned in enumerate([None, actual, 'B' if actual == 'A' else 'A']):
            key = 'target' if planned is None else f'{actual}_plan_{planned}'
            ia = common.image_at(ax, pictures[key], [columns[col] - image_width / 2, bottom,
                                                   image_width, image_height])
            if planned is not None:
                ia.contour(mask, [.5], colors='#253139', linewidths=.55, linestyles=[(0, (2.4, 1.8))])
                ax.text(columns[col], bottom - .025, f"{records[key]['surface_mm']:.3f} mm",
                        ha='center', va='top', fontsize=7.4)

    for suffix in ['.png', '.pdf']:
        fig.savefig(output / ('identification_plastic_shaping' + suffix), dpi=300,
                    bbox_inches='tight', pad_inches=.02)
    for name, ax in zip('abcd', panels):
        bounds = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output / f'panel_{name}.png', dpi=300, bbox_inches=bounds.expanded(1.01, 1.01))
    plt.close(fig)
    source = output / 'source'
    source.mkdir()
    for name in ['plate_to_x_figure.py', 'x_common_report.py',
                 'x_shaping_report.py', 'x_shaping_franka.py', 'x_motion_shaping.py']:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    save_json(output / 'provenance.json', dict(identification=str(IDENTIFICATION), shaping=str(SHAPING),
        shaping_manifest_sha256=core.digest(SHAPING / 'checksums.json'),
        identification_fits={k: core.digest(IDENTIFICATION / f'divfree_{k}/identification.json') for k in 'AB'},
        input_camera_sha256={k: core.digest(IDENTIFICATION / f'inputs_{k}/camera_0.npy') for k in 'AB'},
        plate_views=core.read(images / 'plate_views.json'),
        force_files={f'{kind}_{k}': core.digest(IDENTIFICATION / f'{kind}_{k}/force.csv')
                     for kind in ['truth', 'prediction'] for k in 'AB'},
        actions=action_records, results=records, outcome_calibration=calibration,
        outcome_images='All four newly executed grid80 outcomes; one common scale and crop',
        robot_scope='Kinematic hand illustration at recorded collider poses, not hardware execution',
        robot_assets={str(p.relative_to(ROBOT.parent)): core.digest(p)
                      for p in sorted(ROBOT.parent.rglob('*')) if p.is_file()},
        control='Six prescribed openings and shared placement offset; no force or shape feedback',
        new_simulations_during_rendering=False, sources={p.name: core.digest(p) for p in source.iterdir()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shaping', type=Path, default=SHAPING)
    parser.add_argument('--identification', type=Path, default=IDENTIFICATION)
    args = parser.parse_args()
    SHAPING, IDENTIFICATION = args.shaping, args.identification
    report(args.output)
