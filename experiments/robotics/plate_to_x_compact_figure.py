"""Two-row composition of the audited separated-identification study.

Reuses the frozen camera and simulation renders; no simulations or fitting.
The full original figure and force-validation evidence remain in the source.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import numpy as np
import pyvista as pv

from experiments.robotics import x_motion_shaping as core

ROOT = Path(__file__).resolve().parents[2]
INK = '#26343d'
COLORS = {'A': '#2375aa', 'B': '#c96932'}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make(source, output):
    meta = json.loads((source / 'provenance.json').read_text())
    shaping = Path(meta['shaping'])
    assert digest(shaping / 'checksums.json') == meta['shaping_manifest_sha256']
    manifest = json.loads((shaping / 'checksums.json').read_text())
    assert digest(shaping / 'summary.json') == manifest['summary.json']
    rows = {r['material'] + '_plan_' + r['planned_for']: r for r in
            json.loads((shaping / 'summary.json').read_text())['results']}
    assert rows == meta['results']
    assert meta['outcome_calibration']['all_components_included']
    assert not meta['outcome_calibration']['individual_registration_or_rescaling']
    output.mkdir(parents=True, exist_ok=True)
    assets = output / 'renders'
    assets.mkdir(exist_ok=True)
    inputs = {}
    for p in (source / 'renders').glob('*.png'):
        if p.stem in ['target', *rows]:
            assert digest(p) == manifest[f'preview/{p.name}']
        inputs[p.name] = digest(p)
        shutil.copy2(p, assets / p.name)
    archive = Path(meta['plate_views'][0]['source'])
    assert digest(archive) == meta['input_camera_archive_sha256']
    with np.load(archive) as data:
        for i, row in enumerate(meta['plate_views']):
            y0, y1, x0, x1 = row['crop']
            # The original Matplotlib gray-colormap export rounds a few values
            # down by one 8-bit level; verify its exact colormap conversion.
            expected = plt.get_cmap('gray')(
                data[row['archive_key']][y0:y1, x0:x1] / 255., bytes=True)
            np.testing.assert_allclose(
                plt.imread(assets / f'plate_view_{i}.png')[..., 0] * 255,
                expected[..., 0], atol=2e-5)

    calibration = meta['outcome_calibration']
    assert digest(shaping / 'inputs/target.vtp') == manifest['inputs/target.vtp']
    y0, y1, x0, x1 = calibration['crop']
    mask = core.render(pv.read(shaping / 'inputs/target.vtp'),
                       calibration['common_camera_scale_m'], 'black', mask=True)
    mask = mask[y0:y1, x0:x1, :3].mean(-1) < 128
    np.save(assets / 'target_mask.npy', mask)

    W = 7.16
    margin, gap = .015, .03
    column_width = (W - 2 * margin - 4 * gap) / 5
    columns = [margin + i * (column_width + gap) for i in range(5)]
    outcome_height = column_width * mask.shape[1] / mask.shape[0]
    outcome_bottom = .015
    result_label_y = outcome_bottom + outcome_height + .13
    result_heading_y = result_label_y + .163
    image_top = result_heading_y + .05 + column_width * 900 / 1080
    image_label_y = image_top + .132
    H = image_label_y + .20
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.4,
                         'pdf.fonttype': 42, 'text.color': INK})
    fig = plt.figure(figsize=(W, H), facecolor='white')

    def text(x, y, label, **kw):
        return fig.text(x / W, y / H, label, fontsize=7.4, **kw)

    def picture(name, x, y, w, h, crop=None, rotate=False):
        data = plt.imread(assets / name)
        if crop is not None:
            data = data[crop[0]:crop[1], crop[2]:crop[3]]
        if rotate:
            data = np.rot90(data, k=-1)
        ax = fig.add_axes([x / W, y / H, w / W, h / H])
        ax.imshow(data)
        ax.axis('off')
        return ax

    def heading(x, y, label):
        return fig.text(x / W, y / H, label, fontsize=8.5, weight='bold', va='top')

    identification_heading = heading(columns[0], H - .02, '(a) Material identification')
    planning_heading = heading(columns[3], H - .02, '(b) Planned shaping')
    # Keep the right endpoint fixed; balance the gaps using rendered text bounds.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    identification_right = identification_heading.get_window_extent(renderer).x1 / fig.dpi
    planning_left = planning_heading.get_window_extent(renderer).x0 / fig.dpi
    arrow_right = columns[3] - .14
    arrow_gap = planning_left - arrow_right
    arrow_left = identification_right + arrow_gap
    fig.add_artist(FancyArrowPatch(
        (arrow_left / W, (H - .082) / H), (arrow_right / W, (H - .082) / H),
        transform=fig.transFigure, arrowstyle='-|>', mutation_scale=9,
        linewidth=1.1, color='#637e8a', shrinkA=0, shrinkB=0))
    heading(columns[0], result_heading_y, '(c) Shaping results')

    # One strip of observed presses followed by the final two planned pinches.
    # Camera views share an unchanged crop and display scale.
    for i, (name, material) in enumerate([
            ('Early press', 'A'), ('Deep press', 'A'), ('Deep press', 'B')]):
        left, w = columns[i], column_width
        text(left + w / 2, image_label_y, f'{name} ({material})',
             color=COLORS[material], ha='center', va='top')
        height = w * 900 / 1080
        ax = picture(f'plate_view_{i}.png', left, image_top - height, w, height)
        if i == 1:
            arrow = meta['plate_views'][i]['motion_arrow']
            ax.add_patch(Polygon(arrow['vertices_px'], closed=True,
                                 facecolor='#45677a', edgecolor='none'))
    # Common crop removes excess floor and side margins while retaining both
    # contacts and the whole specimen. Its aspect ratio equals the press views.
    action_crop = [60, 560, 50, 650]
    assert (action_crop[1] - action_crop[0]) * 1080 == (action_crop[3] - action_crop[2]) * 900
    for i, (stage, axis) in enumerate([(4, 'y'), (5, 'x')]):
        left, w = columns[i + 3], column_width
        text(left + w / 2, image_label_y, f'Pinch along {axis}', ha='center', va='top')
        height = w * 900 / 1080
        picture(f'action_{stage}.png', left, image_top - height, w, height, action_crop)

    # One common target, followed by matched/swapped pairs for each material.
    order = ['target', 'A_plan_A', 'A_plan_B', 'B_plan_B', 'B_plan_A']
    # A common, lossless camera-image roll, uniformly enlarged to column width.
    # Rotate the target mask identically; never transform an individual result.
    w, h = column_width, outcome_height
    rotated_mask = np.rot90(mask, k=-1)
    for i, key in enumerate(order):
        center = columns[i] + column_width / 2
        label = 'Target' if key == 'target' else (
            ('Matched ID' if rows[key]['material'] == rows[key]['planned_for'] else 'Swapped ID')
            + f" ({rows[key]['material']})")
        color = INK if key == 'target' else COLORS[rows[key]['material']]
        text(center, result_label_y, label, color=color, ha='center', va='top')
        ax = picture(key + '.png', center - w / 2, outcome_bottom, w, h, rotate=True)
        if key != 'target':
            ax.contour(rotated_mask, [.5], colors=INK, linewidths=.55,
                       linestyles=[(0, (2.4, 1.8))])

    for suffix in ['pdf', 'png']:
        fig.savefig(output / f'identification_plastic_shaping.{suffix}', dpi=400,
                    pad_inches=0)
    plt.close(fig)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    provenance = dict(
        source_figure=str(source.resolve()), source_provenance_sha256=digest(source / 'provenance.json'),
        scientific_provenance=meta, input_render_sha256=inputs,
        layout=dict(size_inches=[W, H], body_font_pt=7.4, heading_font_pt=8.5,
                    headings=['Material identification', 'Planned shaping', 'Shaping results'],
                    heading_arrow=dict(left_inches=arrow_left, right_inches=arrow_right,
                                       equal_text_gap_inches=arrow_gap,
                                       identification_text_right_inches=identification_right,
                                       planning_text_left_inches=planning_left),
                    target_and_outcomes_common_clockwise_rotation_deg=90,
                    outcome_common_scale_factor_from_previous_layout=column_width / .98,
                    column_left_inches=columns, common_column_width_inches=column_width,
                    top_images_common_height_inches=column_width * 900 / 1080,
                    numerical_annotations='Opening commands and surface errors in manuscript text',
                    top=['early_A', 'deep_A', 'deep_B', 'pinch_y', 'pinch_x'],
                    bottom=order, action_common_crop=action_crop,
                    target_and_outcomes_common_display_height_inches=h),
        omitted_panel='Force validation; source data and original figure preserved',
        new_simulations=False, individual_shape_registration_or_rescaling=False,
        files={p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != 'provenance.json'})
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'out/press_paper_update_20260913/figure')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    make(args.source, args.output)
