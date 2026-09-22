"""Seven-panel paper strip assembled from audited, frozen simulation renders."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'out/press_shaping_centered_arrow_20260914/figure'
OUT = ROOT / 'out/press_shaping_strip_20260915'
PAPER = ROOT / 'paper/icra2027'
INK = '#26343d'
COLORS = {'A': '#2375aa', 'B': '#c96932'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    before = OUT / 'before'
    before.mkdir(exist_ok=True)
    for relative in ['paper.tex', *[
            'figs/identification_plastic_shaping.' + ext
            for ext in ['pdf', 'png', 'provenance.json']]]:
        dest = before / Path(relative).name
        if not dest.exists():
            shutil.copy2(PAPER / relative, dest)
    meta = json.loads((before / 'identification_plastic_shaping.provenance.json').read_text())
    assets = OUT / 'renders'
    assets.mkdir(exist_ok=True)
    order = ['A_plan_A', 'A_plan_B', 'B_plan_B', 'B_plan_A']
    names = ['plate_view_1.png', 'action_5.png', 'target.png', *[k + '.png' for k in order]]
    for name in names:
        src = SOURCE / 'renders' / name
        assert digest(src) == meta['input_render_sha256'][name]
        shutil.copy2(src, assets / name)
    shutil.copy2(SOURCE / 'renders/target_mask.npy', assets / 'target_mask.npy')
    mask = np.load(assets / 'target_mask.npy')
    # Raw outcomes are portrait. The preceding layout rolled them clockwise;
    # restoring the saved orientation rotates all four views 90 degrees back.
    panels = [plt.imread(assets / names[0]),
              plt.imread(assets / names[1])[60:560, 50:650],
              plt.imread(assets / 'target.png')]
    panels.extend(plt.imread(assets / (k + '.png')) for k in order)
    assert all(p.shape[:2] == mask.shape for p in panels[2:])
    ratios = [p.shape[1] / p.shape[0] for p in panels]
    W, margin, gap = 7.16, .015, .03
    image_height = (W - 2 * margin - (len(panels) - 1) * gap) / sum(ratios)
    H = image_height + .235
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.4,
                         'pdf.fonttype': 42, 'text.color': INK})
    fig = plt.figure(figsize=(W, H), facecolor='white')
    titles = ['Press', 'Shaping', 'Target', 'Matched ID (A)', 'Swapped ID (A)',
              'Matched ID (B)', 'Swapped ID (B)']
    left = margin
    for i, (panel, ratio, title) in enumerate(zip(panels, ratios, titles)):
        width = ratio * image_height
        ax = fig.add_axes([left/W, .015/H, width/W, image_height/H])
        ax.imshow(panel)
        ax.axis('off')
        if i == 0:
            # Start at the upper plate's lower front-left corner, where the
            # two vertical faces meet. End directly below on the lower plate's
            # front upper edge. Coordinates refer to the frozen camera crop.
            half_shaft = (4 / 72) / width * panel.shape[1] / 2
            x, y_top, y_tip, shoulder, half_head = 128., 289., 725., 631., 52.
            vertices = [(x-half_shaft, y_top), (x+half_shaft, y_top),
                        (x+half_shaft, shoulder), (x+half_head, shoulder),
                        (x, y_tip), (x-half_head, shoulder), (x-half_shaft, shoulder)]
            ax.add_patch(Polygon(vertices, facecolor='#287FA0', edgecolor='white',
                                 linewidth=.7, joinstyle='miter'))
        elif i >= 3:
            ax.contour(mask, [.5], colors=INK, linewidths=.55,
                       linestyles=[(0, (2.4, 1.8))])
        fig.text((left+width/2)/W, (H-.025)/H, title, ha='center', va='top',
                 weight='bold', color=INK if i < 3 else COLORS[order[i-3][0]])
        left += width + gap
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / ('identification_plastic_shaping.' + ext), dpi=400, pad_inches=0)
    plt.close(fig)
    provenance = dict(
        source_layout_provenance=str(before / 'identification_plastic_shaping.provenance.json'),
        source_layout_sha256=digest(before / 'identification_plastic_shaping.provenance.json'),
        scientific_provenance=meta['scientific_provenance'],
        input_render_sha256={name: digest(assets/name) for name in names},
        target_mask_sha256=digest(assets/'target_mask.npy'),
        panel_order=titles, action_saved_state='5_close',
        press_frame=meta['scientific_provenance']['plate_views'][1],
        outcome_rotation_from_previous_deg=90,
        outcome_rotation_direction='counterclockwise, including all target outlines',
        outcome_rotation_from_raw_renders_deg=0,
        common_outcome_scale=True, new_simulations=False,
        individual_shape_registration_or_rescaling=False,
        size_inches=[W,H], previous_size_inches=meta['layout']['size_inches'],
        script_sha256=digest(Path(__file__)))
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    for ext in ['pdf', 'png']:
        shutil.copy2(OUT / ('identification_plastic_shaping.'+ext),
                     PAPER / ('figs/identification_plastic_shaping.'+ext))
    shutil.copy2(OUT/'provenance.json', PAPER/'figs/identification_plastic_shaping.provenance.json')
    print(f'Figure height: {H:.3f} in; previous: {meta["layout"]["size_inches"][1]:.3f} in')


if __name__ == '__main__':
    main()
