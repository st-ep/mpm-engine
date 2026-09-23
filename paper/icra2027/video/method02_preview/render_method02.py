"""Review layout for the combined FORM identification / planning section.

Native Python composition using frozen research assets. No model fitting,
simulation, or manuscript mutation. Coordinates are logical 1280 x 720 pixels.
Run with .venv/bin/python; outputs a 2x PNG and a 720p review PNG.
"""
from pathlib import Path
from functools import lru_cache
import io
import json
import sys
import argparse
import subprocess
import cv2

import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import math_to_image, MathTextParser
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image, ImageDraw

P = Path(__file__).resolve().parent
VIDEO = P.parent
sys.path.insert(0, str(VIDEO))
from method01_preview.render_method01 import (
    Canvas, S, BG, INK, TEAL, BLUE, ORANGE, MUTED, LINE,
    data, Projection, lerp_frame, centered_positions, DISPLACEMENT,
)

ASSETS = VIDEO / 'assets'
DURATION = 24
FPS = 25
TEXT_RECORDS = []


@lru_cache(None)
def math_token(source, size, color):
    """Render at a common font size, retaining baseline and subscript depth."""
    from PIL import ImageColor
    raster = MathTextParser('agg').parse('$'+source+'$', dpi=72*S,
                                       prop=FontProperties(size=size))
    mask = np.asarray(raster.image).copy()
    rgba = np.empty((*mask.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = ImageColor.getrgb(color)
    rgba[:, :, 3] = mask
    return Image.fromarray(rgba), raster.depth/S


class ReviewCanvas(Canvas):
    def text(self, x, y, s, n=18, c=INK, bold=False, anchor=None):
        from method01_preview.render_method01 import font
        bounds = self.d.textbbox((x*S, y*S), s, font=font(n, bold), anchor=anchor)
        TEXT_RECORDS.append({'text': s, 'size_px': n, 'bbox': [v/S for v in bounds]})
        super().text(x, y, s, n, c, bold, anchor)


@lru_cache(None)
def equation(source, height, color=INK):
    out = io.BytesIO()
    math_to_image('$'+source+'$', out, dpi=230, format='png', color=color)
    im = Image.open(out).convert('RGBA')
    # math_to_image has an opaque white canvas; convert ink coverage to alpha.
    a = np.asarray(im).copy()
    from PIL import ImageColor
    rgb = np.array(ImageColor.getrgb(color))
    channel = int(np.argmin(rgb))
    alpha = np.clip((255-a[:, :, channel].astype(float))/(255-rgb[channel]), 0, 1)
    a[:, :, :3] = rgb
    a[:, :, 3] = np.rint(alpha*255).astype(np.uint8)
    im = Image.fromarray(a)
    im = im.crop(im.getbbox())
    return im.resize((round(im.width/im.height*height*S), round(height*S)), Image.Resampling.LANCZOS)


def math(c, source, x, y, height, color=INK, center=True, max_width=None):
    tile = equation(source, height, color)
    if max_width:
        assert tile.width <= max_width*S, (source, tile.width/S, max_width)
    xx = round(x*S-tile.width/2) if center else round(x*S)
    yy = round(y*S-tile.height/2)
    assert 0 <= xx and xx+tile.width <= 1280*S and yy >= 0 and yy+tile.height <= 720*S
    c.im.paste(tile, (xx, yy), tile)
    return (xx/S, yy/S, (xx+tile.width)/S, (yy+tile.height)/S)


def asset(c, name, box, white_to_bg=False):
    im = Image.open(ASSETS / (name+'.png')).convert('RGBA')
    if white_to_bg:
        a = np.asarray(im).copy()
        # Only replace near-white background; preserve object geometry and shading.
        a[np.min(a[:, :, :3], axis=2) > 250, 3] = 0
        im = Image.fromarray(a)
    alpha = im.getchannel('A')
    if alpha.getextrema()[0] == 0:
        im = im.crop(alpha.getbbox())
    x, y, w, h = box
    scale = min(w*S/im.width, h*S/im.height)
    im = im.resize((round(im.width*scale), round(im.height*scale)), Image.Resampling.LANCZOS)
    c.im.paste(im, (round((x+w/2)*S-im.width/2), round((y+h/2)*S-im.height/2)), im)


def motion(c, source_time=1.3):
    D = data()
    source, _, _ = lerp_frame(D['hp']['pos'], D['hp']['times'], source_time)
    points = centered_positions(source, D['reference'])
    proj = Projection(D['display_pos'][D['hsel'][::3], ::3].reshape(-1, 3), (327, 196, 203, 132))
    xy = proj(points)
    dep = points @ proj.view
    brightness = .80+.20*(dep-dep.min())/np.ptp(dep)
    displacement = np.linalg.norm(source-D['reference'], axis=1)
    colors = DISPLACEMENT(np.clip(displacement/D['disp_max_m'], 0, 1))[:, :3]*255
    for j in np.argsort(dep):
        c.dot(xy[j], .85, tuple((colors[j]*brightness[j]).astype(int)))


def force(c, source_time=2.):
    f = data()['force']
    use = (f['time'] >= 0) & (f['time'] <= source_time)
    t, v = f['time'][use], f['incremental_normal_force'][use]
    c.line([(76, 211), (76, 312), (252, 312)], '#a9bac3', 1.2)
    if len(t) > 1:
        c.line(np.c_[76+t/2*176, 312-v/12*101], ORANGE, 2.7)
    c.text(66, 201, 'N', 16, MUTED, anchor='rt')
    c.text(68, 228, '10', 16, MUTED, anchor='rm')
    c.text(76, 319, '0', 16, MUTED, anchor='mt')
    c.text(252, 319, '2 s', 16, MUTED, anchor='mt')


def law_curve(c):
    # Explicitly schematic stress/strain response, as in Fig. 2; not fitted data.
    c.arrow((640, 649), (806, 649), '#8197a5', 1.2, 5)
    c.arrow((640, 649), (640, 558), '#8197a5', 1.2, 5)
    c.line([(648, 643), (708, 587), (798, 587)], BLUE, 3)
    c.line([(708, 587), (708, 649)], '#bed3df', 1)
    c.text(669, 605, 'E', 20, BLUE, True)
    c.text(795, 561, 'Y', 20, BLUE, True, 'mt')
    c.text(624, 559, 'Stress', 16, MUTED, anchor='rt')
    c.text(806, 655, 'Strain', 16, MUTED, anchor='rt')


@lru_cache(1)
def layout():
    TEXT_RECORDS.clear()
    im = Image.new('RGB', (1280*S, 720*S), BG)
    c = ReviewCanvas(im)
    c.text(44, 22, 'FORM · From Observed Response to Material laws', 23, TEAL, True)
    c.text(1236, 22, '02 / IDENTIFY & PLAN', 20, TEAL, True, 'rt')
    c.text(44, 56, 'Recover the material law. Plan the robot action.', 41, INK, True)
    c.line([(44, 114), (1236, 114)], LINE, 1)

    # Three physical / mathematical inputs, all contributing to the weak balance.
    c.text(164, 149, 'Measured force', 21, ORANGE, True, 'mt')
    c.text(429, 149, 'Reconstructed motion', 21, BLUE, True, 'mt')
    c.text(711, 149, 'Chosen test field', 21, TEAL, True, 'mt')
    force(c)
    motion(c)
    asset(c, 'field', (599, 190, 223, 148))
    math(c, r'\nabla\!\cdot\!\psi_j=0', 711, 358, 20, TEAL)
    c.text(711, 382, 'Cancel pressure', 17, TEAL, True, 'mt')

    # Match the paper's Fig. 2; only selected terms carry semantic color.
    c.rect((56, 417, 839, 500), '#eaf0f3', 12)
    pieces = [
        (r'b_j', ORANGE),
        (r'\simeq', INK),
        (r'\int\!\sum_p V_p', INK),
        (r'\sigma_p(\theta)', BLUE),
        (r':', INK),
        (r'D[w_j]', TEAL),
        (r'\,dt', INK),
    ]
    tiles = [math_token(s, 31, col) for s, col in pieces]
    total = sum(q.width/S for q, depth in tiles)+6*10
    x = (56+839-total)/2
    centers = []
    for tile, depth in tiles:
        im.paste(tile, (round(x*S), round((460+depth)*S-tile.height)), tile)
        centers.append(x+tile.width/S/2)
        x += tile.width/S+10
    c.arrow((164, 348), (centers[0], 425), ORANGE, 2.2, 8)
    c.arrow((429, 342), (centers[3], 425), BLUE, 2.2, 8)
    c.arrow((711, 409), (centers[5], 425), TEAL, 2.2, 8)
    c.text(centers[0], 479, 'Motion + loads', 17, ORANGE, True, 'mt')
    c.text(557, 479, 'Weighted internal stress', 17, INK, True, 'mt')

    # Reduce to coefficients, then an explicit recovered law.
    c.text(62, 524, 'One linear least-squares solve', 23, INK, True)
    math(c, r'\tau(q;\theta)=\sum_k\theta_k T_k(q)', 278, 578, 36, INK, max_width=440)
    math(c, r'\hat\theta=\arg\min_{\theta}\|A\theta-b\|_2^2', 278, 635, 39, BLUE, max_width=444)
    c.text(278, 678, 'No forward simulation during identification', 17, MUTED, anchor='mt')
    c.arrow((510, 612), (564, 612), BLUE, 2.2, 8)
    c.text(716, 524, 'Identified law', 23, BLUE, True, 'mt')
    law_curve(c)
    c.text(716, 678, 'Stiffness + yield · schematic', 16, MUTED, anchor='mt')

    # Planning uses the recovered law, held fixed; no model refitting here.
    c.line([(866, 145), (866, 550)], LINE, 1.2)
    c.line([(866, 623), (866, 691)], LINE, 1.2)
    c.text(1062, 149, 'Plan with MPM', 25, INK, True, 'mt')
    asset(c, 'target', (945, 190, 232, 106), white_to_bg=True)
    c.text(1210, 227, 'Target', 18, MUTED, anchor='rt')
    c.arrow((1062, 301), (1062, 319), TEAL, 2.2, 7)
    asset(c, 'shape_action', (899, 355, 337, 208), white_to_bg=True)
    c.text(1062, 326, 'Simulate candidate actions', 18, INK, True, 'mt')
    c.arrow((832, 587), (897, 587), BLUE, 2.3, 8)
    math(c, r'u^\star\!\in\!\arg\min_u\,J(S_{\hat\theta}(x_0,u),G)', 1066, 589, 31, INK, max_width=340)
    c.text(1062, 632, 'Keep the law fixed.', 21, BLUE, True, 'mt')
    c.text(1062, 660, 'Search robot actions.', 21, INK, True, 'mt')
    return im


@lru_cache(100)
def animated_inputs(frame):
    im = Image.new('RGB', (1280*S, 720*S), BG)
    c = Canvas(im)
    source_time = min(2., .05+frame/FPS*.5)
    force(c, source_time)
    motion(c, source_time)
    return im.crop((50*S, 191*S, 541*S, 340*S))


@lru_cache(1)
def shape_capture():
    cap = cv2.VideoCapture(str(VIDEO.parents[1]/'videos/06_shaping_simulation_matched_swapped.mp4'))
    assert cap.isOpened()
    return cap


def progress(t, start):
    x = float(np.clip((t-start)/.6, 0, 1))
    return x*x*(3-2*x)


def draw_method02(t=None, reveal=True, high_resolution=False):
    im = layout().copy()
    if t is not None:
        tile = animated_inputs(int(round(t*FPS)) % (4*FPS))
        im.paste(tile, (50*S, 191*S))
        # One frozen matched-A plan shown as an example of MPM action prediction.
        # Playback is 2.5x; there is no new optimization or fabricated search trace.
        cap = shape_capture()
        source_frame = min(int(max(0, t-14)*2.5*FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, bgr = cap.read()
        assert ok, source_frame
        pic = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).crop((56, 56, 816, 576))
        box = (899, 350, 337, 218)
        x, y, w, h = box
        scale = min(w*S/pic.width, h*S/pic.height)
        pic = pic.resize((round(pic.width*scale), round(pic.height*scale)), Image.Resampling.LANCZOS)
        ImageDraw.Draw(im).rectangle((899*S, 350*S, 1236*S, 568*S), fill=BG)
        im.paste(pic, (round((x+w/2)*S-pic.width/2), round((y+h/2)*S-pic.height/2)))
        if reveal:
            # Preserve context in faint silhouettes, as on the approved Observe slide.
            for bounds, start in [((56, 341, 839, 509), 4),
                                  ((56, 514, 830, 703), 9),
                                  ((831, 137, 1244, 703), 14)]:
                region = tuple(round(v*S) for v in bounds)
                tile = im.crop(region)
                opacity = .07+.93*progress(t, start)
                im.paste(Image.blend(Image.new('RGB', tile.size, BG), tile, opacity), region[:2])
    return im if high_resolution else im.resize((1280, 720), Image.Resampling.LANCZOS)


def preview():
    result = draw_method02(high_resolution=True)
    result.save(P / 'method02_identify_plan.png')
    result.resize((1280, 720), Image.Resampling.LANCZOS).save(P / 'method02_identify_plan_720p.png')
    (P/'layout_review.json').write_text(json.dumps({'text': TEXT_RECORDS,
        'reference': 'Final supplied 2027_ICRA_FORM.pdf, Fig. 2 and Eqs. 8–11',
        'status': 'Combined second method section. Static layout and progressive animation.',
        'source_time_s': 1.3,
        'motion': 'Same saved RGB-D reconstructed particles and displacement colors as approved Observe section.',
        'force': 'Same recorded baseline-subtracted normal force as approved Observe section.',
        'test_field': 'Existing Fig. 2 asset. Vectors are mathematical test weights, not material motion.',
        'law_curve': 'Schematic E/Y response; does not represent a fit to the displayed hardware specimen.',
        'shaping': 'Existing simulated shaping illustration from Fig. 2; not a new forward simulation.',
        'claims': 'Linear coefficients are recovered from weak balances. The identified model is held fixed during MPM action search.',
        'animation': {'duration_s': DURATION, 'fps': FPS,
                      'reveals_s': {'weak_balance': 4, 'linear_solve_and_law': 9, 'planning': 14},
                      'fade_s': .6,
                      'inputs': 'Synchronized 4 s loops of the same .05–2 s hardware motion / force at 0.5x.',
                      'test_field': 'Static chosen mathematical weights, not animated physical motion.',
                      'planning': 'Saved matched-A shaping sequence at 2.5x; one example prediction, not an optimization history.',
                      'shape_source': str(VIDEO.parents[1]/'videos/06_shaping_simulation_matched_swapped.mp4'),
                      'shape_crop_xyxy': [56, 56, 816, 576]}
    }, indent=2)+'\n')
    print(P / 'method02_identify_plan.png')


def render():
    output = P/'method02_identify_plan.mp4'
    command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1280x720', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '17', '-threads', '4', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(DURATION*FPS):
        proc.stdin.write(draw_method02(i/FPS).tobytes())
        if i % 100 == 0:
            print(f'Rendered {i}/{DURATION*FPS}', flush=True)
    proc.stdin.close()
    assert proc.wait() == 0
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    preview()
    if args.render:
        render()
