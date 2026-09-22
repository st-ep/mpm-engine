"""Timed reveals of the approved static opening, without changing its layout."""
from functools import lru_cache
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
DURATION = 13
FADE_SECONDS = 0.4
# The full approved text is retained. Rectangles include only numbered steps.
REVEALS = [
    {'step': '01', 'start_s': 3.0, 'box': (40, 390, 554, 466)},
    {'step': '02', 'start_s': 6.0, 'box': (40, 490, 554, 591)},
    {'step': '03', 'start_s': 10.0, 'box': (40, 618, 554, 689)},
]

@lru_cache(maxsize=1)
def approved_frame():
    return Image.open(HERE / 'opening_proposal_720p.png').convert('RGB')

def draw_opening(t):
    approved = approved_frame()
    out = approved.copy()
    for reveal in REVEALS:
        alpha = min(1.0, max(0.0, (t - reveal['start_s']) / FADE_SECONDS))
        if alpha < 1:
            crop = approved.crop(reveal['box'])
            blank = Image.new('RGB', crop.size, '#f4f6f7')
            out.paste(Image.blend(blank, crop, alpha), reveal['box'][:2])
    return out
