"""Timed reveals of the approved static opening, without changing its layout."""
from functools import lru_cache
from pathlib import Path
from PIL import Image
from narrative_captions import caption_at,paint_caption

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
    # Keep the title at its approved size; uniformly fit the complete lower
    # composition above the subtitle band. No photograph is cropped or warped.
    body=out.crop((40,119,1240,706))
    scale=(653-119)/body.height
    body=body.resize((round(body.width*scale),round(body.height*scale)),Image.Resampling.LANCZOS)
    framed=Image.new('RGB',out.size,'#f4f6f7')
    framed.paste(out.crop((0,0,1280,119)),(0,0))
    framed.paste(body,((1280-body.width)//2,119))
    return paint_caption(framed,caption_at('opening',t))
