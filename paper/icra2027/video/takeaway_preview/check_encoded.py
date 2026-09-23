"""Read-only encoded-video checks; evidence writes stay in takeaway_preview."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from render_takeaway import DURATION, FPS, HERE, REVEAL_TIMES, audit, compose, draw_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, default=HERE/'takeaway_preview_v1.mp4')
    parser.add_argument('--offset', type=float, default=0)
    parser.add_argument('--stem', default='encoded')
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FPS) == FPS
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280
    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == round((args.offset+DURATION)*FPS)
    samples = sorted({0, DURATION-1/FPS,
                      *[round(t+dt, 2) for t in REVEAL_TIMES for dt in (-1/FPS, .24, .6)
                        if 0 <= t+dt < DURATION]})
    pictures = []
    comparisons = []
    for t in samples:
        audit(compose(t))
        cap.set(cv2.CAP_PROP_POS_FRAMES, round((args.offset+t)*FPS))
        ok, frame = cap.read()
        assert ok
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        expected = np.asarray(draw_frame(t))
        # Exclude the global progress bar, added during full-film assembly.
        error = float(np.abs(rgb[:715].astype(float)-expected[:715]).mean())
        assert error < 4, (t, error)
        for row in (668, 707):
            assert np.abs(rgb[row,600:680].astype(float)-[38,53,62]).mean() < 10
        pic = Image.fromarray(rgb)
        pic.save(HERE/f'{args.stem}_{t:g}s.png')
        pictures.append((t, pic))
        comparisons.append(dict(local_time_s=t, global_time_s=args.offset+t, mean_pixel_error=error))
    cap.release()
    font = ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf', 19)
    for page in range((len(pictures)+3)//4):
        sheet = Image.new('RGB', (1280, 770), 'white')
        for i, (t, pic) in enumerate(pictures[page*4:page*4+4]):
            x, y = i%2*640, i//2*385
            ImageDraw.Draw(sheet).text((x+8,y+2), f'{t:g} s', font=font, fill='#21333e')
            sheet.paste(pic.resize((640,360), Image.Resampling.LANCZOS), (x,y+25))
        sheet.save(HERE/f'{args.stem}_contact_{page+1}.png')
    result = dict(video=str(args.video.resolve()), offset_s=args.offset, closing_duration_s=DURATION,
                  reveal_times_s=REVEAL_TIMES, frame_dimensions=[1280,720], fps=FPS,
                  caption_position_pass=True, expected_frame_comparisons=comparisons)
    (HERE/f'{args.stem}_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
