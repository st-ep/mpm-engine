"""Render/review opening captions using the approved existing static layout.

All new outputs stay in this directory. Installation/assembly are separate.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
VIDEO = HERE.parent
sys.path.insert(0, str(VIDEO))
from opening_preview.render_opening import draw_opening, DURATION
from narrative_captions import CUES, OPENING_CAPTION_RATE_LIMITS

FPS = 25
OUTPUT = HERE/'opening_captions_updated.mp4'


def check(path, stem):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == FPS
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= DURATION*FPS
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (1280,720)
    times = [0,1.5,2.96,3.24,3.6,5.96,6.24,6.6,9.96,10.24,10.6,12.96]
    rows=[];frames=[]
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(t*FPS))
        ok, bgr=cap.read();assert ok
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        expected=np.asarray(draw_opening(t))
        error=float(np.abs(rgb[:715].astype(float)-expected[:715]).mean())
        assert error < 4, (t,error)
        for y in (668,707):
            assert np.abs(rgb[y,600:680].astype(float)-[38,53,62]).mean()<10
        im=Image.fromarray(rgb);im.save(HERE/f'{stem}_{t:g}s.png')
        frames.append((t,im));rows.append(dict(time_s=t,mean_pixel_error=error))
    cap.release()
    font=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf',19)
    for page in range(3):
        sheet=Image.new('RGB',(1280,770),'white')
        for j,(t,im) in enumerate(frames[page*4:page*4+4]):
            x=j%2*640;y=j//2*385
            ImageDraw.Draw(sheet).text((x+8,y+2),f'{t:g} s',font=font,fill='#21333e')
            sheet.paste(im.resize((640,360)),(x,y+25))
        sheet.save(HERE/f'{stem}_contact_{page+1}.png')
    report=dict(path=str(path),duration_s=DURATION,fps=FPS,captions=CUES['opening'],
                caption_rate_limits_cps=OPENING_CAPTION_RATE_LIMITS,checks=rows)
    (HERE/f'{stem}_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Checked {len(rows)} encoded opening samples: {path}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--installed',action='store_true')
    args=parser.parse_args()
    for a,b,s in CUES['opening']:
        assert len(s)/(b-a)<=OPENING_CAPTION_RATE_LIMITS.get(a,20)
        draw_opening((a+b)/2)
    if args.render:
        command=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720',
                 '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','17',
                 '-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(OUTPUT)]
        proc=subprocess.Popen(command,stdin=subprocess.PIPE)
        for i in range(DURATION*FPS):proc.stdin.write(draw_opening(i/FPS).tobytes())
        proc.stdin.close();assert proc.wait()==0
    check(VIDEO/'ICRA2027_video.mp4' if args.installed else OUTPUT,
          'opening_installed' if args.installed else 'opening_updated')


if __name__=='__main__':main()
