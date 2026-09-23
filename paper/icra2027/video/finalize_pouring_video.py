"""Finish the current pouring edit on the server after all six renders arrive.

One-shot job: validates the frozen replays, rebuilds the slide and full movie,
and saves an encoded review sheet. Safe to run independently of the desktop.
"""
from pathlib import Path
import json,os,shutil,subprocess,time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
WORK=HERE/'assets/pouring_targets'
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
STATUS=WORK/'finalize_status.json'


def status(stage,**fields):
    STATUS.write_text(json.dumps({'stage':stage,**fields},indent=2)+'\n')


def run(script,*args):
    subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/script),*args],cwd=ROOT,env=ENV,check=True)


def main():
    status('waiting_for_target_clips')
    deadline=time.monotonic()+7200
    while not all((WORK/f'target_{n}/target.mp4').exists() for n in [60,80,100,120,140,160]):
        if time.monotonic()>deadline:raise TimeoutError('Target clips did not finish within two hours.')
        time.sleep(5)
    # Let the final FFmpeg writer close before starting the strict decode check.
    time.sleep(3)
    status('checking_frozen_replays');run('check_pouring_targets.py')
    validated=json.loads((WORK/'validated_clips.json').read_text())
    path=HERE/'assets/pouring_storyboard_provenance.json';info=json.loads(path.read_text())
    info['target_replays']=validated
    info['lower_video_style']='Same paper camera, original glass/lighting/robot materials, 64-sample Cycles renders, original blue liquid. Actual captured particles reconstructed by the same audited surface method as the upper replay.'
    path.write_text(json.dumps(info,indent=2)+'\n')
    status('rendering_slide_and_full_video');run('build_video.py','--render','pouring','--assemble')
    status('validating_full_video');run('validate_video.py','--working-cut')
    shutil.copy2(HERE/'sections/pouring.mp4',HERE/'review/pouring_mpm_four_sections.mp4')
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    times=[.5,4.8,7,12.8,13.2,14.28,14.6,17.6,20.6,23.6,26.6,29.6,31.8,35.8]
    sheet=Image.new('RGB',(1280,385*7),'white')
    font=ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf',18)
    cap=cv2.VideoCapture(str(HERE/'sections/pouring.mp4'))
    for i,t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read();assert ok
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        im.save(HERE/f'review/pouring_mpm_encoded_{t:g}.png')
        x=(i%2)*640;y=(i//2)*385
        sheet.paste(im.resize((640,360)),(x,y+25))
        ImageDraw.Draw(sheet).text((x+8,y+3),f'{t:.2f} s',font=font,fill='#21333e')
    cap.release();sheet.save(HERE/'review/pouring_mpm_encoded_contact.jpg',quality=94)
    status('complete',preview=str(HERE/'review/pouring_mpm_four_sections.mp4'),full_video=str(HERE/'ICRA2027_video.mp4'),duration_s=36)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('failed',error=repr(error));raise
