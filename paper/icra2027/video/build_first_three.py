"""Render and join the first three approved sections with spoken-style captions.

Run with --render to regenerate all three clips. Without it, join existing clips.
This review export does not replace the full conference film.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess,sys
import cv2
from PIL import Image
from narrative_captions import SECTIONS,CUES,export_captions

P=Path(__file__).resolve().parent
OUT=P/'first_three_preview'
CLIPS=[P/'sections/opening.mp4',P/'method01_preview/method01_animated.mp4',P/'method02_preview/method02_block1_animated.mp4']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.render:
        commands=[
            [sys.executable,str(P/'build_video.py'),'--render','opening'],
            [sys.executable,str(P/'method01_preview/render_method01.py'),'--preview','--render'],
            [sys.executable,str(P/'method02_preview/render_method02_animated.py'),'--render'],
        ]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(subprocess.run,cmd,check=True) for cmd in commands]
            for future in futures:future.result()
    for clip,(_,duration) in zip(CLIPS,SECTIONS):
        cap=cv2.VideoCapture(str(clip))
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==duration*25,str(clip)
        assert cap.get(cv2.CAP_PROP_FPS)==25,str(clip)
        assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==(1280,720)
        cap.release()
    listing=OUT/'concat.txt'
    assert all("'" not in str(path) for path in CLIPS)
    listing.write_text(''.join(f"file '{path}'\n" for path in CLIPS))
    video=OUT/'first_three_slides.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(video)],check=True)
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
    export_captions(OUT)
    cap=cv2.VideoCapture(str(video));duration=sum(d for _,d in SECTIONS)
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==duration*25
    # One actual encoded frame per cue, including both sides of the slide cuts.
    cue_count=sum(len(CUES[name]) for name,_ in SECTIONS)
    sheet=Image.new('RGB',(1920,360*((cue_count+2)//3)),'white');offset=0;index=0;review=[]
    for section,length in SECTIONS:
        previous=0
        for start,end,text in CUES[section]:
            assert start==previous and end>start
            previous=end
            assert len(text)/(end-start)<=20,(text,'reading rate')
            t=offset+(start+end)/2
            cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read();assert ok
            im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            im.save(OUT/f'cue_{index+1:02d}.png')
            sheet.paste(im.resize((640,360)),((index%3)*640,(index//3)*360))
            review.append({'section':section,'start_s':offset+start,'end_s':offset+end,'caption':text,'characters_per_second':round(len(text)/(end-start),2)})
            index+=1
        assert previous==length
        offset+=length
    cap.release();sheet.save(OUT/'caption_contact_sheet.png')
    boundaries=[0]
    for _,length in SECTIONS:boundaries.append(boundaries[-1]+length)
    (OUT/'review.json').write_text(json.dumps({'duration_s':duration,'frames':duration*25,'fps':25,'full_decode_passed':True,'caption_cues':review,'section_boundaries_s':boundaries,'sources':[str(p) for p in CLIPS]},indent=2)+'\n')
    (OUT/'README.md').write_text(f'First three slides with spoken-style subtitles\n\nOpen first_three_slides.mp4 ({duration} seconds, 1280 × 720, 25 fps).\nThe captions are burned in; first_three_slides.srt and\nfirst_three_slides_transcript.txt are editable exports.\n\nSource: ../narrative_captions.py. Rebuild with:\nOPENBLAS_NUM_THREADS=1 .venv/bin/python paper/icra2027/video/build_first_three.py --render\n\nThe opening body is uniformly reduced by about 9% to reserve subtitle space;\nits title and the method diagrams retain their previous sizes.\nThe full conference film is not replaced by this review export.\n')
    print(video,flush=True)


if __name__=='__main__':main()
