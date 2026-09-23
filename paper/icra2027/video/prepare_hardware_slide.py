"""Display cache of the actual identification presses; no inference or simulation."""
from pathlib import Path
import json,hashlib,subprocess
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'assets/hardware_identification.mp4'

def prepare():
    protocol=json.loads((ROOT/'out/press_weakform_restart_20260914/protocol.json').read_text())
    records=[];readers=[]
    for material in protocol['materials']:
        ep=protocol['train'][material][0]
        folder=ROOT/'press_real_data'/ep
        assessment=json.loads((ROOT/f'out/press_observation_assessment_20260913/{ep}/assessment.json').read_text())
        hosts=np.array([json.loads(line)['t_host'] for line in (folder/'frames_hand.jsonl').read_text().splitlines()])
        times=np.linspace(0,10,150)
        indices=[int(np.argmin(abs(hosts-assessment['t0_host']-t))) for t in times]
        cap=cv2.VideoCapture(str(folder/'hand_rgb.mp4'));assert cap.isOpened()
        readers.append((cap,indices))
        records.append({'material':material,'episode':ep,'path':str(folder/'hand_rgb.mp4'),'source_sha256':hashlib.sha256((folder/'hand_rgb.mp4').read_bytes()).hexdigest(),'frame_indices':indices,'relative_host_time_s':times.tolist()})
    proc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1368x280','-r','25','-i','-','-an','-c:v','libx264','-crf','17','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT)],stdin=subprocess.PIPE)
    for i in range(150):
        strip=np.full((280,1368,3),255,np.uint8)
        for j,(cap,indices) in enumerate(readers):
            cap.set(cv2.CAP_PROP_POS_FRAMES,indices[i]);ok,raw=cap.read();assert ok
            strip[:,460*j:460*j+448]=cv2.resize(raw[140:280,348:572],(448,280))
        proc.stdin.write(strip.tobytes())
    proc.stdin.close();assert proc.wait()==0
    for cap,_ in readers:cap.release()
    (OUT.with_suffix('.json')).write_text(json.dumps({'scope':'Actual independent 31 N identification recordings; first 10 s after the recorded baseline, shown in six seconds. No geometry changes, per-frame crops, model refits or physics. Evaluation recordings remain separate.','crop_xyxy':[348,140,572,280],'frames':150,'fps':25,'records':records},indent=2)+'\n')
    print(OUT)
def prepare_execution():
    # Same complete source and 4.5x display speed as the approved execution.
    source=HERE/'sources/hardware_shaping/side_rgb.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(source),'-an',
        '-vf','crop=1050:520:480:180,setpts=PTS/4.5,fps=25,scale=424:210',
        '-c:v','libx264','-crf','17','-preset','fast','-threads','4',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(HERE/'assets/hardware_execution.mp4')],check=True)

if __name__=='__main__':
    prepare()
    prepare_execution()
