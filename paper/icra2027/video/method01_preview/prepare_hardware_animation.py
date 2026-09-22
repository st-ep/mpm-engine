"""Cache timestamp-matched RGB crops for the existing hardware reconstruction."""
from pathlib import Path
import json,cv2,numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
rows=[json.loads(s) for s in (ROOT/'press_real_data/ep0001/frames_hand.jsonl').read_text().splitlines()]
t0=json.loads((ROOT/'out/press_observation_assessment_20260913/ep0001/assessment.json').read_text())['t0_host']
time=np.array([r['t_host']-t0 for r in rows]);use=np.flatnonzero((time>=0)&(time<=2.06))
cap=cv2.VideoCapture(str(ROOT/'press_real_data/ep0001/hand_rgb.mp4'));rgb=[]
cap.set(cv2.CAP_PROP_POS_FRAMES,rows[use[0]]['frame_idx']);previous=rows[use[0]]['frame_idx']-1
for j in use:
 idx=rows[j]['frame_idx']
 if idx!=previous+1:cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
 ok,bgr=cap.read();assert ok;frame=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
 # Fixed wider crop includes the upper plate and full material at every time.
 rgb.append(frame[125:286,330:590]);previous=idx
cap.release();np.savez_compressed(P/'animation_hardware_rgb.npz',rgb=np.stack(rgb),time=time[use],frame_idx=np.array([rows[j]['frame_idx'] for j in use]))
(P/'hardware_animation_cache_provenance.json').write_text(json.dumps({'episode':'ep0001','source':'press_real_data/ep0001/hand_rgb.mp4','time_base':'Robot first active baseline host timestamp, same convention as the existing hardware observation assessment.','t0_host':t0,'rgb_frames':[int(rows[j]['frame_idx']) for j in use],'raw_crop':[330,125,590,286],'uniform_resize':None,'secondary_crop':None,'note':'Fixed crop and scale for every frame; no shape/image warping.'},indent=2)+'\n')
print('Cached',len(rgb),'timestamped RGB frames.')
