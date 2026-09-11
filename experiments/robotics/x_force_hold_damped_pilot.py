"""Identified-model comparison of normalized and unnormalized force PI gains."""
import json
from pathlib import Path
import shutil
from experiments.robotics import x_force_hold_damped_study as study
from experiments.robotics.plastic_shaping_study import save_json
f=study.OUT;study.check(f)
src=study.ROOT/'out/x_force_hold_study_20260910/robust/initial_B.json'
r=json.loads(src.read_text())
save_json(f/'gain_pilot_protocol.json',dict(source=str(src),source_sha256=study.digest(src),
    seed=r,scope='Same commands and identified-B law; change only the common command-dependent PI gain rule. No true-material outcomes.'))
shutil.copy2(__file__,f/'gain_pilot_source.py')
rows=[]
for name,(grid,dt) in {'planning64':(64,.0001),'grid80':(80,.00005)}.items():
    _,result=study.rollout(f,f/'gain_pilot'/name,r['law'],r['peaks_n'],r['durations_s'],'cuda:1',grid,dt)
    rows.append(dict(**result,setting=name));save_json(f/'gain_pilot_results.json',rows)
    print(json.dumps(rows[-1],indent=2),flush=True)
