"""Evaluate one predefined late-hold fallback early, using identified B only."""
import json
from pathlib import Path
import shutil
from experiments.robotics import x_force_hold_robust as robust
from experiments.robotics.plastic_shaping_study import save_json
f=robust.planning.study.OUT;robust.check(f)
r=next(r for r in json.loads((f/'robust/B/evaluations.json').read_text()) if r['label']=='seed_3')
source=r['checks'][0]
peaks=source['peaks_n'];dwell=list(source['durations_s']);dwell[-1]=max(.4,dwell[-1])
save_json(f/'robust/B/early_late_hold_protocol.json',dict(candidate=r,forces_n=peaks,dwells_s=dwell,
    scope='One fallback already specified in robust protocol, evaluated early to diagnose final-hold tracking; identified B only'))
shutil.copy2(__file__,f/'robust/B/early_late_hold_source.py')
rows=[]
for setting,(grid,dt) in robust.PROTOCOL['settings'].items():
    _,result=robust.planning.study.rollout(f,f/'robust/B'/setting,source['law'],peaks,dwell,'cuda:1',grid,dt)
    rows.append(dict(**result,setting=setting));save_json(f/'robust/B/early_late_hold_results.json',rows)
    print(json.dumps(rows[-1],indent=2),flush=True)
