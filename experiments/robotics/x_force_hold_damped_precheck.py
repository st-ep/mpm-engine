"""Early identified-B grid check while the model-only search finishes."""
import json
from pathlib import Path
import shutil
from experiments.robotics import x_force_hold_damped_plan as planning
from experiments.robotics.plastic_shaping_study import save_json
f=planning.study.OUT;planning.stage(f)
rows=json.loads((f/'plans/B/evaluations.json').read_text())
r=min((r for r in rows if r['feasible'] and len(r['peaks_n'])==4),key=lambda r:r['surface_mm'])
assert r['law']==json.loads((f/'inputs/initialization_B.json').read_text())['law']
save_json(f/'early_B_grid_check_protocol.json',dict(candidate=r,completed_model_objective_calls=len(rows),
    scope='Preliminary identified-model check, before final selection or any true execution; no command changes'))
shutil.copy2(__file__,f/'early_B_grid_check_source.py')
_,result=planning.study.rollout(f,f/'model_checks/B/grid80',r['law'],r['peaks_n'],r['durations_s'],'cuda:0',80,.00005)
save_json(f/'early_B_grid_check.json',result)
print(json.dumps(result,indent=2),flush=True)
