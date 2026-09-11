"""Exploratory identified-model grid check while the fixed-budget search runs."""
import argparse
import json
from pathlib import Path
import shutil
from experiments.robotics import x_force_hold_plan as planning
from experiments.robotics.plastic_shaping_study import save_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,default=planning.study.OUT)
p.add_argument('--material',choices=['A','B'],required=True)
p.add_argument('--device',required=True)
a=p.parse_args();f=a.out;m=a.material
planning.stage(f)
rows=json.loads((f/f'plans/{m}/initialization_evaluations.json').read_text())
r=min((r for r in rows if r['feasible'] and len(r['peaks_n'])==4),key=lambda r:r['surface_mm'])
initial=json.loads((f/f'inputs/initialization_{m}.json').read_text())
dest=f/'exploratory_model_grid_check'/m;dest.mkdir(parents=True,exist_ok=True)
save_json(dest/'protocol.json',dict(selection='Lowest feasible complete initialization error, identified model only',candidate=r,grid=80,dt=.00005,scope='Early diagnostic; does not replace final three-candidate selection or use true-material executions'))
shutil.copy2(__file__,dest/'source.py')
_,result=planning.study.rollout(f,dest,initial['law'],r['peaks_n'],r['durations_s'],a.device,80,.00005)
save_json(dest/'result.json',result)
print(json.dumps(result,indent=2),flush=True)
