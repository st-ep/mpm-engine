"""Replay frozen force-and-hold plans in fresh true-material simulations.

This command imports archived calibration/selection records as fixed inputs;
it does not claim to rerun identification or optimization. The separate
planning entrypoints and their development archives reproduce those stages.
It never writes the active manuscript or figure.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from experiments.robotics import x_force_hold_damped_select as selection
from experiments.robotics.plastic_shaping_study import save_json


def replay(source,folder,render,devices,prepare_only=False):
    selection.check(source)
    assert not folder.exists() and not render.exists(), 'Use fresh output directories'
    # The frozen source archive is sufficient; no other output archive is read.
    manifest=json.loads((source/'checksums.json').read_text())
    for rel,sha in manifest.items():assert selection.planning.study.digest(source/rel)==sha,rel
    folder.mkdir(parents=True)
    for name in ['inputs','validation','source_snapshot','selection_inputs']:
        shutil.copytree(source/name,folder/name)
    shutil.copytree(source/'franka/panda_model_snapshot',folder/'franka/panda_model_snapshot')
    files=['protocol.json','source_sha256.json','input_sha256.json','robot_input_sha256.json',
           'planning_protocol.json','planning_source.py','selection_protocol.json','selection_source.py',
           'selection_input_sha256.json','target.vtp','target.json','target_outline_m.npy',
           'identification_validation.json','observation_isolation.json','packages.txt','commit.txt']
    for name in files:shutil.copy2(source/name,folder/name)
    for material in 'AB':
        dst=folder/'plans'/material;dst.mkdir(parents=True)
        for name in ['selected.json','selection_complete.json']:shutil.copy2(source/'plans'/material/name,dst/name)
    save_json(folder/'replay_provenance.json',dict(source_archive=str(source.resolve()),
        source_manifest_sha256=selection.planning.study.digest(source/'checksums.json'),
        scope='Calibration and planning/selection records are imported frozen inputs. Their model-data paths refer to the source/development archives. True-material executions, audit and figures below are newly computed.',
        devices=devices))
    (folder/'replay_packages.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())))
    shutil.copy2(__file__,folder/'replay_source.py')
    selection.check(folder)
    for material in 'AB':
        frozen=json.loads((folder/f'plans/{material}/selection_complete.json').read_text())
        assert frozen['selected_sha256']==selection.planning.study.digest(folder/f'plans/{material}/selected.json')
    save_json(folder/'replay_staging_audit.json',dict(source_and_input_hashes_checked=True,selected_plan_hashes_checked=True,scope='Staging check before simulation'))
    if prepare_only:
        print(json.dumps(dict(prepared=str(folder),simulations_run=0)),flush=True)
        return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    commands=[]
    def run(module,stage=None,material=None,device=None,is_render=False):
        cmd=[sys.executable,'-m','experiments.robotics.'+module]
        if stage:cmd.append(stage)
        cmd+=['--out',str(folder)]
        if material:cmd+=['--material',material,'--device',device]
        if is_render:cmd+=['--render-out',str(render)]
        print(json.dumps(dict(command=cmd)),flush=True)
        subprocess.run(cmd,check=True,env=env)
        commands.append(cmd)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,'x_force_hold_damped_plan','evaluate',m,d)
                 for m,d in zip('AB',devices,strict=True)]
        for future in futures:future.result()
    run('x_force_hold_audit','audit')
    run('x_force_hold_audit','diagnostics')
    run('x_force_hold_report',is_render=True)
    comparisons=[]
    for setting in selection.planning.study.SETTINGS:
        for actual in 'AB':
            for planned in 'AB':
                name=f'{setting}_{actual}_plan_{planned}.json'
                before=json.loads((source/name).read_text());after=json.loads((folder/name).read_text())
                comparisons.append(dict(case=name,surface_error_difference_mm=abs(before['surface_mm']-after['surface_mm']),
                                        same_stops=before['stops']==after['stops']))
    save_json(folder/'replay_comparison.json',dict(results=comparisons,
        scope='GPU floating-point reductions need not be bit-identical; report all differences without relabeling the original results'))
    save_json(folder/'replay_commands.json',commands)
    print(json.dumps(dict(completed=str(folder),figure=str(render/'identification_plastic_shaping.png'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=selection.planning.study.OUT)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--render-out',type=Path,required=True)
    p.add_argument('--devices',nargs=2,default=['cuda:0','cuda:1'])
    p.add_argument('--prepare-only',action='store_true',help='Validate and stage frozen inputs without launching simulations')
    a=p.parse_args();replay(a.source,a.out,a.render_out,a.devices,a.prepare_only)
