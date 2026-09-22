"""Re-run the frozen putting protocol, including both searches and all checks.

The source bundle supplies material estimates and robot assets. No archived
particle trajectories or reported stopping positions are used as inputs.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
from experiments.elastic.putting import ROOT,digest
from experiments.elastic.putting_study import prepare,read

def reproduce(source,out,devices):
    hashes=read(source/'reproduction_source_hashes.json')
    different=[name for name,h in hashes.items() if digest(ROOT/name)!=h]
    if different:raise RuntimeError(f'Source differs from the archived experiment: {different}. Use the archived source revision before reproducing.')
    prepare(out,source);logs=out/'logs';logs.mkdir()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def pair(commands,label):
        running=[]
        for k,args in zip('AB',commands):
            log=(logs/f'{label}_{k}.log').open('w')
            proc=subprocess.Popen([sys.executable,'-m','experiments.elastic.putting_study',*args,'--out',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            running.append((proc,log))
        for proc,log in running:
            code=proc.wait();log.close()
            if code:raise RuntimeError(f'{label} failed; see {logs}')
    pair([['plan','--material',m,'--device',dev] for m,dev in zip('AB',devices)],'planning')
    for swapped in [False,True]:
        pair([['execute','--material',m,'--plan',('B' if m=='A' else 'A') if swapped else m,'--device',dev]
              for m,dev in zip('AB',devices)],'swapped' if swapped else 'matched')
    for prefix,extra in [('fine',['--grid','320']),('half_contact',['--coupling','.0000625'])]:
        for swapped in [False,True]:
            pair([['check','--material',m,'--plan',('B' if m=='A' else 'A') if swapped else m,'--device',dev,'--prefix',prefix,*extra]
                  for m,dev in zip('AB',devices)],f'{prefix}_{swapped}')
    pair([['check','--material',m,'--plan',m,'--device',dev,'--prefix','combined','--grid','320','--coupling','.0000625']
          for m,dev in zip('AB',devices)],'combined_matched')
    from experiments.elastic.putting_report import render_case,compose,overview,swing_sequence,paper_demo
    for view in ['course','detail']:
        for m in 'AB':
            for k in 'AB':render_case(out,m,k,view=view)
        compose(out,view=view)
    compose(out,view='detail',slow=2)
    swing_sequence(out)
    render_case(out,'A','A',view='overview');overview(out)
    render_case(out,'A','A',view='hero');paper_demo(out)
    from experiments.elastic.putting_audit import summarize
    summarize(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--devices',nargs=2,default=['cuda:0','cuda:1'])
    a=ap.parse_args();reproduce(a.source.resolve(),a.out.resolve(),a.devices)
