"""Reproduce six frozen-plan MPM clips, with capture on two available GPUs.

Resume completed captures/renders; incomplete captures stop for inspection.
Logs and outputs stay under assets/pouring_targets. No paper data are replaced.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,os,subprocess,threading,time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
WORK=HERE/'assets/pouring_targets'
ENV=dict(os.environ,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
STOP=threading.Event()


def run(stage,index,logname,extra=()):
    with (WORK/logname).open('w') as log:
        subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/'prepare_pouring_targets.py'),stage,'--case',str(index),*extra],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)


def captures(gpu,indices):
    try:
        for i in indices:
            if STOP.is_set():return
            run('capture',i,f'capture_{i}.log',('--device',f'cuda:{gpu}'))
            print('Captured case',i,flush=True)
    except BaseException:
        STOP.set();raise


def renders():
    try:
        for i,target in enumerate([60,80,100,120,140,160]):
            out=WORK/f'target_{target}'
            while not (out/'capture/states/state_0000.npz').exists():
                if STOP.wait(2):return
            run('prepare',i,f'prepare_{i}.log')
            with (WORK/f'render_{i}.log').open('w') as log:
                subprocess.run(['/opt/blender/blender','--background','--python-exit-code','1','--python',str(HERE/'render_pouring_targets.py'),'--','--output',str(out),'--device-index','1'],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
            run('compose',i,f'compose_{i}.log')
            print('Rendered target',target,flush=True)
    except BaseException:
        STOP.set();raise


if __name__=='__main__':
    WORK.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(captures,0,[0,2,4]),pool.submit(captures,1,[1,3,5]),pool.submit(renders)]
        for f in as_completed(jobs):f.result()
