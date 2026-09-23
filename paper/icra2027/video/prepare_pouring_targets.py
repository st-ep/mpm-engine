"""Capture/render the six frozen target plans for the ICRA pouring slide.

No search, calibration or result replacement. Physics settings and the archived
kernels are frozen; every replay is compared with its original volume history.
Only 50 actual states per plan are exported for 2-second display clips.
"""
from pathlib import Path
import argparse,csv,hashlib,inspect,json,os,shutil,subprocess,sys,time
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/'out/pour_physics_audit/aflip_blend_time_20260907'
SOURCE=BASE/'isolated_src'
WORK=HERE/'assets/pouring_targets'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(SOURCE))

def write(p,v):
    temp=p.with_suffix('.pending.json');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)

def commands():
    return list(csv.DictReader((ROOT/'out/pour_hardware_receiver_remap_review_20260911/verified_commands.csv').open()))[:6]

def case_dir(index):return WORK/f'target_{int(commands()[index]["target_ml"])}'

def capture(index,device):
    from experiments.pour import pour_transfer_recalibration_reference as ref
    from warpmpm.kernels import mpm_utils
    assert Path(mpm_utils.__file__).resolve().is_relative_to(SOURCE)
    row=commands()[index];frozen_path=ROOT/row['result_path'];frozen=json.loads(frozen_path.read_text())
    out=case_dir(index);out.mkdir(parents=True,exist_ok=True)
    if (out/'verification.json').exists():
        assert json.loads((out/'verification.json').read_text())['accepted'];return
    assert not (out/'forward').exists(),'A partial replay exists; inspect before resuming.'
    (out/'capture/states').mkdir(parents=True,exist_ok=True)
    expected=list(csv.DictReader((frozen_path.parent/'planned_episode/metrics.csv').open()))
    chosen=np.rint(np.linspace(0,len(expected)-1,50)).astype(int)
    mapping={int(f):i for i,f in enumerate(chosen)}
    # Every original dynamics input must match. The three changed live kernel
    # modules are loaded instead from the archived source used for the paper.
    inputs=[]
    archived_hashes=json.loads((BASE/'prototype_provenance.json').read_text())['prototype_sha256']
    for rel,digest in frozen['input_sha256'].items():
        path=ROOT/rel
        if rel.startswith('src/warpmpm/'):
            archived=SOURCE/rel.removeprefix('src/')
            if archived.exists():
                path=archived
                digest=archived_hashes[str(path.relative_to(ROOT))]
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual==digest,(rel,str(path))
        inputs.append({'path':str(path.relative_to(ROOT)),'sha256':actual})
    twin=ref.twin
    def save(l):
        frame=l['frame']
        if frame not in mapping:return
        i=mapping[frame];x=l['s'].x()-l['w2m']
        assert len(x)==frozen['particle_count'] and np.isfinite(x).all()
        if not (out/'capture/particles.npz').exists():
            np.savez(out/'capture/particles.npz',vol=l['vol'],h=l['h'],receiver_pos=l['receiver_pos'],receiver_quat=twin.Q_RCV,world_to_grid_offset=l['w2m'])
        dest=out/f'capture/states/state_{i:04d}.npz'
        tmp=dest.with_suffix('.pending.npz')
        np.savez_compressed(tmp,x_world=x,source_pos=l['p_now'],source_quat=l['q_now'],t=l['t_now'],replay_frame=frame)
        tmp.replace(dest)
    original=inspect.getsource(twin.run);anchor='        if video:\n            img = render_frame'
    assert original.count(anchor)==1
    code=original.replace(anchor,'        _icra_capture(locals())\n\n'+anchor)
    (out/'original_run.py').write_text(original);(out/'instrumented_run.py').write_text(code)
    twin.__dict__['_icra_capture']=save
    exec(compile(code,str(out/'instrumented_run.py'),'exec'),twin.__dict__)
    write(out/'provenance.json',{'target_ml':float(row['target_ml']),'angle_deg':float(row['command_angle_deg']),
        'source_result':str(frozen_path.relative_to(ROOT)),'dynamics_inputs':inputs,
        'selected_replay_frames':chosen.tolist(),
        'saved_episode_inputs':[{"path":str((frozen_path.parent/'planned_episode'/n).relative_to(ROOT)),"sha256":hashlib.sha256((frozen_path.parent/'planned_episode'/n).read_bytes()).hexdigest()} for n in ['meta.json','actions.jsonl','states.jsonl','joint_trajectory.csv']],'display_frames':50,'display_seconds':2,
        'unchanged_parameters':{'viscosity_pa_s':frozen['eta_pa_s'],'source_contact':frozen['source_coulomb_friction'],'grid':160,'dt_scale':1.0},
        'purpose':'Visualization replay of frozen command; no optimization or new calibration. Original quantitative results remain authoritative.',
        'verification_threshold_ml':.5})
    ref.OUT=out/'forward'
    # Reuse the exact saved plan timestamps and joints. Regenerating the old
    # motion through today's helper can change acknowledgement-time offsets.
    saved_episode=frozen_path.parent/'planned_episode'
    ref.planned_motion=lambda max_angle:(None,None)
    def copy_frozen_episode(reference,destination,motion,ep,angle):
        destination.mkdir(parents=True,exist_ok=True)
        for name in ['meta.json','actions.jsonl','states.jsonl','joint_trajectory.csv']:
            shutil.copy2(saved_episode/name,destination/name)
    ref.write_planned_episode=copy_frozen_episode
    ref.run(SimpleNamespace(source_friction=frozen['source_coulomb_friction'],grid=160,phase=0.,wall='original-separable',slip_mm=None,angle=float(row['command_angle_deg']),max_angle=70.,dt_scale=1.,device=device))
    result_path=next((out/'forward').rglob('result.json'))
    actual=list(csv.DictReader((result_path.parent/'planned_episode/metrics.csv').open()))
    assert len(actual)==len(expected)
    n=frozen['particle_count'];errors={k:max(abs(int(a[k])-int(e[k]))*300/n for a,e in zip(actual,expected,strict=True)) for k in ['n_src','n_rcv','n_air_spill']}
    endpoint=abs(json.loads(result_path.read_text())['receiver_ml']-frozen['receiver_ml'])
    passed=max(errors.values())<=.5 and endpoint<=.5
    write(out/'verification.json',{'accepted':passed,'threshold_ml':.5,'max_history_difference_ml':errors,'endpoint_difference_ml':endpoint,'frames':len(actual),'captured_states':len(list((out/'capture/states').glob('*.npz'))),'scope':'Display reproducibility only; no replacement of original numerical results.'})
    assert passed,errors
    print('CAPTURE_COMPLETE',index,errors,flush=True)


def poses(index):
    from experiments.pour import pour_share_video_prepare as prep
    twin=prep.twin;ref=prep.still.reference
    out=case_dir(index)
    info=json.loads((prep.PAPER/'render_assets/scene.json').read_text())
    geometry=json.loads(ref.GEOMETRY.read_text())
    objects=[o for o in info['objects'] if o['material']=='robot']
    ep=twin.load_episode(ref.REFERENCE,twin.PRE_ROLL,twin.HOLD_SECONDS)
    arm=twin.RecordedPanda(ep,ref.MESH,height=64,width=64,max_geom=4000,cup_reference_pos=geometry['cup_reference_pos'],cup_reference_quat=geometry['cup_reference_quat'])
    def matrices(arm,t):
        p,q=arm.cup_pose_at(t);arm.set_time(t)
        result={o['name']:prep.transform(arm.data.geom_xpos[int(o['name'].split('_')[1])],arm.data.geom_xmat[int(o['name'].split('_')[1])].reshape(3,3)) for o in objects}
        result['source_cup']=prep.transform(p,twin.quat_to_mat(q));return result
    inverse={k:np.linalg.inv(v) for k,v in matrices(arm,info['time_s']).items()}
    hand_cup,grasp=arm._hand_cup.copy(),arm._grasp.copy();arm.close()
    episode=next((out/'forward').rglob('planned_episode'))
    target_ep=twin.load_episode(episode,twin.PRE_ROLL,twin.HOLD_SECONDS)
    arm=twin.RecordedPanda(target_ep,ref.MESH,height=64,width=64,max_geom=4000,cup_reference_pos=geometry['cup_reference_pos'],cup_reference_quat=geometry['cup_reference_quat'])
    arm._hand_cup,arm._grasp=hand_cup,grasp
    values=[]
    selected=json.loads((out/'provenance.json').read_text())['selected_replay_frames']
    for i in range(50):
        path=out/f'capture/states/state_{i:04d}.npz'
        state=dict(np.load(path)) if path.exists() else {'t':(selected[i]+1)/60}
        m=matrices(arm,float(state['t']))
        if 'source_pos' in state:np.testing.assert_allclose(m['source_cup'][:3,3],state['source_pos'],atol=1e-7)
        values.append({k:(v@inverse[k]).tolist() for k,v in m.items()})
    arm.close()
    scene=json.loads((ROOT/'out/pour_share_video_side_by_side_20260911/scene.json').read_text())
    scene.update(frame_count=50,display_fps=25,render_scale=1.0,upright_crop_xyxy=[0,181,480,575],poses=values)
    write(out/'scene.json',scene)
    for name in ['meshes','frames','rendered']:(out/name).mkdir(exist_ok=True)
    return values


def mesh_frame(job):
    index,i,pose=job
    from experiments.pour import pour_share_video_prepare as prep
    out=case_dir(index);dest=out/f'frames/frame_{i:04d}.json'
    if dest.exists():return
    s=dict(np.load(out/f'capture/states/state_{i:04d}.npz'));common=dict(np.load(out/'capture/particles.npz'))
    cups=[('source',s['source_pos'],s['source_quat']),('receiver',common['receiver_pos'],common['receiver_quat'])]
    vertices,faces,audit=prep.surface(s['x_world'],common['vol'],cups)
    mesh=out/f'meshes/liquid_{i:04d}.ply';prep.still.write_ply(mesh,vertices,faces)
    write(dest,{'index':i,'t':float(s['t']),'mesh':str(mesh),'poses':pose,'surface':audit})


def prepare(index):
    from concurrent.futures import ProcessPoolExecutor
    out=case_dir(index);values=poses(index)
    with ProcessPoolExecutor(max_workers=6) as pool:
        pending={};next_index=0;done=0
        while done<50:
            while next_index<50 and len(pending)<8:
                path=out/f'capture/states/state_{next_index:04d}.npz'
                # The first two captures predate atomic export. A completed NPZ
                # zip end-of-directory marker ensures those reads are safe too.
                if not path.exists():break
                try:
                    with np.load(path) as state: assert state['x_world'].shape==(229280,3)
                except (OSError,ValueError,EOFError):break
                pending[pool.submit(mesh_frame,(index,next_index,values[next_index]))]=next_index
                next_index+=1
            for future in list(pending):
                if future.done():
                    future.result();del pending[future];done+=1
                    print('Prepared surface',index,done,'/ 50',flush=True)
            if done<50:time.sleep(1)
    while not (out/'verification.json').exists():time.sleep(1)
    assert json.loads((out/'verification.json').read_text())['accepted']
    print('MESHES_COMPLETE',index,flush=True)


def compose(index):
    from PIL import Image
    out=case_dir(index);dest=out/'target.mp4'
    cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','480x394','-r','25','-i','-','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for i in range(50):
        with Image.open(out/f'rendered/frame_{i:04d}.png') as pic:
            im=pic.convert('RGB').transpose(Image.Transpose.ROTATE_90).resize((480,394),Image.Resampling.LANCZOS)
            proc.stdin.write(im.tobytes())
    proc.stdin.close();assert proc.wait()==0
    print('CLIP_COMPLETE',index,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['capture','prepare','compose']);p.add_argument('--case',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    {'capture':lambda:capture(a.case,a.device),'prepare':lambda:prepare(a.case),'compose':lambda:compose(a.case)}[a.stage]()
