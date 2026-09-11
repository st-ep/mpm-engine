"""Full MPM reference in a compact domain, with explicit crop verification.

Both cups, the recorded motion and gravity remain in the world frame. A static
coordinate translation removes unused grid space. No source-only/outlet model,
moving frame, receiver-volume correction or simulator viscosity fitting is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from examples import pour_recorded_twin as twin
from warpmpm.geometry.measuring_cup import make_cup_mesh

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'out/pour_weakform_recovery/identified'
OUT=ROOT/'out/pour_physics_audit/compact'
EPISODE=ROOT/'pouring_real_data/09-04-60-2s'


def swept_bounds(arm,receiver):
    # Use the largest (historical) collision shell for every grid, fixing the
    # geometric crop independently of resolution and measured liquid volumes.
    vertices,_=make_cup_mesh(twin.SPEC,*twin.collision_extras(.7/384),n_theta=96,n_z=24)
    lower=np.full(3,np.inf);upper=-lower
    for t in np.arange(0,arm.duration+1e-9,1/60):
        pos,quat=arm.cup_pose_at(float(t))
        points=vertices@twin.quat_to_mat(quat).T+pos
        lower=np.minimum(lower,points.min(0));upper=np.maximum(upper,points.max(0))
    points=vertices@twin.quat_to_mat(twin.Q_RCV).T+receiver
    return np.minimum(lower,points.min(0)),np.maximum(upper,points.max(0))


def run(args):
    identity_path=BASE/'identify.json'
    identity=json.loads(identity_path.read_text())
    if (identity['calibration_pour_count']!=1 or identity['measured_endpoints_used'] or
            identity['validation_measurements_used'] or identity['viscosity_fitted_by_simulator']):
        raise ValueError('Requires endpoint-free, one-pour weak identification')
    eta=identity['eta']
    if args.refined_identification:
        identity_path=ROOT/'out/pour_physics_audit/optical_full_rate/results.json'
        refined=json.loads(identity_path.read_text())
        protocol=refined['protocol']
        if (protocol['calibration_pour_count']!=1 or protocol['measured_endpoints_used']
                or protocol['other_recordings_used'] or protocol['video_frame_stride']!=1):
            raise ValueError('Requires a single-video optical refinement')
        eta=next(r for r in refined['cases'] if r['ray_steps']==4096)['eta_pa_s']
    geometry=json.loads((BASE/'geometry.json').read_text())
    extent=args.grid*(.7/384) if args.crop_check else .35
    dx=extent/args.grid
    if args.crop_check and (args.grid!=192 or args.phase!=0 or args.refined_identification or args.sdf_res!=160):
        raise ValueError('Crop check fixes the original cell size, grid phase, eta and SDF')
    # The verification cube is 0.35 m = 192 original cells. Finer runs retain the
    # same cube but use the explicitly specified MPM and geometry resolutions.
    name=('crop_check' if args.crop_check else f'refined_n{args.grid}_phase{args.phase:g}_sdf{args.sdf_res}')
    destination=OUT/name;destination.mkdir(parents=True,exist_ok=False)
    ep=twin.load_episode(EPISODE,twin.PRE_ROLL,twin.HOLD_SECONDS)
    arm=twin.RecordedPanda(ep,ROOT/'out/pour_wf/09-04-60-2s/cup_render.obj',height=64,width=64,max_geom=4000,
                           cup_reference_pos=geometry['cup_reference_pos'],cup_reference_quat=geometry['cup_reference_quat'])
    receiver=np.r_[geometry['receiver_xy'],geometry['table_z']]
    lower,upper=swept_bounds(arm,receiver)
    ideal=.5*(extent-lower-upper)
    original_offset=twin.world_to_mpm_offset(arm,receiver,.7/384)
    # Anchor the cube to the original grid; the crop-only case retains identical
    # physical grid nodes, including the fractional placement of the cup.
    coarse_dx=.7/384
    offset=original_offset-np.rint((original_offset-ideal)/coarse_dx)*coarse_dx
    offset+=args.phase*dx*np.array([1.,0.,1.])
    arm.close()
    margin=np.minimum(lower+offset,extent-upper-offset)
    if margin.min()<.035:
        raise ValueError(f'Insufficient independently specified geometric crop margin: {margin}')
    paths=[Path(__file__),Path(twin.__file__),identity_path,BASE/'geometry.json',
           ROOT/'experiments/pour/pour_transport_identify.py',
           ROOT/'experiments/pour/pour_weakform_transport.py',
           ROOT/'experiments/pour/pour_weakform_identify.py',
           *sorted((ROOT/'src/warpmpm').rglob('*.py')),
           *[EPISODE/k for k in ['states.jsonl','actions.jsonl','meta.json']]]
    protocol=dict(name=name,episode=EPISODE.name,eta_pa_s=eta,initial_volume_ml=300.,
                  calibration_pour_count=1,measured_endpoints_used=[],validation_outcomes_used=[],
                  viscosity_fitted_by_simulator=False,source_wall='sticky',receiver_wall='separable',
                  n_grid=args.grid,extent_m=extent,dx_m=dx,sdf_resolution=args.sdf_res,
                  phase_cells=args.phase,world_to_grid_offset=offset.tolist(),
                  geometry_margin_m=margin.tolist(),swept_geometry_lower=lower.tolist(),
                  swept_geometry_upper=upper.tolist(),boundary_clearance_gate_cells=3.,
                  purpose='Full-scene MPM numerical verification; no parameter fit to receiver outcomes',
                  input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    old=(twin.Solver,twin.OUT_ROOT,twin.GRID_LIM,twin.SDF_RES,twin.world_to_mpm_offset)

    class AuditedSolver(old[0]):
        def add_sdf_collider(self,*positional,**keywords):
            index=getattr(self,'_source_index',0)
            if index==0:keywords=dict(keywords,surface='sticky',friction=0.)
            self._source_index=index+1
            return super().add_sdf_collider(*positional,**keywords)

        def step(self,*positional,**keywords):
            value=super().step(*positional,**keywords)
            x=self.x()
            clearance=float(min(x.min(),extent-x.max()))
            self.minimum_clearance_m=min(getattr(self,'minimum_clearance_m',np.inf),clearance)
            if clearance<3*dx:
                raise RuntimeError(f'Liquid reached the crop boundary ({clearance/dx:.2f} cells); reject this crop')
            boundary_clearances.append(clearance)
            return value

    boundary_clearances=[]
    twin.Solver,twin.OUT_ROOT,twin.GRID_LIM,twin.SDF_RES=AuditedSolver,destination,extent,args.sdf_res
    twin.world_to_mpm_offset=lambda arm,receiver,dx:offset.copy()
    started=time.monotonic()
    try:
        simulation=twin.run(EPISODE,device=args.device,n_grid=args.grid,video=False,
                            side_by_side=False,rebake=True,eta=eta,volume_ml=300.,**geometry)
    finally:
        twin.Solver,twin.OUT_ROOT,twin.GRID_LIM,twin.SDF_RES,twin.world_to_mpm_offset=old
    rows=simulation['rows'];counts=[r['n_src']+r['n_rcv']+r['n_air_spill'] for r in rows]
    if len(set(counts))!=1:raise RuntimeError('Particle ledger changed')
    n=counts[0];last=rows[-1];tail=[r['n_rcv'] for r in rows if r['t']>=last['t']-.5]
    result=dict(**protocol,receiver_ml=300*last['n_rcv']/n,
                source_depletion_ml=300*(1-last['n_src']/n),outside_ml=300*last['n_air_spill']/n,
                tail_variation_ml=300*float(np.ptp(tail))/n,particle_count=n,
                minimum_fluid_boundary_clearance_m=min(boundary_clearances),
                elapsed_s=time.monotonic()-started,status='Numerical diagnostic; no robot table released')
    (destination/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid',type=int,required=True)
    p.add_argument('--crop-check',action='store_true')
    p.add_argument('--refined-identification',action='store_true')
    p.add_argument('--phase',type=float,choices=[0.,.5],default=0.)
    p.add_argument('--sdf-res',type=int,default=160)
    p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
