"""Analytical shear-decay verification of viscosity and MPM transfer damping.

A periodic channel with free-slip top/bottom starts with u(z)=A*cos(pi*z/H).
With no gravity, the exact solution decays as exp(-eta/rho*(pi/H)^2*t).
No robot data or fitted parameter enters this numerical diagnostic.
"""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--variant', choices=['original','compensated_position','aflip'],required=True)
p.add_argument('--device',default='cuda:0')
p.add_argument('--eta',type=float,default=3.4392377844275503)
p.add_argument('--grid',type=int,default=16)
p.add_argument('--tag',required=True)
p.add_argument('--mls',type=int,choices=[0,1],default=0)
args=p.parse_args()
source=ROOT/'src' if args.variant=='original' else ROOT/'out/pour_physics_audit'/(args.variant+'_20260907')/'isolated_src'
sys.path.insert(0,str(source))

import hashlib,json,time
import numpy as np
import warp as wp
from warpmpm import Solver,GridConfig,newtonian
from warpmpm.kernels import mpm_utils


def main():
    assert Path(mpm_utils.__file__).resolve().is_relative_to(source)
    out=ROOT/'out/pour_physics_audit/shear_decay_20260907'/args.tag
    out.mkdir(parents=True,exist_ok=False)
    benchmark_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/f'benchmark_{benchmark_sha}.py').write_bytes(Path(__file__).read_bytes())
    grid=GridConfig(n_grid=args.grid,grid_lim=.035)
    dx=grid.dx;hp=dx/2;bottom=.035/4;height=.035/2
    y0=.035*3/8;y1=.035*5/8
    axes=[np.arange(a+hp/2,b-hp/8,hp) for a,b in [(0,.035),(y0,y1),(bottom,bottom+height)]]
    x0=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
    eta=args.eta;rho=1260.;amplitude=.01;duration=.02
    k=np.pi/height
    phase=np.cos(k*(x0[:,2].astype(float)-bottom))
    v0=np.zeros_like(x0);v0[:,0]=amplitude*phase
    L=np.zeros((len(x0),3,3),np.float32)
    L[:,0,2]=-amplitude*k*np.sin(k*(x0[:,2].astype(float)-bottom))
    expected_ratio=float(np.exp(-eta/rho*k*k*duration))
    protocol=dict(purpose=__doc__,variant=args.variant,grid=args.grid,extent_m=.035,
        dx_m=dx,height_m=height,amplitude_m_s=amplitude,duration_s=duration,
        eta_pa_s=eta,density_kg_m3=rho,expected_amplitude_ratio=expected_ratio,mls_transfer=args.mls,
        initial_affine_and_gradient='Analytical initial velocity gradient',
        relative_amplitude_error_threshold=.05,timestep_amplitude_difference_threshold=.01,
        liquid_data_used=[],physical_parameters_refitted=[],
        actual_kernel_module=str(Path(mpm_utils.__file__).resolve()),
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),*sorted((source/'warpmpm').rglob('*.py'))]})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    for scale in [1.,.5,.25]:
        solver=Solver(grid=grid,device=args.device,periodic_x=True).load_particles(
            x0,np.full(len(x0),hp**3,np.float32))
        solver.set_material(newtonian(eta=eta,density=rho,bulk_modulus=9e5),g=[0,0,0])
        solver._sim.mpm_model.mls_transfer=args.mls
        solver.set_v(v0)
        st=solver._sim.mpm_state
        for name in ['particle_C','particle_L']:
            wp.copy(getattr(st,name),wp.array(L,dtype=wp.mat33,device=args.device))
        for point,normal in [((0,y0,0),(0,1,0)),((0,y1,0),(0,-1,0)),
                             ((0,0,bottom),(0,0,1)),((0,0,bottom+height),(0,0,-1))]:
            solver.add_plane(point,normal,'slip')
        acoustic=.28*dx/np.sqrt(1.1*9e5/rho)
        viscous=np.inf if eta==0 else rho*dx*dx/(6*eta)
        limit=min(acoustic,viscous)
        nominal=(1/60)/np.ceil((1/60)/limit)
        steps=int(np.ceil(duration/(nominal*scale)));dt=duration/steps
        start=time.monotonic()
        for done in range(0,steps,100):solver.step(dt,min(100,steps-done))
        x,v=solver.x().astype(float),solver.v().astype(float)
        assert np.isfinite(x).all() and np.isfinite(v).all()
        mode=np.cos(k*(x[:,2]-bottom))
        measured_amplitude=float(np.dot(v[:,0],mode)/np.dot(mode,mode))
        actual_ratio=measured_amplitude/amplitude
        expected=amplitude*expected_ratio*mode
        row=dict(dt_scale=scale,steps=steps,dt_s=dt,elapsed_s=time.monotonic()-start,
            actual_amplitude_ratio=actual_ratio,expected_amplitude_ratio=expected_ratio,
            relative_amplitude_error=(actual_ratio-expected_ratio)/expected_ratio,
            relative_profile_l2=float(np.linalg.norm(v[:,0]-expected)/np.linalg.norm(expected)),
            max_transverse_speed_m_s=float(np.max(abs(v[:,1:]))),
            max_normal_displacement_m=float(np.max(abs(x[:,2]-x0[:,2]))),
            particle_count=len(x0))
        row['amplitude_check_passed']=abs(row['relative_amplitude_error'])<=protocol['relative_amplitude_error_threshold']
        rows.append(row)
        (out/'results.json').write_text(json.dumps(dict(protocol=protocol,cases=rows),indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
