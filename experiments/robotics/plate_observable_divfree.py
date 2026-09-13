"""Pressure-free weak fit to reconstructed stereo surface motion.

No simulation states or true parameters are read. Polynomial spatial test
fields are divergence-free in current coordinates, not a constraint on motion.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.optimize import minimize_scalar
from experiments.robotics.plate_observable_field import reconstruct,unit_stress,volume_quadrature
from experiments.elastic.strip_camera_observe import save


def virtual(X,p,powers=(1,2,3)):
    h=p['size'][2];r=X-np.array([p['center'][0],p['center'][1],p['floor']]);z=r[...,2]/h
    fields=[];grads=[]
    for m in powers:
        f=z**m;df=m*z**(m-1)/h;ddf=m*(m-1)*z**max(m-2,0)/h**2
        w=np.stack([-r[...,0]*df/2,-r[...,1]*df/2,f],axis=-1)
        g=np.zeros((*X.shape[:-1],3,3));g[...,0,0]=g[...,1,1]=-df/2;g[...,2,2]=df
        g[...,0,2]=-r[...,0]*ddf/2;g[...,1,2]=-r[...,1]*ddf/2
        fields.append(w);grads.append(g)
    return np.stack(fields,axis=1),np.stack(grads,axis=1)


def weak(X,V,t,p,sensor,powers,weights=None):
    w,g=virtual(X,p,powers);mass=np.prod(p['size'])*p['density']
    if weights is None:weights=np.full(X.shape[1],1/X.shape[1])
    momentum=mass*np.einsum('tmn,n->tm',np.einsum('tni,tmni->tmn',V,w),weights)
    load=mass*np.einsum('tmn,n->tm',np.einsum('tni,tmnij,tnj->tmn',V,g,V)-9.81*w[...,2],weights)
    W=[];rhs=[];dt=t[1]-t[0]
    for begin in np.arange(.20,p['fit_end']-.159,.06):
        u=np.clip((t-begin)/.16,0,1);chi=np.sin(np.pi*u)**4
        dchi=4*np.pi/.16*np.sin(np.pi*u)**3*np.cos(np.pi*u)
        b=np.sum(chi[:,None]*load+dchi[:,None]*momentum,axis=0)*dt
        u=(sensor['time']-begin)/.16;use=(u>=0)&(u<=1)
        for j,m in enumerate(powers):
            b[j]-=np.sum(np.sin(np.pi*u[use])**4*sensor['Fz'][use]*(sensor['opening'][use]/p['size'][2])**m)*p['force_tick']
        W.append(chi*dt);rhs.append(b)
    return np.array(W),np.array(rhs).ravel(),g


def fit(root,k,powers=(1,2,3),tag='divfree',quadrature_order=None):
    p=json.loads((root/f'inputs_{k}/known.json').read_text());d=np.load(root/f'fit_{k}/tracks.npz')
    selected=json.loads((root/'field_selection.json').read_text())['selected'];spacing=selected['spacing']
    p['cross_section_degree']=selected.get('cross_section_degree',3)
    q,weights=(None,None) if quadrature_order is None else volume_quadrature(p,spacing,quadrature_order)
    X,V,F,q,error=reconstruct(d,p,spacing,quadrature=q,smoothing=selected.get('smoothing',0.));sensor=np.genfromtxt(root/f'inputs_{k}/force.csv',delimiter=',',names=True)
    if weights is None:weights=np.full(len(q),1/len(q))
    W,b,g=weak(X,V,d['time'],p,sensor,powers,weights);tries=[];start=time.monotonic()
    def score(logratio):
        ratio=np.exp(logratio);stress=unit_stress(F,ratio,p['nu'],full=True)
        internal=np.einsum('tnij,tmnij,n->tm',stress,g,weights)*np.prod(p['size'])
        A=(W@internal).ravel();E=float(np.clip(A@b/(A@A),5000,500000));loss=float(np.sum((E*A-b)**2)/np.sum(b*b))
        tries.append(dict(E_pa=E,yield_pa=ratio*E,ratio=ratio,loss=loss));return loss
    grid=np.linspace(np.log(.001),np.log(.5),33);losses=[score(v) for v in grid];j=int(np.argmin(losses))
    opt=minimize_scalar(score,bounds=(grid[max(0,j-1)],grid[min(32,j+1)]),method='bounded',options={'xatol':1e-6});score(opt.x)
    best=dict(min(tries,key=lambda v:v['loss']));best['residual_relative_l2']=float(np.sqrt(best.pop('loss')))
    best.update(fit_wall_s=time.monotonic()-start,evaluations=len(tries),surface_fit_rmse_mm=float(np.sqrt(np.mean(error**2))*1000),
        min_reconstructed_J=float(np.linalg.det(F).min()),known_nu=p['nu'],estimated=['E','yield_stress'],
        estimator='Nonlinear profile search Y/E with conditional linear E solve; local plastic history inferred from reconstructed motion',
        reconstruction='Square-symmetric low-order volume interpolation from tracked surface; no uniform compression or plane stress',
        virtual_field_powers=list(powers),quadrature_order=quadrature_order,pressure_eliminated=True,forward_simulation_calls=0,simulator_states_used=False)
    dest=root/f'{tag}_{k}';dest.mkdir(exist_ok=False)
    save(dest/'identification.json',best);save(dest/'search.json',tries);print(best,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',choices=['A','B'],required=True)
    ap.add_argument('--powers',type=int,nargs='+',default=[1,2,3]);ap.add_argument('--tag',default='divfree');a=ap.parse_args();fit(a.out,a.material,tuple(a.powers),a.tag)
