"""Evaluation-only separation of numerical and volume-reconstruction errors.

This diagnostic deliberately reads simulation states. None of its reconstructed
fields or parameter estimates are valid observation-only identification results.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
from experiments.robotics.plate_observable_divfree import weak
from experiments.robotics.plate_observable_field import unit_stress
from experiments.elastic.strip_camera_observe import save


def audit(root,k,kind='truth',interior=False):
    p=json.loads((root/f'inputs_{k}/known.json').read_text());dest=root/f'numerical_diagnostic_{"interior_" if interior else ""}{kind}_{k}';dest.mkdir(exist_ok=False)
    source=root/f'{kind}_{k}';raw=np.load(source/'x.npy',mmap_mode='r');elastic=np.load(source/'Fe.npy',mmap_mode='r') if (source/'Fe.npy').exists() else None
    t=np.load(root/f'inputs_{k}/time.npy');axes=[np.unique(raw[0,:,j]) for j in range(3)];shape=tuple(map(len,axes))
    counts=[12,12,24] if interior else [12,12,12]
    grid=[c+s*((np.arange(n)+.5)/n-.5) for c,s,n in zip(p['center'],p['size'],counts,strict=True)]
    q=np.stack(np.meshgrid(*grid,indexing='ij'),axis=-1).reshape(-1,3);X=[];F=[];Fe=[]
    for i in range(len(t)):
        xyz=raw[i].reshape(*shape,3).astype(float)
        G=np.stack(np.gradient(xyz,*axes,axis=(0,1,2),edge_order=2),axis=-1)
        def interp(a):return RegularGridInterpolator(axes,a,bounds_error=False,fill_value=None)(q)
        X.append(interp(xyz));F.append(interp(G))
        if elastic is not None:Fe.append(interp(elastic[i].reshape(*shape,3,3)))
    X=np.array(X);F=np.array(F);Fe=np.array(Fe);V=savgol_filter(X,7,3,deriv=1,delta=t[1]-t[0],axis=0)
    sensor=np.genfromtxt(source/'force.csv',names=True,delimiter=',')[1:];sensor['time']-=p['force_tick']/2
    if interior:
        from experiments.robotics.plate_observable_interior import weak as interior_weak
        W,b,g=interior_weak(X,V,t,p,sensor,np.full(len(q),1/len(q)))
    else:W,b,g=weak(X,V,t,p,sensor,(1,2,3))
    nu=p['nu']
    def column(stress):return (W@(np.einsum('tnij,tmnij->tm',stress,g)/len(q)*np.prod(p['size']))).ravel()
    exact=None
    if elastic is not None:
        U,s,_=np.linalg.svd(Fe);eps=np.log(s);principal=eps/(1+nu)+nu/((1+nu)*(1-2*nu))*eps.sum(-1)[...,None]
        tau=np.einsum('tnik,tnk,tnjk->tnij',U,principal,U)
        A=column(tau);E=float(A@b/(A@A));exact=dict(E_pa=E,residual=float(np.linalg.norm(E*A-b)/np.linalg.norm(b)))
    tries=[]
    def score(lr):
        r=np.exp(lr);A=column(unit_stress(F,r,nu,True));E=float(A@b/(A@A));loss=float(np.sum((E*A-b)**2)/np.sum(b*b))
        tries.append(dict(E_pa=E,yield_pa=E*r,loss=loss));return loss
    grid=np.linspace(np.log(.001),np.log(.5),33);loss=[score(x) for x in grid];j=np.argmin(loss)
    minimize_scalar(score,bounds=(grid[max(j-1,0)],grid[min(j+1,32)]),method='bounded')
    result=dict(scope='EVALUATION ONLY, hidden simulator inputs',supplied_Fe_weak_fit=exact,total_motion_local_history_fit=min(tries,key=lambda x:x['loss']))
    np.savez(dest/'diagnostic_fields.npz',x=X,F=F,Fe=Fe,time=t,q=q)
    save(dest/'result.json',result);print(k,result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',choices=['A','B'],required=True)
    ap.add_argument('--kind',default='truth');ap.add_argument('--interior',action='store_true');a=ap.parse_args();audit(a.out,a.material,a.kind,a.interior)
