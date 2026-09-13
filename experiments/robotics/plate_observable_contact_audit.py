"""Evaluation-only check of force balance using supplied elastic states.

This does NOT satisfy observation-only identification. It checks the numerical
balance for the predeclared interior virtual fields without tuning their support.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import savgol_filter
from experiments.robotics.plate_observable_field import volume_quadrature
from experiments.robotics.plate_observable_interior import weak
from experiments.elastic.strip_camera_observe import save


def audit(root,k):
    p=json.loads((root/f'inputs_{k}/known.json').read_text());src=root/f'truth_{k}'
    raw=np.load(src/'x.npy',mmap_mode='r');rawfe=np.load(src/'Fe.npy',mmap_mode='r');t=np.load(root/f'inputs_{k}/time.npy')
    axes=[np.unique(raw[0,:,j]) for j in range(3)];shape=tuple(map(len,axes));q,w=volume_quadrature(p,p['size'][2]/4,12)
    X=[];stress=[];nu=p['nu']
    for i in range(len(t)):
        X.append(RegularGridInterpolator(axes,raw[i].reshape(*shape,3),bounds_error=False,fill_value=None)(q))
        Fe=RegularGridInterpolator(axes,rawfe[i].reshape(*shape,3,3),bounds_error=False,fill_value=None)(q)
        U,s,_=np.linalg.svd(Fe);eps=np.log(s);dev=eps-eps.mean(-1)[:,None]
        stress.append(np.einsum('nik,nk,njk->nij',U,dev/(1+nu),U))
    X=np.array(X);stress=np.array(stress);V=savgol_filter(X,7,3,deriv=1,delta=t[1]-t[0],axis=0)
    sensor=np.genfromtxt(root/f'inputs_{k}/force.csv',names=True,delimiter=',');W,b,g=weak(X,V,t,p,sensor,w)
    A=(W@np.einsum('tnij,tmnij,n->tm',stress,g,w)).ravel()*np.prod(p['size']);E=float(A@b/(A@A))
    result=dict(scope='EVALUATION ONLY: exact simulator elastic states supplied; not an identification result',E_pa=E,
                relative_weak_residual=float(np.linalg.norm(E*A-b)/np.linalg.norm(b)),quadrature_order=12)
    save(root/f'interior_contact_audit_{k}.json',result);print(k,result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',choices=['A','B'],required=True)
    a=ap.parse_args();audit(a.out,a.material)
