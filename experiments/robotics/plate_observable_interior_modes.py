"""Test whether observed weak balance constrains missing interior motion modes.

The two added displacement modes vanish on all four vertical side faces. Their
time/height dependence is supplied by measured curvature coefficients. Two
scalar amplitudes, shared over the whole probe, are inferred along with E/Y.
This relaxes a fixed interior interpolation; it does not provide simulator states.
The resulting interior representation is still approximate, not uniquely given
by the surface observations. Odd weak windows are reserved for validation.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from experiments.robotics.plate_observable_field import coefficients, height_edges, reconstruct, unit_stress, volume_quadrature
from experiments.robotics.plate_observable_interior import virtual
from experiments.elastic.strip_camera_observe import save


def modes(q,p,spacing,c,velocity):
    h=p['size'][2];half=p['size'][0]/2
    knots=np.r_[[p['floor']]*4,height_edges(p,spacing)[1:-1],[p['floor']+h]*4]
    n=len(knots)-4;s=BSpline(knots,np.eye(n),3);Z=s(q[:,2]);dZ=s(q[:,2],nu=1)
    a=(q[:,0]-p['center'][0])/half;b=(q[:,1]-p['center'][1])/half
    f=(1-a*a)*(1-b*b)
    z=(q[:,2]-p['floor'])/h;zshape=4*z*(1-z)
    phi=[np.column_stack([a*f,b*f,np.zeros(len(a))]),np.column_stack([np.zeros(len(a)),np.zeros(len(a)),f*zshape])]
    G=np.zeros((2,len(q),3,3))
    G[0,:,0,0]=(1-3*a*a)*(1-b*b)/half;G[0,:,0,1]=-2*a*b*(1-a*a)/half
    G[0,:,1,0]=-2*a*b*(1-b*b)/half;G[0,:,1,1]=(1-3*b*b)*(1-a*a)/half
    G[1,:,2,0]=-2*a*(1-b*b)*zshape/half;G[1,:,2,1]=-2*b*(1-a*a)*zshape/half
    G[1,:,2,2]=f*4*(1-2*z)/h
    fields=[]
    for j,group in enumerate([1,4]):  # observed tangential/axial curvature in degree-3 basis
        sl=slice(group*n,(group+1)*n);amplitude=c[:,sl]@Z.T;derivative=c[:,sl]@dZ.T
        dx=amplitude[...,None]*phi[j][None]
        dv=(velocity[:,sl]@Z.T)[...,None]*phi[j][None]
        dF=amplitude[...,None,None]*G[j][None]
        dF[...,2]+=derivative[...,None]*phi[j][None]
        fields.append((dx,dv,dF))
    return fields


def balance(X,V,t,p,sensor,weights):
    # Use both deviatoric-sensitive and full vertical momentum balances.
    # Eliminating pressure is useful for fitting a fixed reconstruction, but
    # would leave an inferred interior free to acquire large volume errors.
    w,g=virtual(X,p);normal=np.zeros_like(w);normal[...,2]=w[...,2]
    normal_g=np.zeros_like(g);normal_g[...,2,2]=g[...,2,2]
    w=np.concatenate([w,normal],axis=1);g=np.concatenate([g,normal_g],axis=1)
    mass=np.prod(p['size'])*p['density']
    momentum=mass*np.einsum('tmn,n->tm',np.einsum('tni,tmni->tmn',V,w),weights)
    load=mass*np.einsum('tmn,n->tm',np.einsum('tni,tmnij,tnj->tmn',V,g,V)-9.81*w[...,2],weights)
    W=[];rhs=[];dt=t[1]-t[0]
    for begin in np.arange(.2,p['fit_end']-.159,.06):
        u=np.clip((t-begin)/.16,0,1);chi=np.sin(np.pi*u)**4;dchi=4*np.pi/.16*np.sin(np.pi*u)**3*np.cos(np.pi*u)
        b=np.sum(chi[:,None]*load+dchi[:,None]*momentum,axis=0)*dt
        u=(sensor['time']-begin)/.16;use=(u>=0)&(u<=1)
        b-=np.sum(np.sin(np.pi*u[use])**4*sensor['Fz'][use])*p['force_tick']
        W.append(chi*dt);rhs.append(b)
    return np.array(W),np.array(rhs).ravel(),g


def fit(root,k,order=6,tag='modes6',start_mode=0.,max_evaluations=240):
    p=json.loads((root/f'inputs_{k}/known.json').read_text());d=np.load(root/f'fit_{k}/tracks.npz')
    selection=json.loads((root/'field_selection.json').read_text())['selected']
    if selection.get('cross_section_degree',3)!=3:raise ValueError('This bounded test enriches the degree-3 reconstruction only')
    spacing=selection['spacing'];smoothing=selection.get('smoothing',0.)
    q,weights=volume_quadrature(p,spacing,order);X,V,F,_,_=reconstruct(d,p,spacing,quadrature=q,smoothing=smoothing)
    coeff,_,_=coefficients(d,p,spacing,smoothing=smoothing);dt=float(d['time'][1]-d['time'][0])
    c=savgol_filter(coeff,7,3,axis=0);c[0]=0;v=savgol_filter(coeff,7,3,deriv=1,delta=dt,axis=0)
    fields=modes(q,p,spacing,c,v);sensor=np.genfromtxt(root/f'inputs_{k}/force.csv',names=True,delimiter=',')
    config=json.loads((root/'identification_protocol.json').read_text())
    baseline=json.loads((root/f'divfree_{k}/identification.json').read_text())
    centers=np.arange(.2,p['fit_end']-.159,.06)+.08
    nfields=6
    train=np.repeat(np.arange(len(centers))%2==0,nfields);valid=~train
    _,reference_load,_=balance(X,V,d['time'],p,sensor,weights)
    scale=np.ones_like(reference_load)
    for lo,hi in config.get('balanced_identification_intervals',[[.2,p['fit_end']]]):
        rows=np.repeat((centers>=lo)&(centers<hi),nfields)
        scale[rows]=1/np.linalg.norm(reference_load[rows&train])
    normalization=float(np.linalg.norm((reference_load*scale)[train]))
    validation_norm=float(np.sum((reference_load*scale)[valid]**2))
    attempts=[];started=time.monotonic()
    def residual(parameters,record=True):
        logr,alpha,beta=parameters;ratio=np.exp(logr)
        if not .001<=ratio<=.5:return None
        xx=X+alpha*fields[0][0]+beta*fields[1][0]
        vv=V+alpha*fields[0][1]+beta*fields[1][1]
        ff=F+alpha*fields[0][2]+beta*fields[1][2]
        minj=float(np.linalg.det(ff).min())
        if minj<=0:return None
        W,b,g=balance(xx,vv,d['time'],p,sensor,weights)
        stress=unit_stress(ff,ratio,p['nu'],True)
        A=(W@np.einsum('tnij,tmnij,n->tm',stress,g,weights)).ravel()*np.prod(p['size'])*scale
        target=b*scale
        E=float(np.clip(A[train]@target[train]/(A[train]@A[train]),5000,500000))
        r=(E*A-target)/normalization
        result=dict(E_pa=E,yield_pa=E*ratio,ratio=ratio,alpha=float(alpha),beta=float(beta),
                    training_loss=float(r[train]@r[train]),validation_loss=float(np.sum((E*A-target)[valid]**2)/validation_norm),min_J=minj)
        if record:attempts.append(result)
        return r,result
    def score(z):
        value=residual(z)
        if value is None:return 1e6+float(np.sum(np.asarray(z)**2))
        if len(attempts)%40==0:print(k,order,len(attempts),value[1],flush=True)
        return value[1]['training_loss']
    initial=np.array([np.log(baseline['yield_pa']/baseline['E_pa']),start_mode,start_mode])
    simplex=np.tile(initial,(4,1));simplex[1,0]+=.06;simplex[2,1]+=.25;simplex[3,2]+=.25
    opt=minimize(score,initial,method='Nelder-Mead',options=dict(maxfev=max_evaluations,xatol=2e-4,fatol=1e-8,initial_simplex=simplex))
    value=residual(opt.x);best=value[1]
    # Local sensitivity uses permitted observables only. It detects a flat
    # parameter/mode direction; it is not a statistical confidence interval.
    J=[]
    for j in range(3):
        dz=np.zeros(3);dz[j]=1e-3
        plus=residual(opt.x+dz,False);minus=residual(opt.x-dz,False)
        J.append((plus[0][train]-minus[0][train])/(2e-3) if plus and minus else np.full(sum(train),np.nan))
    singular=np.linalg.svd(np.column_stack(J),compute_uv=False) if np.isfinite(J).all() else np.full(3,np.nan)
    best.update(optimizer_success=bool(opt.success),optimizer_message=str(opt.message),evaluations=len(attempts),quadrature_order=order,
                start_mode=start_mode,fit_wall_s=time.monotonic()-started,profiled_sensitivity_singular_values=singular.tolist(),
                simulator_states_used=False,pressure_balance_included=True,
                scope='Joint parameter/interior-mode fit; approximate interior, alternating weak windows withheld; fixed normalization from training data')
    dest=root/f'{tag}_{k}';dest.mkdir(exist_ok=False);save(dest/'identification.json',best);save(dest/'search.json',attempts)
    print('FINISHED',k,best,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--material',choices=['A','B'],required=True)
    ap.add_argument('--order',type=int,default=6);ap.add_argument('--tag',default='modes6');ap.add_argument('--start-mode',type=float,default=0.)
    ap.add_argument('--max-evaluations',type=int,default=240);a=ap.parse_args();fit(a.out,a.material,a.order,a.tag,a.start_mode,a.max_evaluations)
