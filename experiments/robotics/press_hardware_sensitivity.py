"""Training-only local sensitivity to force offset and assumed initial height."""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from experiments.robotics.press_hardware_model import geometry,quadrature,virtual,stress_history
from experiments.robotics.press_hardware_observe import read,save


def altered(d,g,p,height_offset,force_offset,quadrature_nz=24,quadrature_nr=4):
    d={k:v.copy() for k,v in d.items()};volume=g['volume_m3']
    if height_offset or quadrature_nz!=24 or quadrature_nr!=4:
        h0=g['h0_m']+height_offset;h=d['height']+height_offset;volume*=h0/g['h0_m']
        s,r,w=quadrature(g['k0'],quadrature_nz,quadrature_nr);pairs=[geometry(hh,kk,volume,s,g['k0'],h0,r) for hh,kk in zip(h,d['shape_k'])]
        d['x']=np.array([v[0] for v in pairs]);d['F']=np.array([v[1] for v in pairs]);d['weights']=w
        d['velocity']=np.gradient(d['x'],d['time'],axis=0,edge_order=2)
        vf,gradient=virtual(d['x'],float(h.min()));d['test_gradient']=gradient
        rho=p['density_kg_m3'];v=d['velocity']
        momentum=volume*rho*np.einsum('tni,tmni,n->tm',v,vf,w)
        convective=volume*rho*np.einsum('tni,tmnij,tnj,n->tm',v,gradient,v,w)
        gravity=-9.81*volume*rho*np.einsum('tmn,n->tm',vf[...,2],w)
        b=[]
        for begin in d['window_begin']:
            u=np.clip((d['time']-begin)/.6,0,1);chi=np.sin(np.pi*u)**4;dc=4*np.pi/.6*np.sin(np.pi*u)**3*np.cos(np.pi*u)
            b.append(np.sum(chi[:,None]*(convective+gravity-d['force'][:,None])+dc[:,None]*momentum,axis=0)*p['export_dt_s'])
        d['target']=np.array(b)
    d['target']-=force_offset*d['windows'].sum(1)[:,None]
    d['volume']=volume
    return d


def check(out,material):
    p=read(out/'protocol.json');best=read(out/material/'fit.json')['selected'];viscous=best['tau_s']>0
    base=[(dict(np.load(out/ep/'motion.npz')),read(out/ep/'geometry.json')) for ep in p['train'][material]]
    results=[]
    for dh,df in [(-.002,0.),(.002,0.),(0.,1.)]:
        data=[altered(d,g,p,dh,df) for d,g in base]
        candidates=[]
        def score(z):
            ratio=np.exp(z[0]);tau=np.exp(z[1]) if viscous else 0.;As=[];bs=[]
            for d in data:
                stress=stress_history(d['F'],ratio,tau,p['nu'],p['export_dt_s'])
                a=d['volume']*np.einsum('tnij,tmnij,n->tm',stress,d['test_gradient'],d['weights'])
                a=d['windows']@a;b=d['target'];keep=(d['window_begin']<10)|(d['window_begin']>=10.8);scale=np.linalg.norm(b[keep])
                As.append(a[keep].ravel()/scale);bs.append(b[keep].ravel()/scale)
            A=np.concatenate(As);b=np.concatenate(bs);E=float(np.clip(A@b/(A@A),1000,2e6));loss=float(np.linalg.norm(E*A-b)/np.sqrt(3))
            candidates.append(dict(E_at_fit_nu_Pa=E,G_Pa=E/(2*(1+p['nu'])),Y_Pa=float(E*ratio),tau_s=float(tau),relative_weak_residual=loss))
            return loss**2
        z=[np.log(best['ratio'])]+([np.log(best['tau_s'])] if viscous else [])
        bounds=[(np.log(1e-5),0)]+([(np.log(.002),np.log(300))] if viscous else [])
        opt=minimize(score,z,method='Nelder-Mead',bounds=bounds,options=dict(maxiter=50,xatol=.01,fatol=1e-6));score(opt.x)
        result=min(candidates,key=lambda q:q['relative_weak_residual']);result.update(height_offset_m=dh,force_offset_N=df,optimizer_success=bool(opt.success))
        results.append(result);print(material,result,flush=True)
    save(out/material/'input_sensitivity.json',dict(results=results,scope='Local training-only re-fits in the selected law; assumptions shifted, not calibrated. Not confidence intervals. A force offset does not reconstruct the missing initial prestress.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--material',required=True);a=p.parse_args();check(a.out,a.material)
