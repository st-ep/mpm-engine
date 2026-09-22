"""Known Hencky/von Mises linear-isotropic-hardening candidate.

q is accumulated Frobenius-norm plastic log-strain increment; Y(q)=Y0+H*q.
Outer ratios Y0/E,H/E retain the conditional linear E solve.
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.optimize import minimize
from experiments.robotics.press_hardware_observe import read,save


def stress_history(F,ratio,hardening_ratio,nu,tau=0.,dt=.05):
    mu=1/(2*(1+nu));Fe=np.tile(np.eye(3),(F.shape[1],1,1));old=Fe.copy();q=np.zeros(F.shape[1]);out=[]
    for f in F:
        trial=f@np.linalg.inv(old)@Fe;U,s,Vh=np.linalg.svd(trial)
        eps=np.log(s);mean=eps.mean(1);dev=eps-mean[:,None];norm=np.linalg.norm(dev,axis=1)
        delta=np.maximum(norm-(ratio+hardening_ratio*q)/(2*mu),0)*dt/(tau+dt*(1+hardening_ratio/(2*mu)))
        q+=delta;dev*= (1-delta/np.maximum(norm,1e-20))[:,None]
        Fe=(U*np.exp(mean[:,None]+dev)[:,None,:])@Vh
        out.append((U*(2*mu*dev)[:,None,:])@U.transpose(0,2,1));old=f
    return np.array(out)


def fit(out,material,rate=False):
    p=read(out/'protocol.json');names=p['train'][material];data=[];records=[]
    for ep in names:
        d=dict(np.load(out/ep/'motion.npz'));d['volume']=read(out/ep/'geometry.json')['volume_m3'];data.append(d)
    def evaluate(v):
        ratio,hr=np.exp(v[:2]);tau=float(np.exp(v[2])) if rate else 0.;AA=[];bb=[]
        for d in data:
            stress=stress_history(d['F'],ratio,hr,p['nu'],tau,p['export_dt_s']);inst=d['volume']*np.einsum('tnij,tmnij,n->tm',stress,d['test_gradient'],d['weights'])
            A=d['windows']@inst;b=d['target'];select=(d['window_begin']<10)|(d['window_begin']>=10.8);scale=np.linalg.norm(b[select])
            AA.append(A[select].ravel()/scale);bb.append(b[select].ravel()/scale)
        A=np.concatenate(AA);b=np.concatenate(bb);E=float(np.clip(A@b/(A@A),1000,2e6));loss=float(np.linalg.norm(E*A-b)**2/len(data))
        records.append(dict(E_Pa=E,Y_Pa=E*ratio,ratio=ratio,H_Pa=E*hr,hardening_ratio=hr,tau_s=tau,eta_Pa_s=E/(1+p['nu'])*tau,law='Hencky/von Mises linear isotropic hardening'+(' and linear overstress' if rate else ''),relative_weak_residual=np.sqrt(loss),training_episodes=names))
        return loss
    bounds=[(np.log(1e-5),np.log(1)),(np.log(1e-4),np.log(20))]
    if rate:
        initial=read(out/material/'fit.json')['candidates'][1]
        starts=[np.log([initial['ratio'],hr,initial['tau_s']]) for hr in [.01,.1,1.]]
        bounds.append((np.log(.002),np.log(300)))
    else:
        grid=[]
        for ratio in np.logspace(-4,-.1,10):
            for hr in np.logspace(-3,1,7):
                v=np.log([ratio,hr]);grid.append((evaluate(v),v))
        starts=[min(grid,key=lambda a:a[0])[1]]
    candidates=[minimize(evaluate,best,method='Nelder-Mead',bounds=bounds,options=dict(maxiter=100,xatol=.005,fatol=1e-7)) for best in starts]
    opt=min(candidates,key=lambda a:a.fun);evaluate(opt.x)
    save(out/material/('rate_hardening_fit.json' if rate else 'hardening_fit.json'),dict(candidate=records[-1],scope='Training-only known-family candidate; not automatically adopted',search=records))
    print(material,records[-1],flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',type=Path,required=True);a.add_argument('--material',default='plasticine');a.add_argument('--rate',action='store_true');args=a.parse_args();fit(args.out,args.material,args.rate)
