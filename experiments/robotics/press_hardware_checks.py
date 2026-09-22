"""Independent algebra and kernel checks for the hardware constitutive pipeline."""
import argparse
from pathlib import Path
import numpy as np
import warp as wp
import tempfile
import shutil
from experiments.robotics.press_hardware_model import geometry,quadrature,virtual,stress_history
from experiments.robotics.press_hardware_forward import relax_overstress
from experiments.robotics.press_hardware_observe import read,save


def check(out):
    checks={}
    h0=.047;k0=.85;h=.028;k=.4;volume=5e-5
    s,r,w=quadrature(k0);x,F=geometry(h,k,volume,s,k0,h0,r)
    checks['volume_jacobian_max_error']=float(abs(np.linalg.det(F)-1).max())
    checks['quadrature_weight_error']=float(abs(w.sum()-1))
    assert checks['volume_jacobian_max_error']<1e-10
    # Derivative at fixed initial radius: rho must change when reference Z changes.
    R0=np.sqrt(volume/(np.pi*h0*(1-k0/3))*(1-k0*(2*s-1)**2));radius=r*R0;eps=1e-7
    shifted=[]
    for dz in [-eps,eps]:
        sp=s+dz/h0;rp=np.sqrt(volume/(np.pi*h0*(1-k0/3))*(1-k0*(2*sp-1)**2))
        shifted.append(geometry(h,k,volume,sp,k0,h0,radius/rp)[0])
    difference=(shifted[1]-shifted[0])/(2*eps)
    checks['geometry_gradient_finite_difference_max_error']=float(abs(difference-F[:,:,2]).max())
    assert checks['geometry_gradient_finite_difference_max_error']<2e-6
    vx=np.array([[[.01,0,0],[.01,0,h]]]);vw,vg=virtual(vx,h)
    checks['virtual_divergence_max']=float(abs(np.trace(vg,axis1=-2,axis2=-1)).max())
    assert checks['virtual_divergence_max']<1e-12
    assert np.allclose(vw[0,:,0],0) and np.allclose(vw[0,:,1],[0,0,1])
    wp.config.quiet=True;wp.init();results=[]
    for device,ratio,tau in [(device,ratio,tau) for device in ['cpu','cuda:0'] for ratio,tau in [(1.,3.),(.02,3.),(.02,0.)]]:
        dev=np.array([.2,-.1,-.1]);F0=np.diag(np.exp(dev))[None].astype('float32')
        nu=.45;mu=1/(2*(1+nu));dt=.02;threshold=ratio/(2*mu)
        norm=np.linalg.norm(dev);expected_dev=dev*(1-dt/(tau+dt)*max(0,norm-threshold)/norm)
        expected_F=np.diag(np.exp(expected_dev));expected_stress=np.diag(2*mu*expected_dev)
        ft=wp.array(F0,dtype=wp.mat33,device=device);fe=wp.array(F0,dtype=wp.mat33,device=device);rounding=wp.zeros(1,dtype=wp.mat33,device=device)
        wp.launch(relax_overstress,dim=1,inputs=[ft,fe,rounding,threshold,dt/(tau+dt)],device=device)
        kernel_error=float(abs(ft.numpy()[0]-expected_F).max())
        fit_error=float(abs(stress_history(F0[None].astype(float),ratio,tau,nu,dt)[0,0]-expected_stress).max())
        assert kernel_error<2e-6 and fit_error<2e-7
        results.append(dict(device=device,ratio=ratio,tau=tau,kernel_error=kernel_error,identifier_error=fit_error))
    checks['analytic_uniaxial_relaxation']=results
    # Independently check time refinement against the exact exponential decay.
    exact=threshold+(norm-threshold)*np.exp(-1/3)
    errors=[]
    for dt in [.05,.025,.002]:
        n=norm
        for _ in range(round(1/dt)):n-=dt/(3+dt)*(n-threshold)
        errors.append(abs(n-exact))
    assert errors[2]<errors[1]<errors[0]
    checks['relaxation_time_refinement_error']=errors
    checks['status']='passed';checks['scope']='Manufactured algebra/constitutive checks, not experimental validation'
    save(out/'implementation_checks.json',checks);print(checks,flush=True)


def extended(out):
    from experiments.robotics import press_hardware_model as model
    from experiments.robotics.press_hardware_sensitivity import altered
    p=read(out/'protocol.json');records={}
    with tempfile.TemporaryDirectory(prefix='press_ident_isolated_',dir='/dev/shm') as folder:
        isolated=Path(folder);save(isolated/'protocol.json',p)
        for material in p['materials']:
            for ep in p['train'][material]:
                (isolated/ep).mkdir();shutil.copy2(out/ep/'motion.npz',isolated/ep/'motion.npz');shutil.copy2(out/ep/'geometry.json',isolated/ep/'geometry.json')
        old_raw,old_base=model.RAW,model.BASE
        model.RAW=Path('/nonexistent_press_raw');model.BASE=Path('/nonexistent_press_observations')
        try:
            for material in p['materials']:
                model.fit(isolated,material);a=read(isolated/material/'fit.json')['selected'];b=read(out/material/'fit.json')['selected']
                errors={k:abs(a[k]-b[k]) for k in ['E_Pa','Y_Pa','tau_s','relative_weak_residual']}
                assert max(errors.values())<1e-6,(material,errors);records[material]=errors
        finally:model.RAW,model.BASE=old_raw,old_base
    save(out/'identification_isolation_check.json',dict(status='passed',absolute_parameter_differences=records,
        scope='Re-fitted from copied training reconstructions and protocol only; no held-out directories or forward outputs present; raw/observation globals redirected to nonexistent paths.'))
    result={}
    for material in p['materials']:
        best=read(out/material/'fit.json')['selected'];a=[];b=[]
        for ep in p['train'][material]:
            d=altered(dict(np.load(out/ep/'motion.npz')),read(out/ep/'geometry.json'),p,0.,0.,48,6)
            stress=stress_history(d['F'],best['ratio'],best['tau_s'],p['nu'],p['export_dt_s'])
            response=d['volume']*np.einsum('tnij,tmnij,n->tm',stress,d['test_gradient'],d['weights']);response=d['windows']@response
            keep=(d['window_begin']<10)|(d['window_begin']>=10.8);scale=np.linalg.norm(d['target'][keep])
            a.append(response[keep].ravel()/scale);b.append(d['target'][keep].ravel()/scale)
        A=np.concatenate(a);B=np.concatenate(b);E=float(A@B/(A@A))
        result[material]=dict(refined_conditional_E_at_fit_nu_Pa=E,relative_change=E/best['E_Pa']-1,
            relative_weak_residual=float(np.linalg.norm(E*A-B)/np.sqrt(3)))
    save(out/'quadrature_check.json',dict(base=[24,4],refined=[48,6],results=result,
        scope='Conditional E re-solve at frozen yield ratio and relaxation time. Not a full re-identification or continuum convergence claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--extended',action='store_true');a=p.parse_args()
    check(a.out)
    if a.extended:extended(a.out)
