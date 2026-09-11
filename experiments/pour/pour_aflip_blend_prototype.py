"""Build two predeclared stabilization checks, without experimental fitting.

Both use the0.99 velocity blend shown in Fei et al.(2021). The fixed variant
applies it each step. The time-scaled variant preserves its decay rate as dt
changes: alpha(dt)=exp(log(0.99)*dt/dt_ref), where dt_ref is the existing
acoustic/viscous stability limit. Neither changes the constitutive stress.
"""
from pathlib import Path
import hashlib,json,shutil
from experiments.pour.pour_stable_volume_prototype import replace_once

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'out/pour_physics_audit/aflip_20260907'

TIME_HELPER='''
@wp.func
def aflip_time_blend(state: MPMStateStruct, model: MPMModelStruct, p: int, dt: float):
    mat = state.particle_material[p]
    rho = state.particle_mass[p]/state.particle_vol[p]
    acoustic = 0.28*model.dx/wp.sqrt(1.1*model.bulk[mat]/rho)
    eta = wp.max(model.plastic_viscosity[mat],1.0e-12)
    viscous = rho*model.dx*model.dx/(6.0*eta)
    reference_dt = wp.min(acoustic,viscous)
    return wp.exp(wp.log(0.99)*dt/reference_dt)

'''


def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    base=json.loads((BASE/'prototype_provenance.json').read_text())
    for rel,expected in base['prototype_sha256'].items():assert sha(ROOT/rel)==expected
    for mode in ['fixed','time']:
        out=BASE.parent/f'aflip_blend_{mode}_20260907'
        dest=out/'isolated_src/warpmpm'
        dest.parent.mkdir(parents=True,exist_ok=False)
        shutil.copytree(BASE/'isolated_src/warpmpm',dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        utils=dest/'kernels/mpm_utils.py'
        alpha='0.99' if mode=='fixed' else 'aflip_time_blend(state,model,p,dt)'
        if mode=='time':replace_once(utils,'@wp.func\ndef g2p_particle(',TIME_HELPER+'@wp.func\ndef g2p_particle(')
        replace_once(utils,'            state.particle_v[p] = new_v+(state.particle_v[p]-old_grid_interpolated)\n',
            f'            state.particle_v[p] = new_v+{alpha}*(state.particle_v[p]-old_grid_interpolated)\n')
        result=dict(purpose=__doc__,mode=mode,reference_alpha=.99,beta=0.,
            selection_basis='Published example blend, followed by analytical/stability verification; no optimization against real volumes',
            reference='https://raymondyfei.github.io/asflip/main.pdf',
            time_scaling_derivation='Multiplicative residual damping alpha raised to dt/dt_ref preserves exponential decay rate under timestep subdivision',
            physical_parameters_refitted=[],liquid_data_used=[],identification_changed=False,
            base_manifest_sha256=sha(BASE/'prototype_provenance.json'),
            source_sha256=base['source_sha256'],base_source_sha256=base['prototype_sha256'],
            prototype_sha256={str(p.relative_to(ROOT)):sha(p) for p in dest.rglob('*.py')},
            builder_sha256=sha(Path(__file__)),
            supported_scope='Fresh Newtonian fluid, no CPIC or rigid coupling; analytical/stability tests required')
        (out/'prototype_provenance.json').write_text(json.dumps(result,indent=2)+'\n')
        print(dest)


if __name__=='__main__':main()
