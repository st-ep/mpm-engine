"""Build an isolated forward-only numerical prototype; preserve production files.

The fluid stores log(det F) separately so small kinematic volume increments are
not rounded out of identity-valued float32 matrices. The same Euler determinant
and barotropic EOS are evaluated with cancellation-resistant formulas. Viscosity
and its identification code are untouched. No measured outcomes are inputs.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'out/pour_physics_audit/stable_volume_20260907'

HELPERS='''
@wp.func
def fluid_logdet_increment(L: wp.mat33, dt: float):
    # det(I+A)-1 = tr(A) + principal minors(A) + det(A).
    # Form the small increment directly, avoiding addition to 1 before rounding.
    A = dt*L
    tr = A[0,0]+A[1,1]+A[2,2]
    minors = (A[0,0]*A[1,1]+A[0,0]*A[2,2]+A[1,1]*A[2,2]
              -A[0,1]*A[1,0]-A[0,2]*A[2,0]-A[1,2]*A[2,1])
    d = tr+minors+wp.determinant(A)
    if wp.abs(d)<0.01:
        return d*(1.0+d*(-0.5+d*(1.0/3.0+d*(-0.25+d*0.2))))
    return wp.log(1.0+d)


@wp.func
def fluid_pressure_from_log_volume(logJ: float, bulk: float):
    u = -1.1*logJ
    em1 = wp.exp(u)-1.0
    if wp.abs(u)<0.01:
        em1 = u*(1.0+u*(0.5+u*(1.0/6.0+u*(1.0/24.0+u/120.0))))
    return bulk*em1

'''


def replace_once(path, old, new):
    text=path.read_text()
    if text.count(old)!=1:
        raise RuntimeError(f'Expected one exact source match in {path}: {text.count(old)}')
    path.write_text(text.replace(old,new))


def main():
    dest=OUT/'isolated_src/warpmpm'
    dest.parent.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ROOT/'src/warpmpm',dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    utils=dest/'kernels/mpm_utils.py'
    structs=dest/'kernels/warp_utils.py'
    solver=dest/'kernels/mpm_solver_warp.py'
    replace_once(structs,'    particle_Jp: wp.array(dtype=float)\n',
        '    particle_Jp: wp.array(dtype=float)\n'
        '    particle_logJ: wp.array(dtype=float)  # fluid volumetric kinematics, initialized at rest\n')
    replace_once(solver,'        self.mpm_state.particle_Jp = wp.zeros(\n',
        '        self.mpm_state.particle_logJ = wp.zeros(shape=n_particles, dtype=float, device=device)\n'
        '        self.mpm_state.particle_Jp = wp.zeros(\n')
    replace_once(solver,'                        st.particle_Jp]\n',
        '                        st.particle_Jp, st.particle_logJ]\n')
    replace_once(utils,'@wp.func\ndef g2p_particle(',HELPERS+'@wp.func\ndef g2p_particle(')
    replace_once(utils,'        F_tmp = (I33 + F_rate * dt) * state.particle_F[p]\n',
        '        mat = state.particle_material[p]\n'
        '        if mat == 6 or mat == 10 or mat == 12:\n'
        '            state.particle_logJ[p] = state.particle_logJ[p]+fluid_logdet_increment(F_rate,dt)\n'
        '        F_tmp = (I33 + F_rate * dt) * state.particle_F[p]\n')
    replace_once(utils,'            J = wp.determinant(state.particle_F_trial[p])\n            Jcbr = J**(1.0 / 3.0)\n',
        '            Jcbr = wp.exp(state.particle_logJ[p]/3.0)\n')
    replace_once(utils,'        J = wp.determinant(state.particle_F[p])\n',
        '        J = wp.determinant(state.particle_F[p])\n'
        '        fluid_pressure = float(0.0)\n'
        '        if mat == 6 or mat == 10 or mat == 12:\n'
        '            J = wp.exp(state.particle_logJ[p])\n'
        '            fluid_pressure = fluid_pressure_from_log_volume(state.particle_logJ[p],model.bulk[mat])\n')
    replace_once(utils,'                J, model.bulk[mat]\n', '                J, 0.0\n')
    replace_once(utils,'                J, model.bulk[mat], state.particle_L[p], model.plastic_viscosity[mat],\n',
        '                J, 0.0, state.particle_L[p], model.plastic_viscosity[mat],\n')
    replace_once(utils,'                J, model.bulk[mat], state.particle_L[p], model.eta_table,\n',
        '                J, 0.0, state.particle_L[p], model.eta_table,\n')
    replace_once(utils,'        state.particle_stress[p] = stress\n',
        '        if mat == 6 or mat == 10 or mat == 12:\n'
        '            stress = stress-J*fluid_pressure*wp.identity(n=3,dtype=float)\n'
        '        state.particle_stress[p] = stress\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    paths=sorted(dest.rglob('*.py'))
    provenance=dict(purpose=__doc__,changed_files=[str(p.relative_to(dest)) for p in [utils,structs,solver]],
        liquid_outcomes_used=[],identification_changed=False,physical_parameters_changed=False,
        prototype_limit='New simulations initialized at F=I only; external arbitrary-F imports not implemented',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/warpmpm').rglob('*.py')},
        prototype_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    (OUT/'prototype_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(dest)


if __name__=='__main__':main()
