"""Isolated stable-volume plus compensated-position numerical prototype.

No constitutive changes or outcome data. Small position increments use Kahan
summation. External position edits reset the compensation on changed coordinates
only; unchanged coordinates retain their sub-ULP motion through cup projections.
"""
from pathlib import Path
import hashlib,json,shutil

from experiments.pour.pour_stable_volume_prototype import replace_once

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'out/pour_physics_audit/stable_volume_20260907'
OUT=ROOT/'out/pour_physics_audit/compensated_position_20260907'

HELPERS='''
@wp.func
def compensated_position_step(state: MPMStateStruct, p: int, dt: float, velocity: wp.vec3):
    old = state.particle_x[p]
    increment = dt*velocity-state.particle_x_comp[p]
    updated = old+increment
    state.particle_x_comp[p] = (updated-old)-increment
    return updated


@wp.kernel
def reset_position_compensation_for_edits(state: MPMStateStruct, edited: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    old = state.particle_x[p]
    new = edited[p]
    c = state.particle_x_comp[p]
    for axis in range(3):
        if new[axis] != old[axis]:
            c[axis] = 0.0
    state.particle_x_comp[p] = c

'''


def main():
    old=json.loads((BASE/'prototype_provenance.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for relative,expected in old['prototype_sha256'].items():
        if sha(ROOT/relative)!=expected:raise RuntimeError('Base prototype changed')
    dest=OUT/'isolated_src/warpmpm'
    dest.parent.mkdir(parents=True,exist_ok=False)
    shutil.copytree(BASE/'isolated_src/warpmpm',dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    structs=dest/'kernels/warp_utils.py';solver=dest/'kernels/mpm_solver_warp.py';utils=dest/'kernels/mpm_utils.py'
    replace_once(structs,'    particle_x: wp.array(dtype=wp.vec3)  # current position\n',
        '    particle_x: wp.array(dtype=wp.vec3)  # current position\n'
        '    particle_x_comp: wp.array(dtype=wp.vec3)  # Kahan summation roundoff\n')
    replace_once(solver,'        self.mpm_state.particle_x = wp.empty(\n',
        '        self.mpm_state.particle_x_comp = wp.zeros(shape=n_particles,dtype=wp.vec3,device=device)\n'
        '        self.mpm_state.particle_x = wp.empty(\n')
    replace_once(solver,'        vec3_arrays = [st.particle_x, st.particle_v, st.particle_x_ref]\n',
        '        vec3_arrays = [st.particle_x, st.particle_v, st.particle_x_ref, st.particle_x_comp]\n')
    replace_once(solver,'            wp.copy(self.mpm_state.particle_x, src)\n',
        '            wp.launch(reset_position_compensation_for_edits,dim=self.n_particles,\n'
        '                      inputs=[self.mpm_state,src],device=device)\n'
        '            wp.copy(self.mpm_state.particle_x, src)\n')
    replace_once(utils,'@wp.func\ndef g2p_particle(',HELPERS+'@wp.func\ndef g2p_particle(')
    replace_once(utils,'        x_new = state.particle_x[p] + dt * new_v\n',
        '        x_new = compensated_position_step(state,p,dt,new_v)\n'
        '        x_unclipped = x_new\n')
    replace_once(utils,'        state.particle_x[p] = x_new\n',
        '        comp = state.particle_x_comp[p]\n'
        '        for axis in range(3):\n'
        '            if x_new[axis] != x_unclipped[axis]:\n'
        '                comp[axis] = 0.0\n'
        '        state.particle_x_comp[p] = comp\n'
        '        state.particle_x[p] = x_new\n')
    provenance=dict(purpose=__doc__,base_prototype_sha256=sha(BASE/'prototype_provenance.json'),
        source_sha256=old['source_sha256'],base_source_sha256=old['prototype_sha256'],
        prototype_sha256={str(p.relative_to(ROOT)):sha(p) for p in dest.rglob('*.py')},
        builder_sha256=sha(Path(__file__)),physical_parameters_changed=False,
        identification_changed=False,liquid_outcomes_used=[],
        changed_from_base=['kernels/warp_utils.py','kernels/mpm_solver_warp.py','kernels/mpm_utils.py'])
    (OUT/'prototype_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(dest)


if __name__=='__main__':main()
