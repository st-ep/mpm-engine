"""Build an isolated, unblended AFLIP fluid-transfer prototype.

Equations10-11 in Fei et al.(2021), alpha=1, beta=0: increment particle
velocity from the grid velocity change; advect and update C from grid velocity.
This is a discrete transfer change, not a viscosity or contact fit. The base
arithmetic corrections are retained. CPIC/rigid coupled scenes are out of scope.
"""
from pathlib import Path
import hashlib,json,shutil
from experiments.pour.pour_stable_volume_prototype import replace_once

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'out/pour_physics_audit/compensated_position_20260907'
OUT=ROOT/'out/pour_physics_audit/aflip_20260907'


def replace_count(path,old,new,count):
    content=path.read_text()
    if content.count(old)!=count:raise ValueError((str(path),old,content.count(old),count))
    path.write_text(content.replace(old,new))


def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    base=json.loads((BASE/'prototype_provenance.json').read_text())
    for name,expected in base['prototype_sha256'].items():assert sha(ROOT/name)==expected
    dest=OUT/'isolated_src/warpmpm'
    dest.parent.mkdir(parents=True,exist_ok=False)
    shutil.copytree(BASE/'isolated_src/warpmpm',dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    structs=dest/'kernels/warp_utils.py';solver=dest/'kernels/mpm_solver_warp.py';utils=dest/'kernels/mpm_utils.py'
    replace_once(structs,'    grid_v_in: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum/velocity\n',
        '    grid_v_in: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum/velocity\n'
        '    grid_q_transfer: wp.array(dtype=wp.vec3, ndim=3)  # momentum before forces\n'
        '    grid_v_transfer: wp.array(dtype=wp.vec3, ndim=3)  # old normalized velocity; survives fused gather\n')
    alloc='''        self.mpm_state.grid_q_transfer = wp.zeros(
            shape=(self.mpm_model.n_grid,self.mpm_model.n_grid,self.mpm_model.n_grid),
            dtype=wp.vec3,device=device)
        self.mpm_state.grid_v_transfer = wp.zeros(
            shape=(self.mpm_model.n_grid,self.mpm_model.n_grid,self.mpm_model.n_grid),
            dtype=wp.vec3,device=device)
'''
    replace_count(solver,'        self.mpm_state.grid_v_in = wp.zeros(\n',alloc+'        self.mpm_state.grid_v_in = wp.zeros(\n',2)
    for coord,count in [('grid_x, grid_y, grid_z',2),('n[0], n[1], n[2]',1)]:
        for field,newfield in [('grid_v_in','grid_q_transfer'),('grid_v_out','grid_v_transfer')]:
            old=f'state.{field}[{coord}] = wp.vec3(0.0, 0.0, 0.0)'
            indent='    ' if coord.startswith('grid') else '        '
            replace_count(utils,old,old+'\n'+indent+f'state.{newfield}[{coord}] = wp.vec3(0.0, 0.0, 0.0)',count)
    replace_once(utils,'                wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)\n',
        '                wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)\n'
        '                wp.atomic_add(state.grid_q_transfer,ix,iy,iz,\n'
        '                              weight*state.particle_mass[p]*(state.particle_v[p]+C*dpos))\n')
    replace_once(utils,'        state.grid_v_out[grid_x, grid_y, grid_z] = v_out\n',
        '        state.grid_v_out[grid_x, grid_y, grid_z] = v_out\n'
        '        state.grid_v_transfer[grid_x,grid_y,grid_z] = state.grid_q_transfer[grid_x,grid_y,grid_z]/state.grid_m[grid_x,grid_y,grid_z]\n')
    replace_once(utils,'            state.grid_v_out[n[0], n[1], n[2]] = v_out\n',
        '            state.grid_v_out[n[0], n[1], n[2]] = v_out\n'
        '            state.grid_v_transfer[n[0],n[1],n[2]] = state.grid_q_transfer[n[0],n[1],n[2]]/state.grid_m[n[0],n[1],n[2]]\n')
    replace_once(utils,'    if m > 0.0:\n        # eps-softened division:',
        '    if m > 0.0:\n'
        '        state.grid_v_transfer[grid_x,grid_y,grid_z] = state.grid_q_transfer[grid_x,grid_y,grid_z]/(m+model.grid_mass_eps)\n'
        '        # eps-softened division:')
    replace_once(utils,'        new_v = wp.vec3(0.0, 0.0, 0.0)\n',
        '        new_v = wp.vec3(0.0, 0.0, 0.0)\n'
        '        old_grid_interpolated = wp.vec3(0.0, 0.0, 0.0)\n')
    replace_once(utils,'                    new_v = new_v + grid_v * weight\n',
        '                    new_v = new_v + grid_v * weight\n'
        '                    old_grid_interpolated = old_grid_interpolated+weight*state.grid_v_transfer[ix,iy,iz]\n')
    replace_once(utils,'        state.particle_v[p] = new_v\n',
        '        transfer_mat = state.particle_material[p]\n'
        '        if transfer_mat == 6 or transfer_mat == 10 or transfer_mat == 12:\n'
        '            state.particle_v[p] = new_v+(state.particle_v[p]-old_grid_interpolated)\n'
        '        else:\n'
        '            state.particle_v[p] = new_v\n')
    result=dict(purpose=__doc__,reference='https://raymondyfei.github.io/asflip/main.pdf',
        alpha=1.,beta=0.,parameters_fitted=[],liquid_data_used=[],
        identification_changed=False,base_manifest_sha256=sha(BASE/'prototype_provenance.json'),
        base_source_sha256=base['prototype_sha256'],source_sha256=base['source_sha256'],
        prototype_sha256={str(f.relative_to(ROOT)):sha(f) for f in dest.rglob('*.py')},
        builder_sha256=sha(Path(__file__)),
        supported_scope='Fresh single-fluid MPM, no CPIC or rigid coupling; requires analytical and integration checks')
    (OUT/'prototype_provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(dest)


if __name__=='__main__':main()
