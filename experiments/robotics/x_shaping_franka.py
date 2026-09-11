"""Panda kinematic feasibility and meshes for the simulated cylindrical fingers.

This does not command hardware. The mount is an explicit design assumption:
cylinder axes are 10 mm outward from each finger carriage and 85 mm along the
hand's local z axis. The 28 mm fingers therefore map physical gap g to stock
gripper travel g+8 mm. A measured mount calibration is required on hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import pyvista as pv
from robot_descriptions import panda_mj_description
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from experiments.robotics.x_shaping import CONFIG
from experiments.robotics.plastic_shaping_study import save_json, save_npz

OFFSET = np.array([.43-CONFIG["domain"]/2, -CONFIG["domain"]/2, .05-CONFIG["floor"]])
MOUNT_OUTWARD = .010
MOUNT_Z = .085


class Panda:
    def __init__(self, xml=None):
        self.model = mujoco.MjModel.from_xml_path(str(xml or panda_mj_description.MJCF_PATH))
        self.data = mujoco.MjData(self.model)
        self.hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.q = np.array([0., -.5, 0., -2.2, 0., 1.7, .8])
        self.initialized = False

    def solve(self, centers, angle):
        if not self.initialized and abs(angle) > 1e-6:
            self.solve(centers, 0.)
        target_position = centers.mean(axis=0)+OFFSET+[0, 0, MOUNT_Z]
        target_rotation = np.array([[-np.sin(angle), np.cos(angle), 0.],
                                    [np.cos(angle), np.sin(angle), 0.], [0., 0., -1.]])
        reference = self.q.copy()

        def residual(q):
            self.data.qpos[:7] = q; mujoco.mj_forward(self.model, self.data)
            position = self.data.xpos[self.hand]-target_position
            rotation = Rotation.from_matrix(target_rotation@self.data.xmat[self.hand].reshape(3, 3).T).as_rotvec()
            return np.r_[position, .1*rotation, .00001*(q-reference)]

        result = least_squares(residual, reference,
                   bounds=(self.model.jnt_range[:7, 0]+1e-5, self.model.jnt_range[:7, 1]-1e-5),
                   xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=100)
        errors = residual(result.x); self.q = result.x
        if np.linalg.norm(errors[:3]) > .0001 or np.linalg.norm(errors[3:6])/.1 > .001:
            raise RuntimeError(f"Panda IK failed: {errors[:6]}")
        self.initialized = True
        width = np.linalg.norm(centers[1]-centers[0])-2*MOUNT_OUTWARD
        if not 0 <= width <= .08000001:
            raise RuntimeError(f"Gripper travel exceeded: {width}")
        self.data.qpos[7:9] = width/2; mujoco.mj_forward(self.model, self.data)
        return dict(q=self.q.copy(), gripper_width=width, position_error_m=float(np.linalg.norm(errors[:3])),
                    rotation_error_rad=float(np.linalg.norm(errors[3:6])/.1))

    def meshes(self, centers, angle, include_arm=False):
        self.solve(centers, angle); meshes = []
        for g in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[g]); name = self.model.body(body).name
            if self.model.geom_group[g] != 2 or "finger" in name:
                continue
            if not include_arm and name != "hand":
                continue
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = int(self.model.geom_dataid[g]); v0 = self.model.mesh_vertadr[mesh_id]
            f0 = self.model.mesh_faceadr[mesh_id]
            vertices = self.model.mesh_vert[v0:v0+self.model.mesh_vertnum[mesh_id]]
            faces = self.model.mesh_face[f0:f0+self.model.mesh_facenum[mesh_id]]
            rotation = self.data.geom_xmat[g].reshape(3, 3)
            vertices = vertices@rotation.T+self.data.geom_xpos[g]-OFFSET
            mesh = pv.PolyData(vertices, np.column_stack([np.full(len(faces), 3), faces]).ravel())
            material = int(self.model.geom_matid[g])
            color = self.model.geom_rgba[g, :3] if material < 0 else self.model.mat_rgba[material, :3]
            meshes.append((mesh, color))
        # Explicit outward adapter brackets, schematic pending manufactured geometry.
        for c in centers:
            upper = c.copy(); upper[2] += CONFIG["finger_height"]/2+.004
            meshes.append((pv.Cylinder(center=upper, direction=(0, 0, 1), radius=.007, height=.008), "#c7cdd0"))
        return meshes


def audit(folder, check_speeds=False):
    results = []
    dest = folder/"franka"; dest.mkdir(exist_ok=True)
    installed = Path(panda_mj_description.MJCF_PATH)
    snapshot = dest/"panda_model_snapshot"
    if not snapshot.exists():
        shutil.copytree(installed.parent, snapshot)
    xml = snapshot/installed.name
    save_json(dest/"robot_asset_sha256.json", {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(snapshot.rglob("*")) if p.is_file()})
    for path in sorted(folder.glob("baseline_*.npz")):
        robot = Panda(xml)
        data = np.load(path); poses = data["tool_centers"]
        # Include all phase endpoints, plus samples every 20 ms.
        ends = np.r_[np.flatnonzero(np.diff(data["phase_id"])), len(poses)-1]
        indices = np.unique(np.r_[np.arange(0, len(poses), 5), ends])
        rows = []
        for i in indices:
            delta = poses[i, 1]-poses[i, 0]; angle = np.arctan2(delta[1], delta[0])
            rows.append(robot.solve(poses[i], angle))
        q = np.array([r["q"] for r in rows]); t = data["time"][indices]
        hand_positions = poses[indices].mean(axis=1)
        directions = poses[indices, 1]-poses[indices, 0]
        angles = np.unwrap(np.arctan2(directions[:, 1], directions[:, 0]))
        joint_speeds = np.max(np.abs(np.diff(q, axis=0)/np.diff(t)[:, None]), axis=0)
        hand_speed = float(np.max(np.linalg.norm(np.diff(hand_positions, axis=0), axis=1)/np.diff(t)))
        angular_speed = float(np.max(np.abs(np.diff(angles)/np.diff(t))))
        # Panda/FER constants from the manufacturer's FCI documentation and
        # franka_ros/franka_description/robots/panda/joint_limits.yaml.
        joint_limits = np.array([2.175]*4+[2.61]*3)
        if check_speeds:
            assert np.all(joint_speeds < joint_limits), (path, joint_speeds)
            assert hand_speed < 1.7 and angular_speed < 2.5, path
        save_npz(dest/f"{path.stem}.npz", dict(time=t, q=q, tool_centers=poses[indices],
                  gripper_width=np.array([r["gripper_width"] for r in rows])))
        result = dict(file=path.name, samples=len(rows), max_position_error_m=max(r["position_error_m"] for r in rows),
                  max_rotation_error_rad=max(r["rotation_error_rad"] for r in rows),
                  peak_joint_speed_rad_s=joint_speeds.tolist(),
                  peak_hand_translation_m_s=hand_speed, peak_hand_rotation_rad_s=angular_speed,
                  gripper_width_range_m=[min(r["gripper_width"] for r in rows), max(r["gripper_width"] for r in rows)])
        results.append(result)
    save_json(dest/"audit.json", dict(results=results, robot_xml=str(xml), robot_xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
              mount=dict(outward_m=MOUNT_OUTWARD, cylinder_center_hand_z_m=MOUNT_Z),
              workpiece_center_robot_m=[.43, 0., .05],
              speed_limits_checked=check_speeds,
              limits_sources=["https://frankarobotics.github.io/docs/robot_specifications.html", "https://github.com/frankarobotics/franka_ros/blob/develop/franka_description/robots/panda/joint_limits.yaml"],
              scope="Position/joint-range and optionally velocity feasibility at sampled poses. Kinematic illustration, not acceleration, jerk, dynamics, collision, torque, or hardware execution validation."))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--check-speeds", action="store_true")
    args = p.parse_args(); audit(args.out, args.check_speeds)
