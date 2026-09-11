"""Check physical units, force sign, both jaws, and frame invariance."""
import numpy as np
from experiments.robotics.hex_shaping_work import work_increment


def test_work_counts_both_opposing_jaws_with_reaction_sign():
    forces=np.array([[-100.,0,0],[100.,0,0]])
    velocities=np.array([[.1,0,0],[-.1,0,0]])
    np.testing.assert_allclose(work_increment(forces,velocities,.008),.16)
    np.testing.assert_allclose(work_increment(-forces,velocities,.008),0)


def test_work_is_invariant_under_rotating_force_and_motion_together():
    angle=np.deg2rad(60)
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    forces=np.array([[-100.,0,0],[100.,0,0]])
    velocities=np.array([[.1,0,0],[-.1,0,0]])
    np.testing.assert_allclose(work_increment(forces@rotation.T,velocities@rotation.T,.008),.16)
