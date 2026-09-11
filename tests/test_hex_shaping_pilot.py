"""Check the rotated contact surfaces used by the separate hexagonal pilot."""
import numpy as np
import pytest
from experiments.robotics.hex_shaping_pilot import BAND, box_sdf, jaw_pose
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation


@pytest.mark.parametrize("angle", [0., 60., 120.])
def test_requested_gap_is_the_rotated_sdf_contact_boundary(angle):
    sdf = box_sdf()
    gap, floor = .09, .01875
    centers, _, quat = jaw_pose(angle, gap, floor)
    rotation = Rotation.from_quat(quat)
    normal = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.])
    tangent = np.array([-normal[1], normal[0], 0.])
    middle = np.array([.15, .15, floor+.065])
    for sign, center in zip([-1, 1], centers, strict=True):
        for tangent_offset in [-.04, 0., .04]:
            face_point = middle + sign*gap/2*normal + tangent_offset*tangent
            distances = []
            for offset in [-.001, 0., .001]:
                body = rotation.inv().apply(face_point+sign*offset*normal-center)
                index = ((body-sdf.origin)/sdf.cell)[:, None]
                distances.append(float(map_coordinates(sdf.values, index, order=1)[0]))
            # The free region is outside the band; moving outward crosses into the jaw.
            assert distances[0] > BAND
            assert distances[1] == pytest.approx(BAND, abs=1e-8)
            assert distances[2] < BAND


def test_sdf_band_does_not_reach_the_stored_field_boundary():
    values = box_sdf().values
    faces = [values[0], values[-1], values[:, 0], values[:, -1], values[:, :, 0], values[:, :, -1]]
    assert min(face.min() for face in faces) > BAND + .01
