"""Conservation cancels a shared gain, but cannot repair unequal optical biases."""

import numpy as np
from experiments.pour.pour_conservation_audit import (
    transferred_from_ratio,
    transferred_from_source,
)


def test_variable_common_gain_cancels_but_differential_gain_does_not():
    receiver = np.array([0.0, 20.0, 80.0, 159.0])
    source = 300.0 - receiver
    gain = np.array([0.7, 0.8, 1.2, 0.9])
    np.testing.assert_allclose(
        transferred_from_ratio(source * gain, receiver * gain, 300), receiver)
    biased = transferred_from_ratio(source, receiver * 0.8, 300)
    assert biased[-1] < receiver[-1] - 10.0


def test_source_normalization_preserves_evidence_of_pose_error():
    source_proxy = np.array([200.0, 180.0, 210.0])
    result = transferred_from_source(source_proxy, 200.0, 300.0)
    np.testing.assert_allclose(result, [0.0, 30.0, -15.0])
    # A negative apparent transfer must not be hidden by clipping or monotonicity.
    assert result[-1] < 0


def test_missing_or_invalid_proxy_is_not_filled_from_conservation():
    result = transferred_from_ratio([np.nan, 0.0, -1.0, 100.0],
                                    [10.0, 0.0, 10.0, np.nan], 300.0)
    assert np.isnan(result).all()
