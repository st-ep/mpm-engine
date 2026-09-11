"""Manufactured time-varying flow verifies weak identification through transit."""
import numpy as np
import pytest

from experiments.pour.pour_weakform_identify import fit_eta_integrated
from experiments.pour.pour_weakform_transport import transported_forcing


@pytest.mark.parametrize('delay0,delay_slope', [(0.,0.),(.2,0.),(.1,.05)])
def test_known_viscosity_recovered_with_constant_and_changing_flight(delay0, delay_slope):
    eta, scale, decay = 3.4, 120., 1.2
    release_t = np.linspace(0.,4.,4001)
    forcing = scale*np.exp(-release_t/decay)
    delay = delay0+delay_slope*release_t
    observed_t = np.linspace(1.,3.,101)
    # Closed-form manufactured receiver trace: a(s)=(1+delay_slope)*s+delay0.
    cutoff = (observed_t-delay0)/(1+delay_slope)
    receiver_true = scale*decay*(1-np.exp(-cutoff/decay))/eta
    source, arrived = transported_forcing(release_t, forcing, delay, observed_t)
    fitted, *_ = fit_eta_integrated(dict(t=observed_t, V=receiver_true, cumF=arrived))
    assert fitted == pytest.approx(eta, rel=1e-6)
    true_source = scale*decay*(1-np.exp(-observed_t/decay))
    assert np.max(abs(source-true_source)) < 2e-5
    assert np.all(source >= arrived-1e-10)
    if delay0 == .2:
        # Omitting transit for a decreasing source flux biases the OLD estimator.
        old, *_ = fit_eta_integrated(dict(t=observed_t, V=receiver_true, cumF=source))
        assert old == pytest.approx(eta*np.exp(-delay0/decay), rel=1e-6)
        assert old < .9*eta


def test_no_receiver_mass_before_first_arrival_and_final_conservation():
    t=np.array([0.,1.,2.])
    source, arrived=transported_forcing(t,np.array([2.,4.,2.]),np.full(3,.5),
                                        np.array([0.,.25,.5,2.,2.5,3.]))
    assert np.all(arrived[:3] == 0)
    assert source[-1] == pytest.approx(6.)
    assert arrived[-1] == pytest.approx(6.)


def test_noncausal_transport_rejected():
    with pytest.raises(ValueError,match='Arrival times'):
        transported_forcing([0,1,2],[1,1,1],[2,0,0],[1,2])
    with pytest.raises(ValueError,match='nonnegative'):
        transported_forcing([0,1,2],[1,1,1],[0,-1,0],[1,2])
