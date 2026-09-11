"""Time-weak source-to-receiver transport, with explicit liquid-in-flight mass.

If eta*Q_source(s)=F(s), and material released at s arrives at a(s), then
eta*[V_receiver(t)-V_receiver(t0)] is the integral of F over releases whose
arrivals lie in (t0,t]. This is still linear in 1/eta for specified forcing and
transport; it takes no derivative of measured receiver volume.
"""
from __future__ import annotations

import numpy as np


def cumulative_piecewise_linear(times, values, query):
    """Integrate the piecewise-linear values exactly, including partial intervals."""
    times, values, query = map(lambda a: np.asarray(a, dtype=float), (times, values, query))
    if (times.ndim != 1 or len(times) < 2 or times.shape != values.shape
            or not np.isfinite(times).all() or not np.isfinite(values).all()
            or not np.isfinite(query).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Finite, aligned samples at increasing times are required')
    dt = np.diff(times)
    cumulative = np.r_[0., np.cumsum(.5*(values[1:]+values[:-1])*dt)]
    q = np.clip(query, times[0], times[-1])
    index = np.clip(np.searchsorted(times, q, side='right')-1, 0, len(times)-2)
    offset = q-times[index]
    slope = (values[index+1]-values[index])/dt[index]
    return cumulative[index]+values[index]*offset+.5*slope*offset**2


def transported_forcing(times, forcing, transit_time, query):
    """Return cumulative source and arrived forcing, before division by viscosity.

Forcing units are mL*Pa.s/s; outputs are mL*Pa.s. Transport delays must be
nonnegative and arrival order must be preserved. No measured endpoint is used.
Adequate prehistory before a fitting window is the caller's responsibility.
"""
    times, forcing, delay, query = map(lambda a: np.asarray(a, dtype=float),
                                       (times, forcing, transit_time, query))
    if (delay.shape != times.shape or not np.isfinite(delay).all()
            or np.any(delay < 0) or np.any(forcing < 0)):
        raise ValueError('Finite, nonnegative delay and forcing are required')
    arrival = times+delay
    if np.any(np.diff(arrival) <= 0):
        raise ValueError('Arrival times must increase; crossing parcels need another transport model')
    release_cutoff = np.interp(query, arrival, times, left=times[0], right=times[-1])
    source = cumulative_piecewise_linear(times, forcing, query)
    receiver = cumulative_piecewise_linear(times, forcing, release_cutoff)
    if np.any(receiver-source > 1e-8):
        raise ValueError('Arrived mass exceeds released mass')
    return source, receiver
