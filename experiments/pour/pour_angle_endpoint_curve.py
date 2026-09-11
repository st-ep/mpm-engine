"""Empirical angle control from the three measured endpoints.

This is the explicitly authorized three-pour fallback, not a one-pour viscosity
identification result. It replaces the grid-sensitive MPM lookup for operation.
The measured interval uses monotone PCHIP; a bounded tangent continuation reaches
160 mL. Accuracy between measurements remains to be tested on future robot pours.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "out/pour_angle_curve_empirical"
TARGETS = [60.0, 80.0, 100.0, 120.0, 140.0, 160.0]


@dataclass
class EndpointCurve:
    angles: np.ndarray
    volumes: np.ndarray
    maximum_volume_ml: float = 160.0

    def __post_init__(self):
        self.angles = np.asarray(self.angles, dtype=float)
        self.volumes = np.asarray(self.volumes, dtype=float)
        if (self.angles.ndim != 1 or self.volumes.shape != self.angles.shape
                or len(self.angles) < 3 or not np.isfinite(self.angles).all()
                or not np.isfinite(self.volumes).all()
                or np.any(np.diff(self.angles) <= 0) or np.any(np.diff(self.volumes) <= 0)):
            raise ValueError("Measured angles and volumes must both increase strictly")
        self.interpolator = PchipInterpolator(self.angles, self.volumes, extrapolate=False)
        self.end_slope = float(self.interpolator.derivative()(self.angles[-1]))
        if self.end_slope <= 0 or self.maximum_volume_ml < self.volumes[-1]:
            raise ValueError("A positive terminal slope and valid upper volume are required")
        self.maximum_angle_deg = (self.angles[-1]
                                  + (self.maximum_volume_ml - self.volumes[-1]) / self.end_slope)

    def volume(self, angle):
        a = np.asarray(angle, dtype=float)
        if (not np.isfinite(a).all() or np.any(a < self.angles[0])
                or np.any(a > self.maximum_angle_deg + 1e-10)):
            raise ValueError("Angle is outside the supported control range")
        interpolation = self.interpolator(np.minimum(a, self.angles[-1]))
        continuation = self.volumes[-1] + self.end_slope * (a - self.angles[-1])
        return np.where(a <= self.angles[-1], interpolation, continuation)

    def angle(self, target):
        if not np.isfinite(target) or not self.volumes[0] <= target <= self.maximum_volume_ml:
            raise ValueError("Requested volume is outside the supported control range")
        if target > self.volumes[-1]:
            return float(self.angles[-1] + (target - self.volumes[-1]) / self.end_slope)
        return float(brentq(lambda a: float(self.volume(a)) - target,
                            self.angles[0], self.angles[-1], xtol=1e-12))


def load_curve():
    measurements = []
    for angle in [45, 50, 60]:
        path = ROOT / f"out/pour_wf/09-04-{angle}-2s/recording_manifest.json"
        manifest = json.loads(path.read_text())
        if manifest["initial_volume_ml"] != 300:
            raise ValueError("All calibration pours must start at 300 mL")
        measurements.append(dict(angle_deg=angle, receiver_ml=manifest["final_receiver_volume_ml"],
                                 role_in_this_model="calibration",
                                 episode=f"09-04-{angle}-2s", archive_sha256=manifest["sha256"],
                                 manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    curve = EndpointCurve([m["angle_deg"] for m in measurements],
                          [m["receiver_ml"] for m in measurements])
    return curve, measurements


def plan_targets(curve, targets):
    rows = []
    for target in targets:
        exact = curve.angle(float(target))
        # Use the lower hundredth for the unmeasured continuation so existing
        # commands stay stable when its supported upper volume is extended.
        command = round(exact, 2)
        if target > curve.volumes[-1] or command > curve.maximum_angle_deg:
            command = float(np.floor(exact * 100) / 100)
        rows.append(dict(target_ml=float(target), command_angle_deg=command,
                         pour_command_duration_s=round(command / 10, 3),
                         dwell_s=2.0, return_command_duration_s=2.0,
                         expected_receiver_ml=float(curve.volume(command))))
    return rows


def export_model():
    curve, measurements = load_curve()
    table = plan_targets(curve, TARGETS)
    angles = np.linspace(curve.angles[0], curve.maximum_angle_deg, 1001)
    volumes = curve.volume(angles)
    if np.any(np.diff(volumes) <= 0):
        raise ValueError("Control curve is not strictly increasing")
    # Verify the inverse across the entire requested interval.
    inverse_error = max(abs(float(curve.volume(curve.angle(v))) - v)
                        for v in np.linspace(min(TARGETS), max(TARGETS), 301))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = dict(method="three-measurement monotone PCHIP control curve",
                  curve_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  calibration_measurements=measurements, calibration_pour_count=3,
                  one_pour_identification_claim=False, viscosity_estimated_by_this_model=False,
                  initial_volume_ml=300, dwell_s=2, tilt_duration_rule="angle_deg / 10 seconds",
                  return_command_duration_s=2, terminal_slope_ml_per_degree=curve.end_slope,
                  continuation=f"linear tangent after the last measurement, limited to "
                               f"{curve.maximum_volume_ml:g} mL",
                  supported_target_range_ml=[min(TARGETS), max(TARGETS)],
                  maximum_angle_deg=curve.maximum_angle_deg,
                  numerical_inverse_max_error_ml=inverse_error,
                  validation_status="future physical validation required; "
                                    "no accuracy bound established",
                  future_validation_measurements_used=[], table=table)
    (OUTPUT / "curve.json").write_text(json.dumps(result, indent=2))
    (OUTPUT / "table.json").write_text(json.dumps(table, indent=2))
    np.savetxt(OUTPUT / "curve_samples.csv", np.c_[angles, volumes], delimiter=",",
               header="command_angle_deg,expected_receiver_ml", comments="", fmt="%.8f")
    return curve, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, nargs="+")
    args = parser.parse_args()
    curve, table = export_model()
    if args.target:
        table = plan_targets(curve, args.target)
    print(json.dumps(table, indent=2))
