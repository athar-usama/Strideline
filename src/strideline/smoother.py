"""The constrained kinematic smoother: pillar 1 of Strideline.

Raw 2D keypoints from a pose model imply, frame to frame, a thigh and shank
that stretch and shrink as the detector jitters. This module jointly recovers
a smooth joint-angle trajectory *and* the runner's two (unknown, but constant)
segment lengths by block-coordinate descent on one energy:

    E(theta_thigh, theta_shank, L_thigh, L_shank) =
        (1/sigma^2) * sum_t || knee_hat(t)  - raw_knee(t)  ||^2
      + (1/sigma^2) * sum_t || ankle_hat(t) - raw_ankle(t) ||^2
      + lambda * sum_t (theta_thigh(t) - theta_thigh(t-1))^2
      + lambda * sum_t (theta_shank(t) - theta_shank(t-1))^2

where knee_hat/ankle_hat are the forward-kinematics reprojection of the
current angle estimate at the current segment lengths (see kinematics.py).

Four blocks are cycled each outer iteration:
  - theta_thigh: one Iterated Extended Kalman Smoother (IEKS) step -
    linearize the reprojection residual around the current angle estimate,
    turn it into a per-frame pseudo-measurement of the angle with a
    noise variance derived from the linearization, then run the linear
    RTS smoother (rts.py) on that pseudo-measurement. A Levenberg-Marquardt
    style step-halving guards every such step so the *true* nonlinear energy
    is never allowed to increase.
  - theta_shank: identical construction against the ankle residual.
  - L_thigh, L_shank: each is the *exact* minimizer of its 1-D slice of E
    given the other blocks, in closed form (ordinary least squares).

Because every block update is either an exact minimizer or a guarded
descent step, E is monotonically non-increasing across outer iterations.
This is checked, not just claimed: test_smoother.py asserts the energy
trace is non-increasing on every synthetic sequence it runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kinematics as kin
from . import rts


def fill_and_unwrap(theta):
    """Linearly interpolate NaN gaps, then unwrap to a continuous angle."""
    theta = np.asarray(theta, dtype=float).copy()
    idx = np.arange(len(theta))
    valid = np.isfinite(theta)
    if valid.sum() < 2:
        theta[~valid] = 0.0
        return theta
    theta[~valid] = np.interp(idx[~valid], idx[valid], theta[valid])
    return np.unwrap(theta)


def _energy(hip, theta_thigh, theta_shank, L_thigh, L_shank, raw_knee, raw_ankle,
            sigma_pixel, lambda_smooth):
    knee_hat, ankle_hat = kin.forward_kinematics(hip, theta_thigh, theta_shank, L_thigh, L_shank)
    d_knee = np.nansum(np.sum((knee_hat - raw_knee) ** 2, axis=-1)) / sigma_pixel ** 2
    d_ankle = np.nansum(np.sum((ankle_hat - raw_ankle) ** 2, axis=-1)) / sigma_pixel ** 2
    s_thigh = np.sum(np.diff(theta_thigh) ** 2)
    s_shank = np.sum(np.diff(theta_shank) ** 2)
    return d_knee + d_ankle + lambda_smooth * (s_thigh + s_shank)


def _length_update(base, target, theta):
    """Closed-form scalar length minimizing sum ||base + L*u(theta) - target||^2."""
    valid = np.all(np.isfinite(target), axis=-1) & np.all(np.isfinite(base), axis=-1)
    if valid.sum() == 0:
        return 1.0
    d = (target - base)[valid]
    a = kin.unit(theta)[valid]
    return float(np.sum(a * d) / valid.sum())


def _ieks_angle_step(theta_bar, dt, sigma_pixel, q_accel, R_theta, pseudo_z):
    """One linear RTS-smoother pass on a pseudo-measurement of an angle."""
    pseudo_z = np.where(np.isfinite(pseudo_z), pseudo_z, np.nan)
    smoothed, _ = rts.smooth_scalar_signal(pseudo_z, dt=dt, q_accel=q_accel, r_meas=R_theta,
                                            x0=np.array([theta_bar[0], 0.0]))
    return smoothed


@dataclass
class SmoothResult:
    theta_thigh: np.ndarray
    theta_shank: np.ndarray
    L_thigh: float
    L_shank: float
    hip: np.ndarray
    energy_trace: list = field(default_factory=list)
    raw_theta_thigh: np.ndarray = None
    raw_theta_shank: np.ndarray = None
    raw_L_thigh_series: np.ndarray = None
    raw_L_shank_series: np.ndarray = None


def constrained_smooth(hip_raw, knee_raw, ankle_raw, dt, sigma_pixel=4.0,
                        q_accel=40.0, lambda_smooth=2.0, max_iters=20, tol=1e-5):
    """Run the constrained kinematic smoother on one leg's raw keypoint track.

    hip_raw, knee_raw, ankle_raw: (T, 2) arrays of raw pixel keypoints, NaN
    where the pose model dropped a detection.
    """
    hip_raw = np.asarray(hip_raw, dtype=float)
    knee_raw = np.asarray(knee_raw, dtype=float)
    ankle_raw = np.asarray(ankle_raw, dtype=float)

    hip_x, _ = rts.smooth_scalar_signal(hip_raw[:, 0], dt, q_accel, sigma_pixel ** 2)
    hip_y, _ = rts.smooth_scalar_signal(hip_raw[:, 1], dt, q_accel, sigma_pixel ** 2)
    hip_smooth = np.stack([hip_x, hip_y], axis=-1)

    theta_thigh_raw, theta_shank_raw = kin.raw_angles(hip_smooth, knee_raw, ankle_raw)
    theta_thigh = fill_and_unwrap(theta_thigh_raw)
    theta_shank = fill_and_unwrap(theta_shank_raw)

    raw_L_thigh_series = kin.segment_length(hip_smooth, knee_raw)
    raw_L_shank_series = kin.segment_length(knee_raw, ankle_raw)
    L_thigh = float(np.nanmedian(raw_L_thigh_series))
    L_shank = float(np.nanmedian(raw_L_shank_series))

    energy_trace = [_energy(hip_smooth, theta_thigh, theta_shank, L_thigh, L_shank,
                             knee_raw, ankle_raw, sigma_pixel, lambda_smooth)]

    for _ in range(max_iters):
        knee_hat, ankle_hat = kin.forward_kinematics(hip_smooth, theta_thigh, theta_shank, L_thigh, L_shank)
        c_knee = knee_hat - knee_raw
        c_ankle = ankle_hat - ankle_raw

        g_thigh = kin.unit_deriv(theta_thigh) * L_thigh
        delta_thigh = -np.sum(g_thigh * (np.nan_to_num(c_knee) + np.nan_to_num(c_ankle)), axis=-1) / (2 * L_thigh ** 2)
        pseudo_thigh = theta_thigh + delta_thigh
        R_thigh = sigma_pixel ** 2 / (2 * L_thigh ** 2)
        cand_thigh = _ieks_angle_step(theta_thigh, dt, sigma_pixel, q_accel, R_thigh, pseudo_thigh)

        def energy_with_thigh(t_thigh, theta_shank=theta_shank, L_thigh=L_thigh, L_shank=L_shank):
            return _energy(hip_smooth, t_thigh, theta_shank, L_thigh, L_shank,
                            knee_raw, ankle_raw, sigma_pixel, lambda_smooth)

        theta_thigh = _damped_accept(theta_thigh, cand_thigh, energy_with_thigh)

        knee_hat, ankle_hat = kin.forward_kinematics(hip_smooth, theta_thigh, theta_shank, L_thigh, L_shank)
        c_ankle = ankle_hat - ankle_raw
        g_shank = kin.unit_deriv(theta_shank) * L_shank
        delta_shank = -np.sum(g_shank * np.nan_to_num(c_ankle), axis=-1) / (L_shank ** 2)
        pseudo_shank = theta_shank + delta_shank
        R_shank = sigma_pixel ** 2 / (L_shank ** 2)
        cand_shank = _ieks_angle_step(theta_shank, dt, sigma_pixel, q_accel, R_shank, pseudo_shank)

        def energy_with_shank(t_shank, theta_thigh=theta_thigh, L_thigh=L_thigh, L_shank=L_shank):
            return _energy(hip_smooth, theta_thigh, t_shank, L_thigh, L_shank,
                            knee_raw, ankle_raw, sigma_pixel, lambda_smooth)

        theta_shank = _damped_accept(theta_shank, cand_shank, energy_with_shank)

        L_thigh = _length_update(hip_smooth, knee_raw, theta_thigh)
        knee_hat, _ = kin.forward_kinematics(hip_smooth, theta_thigh, theta_shank, L_thigh, L_shank)
        L_shank = _length_update(knee_hat, ankle_raw, theta_shank)

        e = _energy(hip_smooth, theta_thigh, theta_shank, L_thigh, L_shank,
                    knee_raw, ankle_raw, sigma_pixel, lambda_smooth)
        energy_trace.append(e)
        if abs(energy_trace[-2] - energy_trace[-1]) < tol:
            break

    return SmoothResult(theta_thigh=theta_thigh, theta_shank=theta_shank, L_thigh=L_thigh,
                         L_shank=L_shank, hip=hip_smooth, energy_trace=energy_trace,
                         raw_theta_thigh=theta_thigh_raw, raw_theta_shank=theta_shank_raw,
                         raw_L_thigh_series=raw_L_thigh_series, raw_L_shank_series=raw_L_shank_series)


def _damped_accept(theta_old, theta_cand, energy_fn, max_halvings=8, slack=1e-9):
    """Levenberg-Marquardt style guard for one IEKS block update.

    Tries the full candidate step, then half, quarter, ... of the way from
    theta_old to theta_cand, accepting the first that does not increase the
    true nonlinear energy. Step size 0 (theta_old unchanged) always
    qualifies, so this can never make the energy worse.
    """
    e_old = energy_fn(theta_old)
    step = 1.0
    for _ in range(max_halvings):
        theta_try = theta_old + step * (theta_cand - theta_old)
        if energy_fn(theta_try) <= e_old + slack:
            return theta_try
        step *= 0.5
    return theta_old


def kinematic_consistency_violation(base, target, true_length_estimate):
    """Frame-to-frame implied-length variance relative to its own median -
    the self-supervised consistency score (pillar 3), needs no ground truth.

    Returns the coefficient of variation (%) of the implied segment length.
    """
    lengths = kin.segment_length(base, target)
    lengths = lengths[np.isfinite(lengths)]
    if len(lengths) < 2 or np.mean(lengths) == 0:
        return 0.0
    return float(100.0 * np.std(lengths) / np.mean(lengths))
