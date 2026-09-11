"""Synthetic runner generator: the exact-ground-truth benchmark.

No motion-capture lab or force plate was available to validate Strideline
against, so validation instead uses a procedural 2D sagittal running-gait
model: a parametric periodic curve for hip and knee angle shaped to match
the *qualitative* signature of published sagittal running kinematics (a
moderate knee-flexion trough at ground contact, a much larger flexion peak
during swing recovery, a smooth hip flexion/extension envelope) rather than
a literal digitization of any one paper's numbers. Because the model is
analytic, the true joint angles and true ground-contact instants are known
to arbitrary precision (found by densely sampling the noiseless curve),
which is what makes it usable as exact ground truth: every number in the
benchmark tables is a comparison against this known truth, never a comparison
generated and checked by the same code path.

The generator then renders that motion through a fixed orthographic camera
(matching the treadmill / planted-tripod scenario Strideline is designed
for), adding independent per-keypoint Gaussian pixel noise and optional
per-frame detection dropout, to produce raw keypoint tracks in the same
shape a real pose model would return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import kinematics as kin


def _leg_angles(phase):
    """One running-gait cycle, phase in [0, 1). Phase 0 = ground contact.

    Both components are single-harmonic in phase by design, so the leg's
    vertical reach (and hence ankle height) has exactly one maximum
    (ground contact) and one minimum (mid-swing recovery) per cycle - a
    deliberately clean synthetic signal for validating the event detector,
    not a literal digitization of a published kinematic curve.

    Returns theta_thigh, theta_shank in radians (see kinematics.py for the
    sign convention: 0 = segment pointing straight down, from vertical).
    """
    p = phase * 2 * np.pi
    lag = 0.07 * 2 * np.pi  # small hip-leads-knee phase lag: breaks the degenerate
                             # flat extremum that results if both terms peak at
                             # exactly the same phase, and is qualitatively true
                             # to real gait (hip motion leads knee motion slightly)
    theta_thigh = np.deg2rad(20) * np.cos(p)
    knee_flexion = np.deg2rad(18) + np.deg2rad(78) * (1 - np.cos(p - lag)) / 2
    theta_shank = theta_thigh - knee_flexion
    return theta_thigh, theta_shank


def _ankle_height(phase, L_thigh, L_shank, shank_offset=0.0):
    theta_thigh, theta_shank = _leg_angles(phase)
    theta_shank = theta_shank + shank_offset
    return L_thigh * np.cos(theta_thigh) + L_shank * np.cos(theta_shank)


def true_contact_phase(L_thigh, L_shank, shank_offset=0.0, search_center=0.0, window=0.25, n=2000):
    """Find the exact phase (within one cycle) where ankle height (the
    downward reach of the leg) is locally maximal, i.e. true ground contact,
    by dense sampling plus a bracketed root find on the derivative."""
    phases = np.linspace(search_center - window, search_center + window, n)
    heights = _ankle_height(phases, L_thigh, L_shank, shank_offset)
    i = int(np.argmax(heights))
    lo, hi = phases[max(i - 2, 0)], phases[min(i + 2, n - 1)]

    def deriv(p, h=1e-6):
        return (_ankle_height(p + h, L_thigh, L_shank, shank_offset)
                 - _ankle_height(p - h, L_thigh, L_shank, shank_offset)) / (2 * h)

    try:
        return brentq(deriv, lo, hi)
    except ValueError:
        return phases[i]


@dataclass
class SyntheticSequence:
    dt: float
    hip_raw: np.ndarray
    knee_raw: dict
    ankle_raw: dict
    theta_thigh_true: dict
    theta_shank_true: dict
    contact_times_true: dict
    L_thigh_true: float
    L_shank_true: float


def generate_sequence(duration_s=6.0, fps=60.0, cadence_spm=170.0,
                       L_thigh=95.0, L_shank=100.0, hip_y0=300.0, hip_bounce_px=6.0,
                       pixel_noise_std=2.5, dropout_prob=0.0, asymmetry_deg=0.0, seed=0):
    """Generate a synthetic two-legged running sequence with exact ground truth.

    cadence_spm counts single-foot strikes per minute per the usual running
    sense (steps per minute across both feet), so the single-leg stride
    period is 2 * 60 / cadence_spm seconds.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    n = int(duration_s * fps)
    t = np.arange(n) * dt

    stride_period = 2 * 60.0 / cadence_spm
    hip_x0 = 160.0

    hip_true = np.stack([
        np.full(n, hip_x0),
        hip_y0 - hip_bounce_px * np.abs(np.cos(np.pi * t / stride_period)),
    ], axis=-1)

    sides = {"L": 0.0, "R": 0.5}
    knee_raw, ankle_raw = {}, {}
    theta_thigh_true, theta_shank_true, contact_times_true = {}, {}, {}

    for side, offset in sides.items():
        phase = (t / stride_period + offset) % 1.0
        extra = np.deg2rad(asymmetry_deg) if side == "R" else 0.0
        tth, tsh = _leg_angles(phase)
        tsh = tsh + extra

        knee_true = hip_true + L_thigh * kin.unit(tth)
        ankle_true = knee_true + L_shank * kin.unit(tsh)

        knee_noisy = knee_true + rng.normal(0, pixel_noise_std, knee_true.shape)
        ankle_noisy = ankle_true + rng.normal(0, pixel_noise_std, ankle_true.shape)

        if dropout_prob > 0:
            drop_k = rng.random(n) < dropout_prob
            drop_a = rng.random(n) < dropout_prob
            knee_noisy[drop_k] = np.nan
            ankle_noisy[drop_a] = np.nan

        theta_thigh_true[side] = tth
        theta_shank_true[side] = tsh
        knee_raw[side] = knee_noisy
        ankle_raw[side] = ankle_noisy

        true_phase = true_contact_phase(L_thigh, L_shank, shank_offset=extra)
        n_cycles = int(np.ceil(duration_s / stride_period)) + 2
        contacts = []
        for k in range(-1, n_cycles):
            tc = (true_phase + k - offset) * stride_period
            if 0 <= tc <= duration_s:
                contacts.append(tc)
        contact_times_true[side] = np.array(sorted(contacts))

    hip_raw = hip_true + rng.normal(0, pixel_noise_std * 0.6, hip_true.shape)

    return SyntheticSequence(dt=dt, hip_raw=hip_raw, knee_raw=knee_raw, ankle_raw=ankle_raw,
                              theta_thigh_true=theta_thigh_true, theta_shank_true=theta_shank_true,
                              contact_times_true=contact_times_true, L_thigh_true=L_thigh, L_shank_true=L_shank)
