"""Gait metrics computed from a smoothed trajectory and its contact events.

Every metric here is either a standard, well-defined quantity (cadence,
stride time, stride-time variability) or is explicitly labelled as a
proposed engineering index rather than a validated clinical one (the Form
Consistency Index). None of this is a diagnostic tool.
"""
from __future__ import annotations

import numpy as np


def cadence_spm(contact_times_L, contact_times_R, duration_s):
    total = len(contact_times_L) + len(contact_times_R)
    return 60.0 * total / duration_s if duration_s > 0 else 0.0


def stride_intervals(contact_times):
    contact_times = np.asarray(contact_times, dtype=float)
    if len(contact_times) < 2:
        return np.array([])
    return np.diff(contact_times)


def stride_time_cv(contact_times):
    intervals = stride_intervals(contact_times)
    if len(intervals) < 2 or np.mean(intervals) == 0:
        return 0.0
    return float(100.0 * np.std(intervals) / np.mean(intervals))


def step_time_asymmetry_pct(contact_times_L, contact_times_R):
    """Mean absolute left/right stride-time difference, as a percent of the
    pooled mean stride time."""
    iL, iR = stride_intervals(contact_times_L), stride_intervals(contact_times_R)
    if len(iL) == 0 or len(iR) == 0:
        return 0.0
    n = min(len(iL), len(iR))
    diff = np.abs(iL[:n] - iR[:n])
    pooled_mean = np.mean(np.concatenate([iL, iR]))
    if pooled_mean == 0:
        return 0.0
    return float(100.0 * np.mean(diff) / pooled_mean)


def knee_angle_at_contacts(theta_thigh, theta_shank, dt, contact_events):
    """Sample the (corrected) knee flexion angle at each detected contact
    instant, interpolating between the two nearest frames."""
    from . import kinematics as kin

    knee_flex = kin.knee_flexion_angle(theta_thigh, theta_shank)
    t = np.arange(len(knee_flex)) * dt
    values = []
    for e in contact_events:
        values.append(float(np.interp(e.t_cross, t, knee_flex)))
    return np.array(values)


def hip_vertical_oscillation_ratio(hip_y, L_thigh, L_shank):
    """Peak-to-peak hip vertical excursion, normalized by leg length so it
    is comparable across cameras/subjects without a pixel-to-metric
    calibration."""
    leg_length = L_thigh + L_shank
    if leg_length == 0:
        return 0.0
    return float((np.max(hip_y) - np.min(hip_y)) / leg_length)


def form_consistency_index(stride_cv_pct, knee_angle_cv_pct, w_stride=1.0, w_knee=1.0):
    """A proposed (not clinically validated) single-number consistency
    score in (0, 100]: 100 means zero measured stride-to-stride variability
    in timing or knee angle at contact, decaying smoothly as either grows.
    """
    penalty = w_stride * stride_cv_pct + w_knee * knee_angle_cv_pct
    return float(100.0 / (1.0 + penalty / 100.0))


def summarize_leg(theta_thigh, theta_shank, hip_y, dt, contact_events, L_thigh, L_shank):
    knee_angles = knee_angle_at_contacts(theta_thigh, theta_shank, dt, contact_events)
    knee_cv = (100.0 * np.std(knee_angles) / np.mean(knee_angles)) if len(knee_angles) > 1 and np.mean(knee_angles) != 0 else 0.0
    contact_times = np.array([e.t_cross for e in contact_events])
    stride_cv = stride_time_cv(contact_times)
    return {
        "n_contacts": len(contact_events),
        "knee_angle_at_contact_deg_mean": float(np.degrees(np.mean(knee_angles))) if len(knee_angles) else float("nan"),
        "knee_angle_at_contact_deg_std": float(np.degrees(np.std(knee_angles))) if len(knee_angles) else float("nan"),
        "stride_time_cv_pct": stride_cv,
        "knee_angle_cv_pct": knee_cv,
        "form_consistency_index": form_consistency_index(stride_cv, knee_cv),
        "hip_vertical_oscillation_ratio": hip_vertical_oscillation_ratio(hip_y, L_thigh, L_shank),
    }
