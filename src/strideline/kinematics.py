"""2D sagittal stick-figure kinematics: keypoints <-> joint angles.

Convention: image coordinates, y increasing downward. A segment's angle is
measured from straight down (the vertical), positive toward the direction of
travel. Each leg is described by two segment angles:

    theta_thigh: hip -> knee segment angle from vertical
    theta_shank: knee -> ankle segment angle from vertical

so that forward kinematics is

    knee  = hip  + L_thigh * u(theta_thigh)
    ankle = knee + L_shank * u(theta_shank)

with u(theta) = [sin(theta), cos(theta)], a unit vector, and u'(theta) =
[cos(theta), -sin(theta)] its derivative (also unit norm, orthogonal to u).
"""
from __future__ import annotations

import numpy as np


def unit(theta):
    return np.stack([np.sin(theta), np.cos(theta)], axis=-1)


def unit_deriv(theta):
    return np.stack([np.cos(theta), -np.sin(theta)], axis=-1)


def angle_from_vertical(vec):
    """Angle of a 2D vector (..., 2) from straight-down vertical."""
    return np.arctan2(vec[..., 0], vec[..., 1])


def segment_length(a, b):
    return np.linalg.norm(b - a, axis=-1)


def raw_angles(hip, knee, ankle):
    """Direct (noisy) per-frame angle measurements from raw 2D keypoints.

    hip, knee, ankle: (T, 2) arrays; NaN rows propagate as NaN angles.
    """
    theta_thigh = angle_from_vertical(knee - hip)
    theta_shank = angle_from_vertical(ankle - knee)
    return theta_thigh, theta_shank


def forward_kinematics(hip, theta_thigh, theta_shank, L_thigh, L_shank):
    """hip: (T, 2); theta_*: (T,); returns knee, ankle: (T, 2)."""
    knee = hip + L_thigh * unit(theta_thigh)
    ankle = knee + L_shank * unit(theta_shank)
    return knee, ankle


def knee_flexion_angle(theta_thigh, theta_shank):
    """Included angle at the knee: 0 = thigh and shank parallel (straight leg),
    increasing with flexion. This is a kinematic proxy, not a clinically
    zeroed goniometric angle (see README limitations)."""
    return np.abs(_wrap(theta_thigh - theta_shank))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
