"""Generic linear-Gaussian Kalman filter and Rauch-Tung-Striebel (RTS) smoother.

State-space model:
    x_t = F x_{t-1} + w_t,      w_t ~ N(0, Q)
    z_t = H x_t + v_t,          v_t ~ N(0, R_t)

Measurements may be missing at any timestep (pass NaN); a missing timestep
falls back to a pure predict step with no update, which is the standard way
a Kalman filter handles dropout without any special-casing of the recursion.
"""
from __future__ import annotations

import numpy as np


class FilterResult:
    def __init__(self, x_filt, P_filt, x_pred, P_pred):
        self.x_filt = x_filt
        self.P_filt = P_filt
        self.x_pred = x_pred
        self.P_pred = P_pred


def kalman_filter(z, F, H, Q, R, x0, P0):
    """Forward Kalman filter pass over a 1D measurement sequence.

    z: (T,) array, np.nan marks a missing measurement.
    F, Q: (n, n) transition and process-noise covariance.
    H: (1, n) measurement matrix.
    R: (T,) or scalar measurement-noise variance.
    x0, P0: (n,), (n, n) initial state mean/covariance.
    """
    T = len(z)
    n = F.shape[0]
    R_seq = np.full(T, R, dtype=float) if np.isscalar(R) else np.asarray(R, dtype=float)

    x_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))
    x_pred = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))

    x_prev, P_prev = np.asarray(x0, dtype=float), np.asarray(P0, dtype=float)

    for t in range(T):
        xp = F @ x_prev
        Pp = F @ P_prev @ F.T + Q
        x_pred[t], P_pred[t] = xp, Pp

        if np.isnan(z[t]):
            x_filt[t], P_filt[t] = xp, Pp
        else:
            y = z[t] - (H @ xp)[0]
            S = (H @ Pp @ H.T)[0, 0] + R_seq[t]
            K = (Pp @ H.T) / S
            xf = xp + (K.flatten() * y)
            Pf = Pp - K @ H @ Pp
            x_filt[t], P_filt[t] = xf, Pf

        x_prev, P_prev = x_filt[t], P_filt[t]

    return FilterResult(x_filt, P_filt, x_pred, P_pred)


def rts_smooth(result: FilterResult, F, Q):
    """Backward RTS smoothing pass given a forward filter result."""
    T, n = result.x_filt.shape
    x_smooth = np.zeros((T, n))
    P_smooth = np.zeros((T, n, n))

    x_smooth[-1], P_smooth[-1] = result.x_filt[-1], result.P_filt[-1]

    for t in range(T - 2, -1, -1):
        P_pred_next = F @ result.P_filt[t] @ F.T + Q
        C = result.P_filt[t] @ F.T @ np.linalg.pinv(P_pred_next)
        x_smooth[t] = result.x_filt[t] + C @ (x_smooth[t + 1] - F @ result.x_filt[t])
        P_smooth[t] = result.P_filt[t] + C @ (P_smooth[t + 1] - P_pred_next) @ C.T

    return x_smooth, P_smooth


def smooth_scalar_signal(z, dt, q_accel, r_meas, x0=None, P0=None):
    """Convenience wrapper: constant-velocity RTS smoothing of one noisy scalar signal.

    State is [value, rate]. Returns the smoothed value sequence (T,).
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q_accel * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])

    finite = z[np.isfinite(z)]
    z0 = finite[0] if len(finite) else 0.0
    if x0 is None:
        x0 = np.array([z0, 0.0])
    if P0 is None:
        P0 = np.diag([10.0, 10.0])

    filt = kalman_filter(z, F, H, Q, r_meas, x0, P0)
    x_smooth, _ = rts_smooth(filt, F, Q)
    return x_smooth[:, 0], x_smooth[:, 1]
