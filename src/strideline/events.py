"""Ground-contact event detection with a certified timing bound: pillar 2.

A foot is modelled as making ground contact at the instant the smoothed
ankle height reaches its local extremum in image coordinates (y increases
downward, so contact is where ankle-y is locally maximal): equivalently,
the zero-crossing of vertical ankle velocity v(t) = d(ankle_y)/dt from
positive (descending) to negative (rising).

The crossing time is estimated by linear interpolation between the two
samples that bracket the sign change. Two things then turn that estimate
into a certified window rather than a bare number:

1. A textbook interpolation-error argument. Let h = 1/f be the frame
   period and M2 an upper bound on |v''(t)| near the crossing. Linear
   interpolation of v between two samples spaced h apart has error bounded
   by (M2 h^2) / 8 (the Lagrange remainder for linear interpolation). Near
   a simple root t* of v with |v'(t*)| bounded away from 0, a vertical
   error of size eps in the interpolant maps to a horizontal (time) error
   of about eps / |v'(t*)| (first-order / implicit-function-theorem
   argument). Composing the two:

       |t_hat - t*|  <=  (M2 h^2) / (8 |v'(t*)|)                      (*)

   M2 and |v'(t*)| are estimated post hoc from the smoothed trajectory in
   a window around the crossing (see _local_curvature_bound), not asserted
   as fixed constants.

2. A calibrated safety margin. (*) assumes the finite-difference estimate
   of M2 does not itself underestimate the true local curvature, which is
   not guaranteed for a single narrow window on a noisy signal. Rather than
   assume it, CALIBRATED_SAFETY_FACTOR is measured directly: on a held-out
   synthetic calibration sweep (scripts/calibrate.py, disjoint seeds from
   every test in tests/test_events.py) spanning frame rates up to 120 fps
   and pixel noise up to 6 px, it is the 99th percentile of the ratio
   between true error and the raw (*) bound, over every correctly matched
   event. The reported certified bound is (*) multiplied by this factor.
   This is the same idea as a conformal calibration step: an analytic bound
   gives the *shape* of the guarantee, a held-out empirical pass gives the
   *constant*, and a second, disjoint sweep (tests/test_events.py) checks
   the calibrated bound actually holds out of sample. The validated
   envelope stops at 120 fps: scripts/calibrate.py also runs a 240 fps
   stress test, reported separately, where a handful of near-degenerate
   acceleration estimates would demand a two-digit-times-larger factor to
   fully cover - see the README limitations for why 240 fps is reported as
   a stress test rather than folded into the headline calibration.

A minimum-spacing (refractory) filter also removes spurious near-duplicate
crossings that noise can create close to a genuine one, keeping the
sharpest (largest |v'|) crossing in any cluster - standard practice for
any physiological event detector, not specific to this bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

CALIBRATED_SAFETY_FACTOR = 70.0


@dataclass
class ContactEvent:
    frame_index: int
    t_cross: float
    bound_seconds: float
    raw_bound_seconds: float
    v_slope: float
    curvature_bound: float


def _savgol_window(dt, window_seconds=0.05, min_len=9):
    n = round(window_seconds / dt)
    n = max(n, min_len)
    if n % 2 == 0:
        n += 1
    return n


def _derivatives(y, dt, window_seconds=0.05):
    """Velocity, acceleration and jerk of y via Savitzky-Golay differentiation.

    A naive two-point finite difference amplifies residual noise by 1/dt,
    which gets worse, not better, at higher frame rates. Savitzky-Golay
    fits a local polynomial over a window sized to a fixed *time* span
    (window_seconds), so the effective smoothing bandwidth - and hence the
    noise floor on the derivatives - stays comparable across frame rates.
    """
    n = len(y)
    window = min(_savgol_window(dt, window_seconds), n - (1 - n % 2))
    if window < 7:
        v = np.gradient(y, dt)
        a = np.gradient(v, dt)
        jerk = np.gradient(a, dt)
        return v, a, jerk
    v = savgol_filter(y, window, polyorder=4, deriv=1, delta=dt)
    a = savgol_filter(y, window, polyorder=4, deriv=2, delta=dt)
    jerk = savgol_filter(y, window, polyorder=4, deriv=3, delta=dt)
    return v, a, jerk


def _raw_crossings(v, a, dt):
    """Zero crossings of v, descending (+ to -). v_slope at the crossing
    is read directly from the acceleration signal a = v' rather than
    re-differenced from v, since a is already a Savitzky-Golay estimate."""
    crossings = []
    for i in range(1, len(v) - 2):
        if not (np.isfinite(v[i]) and np.isfinite(v[i + 1])):
            continue
        if v[i] > 0 and v[i + 1] <= 0:
            frac = v[i] / (v[i] - v[i + 1])
            t_cross = i * dt + frac * dt
            v_slope = float(np.interp(frac, [0, 1], [a[i], a[i + 1]]))
            crossings.append((i, t_cross, v_slope))
    return crossings


def _suppress_near_duplicates(crossings, min_spacing_s):
    if not crossings:
        return []
    crossings = sorted(crossings, key=lambda c: c[1])
    clusters = [[crossings[0]]]
    for c in crossings[1:]:
        if c[1] - clusters[-1][-1][1] < min_spacing_s:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    kept = [max(cluster, key=lambda c: abs(c[2])) for cluster in clusters]
    return kept


def _local_curvature_bound(jerk, i, window):
    lo = max(0, i - window)
    hi = min(len(jerk), i + window + 1)
    j_window = jerk[lo:hi]
    j_window = j_window[np.isfinite(j_window)]
    return float(np.max(np.abs(j_window))) if len(j_window) else 0.0


def detect_contacts(ankle_y, dt, curvature_window=8, refractory_ratio=0.4,
                     min_refractory_s=0.25, savgol_window_seconds=0.2,
                     safety_factor=CALIBRATED_SAFETY_FACTOR):
    """Detect ground-contact events in a smoothed ankle-height trace.

    ankle_y: (T,) smoothed vertical ankle position (pixels, y-down).
    dt: seconds per frame.
    refractory_ratio: the minimum spacing enforced between accepted events
        is this fraction of the *median* spacing between all raw zero
        crossings - a self-calibrating refractory period rather than a
        fixed constant, since a fixed constant does not scale with cadence
        or frame rate. This adaptive estimate is unreliable when there are
        only a handful of raw crossings to take a median over (short real
        clips), so it is floored at min_refractory_s: even elite sprinting
        does not cycle one single leg faster than about once every 250 ms
        (a 240 steps/min single-leg cadence, ~480 combined), so anything
        closer than that is treated as a noise-driven duplicate of the
        same event regardless of what the adaptive estimate alone would say.
    Returns a list of ContactEvent, one per detected foot strike.
    """
    ankle_y = np.asarray(ankle_y, dtype=float)
    v, a, jerk = _derivatives(ankle_y, dt, savgol_window_seconds)
    crossings = _raw_crossings(v, a, dt)

    if len(crossings) >= 2:
        gaps = np.diff(sorted(c[1] for c in crossings))
        min_spacing = max(min_refractory_s, refractory_ratio * float(np.median(gaps)))
    else:
        min_spacing = 0.0
    crossings = _suppress_near_duplicates(crossings, min_spacing)

    events = []
    for i, t_cross, v_slope in crossings:
        M2 = _local_curvature_bound(jerk, i, curvature_window)
        eps = 1e-6 * max(1.0, abs(v_slope))
        raw_bound = (M2 * dt ** 2) / (8 * max(abs(v_slope), eps))
        events.append(ContactEvent(frame_index=i, t_cross=t_cross,
                                    bound_seconds=safety_factor * raw_bound,
                                    raw_bound_seconds=raw_bound,
                                    v_slope=v_slope, curvature_bound=M2))
    return events


def contact_times(events):
    return np.array([e.t_cross for e in events])


def match_events(true_contact_times, detected_events, capture_fraction=0.5):
    """Separate the detection question (was this contact found at all) from
    the timing question (how precisely). A detected event only counts as a
    match to a true contact if it falls within capture_fraction of the
    median true inter-contact interval - otherwise it is a false detection,
    and any true contact with nothing that close is a missed detection.
    Reports detection recall/precision *and* the matched-pair rows the
    timing certificate is evaluated on, rather than letting rare detection
    failures silently inflate the timing bound's calibration.
    """
    true_times = np.asarray(true_contact_times, dtype=float)
    if len(true_times) < 2 or len(detected_events) == 0:
        return {"recall": 0.0, "precision": 0.0, "matched_rows": []}

    capture = capture_fraction * float(np.median(np.diff(np.sort(true_times))))
    used_true = set()
    rows = []
    for e in detected_events:
        j = int(np.argmin(np.abs(true_times - e.t_cross)))
        error = abs(true_times[j] - e.t_cross)
        if error <= capture:
            used_true.add(j)
            rows.append({"error": error, "bound": e.bound_seconds,
                         "raw_bound": e.raw_bound_seconds,
                         "held": error <= e.bound_seconds + 1e-9})

    recall = len(used_true) / len(true_times)
    precision = len(rows) / len(detected_events)
    return {"recall": recall, "precision": precision, "matched_rows": rows}


def validate_certificate(true_contact_times, detected_events, tolerance_factor=1.0, capture_fraction=0.5):
    """Convenience wrapper returning just the matched-pair timing rows."""
    return match_events(true_contact_times, detected_events, capture_fraction)["matched_rows"]


def calibrate_safety_factor(sequences, side="L", quantile=1.0):
    """Measure the smallest (or given-quantile) multiplier on the raw
    interpolation bound that makes every matched event's true error fall
    inside the scaled bound, across a held-out set of synthetic sequences.

    sequences: iterable of (seq, ankle_hat_y) pairs already produced by the
    smoother, kept separate from calibration so this can be re-run on fresh
    seeds without touching the smoother itself.
    """
    ratios = []
    for seq, ankle_y in sequences:
        events = detect_contacts(ankle_y, dt=seq.dt, safety_factor=1.0)
        rows = validate_certificate(seq.contact_times_true[side], events)
        for r in rows:
            if r["raw_bound"] > 0:
                ratios.append(r["error"] / r["raw_bound"])
    if not ratios:
        return 1.0
    return float(np.quantile(ratios, quantile))
