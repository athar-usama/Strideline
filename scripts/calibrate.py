"""Calibrate CALIBRATED_SAFETY_FACTOR (see events.py) on a held-out synthetic
sweep, disjoint in seed from every sweep used in tests/test_events.py and
scripts/benchmark.py. Prints the value to hand-copy into events.py; this
repository does not auto-patch its own source.

    python scripts/calibrate.py
"""
from __future__ import annotations

import numpy as np

from strideline import kinematics as kin
from strideline import simulate
from strideline.events import calibrate_safety_factor, detect_contacts, match_events
from strideline.smoother import constrained_smooth

CALIBRATION_SEEDS = range(100, 108)
FPS_GRID = (30.0, 60.0, 120.0)  # the validated envelope; see README limitations
STRESS_FPS = 240.0
NOISE_GRID = (1.0, 3.0, 6.0)


def _make(fps, noise, seed, side="L"):
    seq = simulate.generate_sequence(duration_s=8.0, fps=fps, pixel_noise_std=noise,
                                      dropout_prob=0.05, seed=seed)
    result = constrained_smooth(seq.hip_raw, seq.knee_raw[side], seq.ankle_raw[side], dt=seq.dt)
    _, ankle_hat = kin.forward_kinematics(result.hip, result.theta_thigh, result.theta_shank,
                                           result.L_thigh, result.L_shank)
    return seq, ankle_hat[:, 1]


def main():
    sequences = [_make(fps, noise, seed)
                 for fps in FPS_GRID for noise in NOISE_GRID for seed in CALIBRATION_SEEDS]

    recalls, precisions = [], []
    for seq, y in sequences:
        events = detect_contacts(y, dt=seq.dt, safety_factor=1.0)
        m = match_events(seq.contact_times_true["L"], events)
        recalls.append(m["recall"])
        precisions.append(m["precision"])

    factor_99 = calibrate_safety_factor(sequences, quantile=0.99)
    factor_max = calibrate_safety_factor(sequences, quantile=1.0)

    print(f"validated envelope: fps in {FPS_GRID}, pixel noise in {NOISE_GRID}")
    print(f"detection recall  (mean/min): {np.mean(recalls):.3f} / {np.min(recalls):.3f}")
    print(f"detection precision (mean/min): {np.mean(precisions):.3f} / {np.min(precisions):.3f}")
    print(f"99th-percentile safety factor: {factor_99:.2f}")
    print(f"max (100% calibration-set coverage) safety factor: {factor_max:.2f}")
    print("\ncopy the chosen value into CALIBRATED_SAFETY_FACTOR in src/strideline/events.py")

    stress_seqs = [_make(STRESS_FPS, noise, seed) for noise in NOISE_GRID for seed in CALIBRATION_SEEDS]
    stress_recalls, stress_precisions = [], []
    for seq, y in stress_seqs:
        events = detect_contacts(y, dt=seq.dt, safety_factor=1.0)
        m = match_events(seq.contact_times_true["L"], events)
        stress_recalls.append(m["recall"])
        stress_precisions.append(m["precision"])
    stress_factor_max = calibrate_safety_factor(stress_seqs, quantile=1.0)
    print(f"\nstress test at {STRESS_FPS} fps (outside the validated envelope):")
    print(f"  detection recall/precision (mean): {np.mean(stress_recalls):.3f} / {np.mean(stress_precisions):.3f}")
    print(f"  safety factor needed for 100% timing coverage here: {stress_factor_max:.1f}x")
    print("  (this is why the certificate's claimed envelope stops at 120 fps)")


if __name__ == "__main__":
    main()
