import numpy as np

from strideline import kinematics as kin
from strideline import simulate
from strideline.smoother import constrained_smooth, fill_and_unwrap, kinematic_consistency_violation


def _run_leg(side="L", noise=3.0, dropout=0.05, seed=0):
    seq = simulate.generate_sequence(duration_s=6.0, fps=60.0, pixel_noise_std=noise,
                                      dropout_prob=dropout, seed=seed)
    result = constrained_smooth(seq.hip_raw, seq.knee_raw[side], seq.ankle_raw[side], dt=seq.dt)
    return seq, result


def test_energy_is_monotonically_non_increasing():
    _, result = _run_leg()
    trace = np.array(result.energy_trace)
    diffs = np.diff(trace)
    assert np.all(diffs <= 1e-6), f"energy increased at steps {np.where(diffs > 1e-6)[0]}: {trace}"


def test_energy_non_increasing_across_seeds_and_noise():
    for seed in range(5):
        for noise in (1.0, 3.0, 6.0):
            _, result = _run_leg(noise=noise, seed=seed)
            trace = np.array(result.energy_trace)
            assert np.all(np.diff(trace) <= 1e-6)


def test_smoothing_recovers_true_segment_lengths():
    seq, result = _run_leg(noise=4.0, dropout=0.1, seed=1)
    assert abs(result.L_thigh - seq.L_thigh_true) / seq.L_thigh_true < 0.05
    assert abs(result.L_shank - seq.L_shank_true) / seq.L_shank_true < 0.05


def test_smoothing_reduces_angle_error_vs_raw():
    seq, result = _run_leg(noise=5.0, dropout=0.05, seed=2)
    true_thigh = np.unwrap(seq.theta_thigh_true["L"])

    raw_thigh = fill_and_unwrap(result.raw_theta_thigh)
    raw_rmse = np.sqrt(np.mean((raw_thigh - true_thigh) ** 2))
    smooth_rmse = np.sqrt(np.mean((result.theta_thigh - true_thigh) ** 2))
    assert smooth_rmse < raw_rmse


def test_kinematic_consistency_violation_drops_after_smoothing():
    seq, result = _run_leg(noise=5.0, dropout=0.0, seed=3)
    raw_cv = kinematic_consistency_violation(seq.hip_raw, seq.knee_raw["L"], seq.L_thigh_true)
    knee_hat, _ = kin.forward_kinematics(result.hip, result.theta_thigh, result.theta_shank,
                                          result.L_thigh, result.L_shank)
    corrected_cv = kinematic_consistency_violation(result.hip, knee_hat, seq.L_thigh_true)
    assert corrected_cv < raw_cv
