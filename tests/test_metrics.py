import numpy as np

from strideline import metrics


def test_cadence_counts_both_feet():
    left = np.array([0.0, 0.7, 1.4])
    right = np.array([0.35, 1.05, 1.75])
    cadence = metrics.cadence_spm(left, right, duration_s=2.1)
    assert abs(cadence - (60.0 * 6 / 2.1)) < 1e-9


def test_stride_time_cv_zero_for_perfectly_even_strides():
    contacts = np.arange(0, 5.0, 0.5)
    assert metrics.stride_time_cv(contacts) < 1e-9


def test_stride_time_cv_positive_for_uneven_strides():
    contacts = np.array([0.0, 0.5, 0.9, 1.5, 1.85])
    assert metrics.stride_time_cv(contacts) > 0


def test_step_time_asymmetry_zero_when_symmetric():
    left = np.array([0.0, 0.7, 1.4, 2.1])
    right = np.array([0.35, 1.05, 1.75, 2.45])
    assert metrics.step_time_asymmetry_pct(left, right) < 1e-9


def test_form_consistency_index_decreases_with_variability():
    high = metrics.form_consistency_index(0.0, 0.0)
    low = metrics.form_consistency_index(20.0, 15.0)
    assert high == 100.0
    assert low < high


def test_hip_vertical_oscillation_ratio_scales_with_leg_length():
    hip_y = np.array([100.0, 106.0, 100.0, 106.0])
    ratio_short_leg = metrics.hip_vertical_oscillation_ratio(hip_y, 50.0, 50.0)
    ratio_long_leg = metrics.hip_vertical_oscillation_ratio(hip_y, 100.0, 100.0)
    assert ratio_short_leg > ratio_long_leg
