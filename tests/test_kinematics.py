import numpy as np

from strideline import kinematics as kin


def test_forward_and_raw_angles_are_inverse():
    hip = np.array([[0.0, 0.0], [1.0, 2.0]])
    theta_thigh = np.array([0.3, -0.5])
    theta_shank = np.array([0.1, 0.8])
    knee, ankle = kin.forward_kinematics(hip, theta_thigh, theta_shank, L_thigh=90.0, L_shank=95.0)
    tth, tsh = kin.raw_angles(hip, knee, ankle)
    np.testing.assert_allclose(tth, theta_thigh, atol=1e-9)
    np.testing.assert_allclose(tsh, theta_shank, atol=1e-9)


def test_segment_length_matches_forward_kinematics_length():
    hip = np.zeros((5, 2))
    theta = np.linspace(-0.5, 0.5, 5)
    knee, _ = kin.forward_kinematics(hip, theta, theta, L_thigh=100.0, L_shank=100.0)
    lengths = kin.segment_length(hip, knee)
    np.testing.assert_allclose(lengths, 100.0, atol=1e-9)


def test_knee_flexion_zero_when_straight():
    theta = np.array([0.2, -0.3, 1.0])
    flex = kin.knee_flexion_angle(theta, theta)
    np.testing.assert_allclose(flex, 0.0, atol=1e-9)


def test_unit_vector_has_unit_norm():
    theta = np.linspace(-3, 3, 50)
    u = kin.unit(theta)
    norms = np.linalg.norm(u, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)
