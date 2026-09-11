
from strideline import kinematics as kin
from strideline import simulate
from strideline.events import detect_contacts, match_events
from strideline.smoother import constrained_smooth


def _detect(fps, noise, dropout=0.05, seed=0, side="L"):
    seq = simulate.generate_sequence(duration_s=8.0, fps=fps, pixel_noise_std=noise,
                                      dropout_prob=dropout, seed=seed)
    result = constrained_smooth(seq.hip_raw, seq.knee_raw[side], seq.ankle_raw[side], dt=seq.dt)
    _, ankle_hat = kin.forward_kinematics(result.hip, result.theta_thigh, result.theta_shank,
                                           result.L_thigh, result.L_shank)
    events = detect_contacts(ankle_hat[:, 1], dt=seq.dt)
    return seq, events


CERTIFIED_FPS_GRID = (30.0, 60.0, 120.0)  # the validated envelope; see README limitations
CERTIFIED_NOISE_GRID = (1.0, 3.0, 6.0)


def test_certified_bound_holds_on_a_held_out_test_sweep():
    """Disjoint seeds from the calibration sweep used to set
    CALIBRATED_SAFETY_FACTOR (see events.py / scripts/calibrate.py), within
    the validated envelope (fps <= 120; see the README's limitations
    section for why 240 fps is reported separately as a stress test)."""
    total, held = 0, 0
    for fps in CERTIFIED_FPS_GRID:
        for noise in CERTIFIED_NOISE_GRID:
            for seed in (7, 21, 55):
                seq, events = _detect(fps=fps, noise=noise, seed=seed)
                m = match_events(seq.contact_times_true["L"], events)
                for r in m["matched_rows"]:
                    total += 1
                    held += int(r["held"])
    assert total > 50, "sanity: expected many detected+matched events across the sweep"
    hold_rate = held / total
    assert hold_rate >= 0.98, f"certified bound held in only {held}/{total} ({hold_rate:.1%}) events"


def test_detection_recall_and_precision_are_high():
    for fps in (60.0, 120.0):
        seq, events = _detect(fps=fps, noise=3.0, dropout=0.05, seed=13)
        m = match_events(seq.contact_times_true["L"], events)
        assert m["recall"] >= 0.9
        assert m["precision"] >= 0.9


def test_detects_roughly_the_expected_number_of_strikes():
    _, events = _detect(fps=60.0, noise=2.0, dropout=0.0, seed=7)
    stride_period = 60.0 / (170.0 / 2)
    expected = 8.0 / stride_period
    assert abs(len(events) - expected) <= 2
