"""End-to-end: a running video in, gait metrics and figures out.

    python scripts/run.py assets/clips/running.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from strideline import kinematics as kin
from strideline import metrics as met
from strideline import pose, viz
from strideline.events import detect_contacts
from strideline.smoother import constrained_smooth, kinematic_consistency_violation

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"


def process_side(tracks, side, dt, pad_frames=6):
    hip_raw = tracks.hip[side]
    knee_raw = tracks.knee[side]
    ankle_raw = tracks.ankle[side]

    result = constrained_smooth(hip_raw, knee_raw, ankle_raw, dt=dt)
    knee_hat, ankle_hat = kin.forward_kinematics(result.hip, result.theta_thigh, result.theta_shank,
                                                  result.L_thigh, result.L_shank)
    events = detect_contacts(ankle_hat[:, 1], dt=dt)

    valid_idx = np.where(np.all(np.isfinite(knee_raw), axis=-1))[0]
    if len(valid_idx):
        t_lo = max(0, valid_idx[0] - pad_frames) * dt
        t_hi = (min(len(knee_raw) - 1, valid_idx[-1] + pad_frames)) * dt
        events = [e for e in events if t_lo <= e.t_cross <= t_hi]

    raw_cv_thigh = kinematic_consistency_violation(hip_raw, knee_raw, result.L_thigh)
    corr_cv_thigh = kinematic_consistency_violation(result.hip, knee_hat, result.L_thigh)

    summary = met.summarize_leg(result.theta_thigh, result.theta_shank, result.hip[:, 1], dt,
                                 events, result.L_thigh, result.L_shank)
    summary["kinematic_consistency_violation_raw_pct"] = raw_cv_thigh
    summary["kinematic_consistency_violation_corrected_pct"] = corr_cv_thigh
    summary["L_thigh_px"] = result.L_thigh
    summary["L_shank_px"] = result.L_shank

    return result, events, ankle_hat, knee_hat, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--model", default="yolo11n-pose.pt")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tracks = pose.extract_tracks(args.video, model_name=args.model)
    dt = 1.0 / tracks.fps

    results = {}
    for side in ("left", "right"):
        results[side] = process_side(tracks, side, dt)

    left_result, left_events, left_ankle_hat, _, left_summary = results["left"]
    right_result, right_events, _, _, right_summary = results["right"]

    contacts_left = np.array([e.t_cross for e in left_events])
    contacts_right = np.array([e.t_cross for e in right_events])

    visible = np.isfinite(tracks.knee["left"][:, 0]) | np.isfinite(tracks.knee["right"][:, 0])
    visible_idx = np.where(visible)[0]
    active_duration = (visible_idx[-1] - visible_idx[0]) * dt if len(visible_idx) > 1 else tracks.n_frames * dt

    cadence = met.cadence_spm(contacts_left, contacts_right, active_duration)
    asymmetry = met.step_time_asymmetry_pct(contacts_left, contacts_right)

    viz.plot_phase_portrait(left_result.raw_theta_thigh, left_result.raw_theta_shank,
                             left_result.theta_thigh, left_result.theta_shank, dt,
                             FIG_DIR / "phase_portrait_left.png")
    viz.plot_ankle_path(tracks.ankle["left"], left_ankle_hat, FIG_DIR / "ankle_path_left.png")
    viz.plot_constraint_violation_strip(left_result.raw_L_thigh_series, left_result.L_thigh, dt,
                                         FIG_DIR / "constraint_violation_left.png")
    viz.plot_certificate_timeline(left_ankle_hat[:, 1], dt, left_events,
                                   FIG_DIR / "certificate_timeline_left.png")
    viz.plot_energy_trace(left_result.energy_trace, FIG_DIR / "energy_trace_left.png")

    labels = ["stride-time CV (%)", "knee-angle CV (%)", "hip oscillation (x100)"]
    values_left = [left_summary["stride_time_cv_pct"], left_summary["knee_angle_cv_pct"],
                   left_summary["hip_vertical_oscillation_ratio"] * 100]
    values_right = [right_summary["stride_time_cv_pct"], right_summary["knee_angle_cv_pct"],
                    right_summary["hip_vertical_oscillation_ratio"] * 100]
    viz.plot_symmetry_radar(labels, values_left, values_right, FIG_DIR / "symmetry_radar.png")

    viz.render_overlay_gif(args.video, tracks,
                            {"left": left_result, "right": right_result},
                            FIG_DIR / "overlay_demo.gif")

    summary = {
        "cadence_spm": cadence,
        "active_duration_s": float(active_duration),
        "step_time_asymmetry_pct": asymmetry,
        "left": left_summary,
        "right": right_summary,
    }
    out_json = ROOT / "assets" / "case_study_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nfigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
