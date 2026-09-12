"""Synthetic ground-truth benchmark: the certified bound validation and the
raw-vs-corrected accuracy tables that back the README.

Everything here runs from this repository alone; nothing needs the real
video clip. Run with:

    python scripts/benchmark.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from strideline import kinematics as kin
from strideline import simulate, viz
from strideline.events import detect_contacts, match_events
from strideline.smoother import constrained_smooth, fill_and_unwrap

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CERTIFIED_FPS_GRID = [30.0, 60.0, 120.0]
STRESS_FPS = 240.0
NOISE_GRID = [1.0, 3.0, 6.0]
TEST_SEEDS = range(6)  # disjoint from scripts/calibrate.py's seeds (100-107)


def run_one(fps, noise, dropout, seed, side="L"):
    seq = simulate.generate_sequence(duration_s=8.0, fps=fps, pixel_noise_std=noise,
                                      dropout_prob=dropout, seed=seed)
    result = constrained_smooth(seq.hip_raw, seq.knee_raw[side], seq.ankle_raw[side], dt=seq.dt)
    _, ankle_hat = kin.forward_kinematics(result.hip, result.theta_thigh, result.theta_shank,
                                           result.L_thigh, result.L_shank)

    true_thigh = np.unwrap(seq.theta_thigh_true[side])
    true_shank = np.unwrap(seq.theta_shank_true[side])
    raw_thigh = fill_and_unwrap(result.raw_theta_thigh)
    raw_shank = fill_and_unwrap(result.raw_theta_shank)

    thigh_rmse_raw = np.degrees(np.sqrt(np.mean((raw_thigh - true_thigh) ** 2)))
    thigh_rmse_corrected = np.degrees(np.sqrt(np.mean((result.theta_thigh - true_thigh) ** 2)))
    shank_rmse_raw = np.degrees(np.sqrt(np.mean((raw_shank - true_shank) ** 2)))
    shank_rmse_corrected = np.degrees(np.sqrt(np.mean((result.theta_shank - true_shank) ** 2)))

    events = detect_contacts(ankle_hat[:, 1], dt=seq.dt)
    match = match_events(seq.contact_times_true[side], events)

    return {
        "fps": fps, "noise": noise, "dropout": dropout, "seed": seed,
        "thigh_rmse_raw_deg": thigh_rmse_raw, "thigh_rmse_corrected_deg": thigh_rmse_corrected,
        "shank_rmse_raw_deg": shank_rmse_raw, "shank_rmse_corrected_deg": shank_rmse_corrected,
        "n_events": len(events), "recall": match["recall"], "precision": match["precision"],
        "matched_rows": match["matched_rows"], "energy_trace": result.energy_trace,
    }


def _sweep(fps_grid, noise_grid, seeds):
    all_rows, table, recalls, precisions = [], [], [], []
    for fps in fps_grid:
        for noise in noise_grid:
            agg = {"thigh_raw": [], "thigh_corr": [], "shank_raw": [], "shank_corr": []}
            for seed in seeds:
                r = run_one(fps, noise, dropout=0.05, seed=seed)
                agg["thigh_raw"].append(r["thigh_rmse_raw_deg"])
                agg["thigh_corr"].append(r["thigh_rmse_corrected_deg"])
                agg["shank_raw"].append(r["shank_rmse_raw_deg"])
                agg["shank_corr"].append(r["shank_rmse_corrected_deg"])
                all_rows.extend(r["matched_rows"])
                recalls.append(r["recall"])
                precisions.append(r["precision"])
            table.append({
                "fps": fps, "noise_px": noise,
                "thigh_rmse_raw_deg": float(np.mean(agg["thigh_raw"])),
                "thigh_rmse_corrected_deg": float(np.mean(agg["thigh_corr"])),
                "shank_rmse_raw_deg": float(np.mean(agg["shank_raw"])),
                "shank_rmse_corrected_deg": float(np.mean(agg["shank_corr"])),
            })
    return all_rows, table, recalls, precisions


def main():
    all_rows, table, recalls, precisions = _sweep(CERTIFIED_FPS_GRID, NOISE_GRID, TEST_SEEDS)

    n_held = int(sum(bool(r["held"]) for r in all_rows))
    n_total = len(all_rows)
    tightness = np.mean([r["bound"] / max(r["error"], 1e-9) for r in all_rows if r["error"] > 1e-9])

    stress_rows, _, stress_recalls, stress_precisions = _sweep([STRESS_FPS], NOISE_GRID, TEST_SEEDS)
    stress_held = int(sum(bool(r["held"]) for r in stress_rows))

    summary = {
        "validated_envelope_fps": CERTIFIED_FPS_GRID,
        "table": table,
        "certificate_events_total": n_total,
        "certificate_events_held": n_held,
        "certificate_hold_rate": n_held / n_total if n_total else None,
        "certificate_avg_tightness_ratio": float(tightness),
        "detection_recall_mean": float(np.mean(recalls)),
        "detection_precision_mean": float(np.mean(precisions)),
        "stress_test_240fps": {
            "hold_rate": stress_held / len(stress_rows) if stress_rows else None,
            "recall_mean": float(np.mean(stress_recalls)),
            "precision_mean": float(np.mean(stress_precisions)),
        },
    }

    out_json = ROOT / "assets" / "benchmark_results.json"
    out_json.write_text(json.dumps(summary, indent=2))

    viz.plot_bound_validation(all_rows, FIG_DIR / "bound_validation.png")

    gallery_seq = simulate.generate_sequence(duration_s=8.0, fps=60.0, pixel_noise_std=7.0,
                                              dropout_prob=0.05, seed=7)
    gallery_result = constrained_smooth(gallery_seq.hip_raw, gallery_seq.knee_raw["L"],
                                         gallery_seq.ankle_raw["L"], dt=gallery_seq.dt)
    viz.plot_synthetic_correction_gallery(gallery_seq, gallery_result, "L", gallery_seq.dt,
                                           FIG_DIR / "synthetic_correction_gallery.png")

    print(json.dumps({k: v for k, v in summary.items() if k != "table"}, indent=2))
    for row in table:
        print(row)
    print(f"\nfull results written to {out_json}")


if __name__ == "__main__":
    main()
