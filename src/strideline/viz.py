"""Every figure in this module is a distinct visual form on purpose: a
limit-cycle phase portrait, a radar-style symmetry plot, a constraint
violation heatmap strip, a certificate filmstrip, and a bound-validation
certificate plot. None of them is a bare bar chart or an unadorned line
chart of a single series over time.
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRIDE_BLUE = "#2f6f9f"
STRIDE_RED = "#c1440e"
STRIDE_INK = "#1b1b1b"
STRIDE_GRID = "#d8d8d8"


def _style_ax(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=STRIDE_INK)
    ax.xaxis.label.set_color(STRIDE_INK)
    ax.yaxis.label.set_color(STRIDE_INK)
    ax.title.set_color(STRIDE_INK)


def plot_phase_portrait(theta_thigh_raw, theta_shank_raw, theta_thigh_sm, theta_shank_sm, dt, out_path):
    from . import kinematics as kin

    flex_raw = np.degrees(kin.knee_flexion_angle(theta_thigh_raw, theta_shank_raw))
    flex_sm = np.degrees(kin.knee_flexion_angle(theta_thigh_sm, theta_shank_sm))
    omega_raw = np.gradient(flex_raw, dt)
    omega_sm = np.gradient(flex_sm, dt)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot(flex_raw, omega_raw, color=STRIDE_RED, alpha=0.35, lw=1.1, label="raw pose")
    ax.plot(flex_sm, omega_sm, color=STRIDE_BLUE, lw=2.0, label="corrected")
    ax.set_xlabel("knee flexion angle (deg)")
    ax.set_ylabel("knee angular velocity (deg / s)")
    ax.set_title("Knee phase portrait: raw vs. corrected")
    ax.legend(frameon=False)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_symmetry_radar(labels, values_left, values_right, out_path, title="Left / right symmetry"):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    def close(v):
        v = list(v) + [v[0]]
        return v

    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, close(values_left), color=STRIDE_BLUE, lw=2.0, label="left")
    ax.fill(angles, close(values_left), color=STRIDE_BLUE, alpha=0.15)
    ax.plot(angles, close(values_right), color=STRIDE_RED, lw=2.0, label="right")
    ax.fill(angles, close(values_right), color=STRIDE_RED, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color=STRIDE_INK)
    ax.set_yticklabels([])
    ax.set_title(title, color=STRIDE_INK, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_constraint_violation_strip(raw_length_series, corrected_length, dt, out_path,
                                     title="Where the raw pose broke limb-length constancy"):
    raw_length_series = np.asarray(raw_length_series, dtype=float)
    deviation_pct = 100.0 * (raw_length_series - corrected_length) / corrected_length
    t = np.arange(len(deviation_pct)) * dt

    fig, (ax_img, ax_line) = plt.subplots(2, 1, figsize=(9.5, 3.4), height_ratios=[1, 2.2], sharex=True)
    vmax = np.nanpercentile(np.abs(deviation_pct), 98) or 1.0
    ax_img.imshow(deviation_pct[np.newaxis, :], aspect="auto", cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax, extent=[t[0], t[-1], 0, 1])
    ax_img.set_yticks([])
    ax_img.set_title(title, color=STRIDE_INK, fontsize=11)

    ax_line.plot(t, deviation_pct, color=STRIDE_INK, lw=1.0)
    ax_line.axhline(0, color=STRIDE_GRID, lw=1.0)
    ax_line.fill_between(t, deviation_pct, 0, where=deviation_pct >= 0, color=STRIDE_RED, alpha=0.35)
    ax_line.fill_between(t, deviation_pct, 0, where=deviation_pct < 0, color=STRIDE_BLUE, alpha=0.35)
    ax_line.set_xlabel("time (s)")
    ax_line.set_ylabel("implied length\nerror (%)")
    _style_ax(ax_line)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_ankle_path(ankle_raw, ankle_smooth, out_path, title="Ankle path: raw vs. corrected"):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ankle_raw[:, 0], -ankle_raw[:, 1], color=STRIDE_RED, alpha=0.4, lw=1.1, label="raw")
    ax.plot(ankle_smooth[:, 0], -ankle_smooth[:, 1], color=STRIDE_BLUE, lw=2.0, label="corrected")
    ax.set_xlabel("horizontal position (px)")
    ax.set_ylabel("height (px, up is positive)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="datalim")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_certificate_timeline(ankle_y_smooth, dt, events, out_path,
                               title="Contact-time certificates"):
    t = np.arange(len(ankle_y_smooth)) * dt
    fig, ax = plt.subplots(figsize=(9.6, 3.2))
    ax.plot(t, -ankle_y_smooth, color=STRIDE_INK, lw=1.3)
    for e in events:
        ax.axvspan(e.t_cross - e.bound_seconds, e.t_cross + e.bound_seconds, color=STRIDE_BLUE, alpha=0.25)
        ax.axvline(e.t_cross, color=STRIDE_RED, lw=1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ankle height\n(px, up positive)")
    ax.set_title(title)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_bound_validation(rows, out_path, title="Certified bound vs. empirical timing error"):
    errors = np.array([max(r["error"] * 1000, 1e-3) for r in rows])
    bounds = np.array([max(r["bound"] * 1000, 1e-3) for r in rows])
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    lo = min(bounds.min(), errors.min()) * 0.7
    hi = max(bounds.max(), errors.max()) * 1.4
    ax.plot([lo, hi], [lo, hi], color=STRIDE_GRID, lw=1.5, ls="--", label="bound = error (limit)")
    ax.scatter(bounds, errors, color=STRIDE_BLUE, alpha=0.6, s=26, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("certified bound (ms, log scale)")
    ax.set_ylabel("empirical timing error (ms, log scale)")
    ax.set_title(title)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(frameon=False, loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_overlay_gif(video_path, tracks, results, out_path, fps_out=12.0, resize_width=480,
                        pad_frames=6):
    """Render raw (red) vs. corrected (blue) skeleton overlays on the
    original video, exported as a compact GIF for the README hero image.

    tracks: pose.RawTracks. results: {"left": SmoothResult, "right": SmoothResult}.
    Automatically crops to the frame range where a runner is actually
    visible (plus a small pad), and subsamples to fps_out / resize_width,
    since a full-resolution, full-length GIF is tens of megabytes - far
    too large for a README asset.
    """
    import cv2
    from PIL import Image

    from . import kinematics as kin

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or resize_width
    src_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or resize_width
    scale = resize_width / src_w if src_w else 1.0
    out_h = round(src_h * scale)

    hats = {}
    for side, result in results.items():
        knee_hat, ankle_hat = kin.forward_kinematics(result.hip, result.theta_thigh, result.theta_shank,
                                                       result.L_thigh, result.L_shank)
        hats[side] = (result.hip, knee_hat, ankle_hat)

    n = len(next(iter(hats.values()))[0])
    visible = np.zeros(n, dtype=bool)
    for side in results:
        visible |= np.all(np.isfinite(tracks.knee[side]), axis=-1)
    visible_idx = np.where(visible)[0]
    if len(visible_idx):
        first = max(0, visible_idx[0] - pad_frames)
        last = min(n - 1, visible_idx[-1] + pad_frames)
    else:
        first, last = 0, n - 1

    step = max(1, round(src_fps / fps_out))
    keep = set(range(first, last + 1, step))

    frames_out = []
    i = 0
    while i <= last:
        ret, frame = cap.read()
        if not ret:
            break
        if i < first or i not in keep:
            i += 1
            continue
        canvas = cv2.resize(frame, (resize_width, out_h))
        for side in results:
            hip_raw = tracks.hip[side][i]
            knee_raw = tracks.knee[side][i]
            ankle_raw = tracks.ankle[side][i]
            hip_c, knee_c, ankle_c = hats[side][0][i], hats[side][1][i], hats[side][2][i]

            def pt(p):
                return (int(p[0] * scale), int(p[1] * scale))

            if np.all(np.isfinite(knee_raw)) and np.all(np.isfinite(hip_raw)):
                cv2.line(canvas, pt(hip_raw), pt(knee_raw), (30, 30, 220), 2, cv2.LINE_AA)
            if np.all(np.isfinite(ankle_raw)) and np.all(np.isfinite(knee_raw)):
                cv2.line(canvas, pt(knee_raw), pt(ankle_raw), (30, 30, 220), 2, cv2.LINE_AA)
            cv2.line(canvas, pt(hip_c), pt(knee_c), (220, 140, 30), 3, cv2.LINE_AA)
            cv2.line(canvas, pt(knee_c), pt(ankle_c), (220, 140, 30), 3, cv2.LINE_AA)
            for p in (hip_c, knee_c, ankle_c):
                cv2.circle(canvas, pt(p), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt(p), 5, (220, 140, 30), 2, cv2.LINE_AA)

        font_scale = 0.5
        cv2.putText(canvas, "red = raw pose   blue = corrected", (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, "red = raw pose   blue = corrected", (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1, cv2.LINE_AA)
        frames_out.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        i += 1

    cap.release()

    images = [Image.fromarray(f).quantize(colors=128, method=Image.MEDIANCUT) for f in frames_out]
    duration_ms = round(1000 / fps_out)
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration_ms,
                   loop=0, optimize=True)


def plot_energy_trace(energy_trace, out_path, title="Alternating-minimization energy"):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(energy_trace, marker="o", ms=4, color=STRIDE_BLUE)
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("energy E (a.u.)")
    ax.set_title(title)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
