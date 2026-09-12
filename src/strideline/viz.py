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
STRIDE_GREEN = "#2c8c4c"
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


def render_overlay_gif(video_path, tracks, results, out_path, fps_out=15.0, resize_width=640,
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

        frames_out.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        i += 1

    cap.release()

    # No burned-in caption text: GIF palette quantization makes small anti-aliased
    # text look blurry/noisy. The red/blue legend lives in the README caption
    # instead, where it renders crisply at any zoom level.
    images = [Image.fromarray(f).quantize(colors=192, method=Image.MEDIANCUT) for f in frames_out]
    duration_ms = round(1000 / fps_out)
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration_ms,
                   loop=0, optimize=True)


def plot_synthetic_correction_gallery(seq, result, side, dt, out_path, n_panels=6):
    """One full gait cycle, small multiples: the true stick figure (green,
    exact simulator ground truth) against the actual raw noisy keypoints
    (red, literally the simulated detector's output) and the corrected
    reconstruction (blue), all recentred on the hip. Unlike the real-video
    case study, this has ground truth, so it is the most direct visual
    evidence the correction is recovering the right skeleton rather than
    merely a smoother-looking wrong one.
    """
    from . import kinematics as kin

    side_name = {"L": "left", "R": "right"}.get(side, side)
    contacts = seq.contact_times_true[side]
    t0, t1 = contacts[1], contacts[2]  # an interior cycle, away from sequence edges
    idxs = [round((t0 + frac * (t1 - t0)) / dt) for frac in np.linspace(0, 1, n_panels, endpoint=False)]

    true_thigh = np.unwrap(seq.theta_thigh_true[side])
    true_shank = np.unwrap(seq.theta_shank_true[side])
    L_total = seq.L_thigh_true + seq.L_shank_true

    fig, axes = plt.subplots(1, n_panels, figsize=(1.85 * n_panels, 2.7), sharex=True, sharey=True)

    def leg(ax, knee, ankle, color, lw, label, z, ls="-"):
        ax.plot([0, knee[0], ankle[0]], [0, -knee[1], -ankle[1]], color=color, lw=lw, ls=ls,
                marker="o", ms=4.5, label=label, zorder=z, solid_capstyle="round",
                dash_capstyle="round")

    for panel, idx in enumerate(idxs):
        ax = axes[panel]
        tk = seq.L_thigh_true * kin.unit(true_thigh[idx])
        ta = tk + seq.L_shank_true * kin.unit(true_shank[idx])
        leg(ax, tk, ta, STRIDE_GREEN, 3.4, "true", 1)

        hip_r, knee_r, ankle_r = seq.hip_raw[idx], seq.knee_raw[side][idx], seq.ankle_raw[side][idx]
        if np.all(np.isfinite(knee_r)) and np.all(np.isfinite(ankle_r)):
            leg(ax, knee_r - hip_r, ankle_r - hip_r, STRIDE_RED, 1.6, "raw", 2)

        ck = result.L_thigh * kin.unit(result.theta_thigh[idx])
        ca = ck + result.L_shank * kin.unit(result.theta_shank[idx])
        leg(ax, ck, ca, STRIDE_BLUE, 2.0, "corrected", 3, ls=(0, (4, 2)))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_xlim(-0.75 * L_total, 0.75 * L_total)
    axes[0].set_ylim(-1.08 * L_total, 0.08 * L_total)
    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False, handlelength=2.0)
    fig.suptitle(f"One gait cycle, ground truth vs. raw vs. corrected ({side_name} leg)", fontsize=13,
                 color=STRIDE_INK, y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_contact_filmstrip(video_path, tracks, result, side, events, dt, out_path,
                              n_events=4, crop_w=260, crop_h=340):
    """A crisp, static filmstrip of real video frames at detected foot-strike
    instants, with the corrected skeleton overlaid and the certified timing
    window printed under each panel. Static PNG, not a GIF, so the text
    renders sharp at any zoom level.
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    from . import kinematics as kin

    if not events:
        return

    chosen_idx = np.linspace(0, len(events) - 1, min(n_events, len(events))).round().astype(int)
    chosen = [events[i] for i in sorted(set(chosen_idx.tolist()))]

    knee_hat, ankle_hat = kin.forward_kinematics(result.hip, result.theta_thigh, result.theta_shank,
                                                  result.L_thigh, result.L_shank)

    cap = cv2.VideoCapture(str(video_path))
    try:
        font = ImageFont.truetype("arial.ttf", 17)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    label_h = 46
    panels = []
    for e in chosen:
        frame_idx = round(e.t_cross / dt)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        cx, cy = ankle_hat[frame_idx]
        cy -= crop_h * 0.32
        x0, y0 = int(cx - crop_w / 2), int(cy - crop_h / 2)
        fh, fw = frame.shape[:2]
        pad_l, pad_t = max(0, -x0), max(0, -y0)
        pad_r, pad_b = max(0, x0 + crop_w - fw), max(0, y0 + crop_h - fh)
        padded = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        x0p, y0p = x0 + pad_l, y0 + pad_t
        crop = padded[y0p:y0p + crop_h, x0p:x0p + crop_w].copy()
        offset = np.array([x0, y0])

        def pt(p, offset=offset):
            return (int(p[0] - offset[0]), int(p[1] - offset[1]))

        hip_c, knee_c, ankle_c = result.hip[frame_idx], knee_hat[frame_idx], ankle_hat[frame_idx]
        knee_r, ankle_r = tracks.knee[side][frame_idx], tracks.ankle[side][frame_idx]
        hip_r = tracks.hip[side][frame_idx]
        if np.all(np.isfinite(knee_r)) and np.all(np.isfinite(hip_r)):
            cv2.line(crop, pt(hip_r), pt(knee_r), (30, 30, 220), 2, cv2.LINE_AA)
        if np.all(np.isfinite(ankle_r)) and np.all(np.isfinite(knee_r)):
            cv2.line(crop, pt(knee_r), pt(ankle_r), (30, 30, 220), 2, cv2.LINE_AA)
        cv2.line(crop, pt(hip_c), pt(knee_c), (220, 140, 30), 3, cv2.LINE_AA)
        cv2.line(crop, pt(knee_c), pt(ankle_c), (220, 140, 30), 3, cv2.LINE_AA)
        for p in (hip_c, knee_c, ankle_c):
            cv2.circle(crop, pt(p), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(crop, pt(p), 5, (220, 140, 30), 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        panel = Image.new("RGB", (crop_w, crop_h + label_h), (250, 250, 250))
        panel.paste(Image.fromarray(rgb), (0, 0))
        draw = ImageDraw.Draw(panel)
        draw.text((10, crop_h + 6), f"t = {e.t_cross:.2f}s", fill=(20, 20, 20), font=font)
        draw.text((10, crop_h + 26), f"certified window: +/-{e.bound_seconds * 1000:.0f} ms",
                   fill=(70, 70, 70), font=font_small)
        panels.append(panel)

    cap.release()
    if not panels:
        return

    gap = 14
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    strip = Image.new("RGB", (total_w, panels[0].height), (255, 255, 255))
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += p.width + gap
    strip.save(out_path)


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
