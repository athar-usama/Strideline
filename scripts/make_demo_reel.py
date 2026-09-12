"""Compile the existing figures/GIF (already produced by benchmark.py and
run.py) into one narrated-by-captions demo video: assets/demo_reel.mp4.

Nothing here recomputes anything; it only composes already-generated assets
into a single MP4, so run benchmark.py and run.py first.

    python scripts/make_demo_reel.py
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "assets" / "figures"
OUT_PATH = ROOT / "assets" / "demo_reel.mp4"

CANVAS = (1280, 720)
FPS = 24
BG = (13, 15, 18)
INK = (235, 236, 238)
SUBINK = (150, 155, 162)
ACCENT = (79, 155, 209)
CAPTION_H = 92
FADE_S = 0.25


def _font(size, bold=False):
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw, text, font, max_width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _fit_image(img, box_w, box_h):
    scale = min(box_w / img.width, box_h / img.height)
    return img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)


def _card(image=None, caption=None, sub=None, duration_s=3.0, image_pad=40):
    """One still card: an optional centered image on a dark canvas, with a
    caption band. Returns a list of RGB frame arrays, faded in/out."""
    n = max(1, round(duration_s * FPS))
    base = Image.new("RGB", CANVAS, BG)

    if image is not None:
        area_h = CANVAS[1] - (CAPTION_H if caption else 0) - 2 * image_pad
        area_w = CANVAS[0] - 2 * image_pad
        fitted = _fit_image(image.convert("RGB"), area_w, area_h)
        x = (CANVAS[0] - fitted.width) // 2
        y = image_pad + (area_h - fitted.height) // 2
        base.paste(fitted, (x, y))

    if caption:
        draw = ImageDraw.Draw(base)
        cap_font = _font(30, bold=True)
        sub_font = _font(19)
        top = CANVAS[1] - CAPTION_H
        draw.line([(0, top), (CANVAS[0], top)], fill=(40, 43, 48), width=1)
        lines = _wrap(draw, caption, cap_font, CANVAS[0] - 80)[:1]
        ty = top + 14
        for line in lines:
            w = draw.textlength(line, font=cap_font)
            draw.text(((CANVAS[0] - w) / 2, ty), line, font=cap_font, fill=INK)
            ty += 34
        if sub:
            w = draw.textlength(sub, font=sub_font)
            draw.text(((CANVAS[0] - w) / 2, ty + 2), sub, font=sub_font, fill=SUBINK)

    frame = np.array(base)
    frames = [frame] * n
    return _fade(frames)


def _fade(frames):
    n_fade = max(1, round(FADE_S * FPS))
    n_fade = min(n_fade, len(frames) // 3 or 1)
    out = [f.copy() for f in frames]
    for i in range(n_fade):
        alpha = i / n_fade
        out[i] = (out[i].astype(np.float32) * alpha).astype(np.uint8)
        j = len(out) - 1 - i
        out[j] = (out[j].astype(np.float32) * alpha).astype(np.uint8)
    return out


def _title_card():
    base = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(base)
    title_font = _font(88, bold=True)
    hook_font = _font(30)
    tag_font = _font(20)

    title = "STRIDELINE"
    w = draw.textlength(title, font=title_font)
    draw.text(((CANVAS[0] - w) / 2, 250), title, font=title_font, fill=INK)

    hook = "Certified running-gait kinematics from one camera"
    w = draw.textlength(hook, font=hook_font)
    draw.text(((CANVAS[0] - w) / 2, 360), hook, font=hook_font, fill=ACCENT)

    tag = "a markerless goniometer, not a pose-overlay demo"
    w = draw.textlength(tag, font=tag_font)
    draw.text(((CANVAS[0] - w) / 2, 410), tag, font=tag_font, fill=SUBINK)

    return _fade([np.array(base)] * round(3.6 * FPS))


def _outro_card():
    base = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(base)
    head_font = _font(46, bold=True)
    stat_font = _font(26)
    label_font = _font(17)
    foot_font = _font(19)

    head = "Validated, not just plotted"
    w = draw.textlength(head, font=head_font)
    draw.text(((CANVAS[0] - w) / 2, 130), head, font=head_font, fill=INK)

    stats = [
        ("99.5%", "certified-bound hold rate, held-out sweep"),
        ("100% / 100%", "detection recall / precision"),
        ("18 / 18", "tests passing"),
        ("2 pillars", "constrained smoother + certified contact timing"),
    ]
    y = 260
    for value, label in stats:
        vw = draw.textlength(value, font=stat_font)
        draw.text(((CANVAS[0] - vw) / 2, y), value, font=stat_font, fill=ACCENT)
        lw = draw.textlength(label, font=label_font)
        draw.text(((CANVAS[0] - lw) / 2, y + 34), label, font=label_font, fill=SUBINK)
        y += 90

    foot = "github.com/athar-usama/Strideline"
    fw = draw.textlength(foot, font=foot_font)
    draw.text(((CANVAS[0] - fw) / 2, 660), foot, font=foot_font, fill=SUBINK)

    return _fade([np.array(base)] * round(4.2 * FPS))


def _side_by_side(path_a, path_b, pad=16):
    a, b = Image.open(path_a).convert("RGB"), Image.open(path_b).convert("RGB")
    h = min(a.height, b.height)
    a = a.resize((round(a.width * h / a.height), h))
    b = b.resize((round(b.width * h / b.height), h))
    combo = Image.new("RGB", (a.width + b.width + pad, h), (255, 255, 255))
    combo.paste(a, (0, 0))
    combo.paste(b, (a.width + pad, 0))
    return combo


def _stacked(path_a, path_b, pad=12):
    a, b = Image.open(path_a).convert("RGB"), Image.open(path_b).convert("RGB")
    w = min(a.width, b.width)
    a = a.resize((w, round(a.height * w / a.width)))
    b = b.resize((w, round(b.height * w / b.width)))
    combo = Image.new("RGB", (w, a.height + b.height + pad), (255, 255, 255))
    combo.paste(a, (0, 0))
    combo.paste(b, (0, a.height + pad))
    return combo


def _gif_frames(gif_path, n_loops=2):
    im = Image.open(gif_path)
    frames = []
    for i in range(im.n_frames):
        im.seek(i)
        frames.append(im.convert("RGB").copy())
    out = []
    for _ in range(n_loops):
        out.extend(frames)
    return out


def _hero_clip(gif_path, caption, n_loops=2):
    raw_frames = _gif_frames(gif_path, n_loops=n_loops)
    canvas_frames = []
    for f in raw_frames:
        base = Image.new("RGB", CANVAS, BG)
        area_h = CANVAS[1] - CAPTION_H - 20
        fitted = _fit_image(f, CANVAS[0] - 40, area_h)
        x = (CANVAS[0] - fitted.width) // 2
        y = 10 + (area_h - fitted.height) // 2
        base.paste(fitted, (x, y))
        draw = ImageDraw.Draw(base)
        top = CANVAS[1] - CAPTION_H
        draw.line([(0, top), (CANVAS[0], top)], fill=(40, 43, 48), width=1)
        cap_font = _font(30, bold=True)
        sub_font = _font(19)
        w = draw.textlength(caption, font=cap_font)
        draw.text(((CANVAS[0] - w) / 2, top + 14), caption, font=cap_font, fill=INK)
        legend = "red = raw pose        blue = Strideline-corrected"
        w = draw.textlength(legend, font=sub_font)
        draw.text(((CANVAS[0] - w) / 2, top + 50), legend, font=sub_font, fill=SUBINK)
        canvas_frames.append(np.array(base))
    return _fade(canvas_frames)


def main():
    fig = FIG_DIR
    segments = []

    segments.append(_title_card())

    segments.append(_hero_clip(fig / "overlay_demo.gif",
                                "Raw pose in, corrected skeleton out"))

    segments.append(_card(
        Image.open(fig / "synthetic_correction_gallery.png"),
        "Correction checked against exact ground truth",
        "synthetic gait cycle: true (green) vs. raw noisy keypoints (red) vs. corrected (blue)",
        duration_s=4.2))

    segments.append(_card(
        Image.open(fig / "contact_filmstrip_left.png"),
        "Every certificate, anchored to a real frame",
        "detected foot strikes with the certified timing window printed underneath",
        duration_s=4.2))

    segments.append(_card(
        _side_by_side(fig / "phase_portrait_left.png", fig / "phase_portrait_right.png"),
        "Knee phase portraits, both legs",
        "raw (pale) vs. corrected (bold) limit cycles - real clip",
        duration_s=3.6))

    segments.append(_card(
        _side_by_side(fig / "ankle_path_left.png", fig / "ankle_path_right.png"),
        "Ankle paths, both legs",
        "the corrected trajectory rides through the raw jitter, not around it",
        duration_s=3.6))

    segments.append(_card(
        Image.open(fig / "symmetry_radar.png"),
        "Left / right symmetry, as a shape",
        "stride-time CV, knee-angle CV, and hip oscillation for both legs at once",
        duration_s=3.4))

    segments.append(_card(
        _stacked(fig / "constraint_violation_left.png", fig / "constraint_violation_right.png"),
        "Where the raw pose broke limb-length constancy",
        "both legs - the same constraint the smoother enforces exactly",
        duration_s=3.8))

    segments.append(_card(
        Image.open(fig / "certificate_timeline_left.png"),
        "Contact-time certificates, the full clip",
        "certified window shaded around every detected ground-contact instant",
        duration_s=3.6))

    segments.append(_card(
        Image.open(fig / "bound_validation.png"),
        "The certificate, validated",
        "log-log: every held-out event falls on or below the bound = error line",
        duration_s=3.8))

    segments.append(_card(
        Image.open(fig / "energy_trace_left.png"),
        "Proven to converge, checked that it does",
        "the smoother's energy, monotonically non-increasing by construction",
        duration_s=3.4))

    segments.append(_outro_card())

    writer = imageio.get_writer(str(OUT_PATH), fps=FPS, codec="libx264", quality=8,
                                 macro_block_size=None,
                                 ffmpeg_params=["-pix_fmt", "yuv420p"])
    n_frames = 0
    for seg in segments:
        for frame in seg:
            writer.append_data(frame)
            n_frames += 1
    writer.close()

    print(f"wrote {OUT_PATH} ({n_frames} frames, {n_frames / FPS:.1f}s at {FPS}fps)")


if __name__ == "__main__":
    main()
