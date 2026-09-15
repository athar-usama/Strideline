"""Run the real pipeline (pose extraction + constrained smoothing, both legs)
on several different real running clips, and compile the resulting skeleton
overlays into one video: assets/action_reel.mp4. This is real Strideline
output on real footage end to end - not a slideshow of the README's static
figures.

Also (re)generates a compact overlay GIF per extra clip for README embedding.

    python scripts/build_action_reel.py
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from strideline import pose, viz
from strideline.smoother import constrained_smooth

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "assets" / "clips"
FIG_DIR = ROOT / "assets" / "figures"
OUT_PATH = ROOT / "assets" / "action_reel.mp4"

CANVAS_W, CANVAS_H = 1280, 720
FPS = 24
CAPTION_H = 64
BG = (13, 15, 18)
INK = (235, 236, 238)
ACCENT = (79, 155, 209)
FADE_S = 0.25

# (clip path, on-screen label, also emit a compact README GIF under this name)
CLIPS = [
    ("running.mp4", "Outdoor athletics track", None),
    ("clip_track2.mp4", "Same track, different runner", "overlay_track2.gif"),
    ("clip_field.mp4", "Athletics field, wide shot", "overlay_field.gif"),
    ("clip_track3.mp4", "Same track, knee-level angle", "overlay_track3.gif"),
]


def _font(size, bold=False):
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fade(frames, fade_s=FADE_S, fps=FPS):
    n_fade = max(1, round(fade_s * fps))
    n_fade = min(n_fade, len(frames) // 3 or 1)
    out = [f.copy() for f in frames]
    for i in range(n_fade):
        alpha = i / n_fade
        out[i] = (out[i].astype(np.float32) * alpha).astype(np.uint8)
        j = len(out) - 1 - i
        out[j] = (out[j].astype(np.float32) * alpha).astype(np.uint8)
    return out


def _title_card(seconds=3.2):
    base = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(base)
    title_font = _font(72, bold=True)
    hook_font = _font(26)

    title = "STRIDELINE IN ACTION"
    w = draw.textlength(title, font=title_font)
    draw.text(((CANVAS_W - w) / 2, 280), title, font=title_font, fill=INK)

    hook = "the same pipeline, run end to end on four different real clips"
    w = draw.textlength(hook, font=hook_font)
    draw.text(((CANVAS_W - w) / 2, 380), hook, font=hook_font, fill=ACCENT)

    return _fade([np.array(base)] * round(seconds * FPS))


def _outro_card(seconds=3.6):
    base = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(base)
    head_font = _font(42, bold=True)
    sub_font = _font(20)
    foot_font = _font(18)

    head = "No retuning between clips"
    w = draw.textlength(head, font=head_font)
    draw.text(((CANVAS_W - w) / 2, 280), head, font=head_font, fill=INK)

    sub = "same model, same smoother, same certificate - on four different videos"
    w = draw.textlength(sub, font=sub_font)
    draw.text(((CANVAS_W - w) / 2, 340), sub, font=sub_font, fill=ACCENT)

    foot = "github.com/athar-usama/Strideline"
    fw = draw.textlength(foot, font=foot_font)
    draw.text(((CANVAS_W - fw) / 2, 640), foot, font=foot_font, fill=(150, 155, 162))

    return _fade([np.array(base)] * round(seconds * FPS))


def _process(clip_path):
    tracks = pose.extract_tracks(str(clip_path))
    dt = 1.0 / tracks.fps
    results = {}
    for side in ("left", "right"):
        results[side] = constrained_smooth(tracks.hip[side], tracks.knee[side], tracks.ankle[side], dt=dt)
    return tracks, results


def _captioned_clip_frames(clip_path, label, tracks, results):
    frames = viz.overlay_frames(str(clip_path), tracks, results, fps_out=FPS,
                                  resize_width=CANVAS_W, pad_frames=6)
    cap_font = _font(26, bold=True)
    sub_font = _font(16)
    out = []
    for f in frames:
        h = f.shape[0]
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        canvas.paste(Image.fromarray(f), (0, max(0, (CANVAS_H - CAPTION_H - h) // 2)))
        draw = ImageDraw.Draw(canvas)
        top = CANVAS_H - CAPTION_H
        draw.rectangle([0, top, CANVAS_W, CANVAS_H], fill=BG)
        draw.line([(0, top), (CANVAS_W, top)], fill=(40, 43, 48), width=1)
        draw.text((20, top + 10), label, font=cap_font, fill=INK)
        draw.text((20, top + 40), "red = raw pose        blue = Strideline-corrected",
                   font=sub_font, fill=(150, 155, 162))
        out.append(np.array(canvas))
    return _fade(out)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    segments = [_title_card()]

    for filename, label, gif_name in CLIPS:
        clip_path = CLIPS_DIR / filename
        print(f"processing {filename} ...")
        tracks, results = _process(clip_path)

        if gif_name:
            viz.render_overlay_gif(str(clip_path), tracks, results, FIG_DIR / gif_name)
            print(f"  wrote {FIG_DIR / gif_name}")

        segments.append(_captioned_clip_frames(clip_path, label, tracks, results))

    segments.append(_outro_card())

    writer = imageio.get_writer(str(OUT_PATH), fps=FPS, codec="libx264", quality=8,
                                 macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
    n_frames = 0
    for seg in segments:
        for frame in seg:
            writer.append_data(frame)
            n_frames += 1
    writer.close()

    print(f"wrote {OUT_PATH} ({n_frames} frames, {n_frames / FPS:.1f}s at {FPS}fps)")


if __name__ == "__main__":
    main()
