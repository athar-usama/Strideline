"""Run the real pipeline (pose extraction + constrained smoothing, both legs)
on several different real running clips, and compile the resulting skeleton
overlays into one video: assets/action_reel.mp4. This is real Strideline
output on real footage end to end - not a slideshow of the README's static
figures. A compact GIF of the same compiled footage (assets/figures/overlay_demo.gif)
is the README's hero image. Neither carries any burned-in text: the red/blue
legend lives in the README's own caption table instead.

    python scripts/build_action_reel.py
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from strideline import pose, viz
from strideline.smoother import constrained_smooth

ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = ROOT / "assets" / "clips"
FIG_DIR = ROOT / "assets" / "figures"
OUT_PATH = ROOT / "assets" / "action_reel.mp4"
GIF_PATH = FIG_DIR / "overlay_demo.gif"

CANVAS_W = 1280
FPS = 24
FADE_S = 0.25

GIF_FPS = 10.0
GIF_WIDTH = 480

CLIPS = [
    "running.mp4",
    "clip_track2.mp4",
    "clip_field.mp4",
    "clip_track3.mp4",
]


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


def _process(clip_path):
    tracks = pose.extract_tracks(str(clip_path))
    dt = 1.0 / tracks.fps
    results = {}
    for side in ("left", "right"):
        results[side] = constrained_smooth(tracks.hip[side], tracks.knee[side], tracks.ankle[side], dt=dt)
    return tracks, results


def _clip_frames(clip_path, tracks, results):
    frames = viz.overlay_frames(str(clip_path), tracks, results, fps_out=FPS,
                                 resize_width=CANVAS_W, pad_frames=6)
    return _fade(frames)


def make_gif_from_video(video_path, out_path, fps_out=GIF_FPS, width=GIF_WIDTH):
    """Build the README hero GIF from the already-rendered action reel MP4,
    rather than from an in-memory frame list - decouples GIF size/fps tuning
    from the (slow) pose extraction pass, and lets the GIF be regenerated on
    its own from the committed MP4.

    Quantizes once against a single palette built from *all* sampled frames
    pooled together (via a temporary strip image), rather than a fresh
    256-color palette per frame: an independent per-frame palette biases
    toward whatever's most populous in that one frame (grass, track, sky),
    which can leave no palette slot close to the thin, comparatively rare
    red/blue skeleton lines, so they get pulled toward a muddy nearby color
    ("brownish"). A single palette shared across the whole clip reserves
    slots for the persistent skeleton colors instead of recomputing a fresh,
    equally biased guess every frame.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    step = max(1, round(src_fps / fps_out))
    src_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width
    src_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or width
    height = round(src_h * (width / src_w))

    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).resize((width, height), Image.LANCZOS))
        i += 1
    cap.release()

    # Build one shared adaptive palette from a strip of sampled frames, then
    # quantize every frame against that fixed palette (no per-frame dithering
    # noise between frames, and the rare accent colors get their own slots
    # because the strip is wide enough for the optimizer to keep them).
    sample_stride = max(1, len(frames) // 24)
    strip_frames = frames[::sample_stride]
    strip = Image.new("RGB", (width * len(strip_frames), height))
    for idx, f in enumerate(strip_frames):
        strip.paste(f, (idx * width, 0))
    palette_img = strip.quantize(colors=255, method=Image.FASTOCTREE)

    images = [f.quantize(palette=palette_img, dither=Image.FLOYDSTEINBERG) for f in frames]

    duration_ms = round(1000 / fps_out)
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration_ms,
                   loop=0, optimize=True)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = []

    for filename in CLIPS:
        clip_path = CLIPS_DIR / filename
        print(f"processing {filename} ...")
        tracks, results = _process(clip_path)
        all_frames.extend(_clip_frames(clip_path, tracks, results))

    writer = imageio.get_writer(str(OUT_PATH), fps=FPS, codec="libx264", quality=8,
                                 macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()
    print(f"wrote {OUT_PATH} ({len(all_frames)} frames, {len(all_frames) / FPS:.1f}s at {FPS}fps)")

    make_gif_from_video(OUT_PATH, GIF_PATH)
    print(f"wrote {GIF_PATH}")


if __name__ == "__main__":
    main()
