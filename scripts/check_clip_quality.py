"""Quick numeric screen for whether a candidate clip is clean enough to use:
detection rate, and how much the raw pose keypoints imply an impossible,
jittering limb length (before any smoothing). Compare the numbers against a
known-good clip (running.mp4) before trusting a new one - this is exactly
the same self-supervised consistency signal the smoother itself reports
(see smoother.kinematic_consistency_violation), just used here as a
pre-flight gate instead of a post-hoc metric.

    python scripts/check_clip_quality.py assets/clips/candidate.mp4
"""
from __future__ import annotations

import argparse

import numpy as np

from strideline import pose
from strideline.smoother import kinematic_consistency_violation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    tracks = pose.extract_tracks(args.video)
    print(f"{args.video}: {tracks.n_frames} frames at {tracks.fps:.1f} fps")

    for side in ("left", "right"):
        hip, knee, ankle = tracks.hip[side], tracks.knee[side], tracks.ankle[side]
        detected = float(np.mean(np.all(np.isfinite(knee), axis=-1)))
        thigh_cv = kinematic_consistency_violation(hip, knee, 1.0)
        shank_cv = kinematic_consistency_violation(knee, ankle, 1.0)
        print(f"  {side}: detection rate {detected:.0%}, "
              f"thigh-length CV {thigh_cv:.1f}%, shank-length CV {shank_cv:.1f}%")


if __name__ == "__main__":
    main()
