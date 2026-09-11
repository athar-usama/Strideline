"""Raw keypoint extraction from real video via a local COCO-17 pose model.

Uses Ultralytics YOLO11-pose, run entirely locally (no API key, no paid
gateway) so the whole pipeline is reproducible offline. COCO-17 keypoint
order: ... 11 left_hip, 12 right_hip, 13 left_knee, 14 right_knee,
15 left_ankle, 16 right_ankle. There is no toe/heel keypoint in COCO-17,
which is why ground contact is estimated from ankle kinematics rather than
a true foot-strike marker (see events.py and the README limitations).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COCO_HIP = {"left": 11, "right": 12}
COCO_KNEE = {"left": 13, "right": 14}
COCO_ANKLE = {"left": 15, "right": 16}


@dataclass
class RawTracks:
    fps: float
    hip: dict
    knee: dict
    ankle: dict
    n_frames: int


def extract_tracks(video_path, model_name="yolo11n-pose.pt", conf=0.3, device=None):
    """Run pose estimation over every frame of video_path and return raw
    (possibly NaN, where the person was not detected) keypoint tracks for
    both legs, in the same shape simulate.generate_sequence produces."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    tracks = {k: {"left": [], "right": []} for k in ("hip", "knee", "ankle")}

    results = model.predict(source=str(video_path), stream=True, conf=conf, verbose=False, device=device)
    n_frames = 0
    for r in results:
        n_frames += 1
        frame_kp = {k: {"left": (np.nan, np.nan), "right": (np.nan, np.nan)} for k in ("hip", "knee", "ankle")}

        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            if r.boxes is not None and len(r.boxes.conf) > 0:
                best = int(np.argmax(r.boxes.conf.cpu().numpy()))
            else:
                best = 0
            kp = r.keypoints.xy[best].cpu().numpy()
            kp_conf = r.keypoints.conf[best].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(kp))

            for name, idx_map in (("hip", COCO_HIP), ("knee", COCO_KNEE), ("ankle", COCO_ANKLE)):
                for side, idx in idx_map.items():
                    if kp_conf[idx] >= conf and np.any(kp[idx] != 0):
                        frame_kp[name][side] = (float(kp[idx][0]), float(kp[idx][1]))

        for name in ("hip", "knee", "ankle"):
            for side in ("left", "right"):
                tracks[name][side].append(frame_kp[name][side])

    out = {}
    for name in ("hip", "knee", "ankle"):
        out[name] = {side: np.array(tracks[name][side], dtype=float) for side in ("left", "right")}

    return RawTracks(fps=fps, hip=out["hip"], knee=out["knee"], ankle=out["ankle"], n_frames=n_frames)
