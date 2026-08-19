"""
pose_common.py
------------------------------------------------------------------
Allain's posture logic, PORTED to the MediaPipe Tasks API.

His original posture.py uses `mp.solutions.pose`, which mediapipe 0.10.35
deleted — so it can't run in the environment Christian's gaze needs. Nothing
about his APPROACH changes here: it still reads the SAME four body points from
the 33-point pose (both shoulders + both wrists) and reports their normalized
coordinates. Only the mediapipe call style is updated (mp.solutions ->
PoseLandmarker), mirroring gaze_common.py's structure.
"""

from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Resolve the model next to this file, so it loads no matter the working dir.
HERE = Path(__file__).resolve().parent
POSE_MODEL_PATH = HERE / "pose_landmarker.task"

# Indices on MediaPipe's 33-point body topology — the SAME numbers Allain's
# `mp_pose.PoseLandmark.LEFT_SHOULDER.value` etc. resolved to. The Tasks API
# doesn't ship the named enum, so we spell them out.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16

# Standard pose skeleton edges, used only to draw the overlay (mp.solutions
# supplied this set before; we list it here since that module is gone).
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def create_pose_landmarker():
    """Build a PoseLandmarker in VIDEO mode — the Tasks-API equivalent of
    Allain's `mp_pose.Pose(...)` context."""
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_posture(result):
    """Turn one PoseLandmarker result into Allain's four tracked points.
    Returns a dict of field name -> 'x, y' text, or None if no body is found
    (so the caller can just skip the frame, like his try/except did)."""
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]
    return {
        "Left shoulder":  f"{lm[LEFT_SHOULDER].x:.2f}, {lm[LEFT_SHOULDER].y:.2f}",
        "Right shoulder": f"{lm[RIGHT_SHOULDER].x:.2f}, {lm[RIGHT_SHOULDER].y:.2f}",
        "Left wrist":     f"{lm[LEFT_WRIST].x:.2f}, {lm[LEFT_WRIST].y:.2f}",
        "Right wrist":    f"{lm[RIGHT_WRIST].x:.2f}, {lm[RIGHT_WRIST].y:.2f}",
    }
