"""
calibrate.py
------------------------------------------------------------------
SMOOTH-PURSUIT calibration. Instead of staring at a handful of fixed dots,
you follow ONE dot that glides smoothly around the whole screen. This records
HUNDREDS of distinct gaze positions (not a few clusters), which gives the
model a truly continuous feature -> screen mapping and much better live
accuracy than the sparse-grid version.

Run:   python calibrate.py
Keys:  SPACE = start     q = quit early

Tip: keep your head comfortable and fairly still, and track the moving dot
SMOOTHLY with your eyes. Don't jump ahead of it - just follow.
"""

import json

import cv2
import numpy as np
import mediapipe as mp

from gaze_common import (create_landmarker, extract_features, FEATURE_NAMES,
                         DATA_PATH, get_screen_size, make_fullscreen_window)

SCREEN_W, SCREEN_H = get_screen_size()
TOTAL_FRAMES = 1200   # ~40s at 30fps of continuous tracking
SWEEPS = 6            # number of horizontal sweeps from top to bottom
MARGIN = 0.08         # keep the dot just inside the screen edges


def dot_position(p):
    """p in [0,1] -> normalized (x, y). A smooth zigzag: x sweeps left<->right
    (triangle wave) while y drifts top->bottom, covering the whole screen with
    no jumps, so the eyes can track it continuously."""
    lo, hi = MARGIN, 1.0 - MARGIN
    y = lo + (hi - lo) * p
    sx = (p * SWEEPS) % 1.0
    tri = 1.0 - abs(2.0 * sx - 1.0)      # triangle wave: 0 -> 1 -> 0
    x = lo + (hi - lo) * tri
    return x, y


landmarker = create_landmarker()
cap = cv2.VideoCapture(0)
make_fullscreen_window("Calibration")

collected = []
timestamp_ms = 0


def wait_for_space():
    while True:
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        cv2.putText(canvas, "Follow the moving dot with your EYES (keep head still).",
                    (60, SCREEN_H // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(canvas, "Press SPACE to start   (q = quit)",
                    (60, SCREEN_H // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Calibration", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q'):
            return False


if wait_for_space():
    frame_i = 0
    while frame_i < TOTAL_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33

        p = frame_i / TOTAL_FRAMES
        target_x, target_y = dot_position(p)

        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        cv2.circle(canvas, (int(target_x * SCREEN_W), int(target_y * SCREEN_H)),
                   14, (0, 255, 0), -1)
        cv2.putText(canvas, f"{int(p * 100)}%", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        features = extract_features(result)   # None during blinks / no face
        if features is not None:
            collected.append({
                **{name: val for name, val in zip(FEATURE_NAMES, features)},
                "target_x": target_x,
                "target_y": target_y,
            })

        cv2.imshow("Calibration", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_i += 1

cap.release()
cv2.destroyAllWindows()

with open(DATA_PATH, "w") as f:
    json.dump(collected, f)
print(f"Saved {len(collected)} samples to {DATA_PATH}")
