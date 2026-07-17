"""
live_gaze.py
------------------------------------------------------------------
Load the trained model and show a red dot where you're looking, live.

Run:  python live_gaze.py    (after calibrate.py + train.py)
Keys: q = quit
"""

import pickle
import time

import cv2
import numpy as np
import mediapipe as mp

from gaze_common import (create_landmarker, extract_features, GAZE_MODEL_PATH,
                         get_screen_size, make_fullscreen_window, OneEuroFilter)

SCREEN_W, SCREEN_H = get_screen_size()

with open(GAZE_MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("Model loaded")

landmarker = create_landmarker()
cap = cv2.VideoCapture(0)
timestamp_ms = 0

# One adaptive filter per axis: smooth when still, responsive when moving.
# Tune here if needed: lower min_cutoff = smoother; higher beta = snappier.
filter_x = OneEuroFilter(min_cutoff=1.5, beta=0.5)
filter_y = OneEuroFilter(min_cutoff=1.5, beta=0.5)

last_sx, last_sy = None, None  # remembered so the dot holds still during blinks

make_fullscreen_window("Gaze")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    canvas = np.full((SCREEN_H, SCREEN_W, 3), 30, dtype=np.uint8)  # dark gray

    features = extract_features(result)
    if features is not None:                         # None during blinks / no face
        pred = model.predict([features])[0]          # raw output (can exceed 0..1)
        raw_px, raw_py = float(pred[0]), float(pred[1])
        px = float(np.clip(raw_px, 0, 1))
        py = float(np.clip(raw_py, 0, 1))

        now = time.perf_counter()
        last_sx = float(np.clip(filter_x(px, now), 0, 1))
        last_sy = float(np.clip(filter_y(py, now), 0, 1))

        # --- DEBUG readout: watch these as you look top / center / bottom ---
        # feature order: left_h,right_h,left_v,right_v,yaw,pitch,roll
        cv2.putText(canvas, f"raw pred: ({raw_px:+.2f}, {raw_py:+.2f})", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(canvas, f"vert feats: L={features[2]:+.3f} R={features[3]:+.3f}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(canvas, f"pitch={features[5]:+.1f}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if last_sx is not None:                          # draw last known position
        cv2.circle(canvas, (int(last_sx * SCREEN_W), int(last_sy * SCREEN_H)), 20, (0, 0, 255), -1)
        cv2.putText(canvas, f"gaze: ({last_sx:.2f}, {last_sy:.2f})", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Gaze", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
