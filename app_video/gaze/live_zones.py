"""
live_zones.py
------------------------------------------------------------------
Show which zone the student is looking at, live, with a confidence score.
Class probabilities are averaged over recent frames so the label doesn't
flicker -- this is the coarse, robust signal AEye's cheating logic consumes.

Run:  python live_zones.py   (after calibrate_zones.py + train_zones.py)
Keys: q = quit
"""

import pickle
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from gaze_common import (create_landmarker, extract_features, ZONE_MODEL_PATH,
                         get_screen_size, make_fullscreen_window)

SCREEN_W, SCREEN_H = get_screen_size()

with open(ZONE_MODEL_PATH, "rb") as f:
    model = pickle.load(f)
classes = list(model.classes_)
print("Model loaded, zones:", classes)

COLORS = {"SCREEN": (0, 200, 0), "DESK": (0, 165, 255), "AWAY": (0, 0, 255)}

landmarker = create_landmarker()
cap = cv2.VideoCapture(0)
timestamp_ms = 0

proba_history = deque(maxlen=10)   # average probabilities over ~1/3 second

make_fullscreen_window("Zone")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    canvas = np.full((SCREEN_H, SCREEN_W, 3), 30, dtype=np.uint8)

    features = extract_features(result)   # None during blinks / no face
    if features is not None:
        proba = model.predict_proba([features])[0]
        proba_history.append(proba)
        avg = np.mean(proba_history, axis=0)     # temporal smoothing
        idx = int(np.argmax(avg))
        zone = classes[idx]
        conf = float(avg[idx])

        color = COLORS.get(zone, (255, 255, 255))
        cv2.putText(canvas, zone, (SCREEN_W // 2 - 220, SCREEN_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 6)
        cv2.putText(canvas, f"confidence: {conf:.0%}", (SCREEN_W // 2 - 220, SCREEN_H // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # per-zone probability readout
        y0 = SCREEN_H // 2 + 130
        for i, cname in enumerate(classes):
            cv2.putText(canvas, f"{cname}: {avg[i]:.0%}", (SCREEN_W // 2 - 220, y0 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.get(cname, (255, 255, 255)), 2)

    cv2.imshow("Zone", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
