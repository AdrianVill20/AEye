"""
calibrate_zones.py
------------------------------------------------------------------
AEye doesn't need exact (x, y) gaze -- it needs to know WHICH region the
student is looking at: the SCREEN, the DESK, or AWAY. That's a classification
problem, and it's far more robust than coordinate regression (coarse decisions
tolerate the head movement / distance / cheap-camera noise that wrecked the
precise version).

For each zone you look AROUND inside that zone for a few seconds while we
record your gaze features, labeled with the zone.

Run:   python calibrate_zones.py
Keys:  SPACE = start each zone     q = quit early
"""

import json

import cv2
import numpy as np
import mediapipe as mp

from gaze_common import (create_landmarker, extract_features, FEATURE_NAMES,
                         ZONES, ZONE_DATA_PATH, get_screen_size, make_fullscreen_window)

SCREEN_W, SCREEN_H = get_screen_size()
FRAMES_PER_ZONE = 200   # ~7 seconds of samples per zone

INSTRUCTIONS = {
    "SCREEN": "Look at the SCREEN. Slowly sweep your eyes over the WHOLE screen.",
    "DESK":   "Look DOWN at your desk / paper. Move eyes + head as if reading or writing.",
    "AWAY":   "Look AWAY from the screen - left, right, up, around the room.",
}

landmarker = create_landmarker()
cap = cv2.VideoCapture(0)
make_fullscreen_window("Zone Calibration")

collected = []
timestamp_ms = 0
quit_early = False


def wait_for_space(lines):
    """Full-screen prompt until SPACE (True) or q (sets quit_early, False)."""
    global quit_early
    while True:
        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        y = SCREEN_H // 2 - 40
        for line in lines:
            cv2.putText(canvas, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            y += 45
        cv2.putText(canvas, "Get ready, then press SPACE   (q = quit)", (60, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Zone Calibration", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q'):
            quit_early = True
            return False


def record_zone(zone):
    """Collect FRAMES_PER_ZONE good frames while the user looks around `zone`."""
    global timestamp_ms, quit_early
    n = 0
    while n < FRAMES_PER_ZONE:
        ret, frame = cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33

        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        cv2.putText(canvas, f"Recording zone: {zone}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(canvas, INSTRUCTIONS[zone], (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(canvas, f"{n}/{FRAMES_PER_ZONE}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        features = extract_features(result)   # None during blinks / no face
        if features is not None:
            collected.append({
                **{name: val for name, val in zip(FEATURE_NAMES, features)},
                "zone": zone,
            })
            n += 1

        cv2.imshow("Zone Calibration", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit_early = True
            return


for zone in ZONES:
    if not wait_for_space([f"Next zone:  {zone}", INSTRUCTIONS[zone]]):
        break
    record_zone(zone)
    if quit_early:
        break

cap.release()
cv2.destroyAllWindows()

with open(ZONE_DATA_PATH, "w") as f:
    json.dump(collected, f)
print(f"Saved {len(collected)} samples across {len(ZONES)} zones to {ZONE_DATA_PATH}")
