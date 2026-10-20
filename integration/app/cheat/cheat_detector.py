# Screen area made with a Graham scan, and a check if the eyes are outside it.

import json
import math
import re

from paths import MODELS_DIR

# values the gaze model uses (same order as training)
FEATURES = ['h_ratio', 'v_openness', 'yaw', 'pitch', 'roll']
# how much to grow the outline (1.11 = exactly the screen edge, lower = stricter)
MARGIN = 1.20


def user_model_path(user_id):
    # File where the student's screen area is saved.
    safe = re.sub(r'[^A-Za-z0-9_-]+', '_', str(user_id)) if user_id else 'unknown'
    return MODELS_DIR / f'screen_area_{safe}.json'


def gaze_model_path(user_id):
    # File where the student's gaze model (scaler + ridge) is saved.
    safe = re.sub(r'[^A-Za-z0-9_-]+', '_', str(user_id)) if user_id else 'unknown'
    return MODELS_DIR / f'cheat_model_{safe}.joblib'


def cross(o, a, b):
    # Positive = left turn, negative = right turn.
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def graham_scan(points):
    # Find the outline around the points.
    points = sorted(set(points))
    # start at the lowest point
    start = min(points, key=lambda p: (p[1], p[0]))
    # sort the rest by angle
    rest = sorted((p for p in points if p != start),
                  key=lambda p: (math.atan2(p[1] - start[1], p[0] - start[0]),
                                 (p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2))
    hull = [start]
    for p in rest:
        # remove points that turn right
        while len(hull) > 1 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull


def grow(hull, margin=MARGIN):
    # Make the outline a bit bigger.
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    return [(cx + (x - cx) * margin, cy + (y - cy) * margin) for x, y in hull]


class CheatDetector:
    # Holds the screen area and checks the eyes against it.

    def __init__(self, hull=None):
        self.hull = hull
        self.scaler = None     # scales the features for the gaze model
        self.gaze_model = None # ridge regression, features -> screen spot

    @property
    def ready(self):
        # True if there is a screen area.
        return bool(self.hull) and len(self.hull) >= 3

    @classmethod
    def load(cls, user_id=None):
        # Load the saved screen area and gaze model (either can be missing).
        detector = cls()
        try:
            with open(user_model_path(user_id), encoding='utf-8') as f:
                data = json.load(f)
            detector.hull = [tuple(p) for p in data['hull']]
            print(f'[CHEAT] Loaded screen area for "{user_id}" ({len(detector.hull)} corners).')
        except Exception as exc:
            print(f'[CHEAT] No screen area for "{user_id}" ({exc}); detection off.')
        try:
            import joblib
            bundle = joblib.load(gaze_model_path(user_id))
            detector.scaler = bundle['scaler']
            detector.gaze_model = bundle.get('gaze')   # older models have none
            print(f'[CHEAT] Loaded gaze model for "{user_id}".')
        except Exception as exc:
            print(f'[CHEAT] No gaze model for "{user_id}" ({exc}); no gaze circle.')
        return detector

    def outside(self, h_ratio, v_openness):
        # True if the eyes are outside the screen area.
        if not self.ready:
            return False
        p = (h_ratio, v_openness)
        # inside = every edge turns left to the point
        return any(cross(self.hull[i], self.hull[(i + 1) % len(self.hull)], p) < 0
                   for i in range(len(self.hull)))

    def gaze(self, values):
        # Where on the screen the student looks, (x, y) 0 to 1. None = no gaze model.
        if self.gaze_model is None:
            return None
        x, y = self.gaze_model.predict(self.scaler.transform([values]))[0]
        return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))


if __name__ == '__main__':      # test: python -m cheat.cheat_detector
    square = graham_scan([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (1, 0)])
    assert len(square) == 4, square                    # middle points removed
    area = CheatDetector(grow(square, 1.0))
    assert not area.outside(1, 1) and not area.outside(0, 0)     # inside, corner
    assert area.outside(3, 1) and area.outside(1, -0.5)          # right, below
    assert not CheatDetector(grow(square, 2.0)).outside(3, 1)    # bigger outline covers it
    assert not CheatDetector().outside(99, 99)                   # no area = no flag
    print('hull ok:', square)
