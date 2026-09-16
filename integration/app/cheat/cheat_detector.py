"""The student's "eyes on the screen" area, as a convex hull.

Calibration records two eye values while the student looks at dots covering
the whole screen:
    h_ratio     iris left / right inside the eye
    v_openness  how open the eye is (drops when looking down)
Each value is corrected for the head (x = h_ratio + kx * yaw,
y = v_openness + ky * pitch), so turning the head towards notes counts the same
as moving the eyes there. kx / ky come from the head-movement dots.

The convex hull of those points - found with a Graham scan - is the shape of
"looking at the screen" for THAT student. During the exam, a reading outside
the shape means the student is looking somewhere off the screen.

The model file is plain JSON: the hull corners plus kx and ky.
"""

import json
import math
import re

from paths import MODELS_DIR

FEATURES = ['h_ratio', 'v_openness']   # the two values the hull is built from
# The dots sit 5% in from the screen edge, so growing their hull by 1.11 reaches
# the real edge; 1.2 adds a little room. Tested on real runs: on-screen frames
# stayed outside at most 0.4 s (the alert needs 2 s).
# Notes beside the screen missed? Lower it (1.11 = exactly the screen edge).
MARGIN = 1.20


def user_model_path(user_id):
    """Per-student area file, e.g. models/screen_area_ichoy.json."""
    safe = re.sub(r'[^A-Za-z0-9_-]+', '_', str(user_id)) if user_id else 'unknown'
    return MODELS_DIR / f'screen_area_{safe}.json'


def cross(o, a, b):
    """Positive if o -> a -> b turns left, negative if it turns right."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def graham_scan(points):
    """Convex hull of 2D points, counter-clockwise (Graham scan): start at the
    lowest point, sort the rest by the angle around it, then walk through them
    and drop any point that makes a right turn."""
    points = sorted(set(points))
    start = min(points, key=lambda p: (p[1], p[0]))
    rest = sorted((p for p in points if p != start),
                  key=lambda p: (math.atan2(p[1] - start[1], p[0] - start[0]),
                                 (p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2))
    hull = [start]
    for p in rest:
        while len(hull) > 1 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull


def grow(hull, margin=MARGIN):
    """Push every corner away from the middle, so readings just outside the
    recorded area (camera noise, sitting slightly differently) are still fine."""
    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    return [(cx + (x - cx) * margin, cy + (y - cy) * margin) for x, y in hull]


class CheatDetector:
    """Loads the student's screen area and says whether a frame is outside it.

    Fail-safe: if the student has no area yet, load() returns a 'not ready'
    detector that never flags, so the camera keeps working with detection off.
    """

    def __init__(self, hull=None, kx=0.0, ky=0.0):
        self.hull = hull
        self.kx = kx
        self.ky = ky

    @property
    def ready(self):
        return bool(self.hull) and len(self.hull) >= 3

    @classmethod
    def load(cls, user_id=None):
        try:
            with open(user_model_path(user_id), encoding='utf-8') as f:
                data = json.load(f)
            hull = [tuple(p) for p in data['hull']]
            print(f'[CHEAT] Loaded screen area for "{user_id}" ({len(hull)} corners).')
            return cls(hull, data.get('kx', 0.0), data.get('ky', 0.0))
        except Exception as exc:
            print(f'[CHEAT] No screen area for "{user_id}" ({exc}); detection off.')
            return cls()

    def point(self, h_ratio, v_openness, yaw, pitch):
        """Eye values corrected for the head - the point that goes in the hull."""
        return (h_ratio + self.kx * yaw, v_openness + self.ky * pitch)

    def outside(self, h_ratio, v_openness, yaw=0.0, pitch=0.0):
        """True = the student looks outside their screen area. Inside a
        counter-clockwise hull every edge turns left towards the point."""
        if not self.ready:
            return False
        p = self.point(h_ratio, v_openness, yaw, pitch)
        return any(cross(self.hull[i], self.hull[(i + 1) % len(self.hull)], p) < 0
                   for i in range(len(self.hull)))


if __name__ == '__main__':      # self-check: python -m cheat.cheat_detector
    square = graham_scan([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (1, 0)])
    assert len(square) == 4, square                    # inside / edge points dropped
    area = CheatDetector(grow(square, 1.0))
    assert not area.outside(1, 1) and not area.outside(0, 0)     # inside, corner
    assert area.outside(3, 1) and area.outside(1, -0.5)          # right, below
    assert not CheatDetector(grow(square, 2.0)).outside(3, 1)    # margin covers it
    assert not CheatDetector().outside(99, 99)                   # no area = never flags
    turned = CheatDetector(grow(square, 1.0), kx=0.1)
    assert not turned.outside(1, 1) and turned.outside(1, 1, yaw=15)   # head turned away
    print('hull ok:', square)
