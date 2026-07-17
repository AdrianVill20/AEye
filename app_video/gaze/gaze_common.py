"""
gaze_common.py
------------------------------------------------------------------
Shared pieces for the AEye eye-gaze pipeline:
  - building the MediaPipe FaceLandmarker
  - turning face landmarks into a FIXED list of gaze "features"

calibrate.py, train.py and live_gaze.py all import from here so the
feature definition lives in ONE place. If calibration and live prediction
ever computed features differently, the model would be garbage. Keep it
shared and there is only one source of truth.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Every path is relative to THIS file, not your current working directory.
# That is why the "FileNotFound: face_landmarker.task" problem can't come
# back: no matter which folder you run a script from, these still resolve.
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "face_landmarker.task"
DATA_PATH = HERE / "calibration_data.json"      # coordinate-regression data
GAZE_MODEL_PATH = HERE / "gaze_model.pkl"        # coordinate-regression model
ZONE_DATA_PATH = HERE / "zone_data.json"         # zone-classification data
ZONE_MODEL_PATH = HERE / "zone_model.pkl"        # zone-classification model

# The attention regions AEye classifies. Order is not important.
ZONES = ["SCREEN", "DESK", "AWAY"]

# --- landmark indices on the 478-point mesh (with iris) ---
# "LEFT"/"RIGHT" = the subject's own left / right eye.
LEFT_IRIS,  LEFT_OUTER,  LEFT_INNER,  LEFT_TOP,  LEFT_BOTTOM  = 468, 33,  133, 159, 145
RIGHT_IRIS, RIGHT_INNER, RIGHT_OUTER, RIGHT_TOP, RIGHT_BOTTOM = 473, 362, 263, 386, 374

# Fixed feature order, shared by calibrate / train / live.
# NOTE: head TRANSLATION (tx,ty,tz) was removed. With single-pose calibration
# it barely varies, so StandardScaler amplifies tiny live head shifts into huge
# values -> the model extrapolates and the dot sticks to screen edges. Rotation
# (yaw/pitch/roll) is kept because it varies naturally and stays well-behaved.
FEATURE_NAMES = ["left_h", "right_h", "left_v", "right_v", "yaw", "pitch", "roll"]


class OneEuroFilter:
    """Adaptive low-pass filter for a single value over time.
    - When the value is steady, it smooths aggressively -> jitter disappears.
    - When the value moves fast, it barely smooths -> no lag.
    Tuning: lower `min_cutoff` = smoother (but laggier) when still;
            higher `beta`      = snappier when moving fast.
    """

    def __init__(self, min_cutoff=1.5, beta=0.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:            # first sample: nothing to filter yet
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev     # smoothed speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)  # move fast -> filter less
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def get_screen_size():
    """Real monitor resolution on Windows (DPI-aware), so calibration dots
    and the live gaze dot use the FULL physical screen. Falls back to 1280x720."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()          # report true pixels on high-DPI displays
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1280, 720


def make_fullscreen_window(name):
    """Create a borderless fullscreen OpenCV window. Fullscreen matters:
    it forces your eyes across the whole screen, which is what makes the
    calibration mapping match live use."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def create_landmarker():
    """Build a FaceLandmarker in VIDEO mode (tracks the face across frames)."""
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,               # expression scores (used later)
        output_facial_transformation_matrixes=True,  # head pose (used below)
    )
    return vision.FaceLandmarker.create_from_options(options)


def _h_ratio(lm, iris, a, b):
    """Horizontal iris position between two eye corners, 0..1 (scale-invariant)."""
    return (lm[iris].x - lm[a].x) / (lm[b].x - lm[a].x)


def _v_ratio(lm, iris, corner_a, corner_b):
    """Vertical iris position relative to the EYE-CORNER line, normalized by
    eye width. Corners are anchored to the eye socket, so (unlike the eyelids)
    they stay put when you glance up/down -> a reliable vertical gaze signal.
    Positive = iris below the corner line (looking down)."""
    eye_center_y = (lm[corner_a].y + lm[corner_b].y) / 2.0
    eye_width = abs(lm[corner_b].x - lm[corner_a].x)
    return (lm[iris].y - eye_center_y) / eye_width


def _openness(lm, top, bottom, corner_a, corner_b):
    """Eyelid gap divided by eye width. ~0.3+ open, drops toward 0 when closed.
    Normalized by width so it means the same at any distance."""
    return abs(lm[bottom].y - lm[top].y) / abs(lm[corner_b].x - lm[corner_a].x)


# Eyes below this openness are treated as closed (blink) and the frame skipped.
# Kept low so it rejects real blinks without penalizing naturally squinty eyes
# (a fairness concern). Ideally personalized per student from their calibration.
BLINK_THRESHOLD = 0.10


def _head_angles(matrix):
    """Yaw / pitch / roll (degrees) from the 4x4 facial transformation matrix.
    The top-left 3x3 is the rotation; we decompose it into Euler angles."""
    R = np.array(matrix)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal-lock fallback
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _head_translation(matrix):
    """Head position (x, y) and distance (z) from the camera, read from the
    4th column of the 4x4 transformation matrix. Giving these to the model is
    what lets it stay accurate when you lean in/out or shift in your seat."""
    t = np.array(matrix)[:3, 3]
    return float(t[0]), float(t[1]), float(t[2])


def extract_features(result):
    """Turn one FaceLandmarker result into the fixed feature list.
    Returns None if there is no face (or no head-pose matrix) this frame,
    so callers can simply skip bad frames."""
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]

    # Blink rejection: a closed/blinking eye gives garbage iris landmarks that
    # otherwise make the gaze dot jump. If EITHER eye is closed, skip the frame.
    left_open  = _openness(lm, LEFT_TOP,  LEFT_BOTTOM,  LEFT_OUTER, LEFT_INNER)
    right_open = _openness(lm, RIGHT_TOP, RIGHT_BOTTOM, RIGHT_INNER, RIGHT_OUTER)
    if min(left_open, right_open) < BLINK_THRESHOLD:
        return None

    # Corner order chosen so BOTH eyes increase in the same screen direction.
    left_h  = _h_ratio(lm, LEFT_IRIS,  LEFT_OUTER, LEFT_INNER)
    right_h = _h_ratio(lm, RIGHT_IRIS, RIGHT_INNER, RIGHT_OUTER)
    left_v  = _v_ratio(lm, LEFT_IRIS,  LEFT_OUTER, LEFT_INNER)
    right_v = _v_ratio(lm, RIGHT_IRIS, RIGHT_INNER, RIGHT_OUTER)

    if not result.facial_transformation_matrixes:
        return None
    matrix = result.facial_transformation_matrixes[0]
    yaw, pitch, roll = _head_angles(matrix)

    return [left_h, right_h, left_v, right_v, yaw, pitch, roll]
