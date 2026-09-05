import math
import time
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

MODEL = Path(__file__).resolve().parent.parent / 'head_pose' / 'face_landmarker.task'

RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (263, 362)

RIGHT_UPPER_LIDS = [159, 160, 161, 186]
RIGHT_LOWER_LIDS = [145, 144, 146, 153]
LEFT_UPPER_LIDS = [386, 374, 373, 382]
LEFT_LOWER_LIDS = [374, 373, 372, 380]

H_LEFT_THRESH = 0.42
H_RIGHT_THRESH = 0.58
V_UP_THRESHOLD = 0.01
V_DOWN_THRESHOLD = -0.01
CALIB_FRAMES = 40
SMOOTHING = 0.3
HEAD_TH = 5

# --- Cheat-detection tuning ---
ALERT_SECONDS = 2.0      # a flag must persist this long before it counts
PITCH_DOWN_TH = 8        # pitch (deg) above this = looking down at desk / notes

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _ema(prev, new, alpha=SMOOTHING):
    return prev * (1 - alpha) + new * alpha


def _avg_val(landmarks, indices, axis, size):
    vals = [getattr(landmarks[i], axis) * size for i in indices]
    return np.mean(vals)


def _get_eye_data(landmarks, corners_idx, upper_lids, lower_lids, img_w, img_h):
    outer_x = landmarks[corners_idx[0]].x * img_w
    inner_x = landmarks[corners_idx[1]].x * img_w
    eye_width = inner_x - outer_x

    upper_y = _avg_val(landmarks, upper_lids, 'y', img_h)
    lower_y = _avg_val(landmarks, lower_lids, 'y', img_h)
    eye_height = lower_y - upper_y

    openness = eye_height / eye_width if eye_width != 0 else 0.3
    return openness


def _get_h_ratio(landmarks, pupil_px, corners_idx, img_w):
    outer_x = landmarks[corners_idx[0]].x * img_w
    inner_x = landmarks[corners_idx[1]].x * img_w
    eye_width = inner_x - outer_x
    return (pupil_px[0] - outer_x) / eye_width if eye_width != 0 else 0.5


class FrontCamWorker(QThread):
    """Combined iris-based eye gaze + head pose on a single front camera.

    Eye gaze:
      - Horizontal: iris center (landmark 468/473) relative to eye corners
      - Vertical: eye openness ratio calibrated against a baseline
    Head pose: yaw / pitch / roll from the facial transformation matrix
    """

    frame_ready = Signal(QImage)
    stats_ready = Signal(dict)
    record_ready = Signal(object)
    features_ready = Signal(object)   # per-frame features (for calibration JSON)
    cheat_detected = Signal(object)   # {'user','timestamp'} on a confirmed episode

    def __init__(self, camera_index=0, session_user_id=None, detect=False, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.session_user_id = session_user_id
        self.detect = detect              # run cheat detection this session?
        self._running = False
        self._prev_h = 0.5
        self._calib_openness = []
        self._baseline = None

    def stop(self):
        self._running = False

    def recalibrate(self):
        self._prev_h = 0.5
        self._calib_openness = []
        self._baseline = None

    def _open_camera(self):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            for attempt in range(3):
                cap = cv2.VideoCapture(self.camera_index, backend)
                if not cap.isOpened():
                    cap.release()
                    time.sleep(0.5)
                    continue
                time.sleep(0.5)
                for _ in range(10):
                    ret, _ = cap.read()
                    if ret:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f'[CAM] Camera {self.camera_index} opened ({w}x{h}) backend={backend}')
                        return cap
                    time.sleep(0.2)
                cap.release()
        return cv2.VideoCapture()

    def run(self):
        self._running = True
        self._prev_h = 0.5
        self._calib_openness = []
        self._baseline = None

        # Cheat-detection state (only used when detect=True). Loads this
        # student's personal model; if none exists yet, detection stays off.
        self._detector = None
        self._anom_since = None
        self._alert_active = False
        if self.detect:
            from cheat_detector import CheatDetector
            self._detector = CheatDetector.load(self.session_user_id)

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)

        cap = self._open_camera()
        if not cap.isOpened():
            self.stats_ready.emit({
                'Gaze Direction': f'Camera {self.camera_index} unavailable',
                'H Ratio': '--', 'V Direction': '--',
                'Head Direction': '--', 'Yaw': '--', 'Pitch': '--', 'Roll': '--',
                'Landmarks': '--',
            })
            landmarker.close()
            return

        timestamp_ms = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, timestamp_ms)
            timestamp_ms += 33

            stats = {
                'Gaze Direction': '--', 'H Ratio': '--', 'V Direction': '--',
                'Head Direction': '--', 'Yaw': '--', 'Pitch': '--', 'Roll': '--',
                'Landmarks': '--',
            }

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                stats['Landmarks'] = str(len(lm))

                # --- Eye gaze (iris-based, matching sample.py) ---
                r_cx = int(lm[RIGHT_IRIS_CENTER].x * w)
                r_cy = int(lm[RIGHT_IRIS_CENTER].y * h)
                l_cx = int(lm[LEFT_IRIS_CENTER].x * w)
                l_cy = int(lm[LEFT_IRIS_CENTER].y * h)

                cv2.circle(frame, (r_cx, r_cy), 2, (0, 0, 255), -1)
                cv2.circle(frame, (l_cx, l_cy), 2, (0, 0, 255), -1)

                r_open = _get_eye_data(lm, RIGHT_EYE_CORNERS,
                                       RIGHT_UPPER_LIDS, RIGHT_LOWER_LIDS, w, h)
                l_open = _get_eye_data(lm, LEFT_EYE_CORNERS,
                                       LEFT_UPPER_LIDS, LEFT_LOWER_LIDS, w, h)

                h_ratio = _get_h_ratio(lm, (r_cx, r_cy), RIGHT_EYE_CORNERS, w)
                self._prev_h = _ema(self._prev_h, h_ratio)

                if self._prev_h < H_LEFT_THRESH:
                    h_dir = "Left"
                elif self._prev_h > H_RIGHT_THRESH:
                    h_dir = "Right"
                else:
                    h_dir = "Center"

                avg_open = (r_open + l_open) / 2.0

                if self._baseline is None:
                    self._calib_openness.append(avg_open)
                    if len(self._calib_openness) >= CALIB_FRAMES:
                        self._baseline = np.mean(self._calib_openness)
                    v_dir = "Calibrating..."
                else:
                    diff = avg_open - self._baseline
                    if diff > V_UP_THRESHOLD:
                        v_dir = "Up"
                    elif diff < V_DOWN_THRESHOLD:
                        v_dir = "Down"
                    else:
                        v_dir = "Center"

                gaze_dir = f"{v_dir} / {h_dir}"

                stats['Gaze Direction'] = gaze_dir
                stats['H Ratio'] = f'h={self._prev_h:.3f} open={r_open:.4f}/{l_open:.4f}'
                stats['V Direction'] = v_dir

                color = (0, 255, 255)
                cv2.putText(frame, f"Gaze: {h_dir} / {v_dir}", (10, 28), FONT, 0.7, color, 2)
                if self._baseline is not None:
                    cv2.putText(frame, f"baseline={self._baseline:.4f}",
                                (10, 55), FONT, 0.5, (150, 150, 150), 1)

                # --- Head pose ---
                yaw = pitch = roll = None
                head_dir = None
                if result.facial_transformation_matrixes:
                    R = np.array(result.facial_transformation_matrixes[0])[:3, :3]
                    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
                    yaw = math.degrees(math.atan2(-R[2, 0], sy))
                    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
                    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))

                    stats['Yaw'] = f'{yaw:+.1f}'
                    stats['Pitch'] = f'{pitch:+.1f}'
                    stats['Roll'] = f'{roll:+.1f}'

                    vert = 'down' if pitch > HEAD_TH else 'up' if pitch < -HEAD_TH else ''
                    horiz = 'right' if yaw > HEAD_TH else 'left' if yaw < -HEAD_TH else ''
                    words = ' '.join((w for w in (vert, horiz) if w))
                    head_dir = f'looking {words}' if words else 'looking center'
                    stats['Head Direction'] = head_dir

                    bar_y = 80
                    cv2.putText(frame, head_dir, (10, bar_y), FONT, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f'Y {yaw:+.0f}  P {pitch:+.0f}  R {roll:+.0f}', (10, bar_y + 28), FONT, 0.6, (200, 200, 200), 1)

                record = (
                    self.session_user_id,
                    datetime.now(),
                    h_dir.lower(),
                    v_dir.lower() if v_dir != 'Calibrating...' else 'calibrating',
                    float(self._prev_h),
                    float(avg_open),
                    0,
                    float(yaw) if yaw is not None else None,
                    float(pitch) if pitch is not None else None,
                    float(roll) if roll is not None else None,
                    head_dir,
                    len(lm),
                    1,
                )
                self.record_ready.emit(record)

                # --- Per-frame features (calibration JSON) + cheat detection ---
                if yaw is not None:
                    feats = {
                        'h_ratio': float(self._prev_h),
                        'v_openness': float(avg_open),
                        'yaw': float(yaw),
                        'pitch': float(pitch),
                        'roll': float(roll),
                    }
                    self.features_ready.emit(feats)   # calibration collects these

                    # Detection = personalised MODEL + RULE gate + 2s TIME gate.
                    if self.detect and self._detector is not None and self._detector.ready:
                        unusual = self._detector.is_anomaly(
                            (feats['h_ratio'], feats['v_openness'],
                             feats['yaw'], feats['pitch'], feats['roll']))
                        # Rule gate: only count it if the EYES are off the screen -
                        # looking down (desk/notes) or gaze to a side. A head turn
                        # with eyes still on screen (thinking) is not cheating.
                        looking_down = (pitch > PITCH_DOWN_TH) or (v_dir == 'Down')
                        gaze_to_side = (h_dir != 'Center')
                        suspicious = unusual and (looking_down or gaze_to_side)

                        now = time.time()
                        if suspicious:
                            if self._anom_since is None:
                                self._anom_since = now
                            held = now - self._anom_since
                        else:
                            self._anom_since = None
                            held = 0.0

                        if held >= ALERT_SECONDS:
                            cv2.putText(frame, 'CHEATING DETECTED', (10, h - 20),
                                        FONT, 0.9, (0, 0, 255), 3)
                            # Fire ONE event per episode (rising edge), so MySQL
                            # gets one row per incident, not one per frame.
                            if not self._alert_active:
                                self._alert_active = True
                                self.cheat_detected.emit({
                                    'user': self.session_user_id,
                                    'timestamp': datetime.now(),
                                })
                        else:
                            self._alert_active = False

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_out.shape[:2]
            qimg = QImage(rgb_out.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            self.stats_ready.emit(stats)

        cap.release()
        landmarker.close()
