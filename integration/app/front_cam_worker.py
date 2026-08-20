import math
import time
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from datetime import datetime

MODEL = Path(__file__).resolve().parent.parent / 'head_pose' / 'face_landmarker.task'

RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (263, 362)

RIGHT_UPPER_LIDS = [159, 160, 161, 186]
RIGHT_LOWER_LIDS = [145, 144, 146, 153]
LEFT_UPPER_LIDS = [386, 374, 373, 382]
LEFT_LOWER_LIDS = [374, 373, 372, 380]

FONT = cv2.FONT_HERSHEY_SIMPLEX

H_LEFT_THRESH = 0.42
H_RIGHT_THRESH = 0.58
V_UP_THRESHOLD = 0.01
V_DOWN_THRESHOLD = -0.01
CALIB_FRAMES = 40
SMOOTHING = 0.3
HEAD_TH = 10


def _ema(prev, new, alpha=SMOOTHING):
    return prev * (1 - alpha) + new * alpha


def _avg_val(landmarks, indices, axis, size):
    vals = [getattr(landmarks[i], axis) * size for i in indices]
    return np.mean(vals)


def _to_px(landmarks, idx, w, h):
    return int(landmarks[idx].x * w), int(landmarks[idx].y * h)


def _pupil_from_eye_region(gray, contour_pts):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [contour_pts], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    x_min, y_min = contour_pts.min(axis=0)
    x_max, y_max = contour_pts.max(axis=0)
    crop = eye[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    _, thresh = cv2.threshold(crop, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < 20:
        return None

    M = cv2.moments(best)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"]) + x_min
    cy = int(M["m01"] / M["m00"]) + y_min
    return cx, cy


class FrontCamWorker(QThread):
    """Combined eye gaze + head pose on a single front camera.

    Runs FaceLandmarker once per frame, then extracts:
      - Eye gaze: pupil position (horizontal) + eye openness (vertical)
      - Head pose: yaw / pitch / roll from the facial transformation matrix
    """

    frame_ready = Signal(QImage)
    stats_ready = Signal(dict)
    record_ready = Signal(tuple) 

    def __init__(self, camera_index=0, session_user_id=None, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.session_user_id = session_user_id
        self._running = False
        self._prev_h = 0.5
        self._calib_openness = []
        self._baseline = None

    def stop(self):
        self._running = False

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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, timestamp_ms)
            timestamp_ms += 33

            stats = {
                'Gaze Direction': '--', 'H Ratio': '--', 'V Direction': '--',
                'Head Direction': '--', 'Yaw': '--', 'Pitch': '--', 'Roll': '--',
                'Landmarks': '--',
            }
            captured_at = datetime.now()

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                stats['Landmarks'] = str(len(lm))

                # --- Eye gaze ---
                right_contour = np.array([_to_px(lm, i, w, h) for i in RIGHT_EYE_CONTOUR], dtype=np.int32)
                left_contour = np.array([_to_px(lm, i, w, h) for i in LEFT_EYE_CONTOUR], dtype=np.int32)

                r_outer_px = lm[RIGHT_EYE_CORNERS[0]].x * w
                r_inner_px = lm[RIGHT_EYE_CORNERS[1]].x * w
                r_eye_width = r_inner_px - r_outer_px

                l_outer_px = lm[LEFT_EYE_CORNERS[0]].x * w
                l_inner_px = lm[LEFT_EYE_CORNERS[1]].x * w
                l_eye_width = l_inner_px - l_outer_px

                r_pupil = _pupil_from_eye_region(gray, right_contour)
                l_pupil = _pupil_from_eye_region(gray, left_contour)

                h_ratio = 0.5
                if r_pupil:
                    h_ratio = (r_pupil[0] - r_outer_px) / r_eye_width if r_eye_width != 0 else 0.5
                    cv2.circle(frame, r_pupil, 3, (0, 0, 255), -1)
                if l_pupil:
                    cv2.circle(frame, l_pupil, 3, (0, 0, 255), -1)

                self._prev_h = _ema(self._prev_h, h_ratio)

                if self._prev_h < H_LEFT_THRESH:
                    h_dir = "Left"
                elif self._prev_h > H_RIGHT_THRESH:
                    h_dir = "Right"
                else:
                    h_dir = "Center"

                # Eye openness
                r_upper_y = _avg_val(lm, RIGHT_UPPER_LIDS, 'y', h)
                r_lower_y = _avg_val(lm, RIGHT_LOWER_LIDS, 'y', h)
                l_upper_y = _avg_val(lm, LEFT_UPPER_LIDS, 'y', h)
                l_lower_y = _avg_val(lm, LEFT_LOWER_LIDS, 'y', h)

                r_open = (r_lower_y - r_upper_y) / r_eye_width if r_eye_width != 0 else 0.3
                l_open = (l_lower_y - l_upper_y) / l_eye_width if l_eye_width != 0 else 0.3
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
                cv2.polylines(frame, [right_contour], True, (0, 255, 0), 1)
                cv2.polylines(frame, [left_contour], True, (255, 0, 0), 1)

                stats['Gaze Direction'] = gaze_dir
                stats['H Ratio'] = f'{self._prev_h:.3f}'
                stats['V Direction'] = v_dir

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

                    cv2.putText(frame, head_dir, (10, 28), FONT, 0.7, (0, 200, 255), 2)
                    cv2.putText(frame, f'Y {yaw:+.0f} P {pitch:+.0f} R {roll:+.0f}', (10, 55), FONT, 0.6, (0, 255, 0), 1)

                # Combined direction bar
                color = (0, 255, 0) if h_dir == "Left" else (0, 0, 255) if h_dir == "Right" else (255, 0, 0)
                cv2.rectangle(frame, (0, 0), (w, 20), color, -1)
                cv2.putText(frame, gaze_dir, (10, 16), FONT, 0.5, (255, 255, 255), 1)

                # --- Build the DB record (raw values, not display strings) ---
                v_direction_db = 'calibrating' if v_dir == 'Calibrating...' else v_dir.lower()
                record = (
                    self.session_user_id,
                    captured_at,
                    h_dir.lower(),
                    v_direction_db,
                    float(self._prev_h),
                    float(avg_open),
                    None,                       # is_blinking — not detected yet
                    yaw, pitch, roll,
                    head_dir,
                    len(lm),
                    1,                          # signal_ok
                )
            else:
                # No face this frame — log a gap rather than silently dropping it
                record = (
                    self.session_user_id, captured_at,
                    None, None, None, None, None,
                    None, None, None, None,
                    None, 0,
                )

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_out.shape[:2]
            qimg = QImage(rgb_out.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            self.stats_ready.emit(stats)
            self.record_ready.emit(record)

        cap.release()
        landmarker.close()
