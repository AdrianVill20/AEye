import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from posture_logger import PostureLogWriter

MODEL = Path(__file__).resolve().parent.parent / 'head_pose' / 'pose_landmarker_heavy.task'

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16

MIN_VISIBILITY = 0.5
LOST_FRAMES = 5

# Setup mode: where the student's hands have to sit in the frame before the side
# camera counts as aimed properly. Normalized, so the zone holds at any webcam
# resolution. The student cannot move this box - only the tripod moves.
ZONE_LEFT = 0.15
ZONE_RIGHT = 0.85
ZONE_TOP = 0.62
ZONE_BOTTOM = 0.98

# How long the alignment has to hold before setup passes.
HOLD_SECONDS = 3.0


def is_visible(landmark):
    return landmark.visibility >= MIN_VISIBILITY


def count_visible(lm):
    return sum(1 for p in lm if is_visible(p))


def show(landmark):
    if not is_visible(landmark):
        return 'hidden'
    return f'{landmark.x:.2f}, {landmark.y:.2f}'


def db_values(landmark):
    if not is_visible(landmark):
        return (None, None, landmark.visibility)
    return (landmark.x, landmark.y, landmark.visibility)


def in_hand_zone(landmark):
    return (ZONE_LEFT <= landmark.x <= ZONE_RIGHT
            and ZONE_TOP <= landmark.y <= ZONE_BOTTOM)


def check_alignment(pose_lm):
    """Setup mode check. Returns (shoulders_ok, wrists_ok, message)."""
    l_sh = pose_lm[LEFT_SHOULDER]
    r_sh = pose_lm[RIGHT_SHOULDER]
    l_wr = pose_lm[LEFT_WRIST]
    r_wr = pose_lm[RIGHT_WRIST]

    # Shoulders only have to be somewhere in the upper half, not in a box.
    # MediaPipe needs the torso in frame to track the wrists at all, and
    # signal_ok is shoulder visibility, so a desk-only view logs nothing usable.
    shoulders_ok = (is_visible(l_sh) and is_visible(r_sh)
                    and l_sh.y < 0.5 and r_sh.y < 0.5)
    wrists_ok = (is_visible(l_wr) and is_visible(r_wr)
                 and in_hand_zone(l_wr) and in_hand_zone(r_wr))

    if not shoulders_ok and not wrists_ok:
        message = 'Sit facing the screen and move the camera back'
    elif not shoulders_ok:
        message = 'Shoulders out of frame - raise the camera'
    elif not wrists_ok:
        message = 'Hands not in the zone - tilt the camera down'
    else:
        message = 'Hold still'
    return shoulders_ok, wrists_ok, message


def draw_setup_guide(frame, h, w, wrists_ok, aligned, message):
    box_color = (0, 255, 0) if wrists_ok else (0, 0, 255)
    text_color = (0, 255, 0) if aligned else (0, 0, 255)
    pt1 = (int(ZONE_LEFT * w), int(ZONE_TOP * h))
    pt2 = (int(ZONE_RIGHT * w), int(ZONE_BOTTOM * h))
    cv2.rectangle(frame, pt1, pt2, box_color, 2)
    cv2.putText(frame, 'HANDS HERE', (pt1[0], pt1[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    cv2.putText(frame, message, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


def draw_landmarks(frame, landmarks, h, w):
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 3, (245, 66, 230), -1)
    for a, b in POSE_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            pt1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
            pt2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
            cv2.line(frame, pt1, pt2, (255, 255, 255), 1)


def extract_posture(pose_lm):
    l_shoulder = pose_lm[LEFT_SHOULDER]
    r_shoulder = pose_lm[RIGHT_SHOULDER]
    l_wrist = pose_lm[LEFT_WRIST]
    r_wrist = pose_lm[RIGHT_WRIST]

    coords = {
        'Left shoulder': show(l_shoulder),
        'Right shoulder': show(r_shoulder),
        'Left wrist': show(l_wrist),
        'Right wrist': show(r_wrist),
    }

    raw = (db_values(l_shoulder) + db_values(r_shoulder)
           + db_values(l_wrist) + db_values(r_wrist))
    return coords, raw


class SideCameraWorker(QThread):
    frame_ready = Signal(QImage)
    stats_ready = Signal(dict)
    setup_ready = Signal(bool)   # setup mode only: the camera is aimed correctly

    def __init__(self, camera_index=1, session_user_id=None, log_to_db=False,
                 setup_mode=False, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.session_user_id = session_user_id
        self.setup_mode = setup_mode
        self._running = False
        # Only the Front + Side Cam tab logs to MySQL. The other tabs are for
        # testing the camera, so they run without a writer and save nothing.
        # Setup mode never logs either - it runs before the exam starts.
        if setup_mode:
            log_to_db = False
        self._log_writer = PostureLogWriter() if log_to_db else None

    def stop(self):
        self._running = False

    def _stop_log_writer(self):
        # Nothing to do on the tabs that run without a writer.
        if self._log_writer is not None:
            self._log_writer.stop()
            self._log_writer.wait()

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

    @staticmethod
    def _blank_stats():
        return {'Signal': '--', 'Left shoulder': '--', 'Right shoulder': '--', 'Left wrist': '--', 'Right wrist': '--', 'Landmarks detected (/33)': '--'}

    def run(self):
        self._running = True
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
        if self._log_writer is not None:
            self._log_writer.start()

        cap = self._open_camera()
        if not cap.isOpened():
            stats = self._blank_stats()
            stats['Left shoulder'] = f'Camera {self.camera_index} unavailable'
            self.stats_ready.emit(stats)
            landmarker.close()
            self._stop_log_writer()
            return

        upd_interval = 0.5
        last_upd_time = 0.0
        display_stats = self._blank_stats()
        signal = 'LOST'
        lost_count = 0
        read_fails = 0
        timestamp_ms = 0
        hold_start = None   # setup mode: when the alignment first went good

        while self._running:
            ret, frame = cap.read()
            if not ret:
                read_fails += 1
                if read_fails > 30:
                    break
                continue
            read_fails = 0

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33

            if self.setup_mode:
                h, w = frame.shape[:2]
                shoulders_ok = False
                wrists_ok = False
                message = 'No person detected - check the camera view'

                if result.pose_landmarks:
                    pose_lm = result.pose_landmarks[0]
                    draw_landmarks(frame, pose_lm, h, w)
                    shoulders_ok, wrists_ok, message = check_alignment(pose_lm)

                aligned = shoulders_ok and wrists_ok
                if aligned:
                    if hold_start is None:
                        hold_start = time.time()
                    left = HOLD_SECONDS - (time.time() - hold_start)
                    message = f'Hold still - {max(0.0, left):.1f}s'
                else:
                    hold_start = None
                    left = HOLD_SECONDS

                passed = aligned and left <= 0
                if passed:
                    message = 'ALIGNED'
                draw_setup_guide(frame, h, w, wrists_ok, aligned, message)

                rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb_out.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                self.frame_ready.emit(qimg)
                self.stats_ready.emit({'Setup': message})

                if passed:
                    # Show the ALIGNED frame, then end the loop - the student
                    # presses Start themselves after this.
                    self.setup_ready.emit(True)
                    break
                continue

            stats = self._blank_stats()
            good_frame = False

            if result.pose_landmarks:
                for pose_lm in result.pose_landmarks:
                    h, w = frame.shape[:2]
                    draw_landmarks(frame, pose_lm, h, w)
                    good_frame = (is_visible(pose_lm[LEFT_SHOULDER])
                                  or is_visible(pose_lm[RIGHT_SHOULDER]))

                    curr_time = time.time()
                    if curr_time - last_upd_time > upd_interval:
                        last_upd_time = curr_time
                        coords, raw = extract_posture(pose_lm)
                        if coords is not None:
                            display_stats.update(coords)
                        if raw is not None and self._log_writer is not None:
                            # Hidden joints go in as NULL rather than being
                            # skipped, so the gaps stay visible in the data.
                            record = (self.session_user_id, datetime.now()) + raw + (
                                count_visible(pose_lm),
                                1 if good_frame else 0,
                            )
                            self._log_writer.enqueue(record)
                    stats.update(display_stats)
                    stats['Landmarks detected (/33)'] = str(count_visible(pose_lm))
                    break

            if good_frame:
                lost_count = 0
                signal = 'OK'
            else:
                lost_count += 1
                if lost_count > LOST_FRAMES:
                    signal = 'LOST'
                    display_stats = self._blank_stats()
            stats['Signal'] = signal

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            qimg = QImage(rgb_out.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            self.stats_ready.emit(stats)

        cap.release()
        landmarker.close()
        self._stop_log_writer()
