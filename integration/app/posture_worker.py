import cv2
import mediapipe as mp
import time
import numpy as np
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from datetime import datetime
from posture_logger import PostureLogWriter

MP_DRAWING = mp.solutions.drawing_utils
MP_POSE = mp.solutions.pose
POSE_CONNECTIONS = MP_POSE.POSE_CONNECTIONS

# MediaPipe always gives back all 33 landmarks, even the ones it cannot see -
# hidden ones are just guesses. Only trust a landmark above this visibility.
MIN_VISIBILITY = 0.5

# How many bad frames in a row before we say the camera signal is lost.
LOST_FRAMES = 5


def create_pose_landmarker():
    return MP_POSE.Pose(min_detection_confidence=0.7,
                        min_tracking_confidence=0.7,
                        model_complexity=1,
                        smooth_landmarks=True,
                        )

def is_visible(landmark):
    return landmark.visibility >= MIN_VISIBILITY

def count_visible(lm):
    total = 0
    for point in lm:
        if is_visible(point):
            total += 1
    return total

def show(landmark):
    if not is_visible(landmark):
        return 'hidden'
    return f'{landmark.x:.2f}, {landmark.y:.2f}'

def db_values(landmark):
    # x and y are stored as NULL when the landmark was only guessed, but the
    # visibility always goes in - that is what says how far to trust the row.
    if not is_visible(landmark):
        return (None, None, landmark.visibility)
    return (landmark.x, landmark.y, landmark.visibility)

def extract_posture(result):
    if not result.pose_landmarks:
        return None, None
    lm = result.pose_landmarks.landmark
    l_shoulder = lm[MP_POSE.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = lm[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value]
    l_wrist = lm[MP_POSE.PoseLandmark.LEFT_WRIST.value]
    r_wrist = lm[MP_POSE.PoseLandmark.RIGHT_WRIST.value]

    coords = {
        'Left shoulder': show(l_shoulder),
        'Right shoulder': show(r_shoulder),
        'Left wrist': show(l_wrist),
        'Right wrist': show(r_wrist),
    }

    # The row is kept even when an arm is hidden: the coordinate columns are
    # nullable and each carries its own visibility score.
    raw = (db_values(l_shoulder) + db_values(r_shoulder)
           + db_values(l_wrist) + db_values(r_wrist))
    return coords, raw


class SideCameraWorker(QThread):
    frame_ready = Signal(QImage)
    stats_ready = Signal(dict)

    def __init__(self, camera_index=1, session_user_id=None, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.session_user_id = session_user_id
        self._running = False
        self._log_writer = PostureLogWriter()

    def stop(self):
        self._running = False

    def _open_camera(self):
        for _ in range(5):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
            time.sleep(0.3)
        return cap

    @staticmethod
    def _blank_stats():
        return {'Signal': '--', 'Left shoulder': '--', 'Right shoulder': '--', 'Left wrist': '--', 'Right wrist': '--', 'Landmarks detected (/33)': '--'}

    def run(self):
        self._running = True
        pose = create_pose_landmarker()
        self._log_writer.start()
        cap = self._open_camera()   # retries briefly; camera can be slow to free
        if not cap.isOpened():
            stats = self._blank_stats()
            stats['Left shoulder'] = f'Camera {self.camera_index} unavailable'
            self.stats_ready.emit(stats)
            pose.close()
            self._log_writer.stop()   # already started, so shut it down too
            self._log_writer.wait()
            return
        upd_interval = 0.5
        last_upd_time = 0.0
        display_stats = self._blank_stats()
        signal = 'LOST'
        lost_count = 0
        read_fails = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                # Camera unplugged or died - give up instead of looping forever.
                read_fails += 1
                if read_fails > 30:
                    break
                continue
            read_fails = 0
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            stats = self._blank_stats()
            good_frame = False
            if result.pose_landmarks:
                MP_DRAWING.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    POSE_CONNECTIONS,
                    MP_DRAWING.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=3),
                    MP_DRAWING.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                )
                lm = result.pose_landmarks.landmark
                # If we cannot even see a shoulder, we are basically blind.
                good_frame = (is_visible(lm[MP_POSE.PoseLandmark.LEFT_SHOULDER.value])
                              or is_visible(lm[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value]))

                curr_time = time.time()
                if curr_time - last_upd_time > upd_interval:
                    last_upd_time = curr_time
                    coords, raw = extract_posture(result)
                    if coords is not None:
                        display_stats.update(coords)
                    if raw is not None:
                        # Every sample with a pose goes in - hidden joints are
                        # NULL, not skipped, so the gaps stay visible in the data.
                        record = (self.session_user_id, datetime.now()) + raw + (
                            count_visible(lm),
                            1 if good_frame else 0,
                        )
                        self._log_writer.enqueue(record)
                stats.update(display_stats)
                stats['Landmarks detected (/33)'] = str(count_visible(lm))

            # A few bad frames in a row means lost, not just a flicker.
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
        pose.close()
        self._log_writer.stop()
        self._log_writer.wait()
