import cv2
import mediapipe as mp
import time
import numpy as np
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL = Path(__file__).resolve().parent.parent / 'head_pose' / 'pose_landmarker_heavy.task'

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
]


def draw_landmarks(frame, landmarks, h, w):
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 3, (245, 66, 230), -1)
    for a, b in POSE_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            pt1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
            pt2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
            cv2.line(frame, pt1, pt2, (255, 255, 255), 1)


def extract_posture(landmarks):
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_wrist = landmarks[15]
    r_wrist = landmarks[16]
    return {
        'Left shoulder': f'{l_shoulder.x:.2f}, {l_shoulder.y:.2f}',
        'Right shoulder': f'{r_shoulder.x:.2f}, {r_shoulder.y:.2f}',
        'Left wrist': f'{l_wrist.x:.2f}, {l_wrist.y:.2f}',
        'Right wrist': f'{r_wrist.x:.2f}, {r_wrist.y:.2f}',
    }


class SideCameraWorker(QThread):
    frame_ready = Signal(QImage)
    stats_ready = Signal(dict)

    def __init__(self, camera_index=1, session_user_id=None, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.session_user_id = session_user_id
        self._running = False

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
        return {'Left shoulder': '--', 'Right shoulder': '--', 'Left wrist': '--', 'Right wrist': '--', 'Landmarks detected (/33)': '--'}

    def run(self):
        self._running = True
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
        cap = self._open_camera()
        if not cap.isOpened():
            stats = self._blank_stats()
            stats['Left shoulder'] = f'Camera {self.camera_index} unavailable'
            self.stats_ready.emit(stats)
            landmarker.close()
            return
        upd_interval = 0.5
        last_upd_time = 0.0
        display_stats = self._blank_stats()
        timestamp_ms = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33
            stats = self._blank_stats()
            if result.pose_landmarks:
                for pose_lm in result.pose_landmarks:
                    h, w = frame.shape[:2]
                    draw_landmarks(frame, pose_lm, h, w)
                    stats['Landmarks detected (/33)'] = str(len(pose_lm))
                    curr_time = time.time()
                    if curr_time - last_upd_time > upd_interval:
                        last_upd_time = curr_time
                        coords = extract_posture(pose_lm)
                        if coords is not None:
                            display_stats.update(coords)
                    stats.update(display_stats)
                    break
            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            qimg = QImage(rgb_out.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            self.stats_ready.emit(stats)
        cap.release()
        landmarker.close()
