import sys
from pathlib import Path
import cv2
import mediapipe as mp
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
POSTURE_DIR = Path(__file__).resolve().parent.parent / 'posture'
sys.path.append(str(POSTURE_DIR))
from pose_common import create_pose_landmarker, extract_posture, POSE_CONNECTIONS

class SideCameraWorker(QThread):
    frame_ready = Signal(QImage)
    stats_ready = Signal(dict)

    def __init__(self, camera_index=1, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False

    def stop(self):
        self._running = False

    @staticmethod
    def _blank_stats():
        return {'Left shoulder': '--', 'Right shoulder': '--', 'Left wrist': '--', 'Right wrist': '--', 'Landmarks detected (/33)': '--'}

    @staticmethod
    def _draw_skeleton(frame, lm, w, h):
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in POSE_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 0, 0), 1)
        for x, y in pts:
            cv2.circle(frame, (x, y), 3, (245, 66, 230), -1)

    def run(self):
        self._running = True
        landmarker = create_pose_landmarker()
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            stats = self._blank_stats()
            stats['Left shoulder'] = f'Camera {self.camera_index} unavailable'
            self.stats_ready.emit(stats)
            return
        timestamp_ms = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33
            stats = self._blank_stats()
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                self._draw_skeleton(frame, lm, w, h)
                stats['Landmarks detected (/33)'] = str(len(lm))
                coords = extract_posture(result)
                if coords is not None:
                    stats.update(coords)
            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_out.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            self.stats_ready.emit(stats)
        cap.release()
