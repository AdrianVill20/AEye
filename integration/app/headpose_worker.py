import math
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
MODEL = Path(__file__).resolve().parent.parent / 'head_pose' / 'face_landmarker.task'

class HeadPoseWorker(QThread):
    frame_ready = Signal(QImage)
    stats_ready = Signal(dict)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO, output_facial_transformation_matrixes=True)
        landmarker = vision.FaceLandmarker.create_from_options(options)
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.stats_ready.emit({'Yaw': f'Camera {self.camera_index} unavailable', 'Pitch': '--', 'Roll': '--', 'Landmarks detected (/478)': '--'})
            return
        timestamp_ms = 0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33
            stats = {'Direction': '--', 'Yaw': '--', 'Pitch': '--', 'Roll': '--', 'Landmarks detected (/478)': '--'}
            if result.face_landmarks and result.facial_transformation_matrixes:
                stats['Landmarks detected (/478)'] = str(len(result.face_landmarks[0]))
                R = np.array(result.facial_transformation_matrixes[0])[:3, :3]
                sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
                yaw = math.degrees(math.atan2(-R[2, 0], sy))
                pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
                roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                stats['Yaw'], stats['Pitch'], stats['Roll'] = (f'{yaw:+.1f}°', f'{pitch:+.1f}°', f'{roll:+.1f}°')
                TH = 10
                vert = 'down' if pitch > TH else 'up' if pitch < -TH else ''
                horiz = 'right' if yaw > TH else 'left' if yaw < -TH else ''
                words = ' '.join((w for w in (vert, horiz) if w))
                direction = f'looking {words}' if words else 'looking center'
                stats['Direction'] = direction
                cv2.putText(frame, direction, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                cv2.putText(frame, f'Y {yaw:+.0f}  P {pitch:+.0f}  R {roll:+.0f}', (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_out.shape[:2]
            qimg = QImage(rgb_out.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            self.stats_ready.emit(stats)
        cap.release()
