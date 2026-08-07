import sys
from pathlib import Path
import cv2
import dlib
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
REVISED_DIR = Path(__file__).resolve().parent.parent / 'eye_gaze'
sys.path.append(str(REVISED_DIR))
from gaze_core import blink, gaze_ratio
PREDICTOR_PATH = REVISED_DIR / 'shape_predictor_68_face_landmarks.dat'
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
FONT = cv2.FONT_HERSHEY_COMPLEX

class GazeWorker(QThread):
    frame_ready = Signal(QImage)
    stats_ready = Signal(dict)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(PREDICTOR_PATH))

    def stop(self):
        self._running = False

    @staticmethod
    def _blank_stats():
        return {'Direction': '--', 'Gaze ratio': '--', 'Blink': '--'}

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            stats = self._blank_stats()
            stats['Direction'] = f'Camera {self.camera_index} unavailable'
            self.stats_ready.emit(stats)
            return
        while self._running:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            stats = self._blank_stats()
            for face in faces:
                landmarks = self.predictor(gray, face)
                left_div = blink(LEFT_EYE, landmarks)
                right_div = blink(RIGHT_EYE, landmarks)
                blinking = right_div and left_div > 5.5
                gaze = (gaze_ratio(RIGHT_EYE, landmarks, frame, gray) + gaze_ratio(LEFT_EYE, landmarks, frame, gray)) / 2
                if gaze < 0.45:
                    direction, color = ('looking left', (0, 0, 255))
                elif 0.45 < gaze < 2:
                    direction, color = ('looking center', (255, 0, 0))
                else:
                    direction, color = ('looking right', (0, 255, 0))
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 40), color, -1)
                cv2.putText(frame, direction, (10, 28), FONT, 0.8, (255, 255, 255), 2)
                if blinking:
                    cv2.putText(frame, 'BLINKING', (10, 80), FONT, 0.8, (0, 255, 0), 2)
                stats['Direction'] = direction
                stats['Gaze ratio'] = f'{gaze:.2f}'
                stats['Blink'] = 'BLINKING' if blinking else 'open'
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            self.stats_ready.emit(stats)
        cap.release()
