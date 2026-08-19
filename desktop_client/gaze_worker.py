"""
gaze_worker.py
------------------------------------------------------------------
Runs Christian's dlib eye-gaze INSIDE the AEye kiosk.

His Revised_Gaze/main.py is a standalone `while True: cv2.imshow(...)` loop.
To show it embedded in a tab (instead of a separate window that the locked
fullscreen would hide), we run the same per-frame work on a background QThread
and emit each annotated frame + readout back to the GUI. His color feedback
(red=right / blue=center / green=left) is reproduced as a banner on the frame.

Detection math is entirely his (Revised_Gaze/gaze_core.py).
"""

import sys
from pathlib import Path

import cv2
import dlib
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

# Append (not insert) so Revised_Gaze's generic main.py can't shadow ours.
REVISED_DIR = Path(__file__).resolve().parent.parent / "Revised_Gaze"
sys.path.append(str(REVISED_DIR))

from gaze_core import blink, gaze_ratio          # noqa: E402  (his functions)

PREDICTOR_PATH = REVISED_DIR / "shape_predictor_68_face_landmarks.dat"
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
        # Built once (loading the 99 MB predictor every frame would be far too
        # slow). dlib's HOG face detector + his 68-point landmark predictor.
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(PREDICTOR_PATH))

    def stop(self):
        self._running = False

    @staticmethod
    def _blank_stats():
        return {"Direction": "--", "Gaze ratio": "--", "Blink": "--"}

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            stats = self._blank_stats()
            stats["Direction"] = f"Camera {self.camera_index} unavailable"
            self.stats_ready.emit(stats)
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            stats = self._blank_stats()

            for face in faces:
                landmarks = self.predictor(gray, face)

                left_div = blink(LEFT_EYE, landmarks)
                right_div = blink(RIGHT_EYE, landmarks)
                blinking = right_div and left_div > 5.5      # his exact test

                gaze = (gaze_ratio(RIGHT_EYE, landmarks, frame, gray)
                        + gaze_ratio(LEFT_EYE, landmarks, frame, gray)) / 2

                # His direction bands + colors (BGR): red / blue / green.
                if gaze < 1:
                    direction, color = "looking right", (0, 0, 255)
                elif 1 < gaze < 3:
                    direction, color = "looking center", (255, 0, 0)
                else:
                    direction, color = "looking left", (0, 255, 0)

                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 40), color, -1)
                cv2.putText(frame, direction, (10, 28), FONT, 0.8,
                            (255, 255, 255), 2)
                if blinking:
                    cv2.putText(frame, "BLINKING", (10, 80), FONT, 0.8,
                                (0, 255, 0), 2)

                stats["Direction"] = direction
                stats["Gaze ratio"] = f"{gaze:.2f}"
                stats["Blink"] = "BLINKING" if blinking else "open"
                break   # one face is enough for the readout

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

            self.frame_ready.emit(qimg)
            self.stats_ready.emit(stats)

        cap.release()
