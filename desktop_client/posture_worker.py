"""
posture_worker.py
------------------------------------------------------------------
Runs Allain's posture (ported to PoseLandmarker) INSIDE the AEye kiosk, exactly
like gaze_worker.py does for Christian's gaze: his per-frame logic runs on a
background QThread and each annotated frame + readout is emitted to the GUI, so
it renders in the Posture tab with no separate window.

The pose logic is his (see app_video/posture/pose_common.py).
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

# Append (not insert) so nothing here can shadow our own modules.
POSTURE_DIR = Path(__file__).resolve().parent.parent / "app_video" / "posture"
sys.path.append(str(POSTURE_DIR))

from pose_common import (              # noqa: E402
    create_pose_landmarker, extract_posture, POSE_CONNECTIONS,
)


class SideCameraWorker(QThread):
    frame_ready = Signal(QImage)
    stats_ready = Signal(dict)

    def __init__(self, camera_index=1, parent=None):
        # Side camera = a SECOND webcam (index 1); index 0 is the front camera
        # used for gaze. A parameter so it's easy to change for one-camera tests.
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False

    def stop(self):
        self._running = False

    @staticmethod
    def _blank_stats():
        return {
            "Left shoulder": "--", "Right shoulder": "--",
            "Left wrist": "--", "Right wrist": "--",
            "Landmarks detected (/33)": "--",
        }

    @staticmethod
    def _draw_skeleton(frame, lm, w, h):
        """Draw the 33 body points + skeleton edges (replaces the old
        mp_drawing.draw_landmarks, which lived in the removed mp.solutions).
        Styled to match Allain's original: BLACK connections, colored dots."""
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in POSE_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 0, 0), 1)          # his black lines
        for (x, y) in pts:
            cv2.circle(frame, (x, y), 3, (245, 66, 230), -1)      # his colored dots

    def run(self):
        self._running = True
        landmarker = create_pose_landmarker()

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            stats = self._blank_stats()
            stats["Left shoulder"] = f"Camera {self.camera_index} unavailable"
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
                stats["Landmarks detected (/33)"] = str(len(lm))
                coords = extract_posture(result)      # Allain's 4 points
                if coords is not None:
                    stats.update(coords)

            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_out.data, w, h, 3 * w, QImage.Format_RGB888).copy()

            self.frame_ready.emit(qimg)
            self.stats_ready.emit(stats)

        cap.release()
