import cv2
import mediapipe as mp
import time
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

MP_DRAWING = mp.solutions.drawing_utils
MP_POSE = mp.solutions.pose
POSE_CONNECTIONS = MP_POSE.POSE_CONNECTIONS


def create_pose_landmarker():
    return MP_POSE.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def extract_posture(result):
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks.landmark
    l_shoulder = lm[MP_POSE.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = lm[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value]
    l_wrist = lm[MP_POSE.PoseLandmark.LEFT_WRIST.value]
    r_wrist = lm[MP_POSE.PoseLandmark.RIGHT_WRIST.value]
    return {
        'Left shoulder': f'{l_shoulder.x:.2f}, {l_shoulder.y:.2f}',
        'Right shoulder': f'{r_shoulder.x:.2f}, {r_shoulder.y:.2f}',
        'Left wrist': f'{l_wrist.x:.2f}, {l_wrist.y:.2f}',
        'Right wrist': f'{r_wrist.x:.2f}, {r_wrist.y:.2f}',
    }

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

    def run(self):
        self._running = True
        pose = create_pose_landmarker()
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            stats = self._blank_stats()
            stats['Left shoulder'] = f'Camera {self.camera_index} unavailable'
            self.stats_ready.emit(stats)
            pose.close()
            return
        upd_interval = 0.5
        last_upd_time = 0.0
        display_stats = self._blank_stats()
        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)
            rgb.flags.writeable = True
            stats = self._blank_stats()
            if result.pose_landmarks:
                MP_DRAWING.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    POSE_CONNECTIONS,
                    MP_DRAWING.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=3),
                    MP_DRAWING.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                )
                stats['Landmarks detected (/33)'] = str(len(result.pose_landmarks.landmark))
                
                curr_time = time.time()
                if curr_time - last_upd_time > upd_interval:
                    last_upd_time = curr_time
                    coords = extract_posture(result)
                    if coords is not None:
                        display_stats.update(coords)
                stats.update(display_stats)
            rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            qimg = QImage(rgb_out.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            self.stats_ready.emit(stats)
        cap.release()
        pose.close()
