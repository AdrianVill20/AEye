from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
HERE = Path(__file__).resolve().parent
POSE_MODEL_PATH = HERE / 'pose_landmarker.task'
LEFT_SHOULDER, RIGHT_SHOULDER = (11, 12)
LEFT_WRIST, RIGHT_WRIST = (15, 16)
POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

def create_pose_landmarker():
    options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)), running_mode=vision.RunningMode.VIDEO, num_poses=1)
    return vision.PoseLandmarker.create_from_options(options)

def extract_posture(result):
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]
    return {'Left shoulder': f'{lm[LEFT_SHOULDER].x:.2f}, {lm[LEFT_SHOULDER].y:.2f}', 'Right shoulder': f'{lm[RIGHT_SHOULDER].x:.2f}, {lm[RIGHT_SHOULDER].y:.2f}', 'Left wrist': f'{lm[LEFT_WRIST].x:.2f}, {lm[LEFT_WRIST].y:.2f}', 'Right wrist': f'{lm[RIGHT_WRIST].x:.2f}, {lm[RIGHT_WRIST].y:.2f}'}
