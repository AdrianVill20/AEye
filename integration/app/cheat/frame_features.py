# The 5 numbers AEye reads from every frame: eyes (h_ratio, v_openness) and
# head pose (yaw, pitch, roll).
#
# This file has NO Qt in it, so the SAME code runs in two places:
#   - the app (workers/front_cam_worker.py imports the helpers below)
#   - Google Colab (colab/AEye_Isolation_Forest.ipynb gets a copy of this file)
# That way the Isolation Forest is trained on exactly the numbers the app sees.
import math

import numpy as np

# the 5 numbers, in this order, everywhere (forest, gaze model, CSVs)
FEATURES = ['h_ratio', 'v_openness', 'yaw', 'pitch', 'roll']

RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)

RIGHT_UPPER_LIDS = [160, 159, 158]
RIGHT_LOWER_LIDS = [144, 145, 153]
LEFT_UPPER_LIDS = [385, 386, 387]
LEFT_LOWER_LIDS = [380, 374, 373]

SMOOTHING = 0.3          # EMA weight of the newest frame
FRAME_STEP_MS = 33       # MediaPipe timestamp step per frame (same as the app)


def _ema(prev, new, alpha=SMOOTHING):
    return prev * (1 - alpha) + new * alpha


def _avg_val(landmarks, indices, axis, size):
    vals = [getattr(landmarks[i], axis) * size for i in indices]
    return np.mean(vals)


def _get_eye_data(landmarks, corners_idx, upper_lids, lower_lids, img_w, img_h):
    outer_x = landmarks[corners_idx[0]].x * img_w
    inner_x = landmarks[corners_idx[1]].x * img_w
    eye_width = inner_x - outer_x

    upper_y = _avg_val(landmarks, upper_lids, 'y', img_h)
    lower_y = _avg_val(landmarks, lower_lids, 'y', img_h)
    eye_height = lower_y - upper_y

    openness = eye_height / eye_width if eye_width != 0 else 0.3
    return openness


def _get_h_ratio(landmarks, pupil_px, corners_idx, img_w):
    outer_x = landmarks[corners_idx[0]].x * img_w
    inner_x = landmarks[corners_idx[1]].x * img_w
    eye_width = inner_x - outer_x
    return (pupil_px[0] - outer_x) / eye_width if eye_width != 0 else 0.5


def _get_v_ratio(landmarks, iris_idx, corners_idx):
    # Iris height above the eye corners (corners don't move with eyebrows). Bigger = looking up.
    a, b = landmarks[corners_idx[0]], landmarks[corners_idx[1]]
    eye_width = abs(b.x - a.x)
    mid_y = (a.y + b.y) / 2
    return (mid_y - landmarks[iris_idx].y) / eye_width if eye_width != 0 else 0.0


def compute_head_pose(R):
    """Same formula as workers/head_pose.py. R is the 3x3 rotation matrix
    (top-left of MediaPipe's facial transformation matrix). Returns
    (yaw, pitch, roll) in degrees."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def raw_eye_values(lm, w, h):
    """Unsmoothed (h_ratio, v_ratio, avg_openness) of one face."""
    r_cx = int(lm[RIGHT_IRIS_CENTER].x * w)
    r_cy = int(lm[RIGHT_IRIS_CENTER].y * h)
    l_cx = int(lm[LEFT_IRIS_CENTER].x * w)
    l_cy = int(lm[LEFT_IRIS_CENTER].y * h)
    h_ratio = (_get_h_ratio(lm, (r_cx, r_cy), RIGHT_EYE_CORNERS, w)
               + _get_h_ratio(lm, (l_cx, l_cy), LEFT_EYE_CORNERS, w)) / 2
    v_ratio = (_get_v_ratio(lm, RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS)
               + _get_v_ratio(lm, LEFT_IRIS_CENTER, LEFT_EYE_CORNERS)) / 2
    r_open = _get_eye_data(lm, RIGHT_EYE_CORNERS, RIGHT_UPPER_LIDS, RIGHT_LOWER_LIDS, w, h)
    l_open = _get_eye_data(lm, LEFT_EYE_CORNERS, LEFT_UPPER_LIDS, LEFT_LOWER_LIDS, w, h)
    return h_ratio, v_ratio, (r_open + l_open) / 2.0


def extract_video_features(video_path, task_path, every=1, progress=None):
    """Play a video the way the app's Video Test does (mirror the frame,
    MediaPipe timestamps +33 ms, EMA smoothing that carries over frames
    without a face) and return one row per frame:
        {'frame', 't', 'face', 'h_ratio', 'v_openness', 'yaw', 'pitch', 'roll'}
    Frames without a face have face=False and NaN features.
    every=2 keeps every 2nd frame in the output (all frames are still
    processed, so the smoothing matches the app)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        landmarker.close()
        raise IOError(f'Cannot open video {video_path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    rows = []
    prev_h, prev_v = 0.5, None
    timestamp_ms = 0
    idx = -1
    nan = float('nan')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                             timestamp_ms)
        timestamp_ms += FRAME_STEP_MS

        row = {'frame': idx, 't': t, 'face': False,
               'h_ratio': nan, 'v_openness': nan, 'yaw': nan, 'pitch': nan, 'roll': nan}
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            h_ratio, v_ratio, _ = raw_eye_values(lm, w, h)
            prev_h = _ema(prev_h, h_ratio)
            prev_v = v_ratio if prev_v is None else _ema(prev_v, v_ratio)
            if result.facial_transformation_matrixes:
                R = np.array(result.facial_transformation_matrixes[0])[:3, :3]
                yaw, pitch, roll = compute_head_pose(R)
                row.update(face=True, h_ratio=float(prev_h), v_openness=float(prev_v),
                           yaw=float(yaw), pitch=float(pitch), roll=float(roll))
        if idx % every == 0:
            rows.append(row)
        if progress and idx % 300 == 0:
            progress(idx, total)

    cap.release()
    landmarker.close()
    return rows
