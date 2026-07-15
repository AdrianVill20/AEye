import cv2
import numpy as np
import mediapipe as mp
import pickle
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

SCREEN_W, SCREEN_H = 1280, 720

# --- same landmark constants as before ---
LEFT_IRIS = 468
LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

LEFT_EYE_OUTLINE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 158, 159, 160, 161, 246]

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)
HEADPOSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def get_iris_ratio(landmarks, w, h):
    iris_x = landmarks[LEFT_IRIS].x
    left_x = landmarks[LEFT_EYE_LEFT_CORNER].x
    right_x = landmarks[LEFT_EYE_RIGHT_CORNER].x
    return (iris_x - left_x) / (right_x - left_x)


def get_vertical_iris_ratio(landmarks, w, h):
    iris_y = landmarks[LEFT_IRIS].y
    top_y = landmarks[LEFT_EYE_TOP].y
    bottom_y = landmarks[LEFT_EYE_BOTTOM].y
    return (iris_y - top_y) / (bottom_y - top_y)


def get_sclera_ratio(landmarks, frame, w, h):
    eye_points = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in LEFT_EYE_OUTLINE
    ], np.int32)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [eye_points], 255)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_only = cv2.bitwise_and(gray, gray, mask=mask)

    min_x, max_x = eye_points[:, 0].min(), eye_points[:, 0].max()
    min_y, max_y = eye_points[:, 1].min(), eye_points[:, 1].max()
    eye_crop = eye_only[min_y:max_y, min_x:max_x]

    if eye_crop.size == 0:
        return 1.0

    _, thresh = cv2.threshold(eye_crop, 60, 255, cv2.THRESH_BINARY)
    height, width = thresh.shape
    left_half = thresh[:, 0:width // 2]
    right_half = thresh[:, width // 2:width]

    left_white = cv2.countNonZero(left_half)
    right_white = max(cv2.countNonZero(right_half), 1)

    return left_white / right_white


def get_euler_angles(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0

    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)


def get_head_pose(landmarks, w, h, prev_rvec, prev_tvec):
    image_points = np.array([
        (landmarks[idx].x * w, landmarks[idx].y * h)
        for idx in HEADPOSE_LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    if prev_rvec is not None:
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            rvec=prev_rvec, tvec=prev_tvec, useExtrinsicGuess=True
        )
    else:
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs
        )

    if not success:
        return None, None, None, prev_rvec, prev_tvec

    pitch, yaw, roll = get_euler_angles(rvec)
    return pitch, yaw, roll, rvec, tvec


# --- load the trained model ---
with open('gaze_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully")

# smoothing buffer for predicted gaze point (reduces jitter)
gaze_x_history = deque(maxlen=5)
gaze_y_history = deque(maxlen=5)

prev_rvec, prev_tvec = None, None
gaze_window = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    gaze_window[:] = 30  # dark gray background

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        iris_ratio = get_iris_ratio(landmarks, w, h)
        vertical_iris_ratio = get_vertical_iris_ratio(landmarks, w, h)
        sclera_ratio = get_sclera_ratio(landmarks, frame, w, h)
        pitch, yaw, roll, prev_rvec, prev_tvec = get_head_pose(
            landmarks, w, h, prev_rvec, prev_tvec
        )

        if pitch is not None:
            features = np.array([[
                iris_ratio, vertical_iris_ratio, sclera_ratio,
                yaw, pitch, roll
            ]])

            predicted = model.predict(features)[0]
            pred_x, pred_y = predicted[0], predicted[1]

            # clip predictions to stay within screen bounds (0 to 1)
            pred_x = np.clip(pred_x, 0, 1)
            pred_y = np.clip(pred_y, 0, 1)

            # smooth over recent frames
            gaze_x_history.append(pred_x)
            gaze_y_history.append(pred_y)
            smooth_x = sum(gaze_x_history) / len(gaze_x_history)
            smooth_y = sum(gaze_y_history) / len(gaze_y_history)

            dot_x = int(smooth_x * SCREEN_W)
            dot_y = int(smooth_y * SCREEN_H)

            cv2.circle(gaze_window, (dot_x, dot_y), 20, (0, 0, 255), -1)
            cv2.putText(gaze_window, f"Predicted: ({smooth_x:.2f}, {smooth_y:.2f})",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Predicted Gaze', gaze_window)
    cv2.imshow('Webcam feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()