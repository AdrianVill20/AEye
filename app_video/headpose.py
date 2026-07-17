import cv2
import numpy as np
import mediapipe as mp
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# generic 3D face model reference points (approximate, in arbitrary units)
# these correspond to: nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left eye left corner
    (225.0, 170.0, -135.0),   # right eye right corner
    (-150.0, -150.0, -125.0), # left mouth corner
    (150.0, -150.0, -125.0)   # right mouth corner
], dtype=np.float64)

# corresponding MediaPipe landmark indices for those same 6 points
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def get_euler_angles(rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # decompose rotation matrix into euler angles
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

    # convert radians to degrees
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    roll = np.degrees(roll)

    return pitch, yaw, roll


# fix 1: keep track of previous solvePnP result to stabilize solution
prev_rotation_vector = None
prev_translation_vector = None

# fix 2: smoothing buffers for yaw/pitch/roll
yaw_history = deque(maxlen=5)
pitch_history = deque(maxlen=5)
roll_history = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # get the 2D pixel coordinates for our 6 reference points
        image_points = np.array([
            (landmarks[idx].x * w, landmarks[idx].y * h)
            for idx in LANDMARK_IDS
        ], dtype=np.float64)

        # approximate camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        # fix 1: use previous frame's result as a starting guess if we have one
        if prev_rotation_vector is not None:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
                rvec=prev_rotation_vector, tvec=prev_translation_vector,
                useExtrinsicGuess=True
            )
        else:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_POINTS, image_points, camera_matrix, dist_coeffs
            )

        if success:
            prev_rotation_vector = rotation_vector
            prev_translation_vector = translation_vector

            # project a 3D axis point (300 units forward on the nose's Z axis)
            # back onto the 2D image to visualize direction
            axis_point_3d = np.array([(0.0, 0.0, 300.0)])
            axis_point_2d, _ = cv2.projectPoints(
                axis_point_3d, rotation_vector, translation_vector,
                camera_matrix, dist_coeffs
            )

            nose_2d = (int(image_points[0][0]), int(image_points[0][1]))
            axis_2d = (int(axis_point_2d[0][0][0]), int(axis_point_2d[0][0][1]))

            pitch, yaw, roll = get_euler_angles(rotation_vector)

            # fix 2: smooth over the last 5 frames
            yaw_history.append(yaw)
            pitch_history.append(pitch)
            roll_history.append(roll)

            yaw = sum(yaw_history) / len(yaw_history)
            pitch = sum(pitch_history) / len(pitch_history)
            roll = sum(roll_history) / len(roll_history)

            cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.1f}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.line(frame, nose_2d, axis_2d, (255, 0, 0), 3)
            cv2.circle(frame, nose_2d, 4, (0, 255, 0), -1)

    cv2.imshow('Stage 1: Head Pose Axis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()