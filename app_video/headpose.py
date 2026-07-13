import cv2
import numpy as np
import mediapipe as mp

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

        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs
        )

        if success:
            # project a 3D axis point (100 units forward on the nose's Z axis)
            # back onto the 2D image to visualize direction
            axis_point_3d = np.array([(0.0, 0.0, 300.0)])
            axis_point_2d, _ = cv2.projectPoints(
                axis_point_3d, rotation_vector, translation_vector,
                camera_matrix, dist_coeffs
            )

            nose_2d = (int(image_points[0][0]), int(image_points[0][1]))
            axis_2d = (int(axis_point_2d[0][0][0]), int(axis_point_2d[0][0][1]))

            cv2.line(frame, nose_2d, axis_2d, (255, 0, 0), 3)
            cv2.circle(frame, nose_2d, 4, (0, 255, 0), -1)

    cv2.imshow('Stage 1: Head Pose Axis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()