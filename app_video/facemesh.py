import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # needed for iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# landmark indices we need
LEFT_IRIS = 468
LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133

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

        iris_x = landmarks[LEFT_IRIS].x
        left_corner_x = landmarks[LEFT_EYE_LEFT_CORNER].x
        right_corner_x = landmarks[LEFT_EYE_RIGHT_CORNER].x

        # ratio: 0 = looking left, 1 = looking right
        ratio = (iris_x - left_corner_x) / (right_corner_x - left_corner_x)

        if ratio < 0.35:
            gaze = "Looking Left"
        elif ratio > 0.65:
            gaze = "Looking Right"
        else:
            gaze = "Looking Center"

        cv2.putText(frame, gaze, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # optional: draw a dot on the iris so you can see what it's tracking
        cx, cy = int(iris_x * w), int(landmarks[LEFT_IRIS].y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    cv2.imshow('Gaze Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()