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

    # get screen resolution for calibration point placement
SCREEN_W, SCREEN_H = 1280, 720  # adjust to match your actual screen/window size

    # 9 calibration points across the screen (3x3 grid)
CALIBRATION_POINTS = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]

LEFT_IRIS = 468
LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133

collected_data = []  # will hold (features, target_x, target_y) tuples

current_point_index = 0
frames_collected_for_point = 0
FRAMES_PER_POINT = 30  # collect 30 frames per calibration point

calibration_window = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

while current_point_index < len(CALIBRATION_POINTS):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

        # draw the calibration dot on a separate fullscreen-ish window
    calibration_window[:] = 0
    target_x_norm, target_y_norm = CALIBRATION_POINTS[current_point_index]
    dot_x = int(target_x_norm * SCREEN_W)
    dot_y = int(target_y_norm * SCREEN_H)
    cv2.circle(calibration_window, (dot_x, dot_y), 15, (0, 255, 0), -1)
    cv2.putText(calibration_window, "Look at the green dot", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(calibration_window, f"Point {current_point_index+1}/9", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        iris_x = landmarks[LEFT_IRIS].x
        left_x = landmarks[LEFT_EYE_LEFT_CORNER].x
        right_x = landmarks[LEFT_EYE_RIGHT_CORNER].x
        iris_ratio = (iris_x - left_x) / (right_x - left_x)

            # collect this frame's data point
        collected_data.append({
            "iris_ratio": iris_ratio,
            "target_x": target_x_norm,
            "target_y": target_y_norm
        })
        frames_collected_for_point += 1

    cv2.imshow('Calibration', calibration_window)
    cv2.imshow('Webcam feed', frame)

    if frames_collected_for_point >= FRAMES_PER_POINT:
        current_point_index += 1
        frames_collected_for_point = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Collected {len(collected_data)} data points across {len(CALIBRATION_POINTS)} calibration targets")

    # save to a file so we can use it in the next stage
import json
with open('calibration_data.json', 'w') as f:
    json.dump(collected_data, f)
print("Saved to calibration_data.json")