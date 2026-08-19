"""
Iris-based pupil detection using MediaPipe Face Mesh (refine_landmarks=True)

Landmark indices (with refine_landmarks=True, 478 total points):
  468        -> right eye iris center (pupil)
  469-472    -> right eye iris boundary ring
  473        -> left eye iris center (pupil)
  474-477    -> left eye iris boundary ring
"""

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Landmark index groups
RIGHT_IRIS_CENTER = 468
RIGHT_IRIS_RING = [469, 470, 471, 472]
LEFT_IRIS_CENTER = 473
LEFT_IRIS_RING = [474, 475, 476, 477]

# Eye corner landmarks: (outer_corner, inner_corner)
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (263, 362)

# Eyelid landmarks: (upper_lid, lower_lid) -- roughly above/below the pupil
RIGHT_EYE_LIDS = (159, 145)
LEFT_EYE_LIDS = (386, 374)

# Thresholds for classifying the ratio into a direction.
# These are a starting point -- tune per-camera / per-student, same as
# you're already doing for the Isolation Forest calibration.
H_LEFT_THRESH = 0.35
H_RIGHT_THRESH = 0.65
V_UP_THRESH = 0.35
V_DOWN_THRESH = 0.65


def get_pupil_data(landmarks, center_idx, ring_idxs, img_w, img_h):
    """Convert normalized landmarks to pixel coords and get center + radius."""
    # Pupil center in pixel coordinates
    center_lm = landmarks[center_idx]
    cx, cy = int(center_lm.x * img_w), int(center_lm.y * img_h)

    # Iris boundary points, used to estimate iris radius
    ring_points = [
        (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
        for i in ring_idxs
    ]
    import numpy as np
    points_array = np.array(ring_points, dtype=np.int32)
    (_, _), radius = cv2.minEnclosingCircle(points_array)

    return (cx, cy), int(radius)


def get_gaze_ratio(landmarks, pupil_px, corners_idx, lids_idx, img_w, img_h):
    """
    Compare pupil position to this eye's own corners/eyelids.
    Returns (h_ratio, v_ratio), each roughly 0.0-1.0.
    h_ratio: 0 = pupil at outer corner, 1 = pupil at inner corner
    v_ratio: 0 = pupil at upper lid,   1 = pupil at lower lid
    """
    outer_idx, inner_idx = corners_idx
    upper_idx, lower_idx = lids_idx

    outer_x = landmarks[outer_idx].x * img_w
    inner_x = landmarks[inner_idx].x * img_w
    upper_y = landmarks[upper_idx].y * img_h
    lower_y = landmarks[lower_idx].y * img_h

    pupil_x, pupil_y = pupil_px

    eye_width = inner_x - outer_x
    eye_height = lower_y - upper_y

    # Guard against div-by-zero on a bad frame
    h_ratio = (pupil_x - outer_x) / eye_width if eye_width != 0 else 0.5
    v_ratio = (pupil_y - upper_y) / eye_height if eye_height != 0 else 0.5

    return h_ratio, v_ratio


def classify_gaze(h_ratio, v_ratio):
    if h_ratio < H_LEFT_THRESH:
        h_dir = "Left"
        print(f'left {h_ratio}')
    elif h_ratio > H_RIGHT_THRESH:
        h_dir = "Right"
        print(f'right {h_ratio}')
    else:
        h_dir = "Center"
        print(f'hcenter {h_ratio}')
    
    if v_ratio < V_UP_THRESH:
        v_dir = "Up"
        print(f'up {v_ratio}')
    elif v_ratio > V_DOWN_THRESH:
        v_dir = "Down"
        print(f'down {v_ratio}')
    else:
        v_dir = "Center"
        print(f'center {v_ratio}')
        
    return h_dir, v_dir 

# def get_face_bbox(landmarks, img_w, img_h, padding=10):
#     """
#     Scan all face landmarks and find the min/max x,y to build a bounding box.
#     padding adds a few extra pixels on each side so the box isn't too tight.
#     """
#     xs = [lm.x * img_w for lm in landmarks]
#     ys = [lm.y * img_h for lm in landmarks]

#     x_min = int(min(xs)) - padding
#     y_min = int(min(ys)) - padding
#     x_max = int(max(xs)) + padding
#     y_max = int(max(ys)) + padding

#     # Clamp so the box never goes off-frame
#     x_min = max(0, x_min)
#     y_min = max(0, y_min)
#     x_max = min(img_w, x_max)
#     y_max = min(img_h, y_max)

#     return x_min, y_min, x_max, y_max


def main():
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,   # <-- this is what enables iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural webcam view
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # x_min, y_min, x_max, y_max = get_face_bbox(landmarks, w, h)
                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 200, 0), 2)

                right_pupil_px = None
                for center_idx, ring_idxs, corners, lids, color in [
                    (RIGHT_IRIS_CENTER, RIGHT_IRIS_RING, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, (0, 255, 0)),
                    (LEFT_IRIS_CENTER, LEFT_IRIS_RING, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, (255, 0, 0)),
                ]:
                    (cx, cy), radius = get_pupil_data(
                        landmarks, center_idx, ring_idxs, w, h
                    )
                    cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)       # pupil dot
                    cv2.circle(frame, (cx, cy), radius, color, 1)        # iris outline

                    if center_idx == RIGHT_IRIS_CENTER:
                        right_pupil_px = (cx, cy)
                        h_ratio, v_ratio = get_gaze_ratio(landmarks, right_pupil_px, corners, lids, w, h)
                        h_dir, v_dir = classify_gaze(h_ratio, v_ratio)
                        cv2.putText(frame,f"Gaze: {h_dir} / {v_dir}",(20, 40), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 255, 255),2,)

            cv2.imshow("Iris / Pupil Detection", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()