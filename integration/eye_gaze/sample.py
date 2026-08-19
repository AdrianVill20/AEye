"""
Iris-based gaze detection using MediaPipe Face Mesh (refine_landmarks=True)

Horizontal (Left/Right): iris center position relative to eye corners
Vertical (Up/Down): eye openness ratio (eyelid droop = looking down)
"""

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (263, 362)

# Averaged eyelid landmarks for stable openness measurement
RIGHT_UPPER_LIDS = [159, 160, 161, 186]
RIGHT_LOWER_LIDS = [145, 144, 146, 153]
LEFT_UPPER_LIDS = [386, 374, 373, 382]
LEFT_LOWER_LIDS = [374, 373, 372, 380]

# Horizontal thresholds (iris position relative to eye width)
H_LEFT_THRESH = 0.42
H_RIGHT_THRESH = 0.58

# Vertical thresholds (eye openness ratio = eye_height / eye_width)
# These are deviations from the calibrated baseline.
V_UP_THRESHOLD = 0.01     # eye wider than baseline by this much = Up
V_DOWN_THRESHOLD = -0.01  # eye narrower than baseline by this much = Down

# Calibration
CALIB_FRAMES = 40         # frames to average for baseline
SMOOTHING = 0.3

_prev_h = 0.5
_calib_openness = []
_baseline = None


def _ema(prev, new, alpha=SMOOTHING):
    return prev * (1 - alpha) + new * alpha


def _avg(landmarks, indices, axis, size):
    vals = [getattr(landmarks[i], axis) * size for i in indices]
    return np.mean(vals)


def get_eye_data(landmarks, corners_idx, upper_lids, lower_lids, img_w, img_h):
    outer_idx, inner_idx = corners_idx
    outer_x = landmarks[outer_idx].x * img_w
    inner_x = landmarks[inner_idx].x * img_w
    eye_width = inner_x - outer_x

    upper_y = _avg(landmarks, upper_lids, 'y', img_h)
    lower_y = _avg(landmarks, lower_lids, 'y', img_h)
    eye_height = lower_y - upper_y

    openness = eye_height / eye_width if eye_width != 0 else 0.3
    return eye_width, openness


def get_h_ratio(landmarks, pupil_px, corners_idx, img_w):
    outer_x = landmarks[corners_idx[0]].x * img_w
    inner_x = landmarks[corners_idx[1]].x * img_w
    eye_width = inner_x - outer_x
    return (pupil_px[0] - outer_x) / eye_width if eye_width != 0 else 0.5


def classify_gaze(h_ratio, right_openness, left_openness):
    global _prev_h, _baseline, _calib_openness

    _prev_h = _ema(_prev_h, h_ratio)

    # Horizontal
    if _prev_h < H_LEFT_THRESH:
        h_dir = "Left"
    elif _prev_h > H_RIGHT_THRESH:
        h_dir = "Right"
    else:
        h_dir = "Center"

    # Average both eyes' openness
    avg_openness = (right_openness + left_openness) / 2.0

    # Calibration phase: collect baseline
    if _baseline is None:
        _calib_openness.append(avg_openness)
        if len(_calib_openness) >= CALIB_FRAMES:
            _baseline = np.mean(_calib_openness)
            print(f"--- Baseline calibrated: {_baseline:.4f} ---")
        return h_dir, "Calibrating..."

    # Vertical: compare current openness to baseline
    diff = avg_openness - _baseline
    if diff > V_UP_THRESHOLD:
        v_dir = "Up"
    elif diff < V_DOWN_THRESHOLD:
        v_dir = "Down"
    else:
        v_dir = "Center"

    print(f'h={_prev_h:.3f} ({h_dir})  open={avg_openness:.4f} diff={diff:+.4f} ({v_dir})')
    return h_dir, v_dir


def main():
    global _prev_h, _baseline, _calib_openness
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Get iris center for display + horizontal ratio
                r_cx = int(landmarks[RIGHT_IRIS_CENTER].x * w)
                r_cy = int(landmarks[RIGHT_IRIS_CENTER].y * h)
                l_cx = int(landmarks[LEFT_IRIS_CENTER].x * w)
                l_cy = int(landmarks[LEFT_IRIS_CENTER].y * h)

                cv2.circle(frame, (r_cx, r_cy), 2, (0, 0, 255), -1)
                cv2.circle(frame, (l_cx, l_cy), 2, (0, 0, 255), -1)

                r_ow, r_open = get_eye_data(landmarks, RIGHT_EYE_CORNERS,
                                            RIGHT_UPPER_LIDS, RIGHT_LOWER_LIDS, w, h)
                l_ow, l_open = get_eye_data(landmarks, LEFT_EYE_CORNERS,
                                            LEFT_UPPER_LIDS, LEFT_LOWER_LIDS, w, h)

                h_ratio = get_h_ratio(landmarks, (r_cx, r_cy), RIGHT_EYE_CORNERS, w)
                h_dir, v_dir = classify_gaze(h_ratio, r_open, l_open)

                color = (0, 255, 255)
                cv2.putText(frame, f"Gaze: {h_dir} / {v_dir}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"h={h_ratio:.3f} open={r_open:.4f}/{l_open:.4f}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                if _baseline is not None:
                    cv2.putText(frame, f"baseline={_baseline:.4f}",
                                (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("Iris / Pupil Detection", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('r'):
                _prev_h = 0.5
                _baseline = None
                _calib_openness.clear()
                print("--- Reset: recalibrating ---")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
