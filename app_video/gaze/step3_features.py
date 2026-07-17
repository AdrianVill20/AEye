import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------
# Landmark indices — MediaPipe's 478-point mesh (with iris)
# ----------------------------------------------------------------------
# Each eye is described by a few key points:
#   - the iris CENTER          (where the eye is pointing)
#   - the inner + outer CORNERS (horizontal reference frame)
#   - the upper + lower EYELID  (vertical reference frame)
#
# "LEFT"/"RIGHT" = the subject's own left/right eye.
LEFT_IRIS,  LEFT_OUTER,  LEFT_INNER,  LEFT_TOP,  LEFT_BOTTOM  = 468, 33,  133, 159, 145
RIGHT_IRIS, RIGHT_INNER, RIGHT_OUTER, RIGHT_TOP, RIGHT_BOTTOM = 473, 362, 263, 386, 374


def horizontal_ratio(lm, iris, corner_a, corner_b):
    """How far the iris sits between the two eye corners, left->right.
       ~0.0 = pinned to corner_a, ~1.0 = pinned to corner_b, 0.5 = centered.
       We divide by the corner-to-corner width so the number stays the same
       whether your face is near or far from the camera (scale-invariant)."""
    return (lm[iris].x - lm[corner_a].x) / (lm[corner_b].x - lm[corner_a].x)


def vertical_ratio(lm, iris, top, bottom):
    """Same idea, vertically: ~0.0 = iris against the upper lid, ~1.0 = lower lid."""
    return (lm[iris].y - lm[top].y) / (lm[bottom].y - lm[top].y)


# ----------------------------------------------------------------------
# Build the landmarker once, before the loop
# ----------------------------------------------------------------------
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        # --- compute horizontal + vertical ratio for EACH eye ---
        # Corner order is chosen so BOTH eyes increase in the same screen
        # direction, so averaging them is meaningful.
        left_h  = horizontal_ratio(lm, LEFT_IRIS,  LEFT_OUTER, LEFT_INNER)
        right_h = horizontal_ratio(lm, RIGHT_IRIS, RIGHT_INNER, RIGHT_OUTER)
        left_v  = vertical_ratio(lm, LEFT_IRIS,  LEFT_TOP, LEFT_BOTTOM)
        right_v = vertical_ratio(lm, RIGHT_IRIS, RIGHT_TOP, RIGHT_BOTTOM)

        # average the two eyes -> one horizontal + one vertical gaze number
        gaze_h = (left_h + right_h) / 2.0
        gaze_v = (left_v + right_v) / 2.0

        # draw the iris centers so you can see what's being measured
        for idx in (LEFT_IRIS, RIGHT_IRIS):
            cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 3, (0, 0, 255), -1)

        # print the live numbers on screen
        cv2.putText(frame, f"H (left<->right): {gaze_h:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"V (up<->down):    {gaze_v:.2f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Step 3: Gaze features', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()