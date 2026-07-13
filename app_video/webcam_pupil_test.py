import cv2
import sys
import os

# add the EyeTracker folder so we can import their function
sys.path.append(r"C:\Users\ICHOY\OneDrive\Desktop\EyeTracker")
from OrloskyPupilDetector import process_frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # rough manual crop around where an eye usually sits - adjust these numbers
    # based on where YOUR eye actually appears in the frame
    x1, y1, x2, y2 = w//2 - 150, h//2 - 100, w//2 + 50, h//2

    eye_crop = frame[y1:y2, x1:x2]

    if eye_crop.size > 0:
        try:
            ellipse = process_frame(eye_crop.copy())
            cv2.ellipse(frame[y1:y2, x1:x2], ellipse, (0, 255, 0), 2)
        except Exception as e:
            cv2.putText(frame, "detection failed", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.imshow('Webcam Pupil Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()