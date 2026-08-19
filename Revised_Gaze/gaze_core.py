"""
gaze_core.py
------------------------------------------------------------------
Christian's dlib gaze functions, lifted out of Revised_Gaze/main.py so the AEye
GUI can call them frame-by-frame and draw the result INSIDE its own window
(instead of his standalone cv2.imshow windows).

The algorithm and every threshold are HIS. The only change is that gaze_ratio
takes `frame`/`gray` as arguments instead of reading them from loop globals, so
the function stands on its own.
"""

from math import hypot

import cv2
import numpy as np


def midpoint(p1, p2):
    return (int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2))


def blink(points, landmarks):
    """His blink ratio = eye width / eye height from 6 dlib eye points."""
    left_point = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
    right_point = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
    center_top = midpoint(landmarks.part(points[1]), landmarks.part(points[2]))
    center_bottom = midpoint(landmarks.part(points[5]), landmarks.part(points[4]))

    hor_line = hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])
    ver_line = hypot(center_top[0] - center_bottom[0], center_top[1] - center_bottom[1])
    return hor_line / ver_line


def gaze_ratio(eye_marks, landmarks, frame, gray):
    """His gaze ratio: isolate the eye, threshold it, and compare white pixels
    on the left half vs the right half. `frame`/`gray` are parameters now."""
    region = np.array([(landmarks.part(i).x, landmarks.part(i).y)
                        for i in eye_marks], np.int32)

    h, w, _ = frame.shape
    mask = np.zeros((h, w), np.uint8)
    cv2.polylines(mask, [region], True, 255, 2)
    cv2.fillPoly(mask, [region], 255)
    eyes = cv2.bitwise_and(gray, gray, mask=mask)

    min_x, max_x = region[:, 0].min(), region[:, 0].max()
    min_y, max_y = region[:, 1].min(), region[:, 1].max()
    gray_eye = eyes[min_y:max_y, min_x:max_x]
    if gray_eye.size == 0:                          # guard so the GUI won't crash
        return 1

    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_white = cv2.countNonZero(threshold_eye[0:height, 0:int(width / 2)])
    right_white = cv2.countNonZero(threshold_eye[0:height, int(width / 2):width])

    if left_white == 0:
        return 1
    elif right_white == 0:
        return 0
    return left_white / right_white
