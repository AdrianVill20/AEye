# Head-pose math, kept separate from front_cam_worker.py so it's easy to
# swap out. Edit compute_head_pose() below to try a different formula -
# nothing else in the app needs to change, as long as it still takes the
# 3x3 rotation matrix R and returns (yaw, pitch, roll) in degrees.

import math


def compute_head_pose(R):
    """R is the 3x3 rotation matrix (top-left block of MediaPipe's 4x4
    facial transformation matrix). Returns (yaw, pitch, roll) in degrees."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll
