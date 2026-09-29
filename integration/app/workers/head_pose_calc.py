# Plug in R matrix values here and run this file to see yaw/pitch/roll.
# Uses the same compute_head_pose() as the real app, so this always matches
# what AEye would actually compute.
#
# Only R(0,0), R(1,0), R(2,0), R(2,1), R(2,2) are used by the formula -
# R(0,1), R(0,2), R(1,1), R(1,2) don't affect the result, so they're left
# at 0 below. Change any value and run: python head_pose_calc.py

import numpy as np
from head_pose import compute_head_pose

R = np.array([
    [0.83, 0.00, 0.00],
    [0.00, 0.98, 0.00],
    [-0.55, -0.18, 0.81],
])

yaw, pitch, roll = compute_head_pose(R)

print(f'yaw   = {yaw:+.2f} degrees')
print(f'pitch = {pitch:+.2f} degrees')
print(f'roll  = {roll:+.2f} degrees')
