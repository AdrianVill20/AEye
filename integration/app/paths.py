"""Single source of truth for where AEye's data and asset folders live.

Every module imports the folder it needs from here instead of computing paths
from its own file location. That way a .py file can be moved into any
sub-package without breaking the paths to models, evidence, calibration data,
or the bundled MediaPipe .task files.
"""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent          # integration/app
INTEGRATION_DIR = APP_DIR.parent                   # integration

# Asset / data folders that live at the app root.
MODELS_DIR = APP_DIR / "models"                    # screen areas + phone model
EVIDENCE_DIR = APP_DIR / "evidence"                # cheating screenshots
CALIB_DIR = APP_DIR / "calibration_data"           # per-student calibration_*.json

# Bundled MediaPipe task files (sit next to app/, in integration/head_pose/).
HEAD_POSE_DIR = INTEGRATION_DIR / "head_pose"
