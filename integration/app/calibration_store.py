"""Read / write per-student calibration data as JSON.

Calibration records a short sample of a student's NORMAL gaze behaviour. It is
stored as JSON (not MySQL) so it can be reused later - to train the student's
personal cheat-detection model now, and for future retraining / analysis.

One file per student:  calibration_data/calibration_<user>.json
"""

import json
import re
from datetime import datetime
from pathlib import Path

CALIB_DIR = Path(__file__).resolve().parent / 'calibration_data'

# Feature order - must match cheat_detector.FEATURES and the training script.
FEATURES = ['h_ratio', 'v_openness', 'yaw', 'pitch', 'roll']


def _safe(user_id):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(user_id)) if user_id else 'unknown'


def calib_path(user_id):
    return CALIB_DIR / f'calibration_{_safe(user_id)}.json'


def save(user_id, samples):
    """samples: list of dicts, each with the FEATURES keys. Written atomically
    (temp file + replace) so a crash mid-write can't corrupt the file."""
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        'user': user_id,
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'features': FEATURES,
        'count': len(samples),
        'samples': samples,
    }
    path = calib_path(user_id)
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    tmp.replace(path)
    return path


def load(user_id):
    """Return the parsed calibration payload for a student (raises if missing)."""
    with open(calib_path(user_id), encoding='utf-8') as f:
        return json.load(f)
