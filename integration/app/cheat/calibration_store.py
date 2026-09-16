"""Read / write per-student calibration data as JSON.

Every dot calibration run is ADDED as a new session instead of overwriting the
old one, so the student's model is trained on more data every exam.

One file per student:  calibration_data/calibration_<user>.json
    {'user': ..., 'sessions': [{'saved_at', 'screen', 'samples': [...]}, ...]}

Each sample is one camera frame taken while a dot was on screen:
    h_ratio, v_openness, yaw, pitch, roll   (from the front camera)
    target  [x, y] where the dot was, 0-1 of the screen
    val     True = validation dot (kept out of the screen area, used to check it)
    head    'lr' / 'ud' = head-movement dot (turn left-right / up-down), '' = normal
"""

import json
import re
from datetime import datetime

from paths import CALIB_DIR

MAX_SESSIONS = 10   # keep the last 10 runs so the file doesn't grow forever


def _safe(user_id):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(user_id)) if user_id else 'unknown'


def calib_path(user_id):
    return CALIB_DIR / f'calibration_{_safe(user_id)}.json'


def load_sessions(user_id):
    """All saved calibration runs for a student, oldest first ([] if none)."""
    path = calib_path(user_id)
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data.get('sessions', [])   # old read-passage files have no sessions


def save_sessions(user_id, sessions):
    """Written atomically (temp file + replace) so a crash can't corrupt it."""
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    payload = {'user': user_id, 'sessions': sessions[-MAX_SESSIONS:]}
    path = calib_path(user_id)
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    tmp.replace(path)
    return path


def add_session(user_id, samples, screen):
    """Append one calibration run. Returns how many runs are saved now."""
    sessions = load_sessions(user_id)
    sessions.append({
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'screen': screen,
        'samples': samples,
    })
    save_sessions(user_id, sessions)
    return min(len(sessions), MAX_SESSIONS)
