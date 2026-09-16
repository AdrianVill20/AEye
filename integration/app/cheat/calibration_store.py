# Save and load each student's calibration runs (a new run is added, never overwritten).

import json
import re
from datetime import datetime

from paths import CALIB_DIR

MAX_SESSIONS = 10   # only keep the last 10 runs


def _safe(user_id):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(user_id)) if user_id else 'unknown'


def calib_path(user_id):
    return CALIB_DIR / f'calibration_{_safe(user_id)}.json'


def load_sessions(user_id):
    # Get all saved runs, oldest first.
    path = calib_path(user_id)
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data.get('sessions', [])   # old files have no runs


def save_sessions(user_id, sessions):
    # Save the runs to the file.
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    payload = {'user': user_id, 'sessions': sessions[-MAX_SESSIONS:]}
    path = calib_path(user_id)
    # write to a temp file first so a crash can't break it
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    tmp.replace(path)
    return path


def add_session(user_id, samples, screen):
    # Add one new run and return how many runs are saved.
    sessions = load_sessions(user_id)
    sessions.append({
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'screen': screen,
        'samples': samples,   # each: h_ratio, v_openness, yaw, pitch, roll, target, val
    })
    save_sessions(user_id, sessions)
    return min(len(sessions), MAX_SESSIONS)
