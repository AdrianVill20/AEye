# Make the student's screen area from their calibration.
# Run by hand from integration/app:  ../.venv/Scripts/python.exe -m cheat.train_cheat_model --user ichoy

import argparse
import json
from statistics import median

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

from cheat.calibration_store import load_sessions
from cheat.cheat_detector import (CheatDetector, FEATURES, MARGIN, forest_model_path,
                                  graham_scan, grow, user_model_path)


def dot_points(session):
    # One point per dot (middle value of its frames).
    dots = {}
    for s in session['samples']:
        # skip test dots, and head turn frames from older calibrations
        if s.get('target') and not s.get('head') and not s.get('val'):
            dots.setdefault(tuple(s['target']), []).append((s['h_ratio'], s['v_openness']))
    return [(median(p[0] for p in d), median(p[1] for p in d))
            for d in dots.values() if len(d) >= 10]


def newest_dots(sessions):
    # Dot points from the newest good calibration.
    for session in reversed(sessions):
        points = dot_points(session)
        if len(points) >= 3:
            return points, session
    return [], None


def train_forest(user, samples, contamination=0.03):
    # Train the isolation forest on every calibration frame (old AEye code).
    X = np.array([[s[f] for f in FEATURES] for run, s in samples], dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42)
    model.fit(Xs)
    flagged = int((model.predict(Xs) == -1).sum())

    out = forest_model_path(user)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'scaler': scaler, 'model': model, 'features': FEATURES}, out)
    return len(X), flagged


def train(user):
    # Make and save the screen area.
    sessions = load_sessions(user)
    if not sessions:
        raise FileNotFoundError(f'No dot calibration saved for "{user}".')

    # screen area uses the newest calibration
    points, session = newest_dots(sessions)
    if not points:
        raise ValueError(f'No usable dot calibration for "{user}" - calibrate again, '
                         f'keeping your face in view.')

    # graham scan, then make it a bit bigger
    area = CheatDetector(grow(graham_scan(points), MARGIN))
    out = user_model_path(user)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'user': user, 'hull': [list(p) for p in area.hull]}, f)

    # check: test dots should be inside
    val = [s for s in session['samples'] if s.get('val')]
    inside = sum(not area.outside(s['h_ratio'], s['v_openness']) for s in val)

    # isolation forest on all frames of all calibrations
    samples = [(run, s) for run, session in enumerate(sessions) for s in session['samples']]
    frames, flagged = train_forest(user, samples)
    return {'dots': len(points), 'corners': len(area.hull), 'runs': len(sessions),
            'val_inside': inside, 'val_total': len(val),
            'frames': frames, 'flagged': flagged, 'model_path': str(out)}


def main():
    # Train from the command line.
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True,
                    help='student id (matches the calibration JSON / login)')
    args = ap.parse_args()

    try:
        result = train(args.user)
    except FileNotFoundError as exc:
        raise SystemExit(f'[TRAIN] {exc} Run calibration first.')
    except ValueError as exc:
        raise SystemExit(f'[TRAIN] {exc}')

    pct = 100 * result['val_inside'] / result['val_total'] if result['val_total'] else 0
    print(f"[TRAIN] Screen area from {result['dots']} dots of the newest run: "
          f"{result['corners']} corners.")
    print(f"[TRAIN] Validation frames inside the area: "
          f"{result['val_inside']}/{result['val_total']} ({pct:.0f}%).")
    print(f"[TRAIN] Isolation forest on {result['frames']} frames; "
          f"{result['flagged']} ({100 * result['flagged'] / result['frames']:.1f}%) count as unusual.")
    print(f"[TRAIN] Saved -> {result['model_path']}")


if __name__ == '__main__':
    main()
