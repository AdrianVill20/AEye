"""Train a student's Isolation Forest model from their calibration JSON.

Usually you don't need this - the calibration screen's "Train Model" button
does it in-app. To run it manually, use the venv that HAS scikit-learn
(app_video/.venv), from integration/app:

    ../../app_video/.venv/Scripts/python.exe train_cheat_model.py --user ichoy
"""

import argparse

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

from calibration_store import load as load_calib, FEATURES
from cheat_detector import user_model_path


def train(user, contamination=0.03):
    """Train and save a student's model from their calibration JSON.

    Returns {'samples', 'flagged', 'model_path'}. Shared by the CLI below and
    the in-app "Train Model" button. Raises FileNotFoundError if there is no
    calibration file, or ValueError if there aren't enough samples.
    """
    data = load_calib(user)              # FileNotFoundError if missing
    samples = data.get('samples', [])
    if len(samples) < 100:
        raise ValueError(
            f'Only {len(samples)} calibration samples for "{user}" - '
            f'record ~2 minutes of calibration first, then train again.')

    X = np.array([[s[f] for f in FEATURES] for s in samples], dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42)
    model.fit(Xs)
    flagged = int((model.predict(Xs) == -1).sum())

    out = user_model_path(user)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'scaler': scaler, 'model': model, 'features': FEATURES}, out)
    return {'samples': len(X), 'flagged': flagged, 'model_path': str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True,
                    help='student id (matches the calibration JSON / login)')
    ap.add_argument('--contamination', type=float, default=0.03,
                    help='expected fraction of unusual frames (default 0.03)')
    args = ap.parse_args()

    try:
        result = train(args.user, args.contamination)
    except FileNotFoundError:
        raise SystemExit(f'[TRAIN] No calibration file for "{args.user}" - run calibration first.')
    except ValueError as exc:
        raise SystemExit(f'[TRAIN] {exc}')

    pct = 100 * result['flagged'] / result['samples']
    print(f"[TRAIN] Trained on {result['samples']} samples; "
          f"flagged {result['flagged']} ({pct:.1f}%).")
    print(f"[TRAIN] Saved model -> {result['model_path']}")


if __name__ == '__main__':
    main()
