"""Train a student's Isolation Forest model from their calibration JSON.

Run AFTER the student finishes calibration (which writes the JSON), from
integration/app:

    ../../venv/Scripts/python.exe train_cheat_model.py --user ichoy

The calibration screen shows the exact command with the right --user filled in.
"""

import argparse

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

from calibration_store import load as load_calib, FEATURES
from cheat_detector import user_model_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True,
                    help='student id (matches the calibration JSON / login)')
    ap.add_argument('--contamination', type=float, default=0.03,
                    help='expected fraction of unusual frames (default 0.03)')
    args = ap.parse_args()

    try:
        data = load_calib(args.user)
    except FileNotFoundError:
        raise SystemExit(f'[TRAIN] No calibration file for "{args.user}" - run calibration first.')

    samples = data.get('samples', [])
    if len(samples) < 100:
        raise SystemExit(
            f'[TRAIN] Only {len(samples)} calibration samples for "{args.user}" - '
            f'record ~2 minutes of calibration first, then train again.')

    X = np.array([[s[f] for f in FEATURES] for s in samples], dtype=float)
    print(f'[TRAIN] Training on {len(X)} samples for "{args.user}". Features = {FEATURES}')

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = IsolationForest(n_estimators=200, contamination=args.contamination,
                            random_state=42)
    model.fit(Xs)

    pred = model.predict(Xs)
    flagged = int((pred == -1).sum())
    print(f'[TRAIN] Flagged {flagged}/{len(X)} training rows as unusual '
          f'({100 * flagged / len(X):.1f}%) - expected ~{100 * args.contamination:.0f}%.')

    out = user_model_path(args.user)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'scaler': scaler, 'model': model, 'features': FEATURES}, out)
    print(f'[TRAIN] Saved model -> {out}')


if __name__ == '__main__':
    main()
