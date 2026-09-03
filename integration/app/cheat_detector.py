"""Isolation Forest cheat / anomaly detector (profile-based).

Each student gets their OWN model file (models/cheat_model_<user>.joblib),
trained by train_cheat_model.py on that student's calibration JSON. The model
flags behaviour that is UNUSUAL compared to their normal sample.

Fail-safe: if the student has no model yet (or scikit-learn / joblib is
missing), load() returns a 'not ready' detector that never flags, so the
camera keeps working with detection simply turned off.
"""

import re
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / 'models'

# Feature order - MUST match training and calibration_store.FEATURES.
FEATURES = ['h_ratio', 'v_openness', 'yaw', 'pitch', 'roll']


def user_model_path(user_id):
    """Per-student model file, e.g. models/cheat_model_ichoy.joblib."""
    safe = re.sub(r'[^A-Za-z0-9_-]+', '_', str(user_id)) if user_id else 'unknown'
    return MODELS_DIR / f'cheat_model_{safe}.joblib'


class CheatDetector:
    def __init__(self, scaler=None, model=None):
        self.scaler = scaler
        self.model = model

    @property
    def ready(self):
        return self.scaler is not None and self.model is not None

    @classmethod
    def load(cls, user_id=None):
        """Load this student's model. Never raises - on any failure returns a
        not-ready detector (is_anomaly always False)."""
        try:
            import joblib
            bundle = joblib.load(user_model_path(user_id))
            print(f'[CHEAT] Loaded model for "{user_id}".')
            return cls(scaler=bundle['scaler'], model=bundle['model'])
        except Exception as exc:
            print(f'[CHEAT] No model for "{user_id}" ({exc}); detection off.')
            return cls()

    def is_anomaly(self, values):
        """values: the 5 features in FEATURES order. True = unusual (outlier)."""
        if not self.ready:
            return False
        try:
            import numpy as np
            x = np.asarray(values, dtype=float).reshape(1, -1)
            x = self.scaler.transform(x)
            return self.model.predict(x)[0] == -1   # IsolationForest: -1 = outlier
        except Exception:
            return False
