# Isolation Forest trained in Google Colab (colab/AEye_Isolation_Forest.ipynb)
# on videos of NORMAL exam behaviour. It answers one question per frame:
# "is this head pose + eye position unusual compared with the normal videos?"
#
# The notebook exports models/isolation_<name>.npz holding
#   - X     : the normal training frames (5 numbers each, see FEATURES)
#   - meta  : JSON text with the forest settings and the chosen cut-off
# The app re-builds the forest from X with the same settings instead of
# un-pickling a Colab model, so it works no matter which numpy / scikit-learn
# versions Colab had.
import json
import re

import numpy as np

from paths import MODELS_DIR
from cheat.frame_features import FEATURES

GENERAL_NAME = 'general'     # the model used when a student has no model of their own


def model_path(name):
    safe = re.sub(r'[^A-Za-z0-9_-]+', '_', str(name)) if name else GENERAL_NAME
    return MODELS_DIR / f'isolation_{safe}.npz'


def available_models():
    # names of every exported forest, e.g. ['general', 'adrian']
    return sorted(p.stem.removeprefix('isolation_') for p in MODELS_DIR.glob('isolation_*.npz'))


class IsolationModel:

    def __init__(self, X, meta):
        from sklearn.ensemble import IsolationForest
        self.meta = meta
        self.features = meta.get('features', FEATURES)
        if list(self.features) != FEATURES:
            raise ValueError(f'model features {self.features} != app features {FEATURES}')
        self.forest = IsolationForest(
            n_estimators=int(meta['n_estimators']),
            max_samples=meta['max_samples'],
            random_state=int(meta['random_state']),
        ).fit(X)
        # anomaly score = -score_samples, about 0.3 (very normal) to 1 (very unusual).
        # The cut-off is stored as a percentile of the training scores, so it
        # means the same thing after the re-build.
        train_scores = -self.forest.score_samples(X)
        self.threshold = float(np.percentile(train_scores, float(meta['threshold_percentile'])))

    @classmethod
    def load(cls, user=None):
        """The student's own forest if there is one, else the general one, else None."""
        for name in (user, GENERAL_NAME):
            if not name:
                continue
            path = model_path(name)
            if not path.exists():
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    X = data['X'].astype(float)
                    meta = json.loads(str(data['meta']))
                model = cls(X, meta)
                model.name = name
                print(f'[FOREST] Loaded {path.name} ({len(X)} normal frames, '
                      f'cut-off {model.threshold:.3f}).')
                return model
            except Exception as exc:
                print(f'[FOREST] Could not load {path.name}: {exc}')
        print(f'[FOREST] No isolation forest for "{user}" or "{GENERAL_NAME}"; forest off.')
        return None

    def score(self, row):
        # anomaly score of one frame (row = the 5 FEATURES in order); higher = more unusual
        return float(-self.forest.score_samples(np.asarray([row], dtype=float))[0])

    def is_unusual(self, row):
        return self.score(row) > self.threshold


if __name__ == '__main__':      # quick check: python -m cheat.isolation_forest
    import sys
    model = IsolationModel.load(sys.argv[1] if len(sys.argv) > 1 else None)
    if model is not None:
        print('settings:', {k: model.meta[k] for k in ('n_estimators', 'max_samples',
                                                       'threshold_percentile')})
        print('validation:', json.dumps(model.meta.get('validation', {}), indent=2))
