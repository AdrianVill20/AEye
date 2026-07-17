"""
train.py
------------------------------------------------------------------
Learn the mapping:  gaze features  ->  screen (x, y).

Model = StandardScaler -> PolynomialFeatures(degree 2) -> Ridge.
Why not plain LinearRegression (what the old code used)?
  - Eye-to-screen mapping curves slightly; a straight line under-fits it.
  - Degree-2 polynomial captures that curve...
  - ...while Ridge keeps the fit SMOOTH, so gaze interpolates nicely
    between the 9 calibration dots instead of snapping to them.

Run:  python train.py     (after calibrate.py)
"""

import json
import pickle

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit

from gaze_common import FEATURE_NAMES, DATA_PATH, GAZE_MODEL_PATH


def build_model():
    # RidgeCV cross-validates its own regularization strength (alpha) from a
    # wide range. Stronger alpha = smoother fit = less overfitting, which is
    # what we need now that we have 10 features + degree-2 interactions.
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        RidgeCV(alphas=np.logspace(-2, 3, 20)),
    )


with open(DATA_PATH) as f:
    data = json.load(f)
print(f"Loaded {len(data)} samples")

# Build X (features) and y (screen targets) in the fixed feature order.
X = np.array([[row[name] for name in FEATURE_NAMES] for row in data])
y = np.array([[row["target_x"], row["target_y"]] for row in data])

# HONEST split: group every sample by which screen CELL its target falls in
# (a 5x5 grid), then hold whole cells out. The test set is therefore screen
# regions the model NEVER trained on -- no near-duplicate frames leaking
# across the split. This is the accuracy number you can trust / cite.
groups = np.array([f"{min(int(r['target_x'] * 5), 4)}_{min(int(r['target_y'] * 5), 4)}"
                   for r in data])
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"{len(set(groups))} screen cells total; {len(set(groups[test_idx]))} held out for test")

model = build_model()
model.fit(X_train, y_train)


def report(name, Xs, ys):
    pred = model.predict(Xs)
    err = np.sqrt(np.sum((pred - ys) ** 2, axis=1))  # distance in 0..1 screen units
    print(f"{name}: R2={model.score(Xs, ys):.3f}  "
          f"mean_err={err.mean():.3f}  max_err={err.max():.3f}")


report("train", X_train, y_train)
report("test ", X_test, y_test)
print(f"chosen alpha (regularization): {model.named_steps['ridgecv'].alpha_}")

# Retrain on ALL data for the final saved model (more data = better).
final = build_model()
final.fit(X, y)
with open(GAZE_MODEL_PATH, "wb") as f:
    pickle.dump(final, f)
print(f"Saved model to {GAZE_MODEL_PATH}")
