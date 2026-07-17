"""
train_zones.py
------------------------------------------------------------------
Train a classifier:  gaze features  ->  which ZONE (SCREEN / DESK / AWAY).

A RandomForest is a good fit here: it handles the non-linear boundaries
between zones, needs no feature scaling, and gives calibrated-ish
probabilities (predict_proba) that we smooth in the live view.

Run:  python train_zones.py   (after calibrate_zones.py)
"""

import json
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from gaze_common import FEATURE_NAMES, ZONE_DATA_PATH, ZONE_MODEL_PATH


def build_model():
    # max_depth caps complexity so the forest can't memorize noise.
    return RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)


with open(ZONE_DATA_PATH) as f:
    data = json.load(f)
print(f"Loaded {len(data)} samples")

X = np.array([[row[name] for name in FEATURE_NAMES] for row in data])
y = np.array([row["zone"] for row in data])

# stratify keeps each zone proportionally represented in train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = build_model()
model.fit(X_train, y_train)

print(f"train accuracy: {model.score(X_train, y_train):.3f}")
print(f"test  accuracy: {model.score(X_test, y_test):.3f}")
print("\nlabels:", list(model.classes_))
print("confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, model.predict(X_test), labels=list(model.classes_)))
print("\n" + classification_report(y_test, model.predict(X_test)))

# retrain on all data for the final saved model
final = build_model()
final.fit(X, y)
with open(ZONE_MODEL_PATH, "wb") as f:
    pickle.dump(final, f)
print(f"Saved model to {ZONE_MODEL_PATH}")
