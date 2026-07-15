import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

with open('calibration_data.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

X = []
y = []

for sample in data:
    features = [
        sample['iris_ratio'],
        sample['vertical_iris_ratio'],
        sample['sclera_ratio'],
        sample['yaw'],
        sample['pitch'],
        sample['roll']
    ]
    target = [sample['target_x'], sample['target_y']]

    X.append(features)
    y.append(target)

X = np.array(X)
y = np.array(y)

# split into 80% training, 20% testing - testing data is "unseen" during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training R2 score: {train_score:.3f}")
print(f"Testing R2 score:  {test_score:.3f}")

# calculate average prediction error in normalized screen units
predictions = model.predict(X_test)
errors = np.sqrt(np.sum((predictions - y_test) ** 2, axis=1))
print(f"Average error: {errors.mean():.3f} (normalized 0-1 screen units)")
print(f"Worst error:   {errors.max():.3f}")

# now retrain on ALL data for the final saved model (more data = better)
final_model = LinearRegression()
final_model.fit(X, y)

with open('gaze_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("\nFinal model (trained on all data) saved to gaze_model.pkl")