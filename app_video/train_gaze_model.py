import json
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# load the calibration data we collected
with open('calibration_data.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# build the feature matrix (X) and target matrix (y)
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

print(f"Feature matrix shape: {X.shape}")  # should be (num_samples, 6)
print(f"Target matrix shape: {y.shape}")   # should be (num_samples, 2)

# train the regression model
model = LinearRegression()
model.fit(X, y)

# check how well it fits the training data itself (rough sanity check)
train_score = model.score(X, y)
print(f"Training R2 score: {train_score:.3f}")  # closer to 1.0 is better fit

# save the trained model so we can use it later without retraining
with open('gaze_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to gaze_model.pkl")

# quick manual test: predict on the first sample and compare to actual target
test_features = X[0].reshape(1, -1)
predicted = model.predict(test_features)
print(f"\nSample prediction check:")
print(f"  Actual target: {y[0]}")
print(f"  Predicted:     {predicted[0]}")