import numpy as np

from ordboost import OrdBoostClassifier, crps_score, pinball_loss

# --- Generate data
# Ordinal targets (e.g., pain scale 0-4)
rng = np.random.default_rng(42)  # For reproducibility
X_train = rng.random((250, 5))
y_train = rng.choice([0, 1, 2, 3, 4], size=250, p=[0.1, 0.2, 0.4, 0.2, 0.1])
X_test = rng.random((3, 5))

# --- Fit model
model = OrdBoostClassifier(monotonicity="isotonic").fit(X_train, y_train)

# --- Predict
# Three ways to make predictions
predictions = model.predict(X_test, method="median")
dist = model.predict_dist(X_test)  # Predictive distributions
medians = dist.median()
ppf_median = dist.ppf(0.5)  # Percent Point Function

print(f"Sample 1 median from predict(): {predictions[1]}")
print(f"Sample 1 median from median(): {medians[1]}")
print(f"Sample 1 median from ppf(0.5): {ppf_median[1]}")
