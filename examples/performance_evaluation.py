import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ordboost import OrdBoostRegressor
from ordboost.metrics import baseline_distribution, crps_score, crps_skill_score

# --- Generate data ---

rng = np.random.default_rng(42)
n_samples = 1000
X = rng.standard_normal((n_samples, 3))
y = X[:, 0] * 8.0 + X[:, 1] * 3.0 + rng.standard_normal(n_samples) * 4.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Fit

model = OrdBoostRegressor(n_bins=10, random_state=42)
model.fit(X_train, y_train)

# --- Predict
y_pred = model.predict(X_test, method="median")
dist_test = model.predict_dist(X_test)

# --- Mean Absolute Error
# point-prediction accuracy
mae = mean_absolute_error(y_test, y_pred)

# --- CRPS ---
# proper scoring rule over the full predicted distribution
crps = crps_score(y_test, dist_test)

# --- CRPS skill score
# skill relative to a naive, covariate-free baseline
dist_baseline = baseline_distribution(y_train, n_samples=len(y_test))
crpss = crps_skill_score(y_test, dist_test, dist_baseline)

print(f"MAE:   {mae:.3f}")
print(f"CRPS:  {crps:.3f}")
print(f"CRPSS: {crpss:.3f}")
