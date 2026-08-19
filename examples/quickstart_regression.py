import numpy as np

from ordboost.models import OrdBoostRegressor

# 1. Generate synthetic continuous regression data with non-linear skew
rng = np.random.default_rng(42)
X_train = rng.standard_normal((300, 3))
y_train = np.exp(X_train[:, 0] * 0.6) + rng.normal(0.0, 0.5, size=300)

X_test = rng.standard_normal((1, 3))

# 2. Instantiate and fit OrdBoostRegressor
reg = OrdBoostRegressor(
    bin_edges=[0.0, 1.0, 2.5, 5.0, 15.0],
    mapper="mean",
    learning_rate=0.05,
    max_iter=50,
    random_state=42,
)
reg.fit(X_train, y_train)

# 3. Generate continuous point predictions
y_pred_median = reg.predict(X_test, method="median")

# Evaluate 80% prediction intervals
dist = reg.predict_dist(X_test)
lower_80, upper_80 = dist.interval(alpha=0.20)
prob_under_3 = dist.cdf(3.0)

# Display predictions for test sample
print(
    f"Predicted Median: {y_pred_median[0]:.2f} "
    f"[80% PI: {lower_80[0]:.2f}, {upper_80[0]:.2f}]"
)
print(f"P(Y <= 3.0): {prob_under_3[0]:.2%}\n")
