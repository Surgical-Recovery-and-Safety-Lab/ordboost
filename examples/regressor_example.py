import numpy as np

from ordboost.models import OrdBoostRegressor

# 1. Generate synthetic continuous regression data with non-linear skew
rng = np.random.default_rng(42)
X_train = rng.standard_normal((300, 3))
y_train = np.exp(X_train[:, 0] * 0.6) + rng.normal(0.0, 0.5, size=300)

X_test = rng.standard_normal((3, 3))

# 2. Instantiate and fit OrdBoostRegressor using dependency injection
reg = OrdBoostRegressor(
    bin_edges=[0.0, 1.0, 2.5, 5.0, 15.0],
    mapper="quantile",
    mapper_kwargs={"quantiles": (0.10, 0.25, 0.50, 0.75, 0.90)},
    learning_rate=0.05,
    max_iter=50,
    random_state=42,
)
reg.fit(X_train, y_train)

# 3. Generate continuous point predictions
y_pred_mean = reg.predict(X_test, method="mean")
y_pred_median = reg.predict(X_test, method="median")

# 4. Extract continuous predictive distribution object
dist = reg.predict_dist(X_test)

# Evaluate 80% prediction intervals and cumulative probability P(Y <= 3.0)
lower_80, upper_80 = dist.interval(alpha=0.20)
prob_under_3 = dist.cdf(3.0)

# Display predictions for test samples
for i in range(len(X_test)):
    print(f"Sample {i + 1}:")
    print(f"  Predicted Mean:      {y_pred_mean[i]:.2f}")
    print(f"  Predicted Median:    {y_pred_median[i]:.2f}")
    print(f"  80% Interval:        [{lower_80[i]:.2f}, {upper_80[i]:.2f}]")
    print(f"  P(Y <= 3.0):         {prob_under_3[i]:.2%}\n")
