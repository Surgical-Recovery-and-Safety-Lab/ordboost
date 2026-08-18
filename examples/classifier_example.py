import numpy as np

from ordboost import OrdBoostClassifier, crps_score, pinball_loss

# 1. Prepare ordinal target dataset (e.g., ratings 1 to 5)
np.random.seed(42)
X_train = np.random.randn(200, 4)
y_train = np.random.choice([1, 2, 3, 4, 5], size=200)

X_test = np.random.randn(50, 4)
y_test = np.random.choice([1, 2, 3, 4, 5], size=50)

# 2. Fit OrdBoostClassifier
model = OrdBoostClassifier(
    max_iter=50,
    learning_rate=0.05,
    monotonicity="isotonic",
    max_leaf_nodes=15,  # Passed to underlying tree estimators
    random_state=42,
)
model.fit(X_train, y_train)

# 3. Predict point estimates (median or mean)
y_pred_median = model.predict(X_test, method="median")
y_pred_mean = model.predict(X_test, method="mean")

# 4. Predict probability distribution object
dist = model.predict_dist(X_test)

# Extract PMF, CDF, and custom quantiles
pmf = dist.pmf  # Shape: (50, 5)
cdf = dist.cdf  # Shape: (50, 5)
y_pred_q_90 = dist.ppf(0.90)  # 90th percentile predictions

# 5. Evaluate probabilistic performance using discrete CRPS
score = crps_score(y_test, dist)
q_90_loss = pinball_loss(y_test, y_pred_q_90, 0.9)

print(f"Discrete CRPS: {score:.4f}")
print(f"Pinball loss at q=0.9: {q_90_loss:.4f}")
