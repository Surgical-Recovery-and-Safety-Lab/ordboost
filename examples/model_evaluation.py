import numpy as np

from ordboost import OrdBoostRegressor
from ordboost.metrics import interval_coverage_rate, winkler_score

# Fit model
X = np.random.randn(300, 3)
y = X[:, 0] * 1.5 + np.random.normal(0, 0.5, 300)

reg = OrdBoostRegressor().fit(X, y)  # Use default OrdBoostRegressor parameters
dist = reg.predict_dist(X)

# Evaluate alpha levels (e.g., 90%, 80%, and 50% intervals)
for alpha in [0.10, 0.20, 0.50]:
    target_coverage = 1.0 - alpha
    actual_coverage = interval_coverage_rate(y, dist, alpha=alpha)
    w_score = winkler_score(y, dist, alpha=alpha)

    print(
        f"Target Coverage: {target_coverage:.0%} | "
        f"Actual Coverage: {actual_coverage:.1%} | "
        f"Winkler Score: {w_score:.3f}"
    )
