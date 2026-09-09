import numpy as np
from sklearn.model_selection import train_test_split

from ordboost import OrdBoostRegressor, interval_coverage_rate, sharpness, winkler_score

# --- Generate data
rng = np.random.default_rng(42)  # For reproducibility
n_samples = 2000

X = rng.standard_normal((n_samples, 3))
y = 10.0 + X[:, 0] * 4.0 + X[:, 1] * 2.5 + rng.standard_normal(n_samples) * 3

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Fit
reg = OrdBoostRegressor().fit(
    X_train, y_train
)  # Use default OrdBoostRegressor parameters
dist = reg.predict_dist(X_test)

# --- Evaluate
# Define alpha levels
alphas = [0.10, 0.20, 0.5]
n_alphas = len(alphas)

# Placeholders for the interval evaluation values
nominal_coverage = np.zeros((n_alphas,))
coverage = np.zeros((n_alphas,))
w_score = np.zeros((n_alphas,))
sharp = np.zeros((n_alphas,))

# Evaluate alpha levels (e.g., 90%, 80%, and 50% intervals)
for i, alpha in enumerate([0.10, 0.20, 0.50]):
    nominal_coverage[i] = 1.0 - alpha
    coverage[i] = interval_coverage_rate(y_test, dist, alpha=alpha)
    w_score[i] = winkler_score(y_test, dist, alpha=alpha)
    sharp[i] = sharpness(dist, alpha=alpha)


# Print results as a table
print("Nominal coverage | Coverage | Sharpness | Winkler score")
print("-------------------------------------------------------")
for i in range(n_alphas):
    n_cov = f"      {nominal_coverage[i]:.0%}        "
    cov = f"  {coverage[i]:.1%}   "
    sh = f"    {sharp[i]:.2f}   "
    wink = f"    {w_score[i]:.2f}"

    print("|".join([n_cov, cov, sh, wink]))
