import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from ordboost import (
    OrdBoostRegressor,
    UniformBinMapper,
    marginal_calibration_curve,
    pit_diagnostics,
)

# --- Generate data
rng = np.random.default_rng(42)  # For reproducibility
n_samples = 6000

X = rng.standard_normal((n_samples, 3))
y = 10.0 + X[:, 0] * 4.0 + X[:, 1] * 2.5 + rng.standard_normal(n_samples) * 3

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Fit
mapper = UniformBinMapper(n_points=3)  # Define the mapper
reg = OrdBoostRegressor(mapper=mapper).fit(X_train, y_train)

# Extract continuous distribution object for test set
dist = reg.predict_dist(X_test)

# --- PIT
# Calculate PIT values
n_bins = 20
pit = pit_diagnostics(y_test, dist, mapper, precision=2)
pit_values = pit.hist_values(bins=n_bins)
alpha_value = pit.alpha_score()

# Plot histogram
plt.figure(figsize=(7, 4))
plt.bar(
    pit_values["bin_centre"].values,
    np.asarray(pit_values.values),
    width=1 / n_bins,
    label=rf"PIT $\alpha$ score = {alpha_value:.2f}",
)
plt.axhline(1 / n_bins, color="red", linestyle="--", label="Ideal Uniformity")
plt.xlabel("PIT")
plt.ylabel("Density")
plt.title("PIT Histogram")
plt.legend()
plt.tight_layout()
plt.show()


# --- Marginal calibration
grid, marginal_calibration = marginal_calibration_curve(y_test, dist)

# Plot marignal calibration
plt.plot(grid, marginal_calibration, linewidth=2, label="Model calibration")
plt.axhline(0, color="red", linestyle="--", label="Ideal calibration")
plt.xlabel("Grid point")
plt.ylabel("Marginal calibration")
plt.title("Marginal calibration curve")
plt.tight_layout()
plt.legend()
plt.show()
