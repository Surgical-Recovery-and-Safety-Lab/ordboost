import matplotlib.pyplot as plt
import numpy as np

from ordboost import OrdBoostRegressor, UniformBinMapper

# 1. Generate test data and fit regressor
rng = np.random.default_rng(42)
X_train, X_test = rng.normal(size=(500, 4)), rng.normal(size=(200, 4))
y_train = X_train[:, 0] * 2 + rng.normal(size=500)
y_test = X_test[:, 0] * 2 + rng.normal(size=200)

mapper = UniformBinMapper()  # Define the mapper instead of using str mapping
reg = OrdBoostRegressor(mapper=mapper).fit(X_train, y_train)

# 2. Extract continuous distribution object for test set
dist = reg.predict_dist(X_test)

# 3. Calculate PIT values: U = CDF(y_true)
pit_values = dist.cdf(y_test)

# 4. Plot PIT Histogram
plt.figure(figsize=(7, 4))
plt.hist(
    pit_values, bins=10, density=True, alpha=0.7, color="skyblue", edgecolor="black"
)
plt.axhline(1.0, color="red", linestyle="--", label="Ideal Uniformity")
plt.xlabel("PIT")
plt.ylabel("Density")
plt.title("PIT Histogram (Calibration Diagnostic)")
plt.legend()
plt.tight_layout()
plt.show()
