import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ordboost import OrdBoostRegressor
from ordboost.metrics import (
    baseline_distribution,
    crps_score,
    crps_skill_score,
    interval_coverage_rate,
    marginal_calibration_curve,
    pit_diagnostics,
    sharpness,
    winkler_score,
)

# --- Generate data
rng = np.random.default_rng(42)
n_samples = 10000
upper_bound = 10.0

regression_data = make_regression(
    n_samples=n_samples,
    n_features=8,
    n_informative=4,
    noise=3.0,
    random_state=42,
)
X = regression_data[0]
latent = regression_data[1]

latent_std = (latent - latent.mean()) / latent.std()  # rescale to a controlled range

# Zero-inflation: a clean, sharp logistic in the single latent driver
# (easy for the classifier to recover).
zero_prob = 1.0 / (1.0 + np.exp(-3.0 * latent_std))
is_zero = rng.random(n_samples) < zero_prob

# Continuous part: Beta parameterized by (mean, concentration) rather than
# a raw shape parameter.
beta_mean = 0.3 + 0.5 * (latent_std - latent_std.min()) / (
    latent_std.max() - latent_std.min()
)
beta_mean = np.clip(beta_mean, 0.05, 0.95)
concentration = 9.0
a = beta_mean * concentration
b = (1.0 - beta_mean) * concentration
continuous_part = upper_bound * rng.beta(a, b, size=n_samples)

y = np.where(is_zero, 0.0, continuous_part)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Plot y_test
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(y_test, bins=20, edgecolor="black")
ax.set_xlabel("Target")
ax.set_ylabel("Density (log)")
ax.set_yscale("log")
ax.set_title("Prediction target on a log scale")
plt.show()

# --- Create and fit models
bin_edges = [1.0, 2.5, 5.0, 7.5, 9.0]

model = OrdBoostRegressor(
    mapper="quantile",
    bin_edges=bin_edges,
    monotonicity="isotonic",
    max_iter=200,
    learning_rate=0.01,
    random_state=42,
    mapper_kwargs={
        "lower_bound": 0.0,
        "floor_atom": True,
        "upper_bound": upper_bound,
        "ceiling_atom": False,
    },
)

model_no_bounds = OrdBoostRegressor(
    mapper="quantile",
    bin_edges=bin_edges,
    monotonicity="isotonic",
    max_iter=200,
    learning_rate=0.01,
    random_state=42,
)

# Fit and predict with the bounded model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
dist_test = model.predict_dist(X_test)

# Fit and predict with the boundless model
model_no_bounds.fit(X_train, y_train)
y_pred_no_bounds = model_no_bounds.predict(X_test)
dist_test_no_bounds = model_no_bounds.predict_dist(X_test)

# --- CRPS and skill score

mae = mean_absolute_error(y_test, y_pred)
crps = crps_score(y_test, dist_test)

dist_baseline = baseline_distribution(y_train, n_samples=len(y_test))
crpss = crps_skill_score(y_test, dist_test, dist_baseline)

mae_no_bounds = mean_absolute_error(y_test, y_pred_no_bounds)
crps_no_bounds = crps_score(y_test, dist_test_no_bounds)
crpss_no_bounds = crps_skill_score(y_test, dist_test_no_bounds, dist_baseline)

print("        Bounded | Boundless")
print(f"MAE:     {mae:.3f}  |    {mae_no_bounds:.3f}")
print(f"CRPS:    {crps:.3f}  |    {crps_no_bounds:.3f}")
print(f"CRPSS:   {crpss:.3f}  |    {crpss_no_bounds:.3f}")

# --- Prediction interval evaluation

coverage_levels = np.arange(10, 91, 10)  # nominal coverage in percent
alphas = 1.0 - coverage_levels / 100.0

empirical_coverage = np.array(
    [interval_coverage_rate(y_test, dist_test, alpha=a) for a in alphas]
)
interval_sharpness = np.array([sharpness(dist_test, alpha=a) for a in alphas])
winkler = np.array([winkler_score(y_test, dist_test, alpha=a) for a in alphas])

empirical_coverage_no_bounds = np.array(
    [interval_coverage_rate(y_test, dist_test_no_bounds, alpha=a) for a in alphas]
)
interval_sharpness_no_bounds = np.array(
    [sharpness(dist_test_no_bounds, alpha=a) for a in alphas]
)
winkler_no_bounds = np.array(
    [winkler_score(y_test, dist_test_no_bounds, alpha=a) for a in alphas]
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(
    coverage_levels,
    empirical_coverage * 100,
    color="b",
    marker="o",
    label="Bounded",
)
axes[0].plot(
    coverage_levels,
    empirical_coverage_no_bounds * 100,
    color="r",
    marker="s",
    label="Boundless",
)
axes[0].plot([0, 100], [0, 100], linestyle="--", color="black")
axes[0].legend()
axes[0].set_xlabel("Nominal coverage (%)")
axes[0].set_ylabel("Empirical coverage (%)")
axes[0].set_title("Coverage reliability")

axes[1].plot(
    coverage_levels, interval_sharpness, color="b", marker="o", label="Bounded"
)
axes[1].plot(
    coverage_levels,
    interval_sharpness_no_bounds,
    color="r",
    marker="s",
    label="Boundless",
)
axes[1].legend()
axes[1].set_xlabel("Nominal coverage (%)")
axes[1].set_ylabel("Mean interval width")
axes[1].set_title("Sharpness")

axes[2].plot(coverage_levels, winkler, marker="o", color="b", label="Bounded")
axes[2].plot(
    coverage_levels,
    winkler_no_bounds,
    color="r",
    marker="s",
    label="Boundless",
)
axes[2].legend()
axes[2].set_xlabel("Nominal coverage (%)")
axes[2].set_ylabel("Winkler score")
axes[2].set_title("Winkler score")

fig.tight_layout()
plt.show()

# --- Calibration diagnostics

# Marginal calibration: aggregate comparison of the empirical CDF against
# the average predicted CDF across all test samples.
grid_y, calibration = marginal_calibration_curve(y_test, dist_test)
grid_y_no_bounds, calibration_no_bounds = marginal_calibration_curve(
    y_test, dist_test_no_bounds
)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(grid_y, calibration, color="b", linewidth=2, label="Bounded")
ax.plot(
    grid_y_no_bounds,
    calibration_no_bounds,
    color="r",
    linewidth=2,
    label="Boundless",
)
ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
ax.legend()
ax.set_xlabel("y")
ax.set_ylabel(r"eCDF $-\ \overline{CDF}$")
ax.set_title("Marginal calibration curve")
plt.show()

# Randomized PIT histogram: per-sample calibration check, aware of the
# floor atom via model.mapper_.floor_atom (True here) and correctly
# treating the ceiling as a plain boundary rather than an atom, since
# model.mapper_.ceiling_atom is False.
pit = pit_diagnostics(y_test, dist_test, model.mapper_)
hist = pit.hist_values(bins=20)
alpha_score = pit.alpha_score()

pit_no_bounds = pit_diagnostics(
    y_test,
    dist_test_no_bounds,
    model_no_bounds.mapper_,
)
hist_no_bounds = pit_no_bounds.hist_values(bins=20)
alpha_score_no_bounds = pit_no_bounds.alpha_score()
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(
    hist["bin_centre"].values - 0.3 / 20,
    np.asarray(hist.values),
    width=0.3 / 20,
    color="b",
    edgecolor="black",
    label=rf"Bounded ($\alpha$ score = {alpha_score:.3f})",
)
ax.bar(
    hist_no_bounds["bin_centre"].values + 0.1 / 20,
    np.asarray(hist_no_bounds.values),
    width=0.3 / 20,
    color="r",
    edgecolor="black",
    label=rf"Boundless ($\alpha$ score = {alpha_score_no_bounds:.3f})",
)
ax.axhline(1.0 / 20, color="black", linestyle="--", linewidth=1, label="Uniform")
ax.set_xlabel("PIT value")
ax.set_ylabel("Proportion")
ax.set_title(f"PIT histograms")
ax.legend()
plt.show()
