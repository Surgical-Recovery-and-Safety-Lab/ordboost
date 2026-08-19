# Welcome to OrdBoost

**ordboost** is a Python library for non-parametric discrete ordinal gradient boosting and continuous regression. 

By reframing continuous targets as discrete ordinal binning problems, **ordboost** models complex, skewed, or multimodal target distributions without forcing restrictive parametric assumptions.

---

## Key Features

* **Scikit-Learn API Compatibility**: Fits seamlessly into standard ML workflows using `fit`, `predict`, and `predict_dist`.
* **Non-Parametric Probabilistic Output**: Obtains complete predictive distribution objects capable of extracting probability mass functions (PMF), cumulative distribution functions (CDF), percentiles (`ppf`), and dynamic prediction intervals.
* **Flexible Continuous Target Mapping**: Maps continuous values to discrete target spaces using configurable binning strategies (`QuantileBinMapper`, `UniformBinMapper`, `EmpiricalMeanBinMapper`, `EmpiricalMedianBinMapper`, `ContinuousBinMapper`).
* **Monotonic Ordinal Constraints**: Supports constrained ordinal boosting (e.g., isotonic constraints) across sequential boundaries.
* **Built-in Probabilistic Evaluation**: Evaluates probabilistic predictions directly using CRPS (`crps_score`), quantile loss (`pinball_loss`), prediction interval coverage (`interval_coverage_rate`), and Winkler scores (`winkler_score`).

---

## Quickstart example

Get started with continuous probabilistic regression in just a few lines of code:

``` py linenums="1", title="OrdBoost regression example"
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
print(f"Predicted Median: {y_pred_median[i]:.2f} [80% PI: {lower_80:.2f}, {upper_80:.2f}]")
```

## Installation

### Standard Installation

Install the published package directly from PyPI:

```bash
pip install ordboost

```

### Installing from source (Developer setup)

To set up a local development environment and contribute to **ordboost**:

1. Clone the GitHub repository:

```bash
git clone [https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost.git](https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost.git)
cd ordboost

```

2. Install the package in editable mode with development dependencies:
```bash
pip install -e ".[dev]"

```

---

## Explore examples

For step-by-step code walkthroughs and diagnostic workflows, visit the **[Examples Page](examples.md)**:

* [Quickstart examples](examples.md#quickstart-examples): Basic discrete ordinal ranking and continuous prediction interval workflows.
* [Evaluating model quality](examples.md#evaluating-model-quality): Model calibration diagnostics using PIT histograms, interval coverage, and Winkler scores.
* [Customising Modelling Strategies](examples.md#customising-modelling-strategies): Creating custom target mappers by extending `BaseBinMapper`.

---

## Contributing

We welcome contributions from the community! Whether you are fixing bugs, improving documentation, or proposing new features:

1. Feel free to open an issue or start a discussion on our **[GitHub Repository](https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost)**.

2. Submit Pull Requests targeting the `main` branch.
3. Ensure all unit tests pass before submitting (`pytest`).

---

## License

This project is licensed under the Apache-2.0 License. Developed and maintained by the **[Surgical Recovery and Safety Lab](https://github.com/Surgical-Recovery-and-Safety-Lab)**.

