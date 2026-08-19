# OrdBoost

[![PyPI Version](https://img.shields.io/pypi/v/ordboost.svg)](https://pypi.org/project/ordboost/)
[![PyPI Python Versions](https://img.shields.io/pypi/pyversions/ordboost.svg)](https://pypi.org/project/ordboost/)
[![License](https://img.shields.io/github/license/Surgical-Recovery-and-Safety-Lab/ordboost)](LICENSE)
[![tests](https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/actions/workflows/run_test.yml/badge.svg)](https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/actions/workflows/run_test.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526CFE?style=flat&logo=materialforgithub)](https://Surgical-Recovery-and-Safety-Lab.github.io/ordboost/)

## Table of content
1. [Overview](#overview)
	1. [Key features](#features)
2. [Installation](#installation)
3. [Quickstart examples](#examples)
	1. [OrdBoostClassifier](#ordboostclassifier)
	2. [OrdBoostRegressor](#ordboostregressor)
4. [Acknowledgements](#acknowledgements)

---

## Overview

**ordboost** is a Python library providing ordinal-binning gradient boosting models for discrete and continuous targets with full scikit-learn compatibility.

By framing continuous regression as a dynamic ordinal binning problem, **ordboost** generates full, non-parametric probabilistic distributions without forcing strict Gaussian or parametric assumptions on target data.

### Key Features

* **Scikit-Learn API Compatibility**: Fits seamlessly into standard ML workflows using `fit`, `predict`, and `predict_dist`.
* **Non-Parametric Probabilistic Output**: Obtains complete predictive distribution objects capable of extracting probability mass functions (PMF), cumulative distribution functions (CDF), percentiles (`ppf`), and dynamic prediction intervals.
* **Flexible Continuous Target Mapping**: Maps continuous values to discrete target spaces using configurable binning strategies (`QuantileBinMapper`, `UniformBinMapper`, `EmpiricalMeanBinMapper`, `EmpiricalMedianBinMapper`, `ContinuousBinMapper`).
* **Monotonic Ordinal Constraints**: Supports constrained ordinal boosting (e.g., isotonic constraints) across sequential boundaries.
* **Built-in Probabilistic Evaluation**: Evaluates probabilistic predictions directly using CRPS (`crps_score`), quantile loss (`pinball_loss`), prediction interval coverage (`interval_coverage_rate`), and Winkler scores (`winkler_score`).

---

## Installation

Install the latest release from PyPI:

```bash
pip install ordboost

```

*Note: It is recommended to install the package inside a virtual environment (`venv` or `conda`).*

---

## Quickstart Examples

### OrdBoostClassifier

``` python linenums=1, title="OrdBoostClassifier example"
import numpy as np
from ordboost import OrdBoostClassifier, crps_score, pinball_loss

# 1. Prepare ordinal target dataset (e.g., discrete ratings 1 to 5)
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
    max_leaf_nodes=15,
    random_state=42,
)
model.fit(X_train, y_train)

# 3. Predict point estimates (mean or median)
y_pred_mean = model.predict(X_test, method="mean")

# 4. Extract discrete probability distribution
dist = model.predict_dist(X_test)
pmf = dist.pmf  # Probability mass function shape: (50, 5)
cdf = dist.cdf  # Cumulative distribution function shape: (50, 5)
y_pred_q90 = dist.ppf(0.90)  # 90th percentile prediction

# 5. Evaluate probabilistic performance
crps = crps_score(y_test, dist)
p_loss = pinball_loss(y_test, y_pred_q90, alpha=0.9)

print(f"Discrete CRPS: {crps:.4f}")
print(f"Pinball Loss (q=0.9): {p_loss:.4f}")

```

### OrdBoostRegressor

``` python linenums=1, title="OrdBoostRegressor example"
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
    
```

---

## Acknowledgements

This package was developed using Gemini 3.6 Thinking. The code was reviewed and edited by humans.
