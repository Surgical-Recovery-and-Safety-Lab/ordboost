## Examples

This page provides end-to-end examples demonstrating advanced features, custom extensions, and diagnostic workflows with **ordboost**.

---

### Quickstart examples

#### 1. Discrete ordinal ranking and CRPS evaluation

Use `OrdBoostClassifier` for discrete ordinal problems (like Likert survey responses or stage rankings) and score probabilistic performance via discrete CRPS.

```python linenums=1 title="OrdBoost classification example"
import numpy as np

from ordboost import OrdBoostClassifier, crps_score, pinball_loss

# Ordinal targets (e.g., pain scale 0-4)
X_train = np.random.randn(250, 5)
y_train = np.random.choice([0, 1, 2, 3, 4], size=250, p=[0.1, 0.2, 0.4, 0.2, 0.1])

model = OrdBoostClassifier(monotonicity="isotonic").fit(X_train, y_train)
dist = model.predict_dist(X_train)

# Calculate metrics
alpha = 0.75
crps = crps_score(y_train, dist)
p_loss_75 = pinball_loss(y_train, dist.ppf(alpha), q=alpha)

print(f"Discrete CRPS: {crps:.4f}")
print(f"Pinball Loss (75th percentile): {p_loss_75:.4f}")

```

#### 2. Continuous predictions with prediction intervals

Use `OrdBoostRegressor` for continuous problems and estimate prediction intervals for the outcome.

```python linenums=1 title="OrdBoost regression example"
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

---

### Evaluating model quality

#### 3. Model diagnostics with PIT histograms

The Probability Integral Transform (PIT) measures how well-calibrated a probabilistic regression model is.
If predictions are properly calibrated, $U_i = F_i(y_i)$ will follow a Uniform distribution $\mathcal{U}(0, 1)$.

```python linenums=1 title="Plotting the PIT histogram"
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
```

#### 4. Evaluating interval coverage and Winkler Scores

Assess prediction interval calibration and tightness across multiple confidence levels using `interval_coverage_rate` and `winkler_score`.

```python linenums=1 title="Interval coverage and Winkler scores"
import numpy as np
from ordboost import OrdBoostRegressor, QuantileBinMapper
from ordboost.metrics import interval_coverage_rate, winkler_score

# Fit model
X = np.random.randn(300, 3)
y = X[:, 0] * 1.5 + np.random.normal(0, 0.5, 300)

reg = OrdBoostRegressor(mapper=QuantileBinMapper(n_bins=8)).fit(X, y)
dist = reg.predict_dist(X)

# Evaluate alpha levels (e.g., 90%, 80%, and 50% intervals)
for alpha in [0.10, 0.20, 0.50]:
    target_coverage = 1.0 - alpha
    actual_coverage = interval_coverage_rate(y, dist, alpha=alpha)
    w_score = winkler_score(y, dist, alpha=alpha)
    
    print(f"Target Coverage: {target_coverage:.0%} | "
          f"Actual Coverage: {actual_coverage:.1%} | "
          f"Winkler Score: {w_score:.3f}")
```

---

### Customising modelling strategies
#### 5. Custom Bin Mapper (Inheriting from `BaseBinMapper`)

`OrdBoostRegressor` accepts any custom mapper that inherits from `BaseBinMapper`. Below is an example of creating a custom version of the EmpiricalMeanBinMapper:

```python linenums=1 title="Custom Bin Mapper example"
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans
from sklearn.base import check_is_fitted

from ordboost.distributions import ContinuousPredictiveDistribution
from ordboost.mappers import BaseBinMapper

class MeanBinMapper(BaseBinMapper):
    """Maps discrete bin probabilities using empirical intra-bin means.

    Computes the empirical mean of continuous targets within each bin during
    fitting. Maps discrete probability mass functions (PMF) to continuous
    expected target values using these fitted means. Empty bins are filled by
    interpolating between adjacent non-empty bin means.

    Parameters
    ----------
    bin_edges : array-like of shape (n_bins + 1,) or None, default=None
        Monotonically increasing boundaries defining continuous bin intervals.

    Attributes
    ----------
    bin_edges_ : np.ndarray
        1D float array of shape (n_bins + 1,) containing validated bin edges.
    bin_means_ : np.ndarray
        1D float array of shape (n_bins,) containing empirical means of
        continuous targets within each bin.
    n_bins_ : int
        Number of discrete bins defined by `bin_edges_`.

    Methods
    -------
    fit(y_continuous, y_binned=None)
        Compute empirical bin means from continuous training targets.
    transform(pmf)
        Map discrete PMF probability matrix to continuous expected values.
    to_continuous_dist(pmf)
        Construct a ContinuousPredictiveDistribution from a discrete PMF matrix.

    """

    def __init__(self, bin_edges: Union[ArrayLike, None] = None) -> None:
        super().__init__(bin_edges=bin_edges)

    def fit(
        self, y_continuous: ArrayLike, y_binned: Union[ArrayLike, None] = None
    ) -> "EmpiricalMeanBinMapper":
        """Compute empirical bin means from continuous training targets.

        Parameters
        ----------
        y_continuous : array-like of shape (n_samples,)
            Unbinned continuous target values (e.g., exact physical units).
        y_binned : array-like of shape (n_samples,), optional
            Corresponding 0-indexed discrete bin labels. If None, labels are
            computed automatically from `bin_edges`.

        Returns
        -------
        EmpiricalMeanBinMapper
            Fitted mapper instance.

        Raises
        ------
        ValueError
            If `bin_edges` is invalid, `y_continuous` is not 1D, or
            `y_binned` shape mismatches `y_continuous`.

        """
        edges = self._validate_edges()
        y_cont = np.asarray(y_continuous, dtype=float)

        if y_cont.ndim != 1:
            raise ValueError("Expected 'y_continuous' to be a 1D array.")

        self.bin_edges_ = edges
        self.n_bins_ = len(edges) - 1

        if y_binned is None:
            # Digitize continuous targets into 0-indexed bins [0, n_bins - 1]
            binned = np.digitize(y_cont, edges[1:-1])
        else:
            binned = np.asarray(y_binned, dtype=int)
            if binned.shape != y_cont.shape:
                raise ValueError(
                    f"Shape mismatch: 'y_binned' shape {binned.shape} "
                    f"does not match 'y_continuous' shape {y_cont.shape}."
                )

        self.bin_means_ = np.empty(self.n_bins_, dtype=float)
        empty_bins = []

        for k in range(self.n_bins_):
            mask = binned == k
            if np.any(mask):
                self.bin_means_[k] = np.mean(y_cont[mask])
            else:
                empty_bins.append(k)

        if empty_bins:
            valid_bins = np.setdiff1d(np.arange(self.n_bins_), empty_bins)
            if len(valid_bins) > 0:
                self.bin_means_[empty_bins] = np.interp(
                    empty_bins, valid_bins, self.bin_means_[valid_bins]
                )
            else:
                # Fallback to geometric midpoints if all bins are empty
                self.bin_means_ = (edges[:-1] + edges[1:]) / 2.0

        return self

    def transform(self, pmf: ArrayLike) -> np.ndarray:
        """Map discrete PMF probability matrix to continuous expected values.

        Parameters
        ----------
        pmf : array-like of shape (n_samples, n_bins)
            Probability mass function matrix where rows sum to 1.0.

        Returns
        -------
        np.ndarray
            1D float array of shape (n_samples,) containing continuous
            expected target values.

        Raises
        ------
        NotFittedError
            If the mapper instance has not been fitted prior to calling transform.
        ValueError
            If `pmf` is not a 2D array or column count does not match `n_bins_`.

        """
        check_is_fitted(self, attributes=["bin_edges_", "bin_means_", "n_bins_"])
        pmf_arr = np.asarray(pmf, dtype=float)

        if pmf_arr.ndim != 2:
            raise ValueError("Expected 'pmf' to be a 2D array.")
        if pmf_arr.shape[1] != self.n_bins_:
            raise ValueError(
                f"PMF column dimension ({pmf_arr.shape[1]}) does not match "
                f"fitted bin count ({self.n_bins_})."
            )

        return np.dot(pmf_arr, self.bin_means_)

    def to_continuous_dist(self, pmf: ArrayLike) -> ContinuousPredictiveDistribution:
        """Construct a ContinuousPredictiveDistribution from a discrete PMF matrix.

        Parameters
        ----------
        pmf : array-like of shape (n_samples, n_bins)
            Discrete probability mass function matrix where rows sum to 1.0.

        Returns
        -------
        ContinuousPredictiveDistribution
            Continuous distribution evaluated over physical target grid.

        Raises
        ------
        NotFittedError
            If the mapper instance has not been fitted prior to calling.
        ValueError
            If `pmf` is not a 2D array or column count does not match `n_bins_`.

        """
        check_is_fitted(self, attributes=["bin_edges_", "bin_means_", "n_bins_"])
        pmf_arr = np.asarray(pmf, dtype=float)

        if pmf_arr.ndim != 2:
            raise ValueError("Expected 'pmf' to be a 2D array.")
        if pmf_arr.shape[1] != self.n_bins_:
            raise ValueError(
                f"PMF column dimension ({pmf_arr.shape[1]}) does not match "
                f"fitted bin count ({self.n_bins_})."
            )

        cum_pmf = np.cumsum(pmf_arr, axis=1)
        grid_cdf = np.hstack(
            [
                np.zeros((pmf_arr.shape[0], 1), dtype=float),
                cum_pmf,
            ]
        )

        return ContinuousPredictiveDistribution(
            grid_y=self.bin_edges_, grid_cdf=grid_cdf
        )

# Usage with OrdBoostRegressor
X_train = np.random.randn(200, 3)
y_train = np.exp(X_train[:, 0]) + np.random.normal(0, 0.2, 200)

custom_mapper = MeanBinMapper()
model = OrdBoostRegressor(mapper=custom_mapper)
model.fit(X_train, y_train)
```

---
