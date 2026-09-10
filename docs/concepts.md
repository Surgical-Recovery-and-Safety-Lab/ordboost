# Concepts

This page explains the ideas behind OrdBoost: how the ordinal classifier works,
how the regressor builds on it via a bin mapper to produce continuous predictive
distributions, and how the package's evaluation metrics assess those distributions.

## Ordinal binning

### The problem

Many continuous clinical and health outcomes (_e.g._ length of stay, days alive and
out of hospital) are difficult to model directly with standard regression when
their distribution is heavily skewed, bounded, or has point masses at one or both
ends of its range. Discretizing the outcome into ordinal bins and modeling it as
a classification problem is a common workaround, but naive multiclass classification
ignores the fact that the bins are *ordered*: an error that confuses adjacent
bins is less severe than one that confuses distant bins, and a model should be
able to share statistical strength across nearby bins.

### Existing approaches

The standard statistical solution to this is the **cumulative link model**
(also known as the proportional odds model, or ordered logit/probit),
introduced by McCullagh (1980) and implemented today in R's
[`ordinal`](https://cran.r-project.org/package=ordinal) package. A cumulative
link model estimates a single latent score per observation together with a
shared vector of ordered thresholds, and derives class probabilities from
where that score falls relative to the thresholds. This idea has since been
extended to gradient boosting: recent work such as
[`OGBoost`](https://arxiv.org/abs/2502.13456) (Sharabiani et al., 2025) jointly
optimizes a boosted latent score and a threshold vector via coordinate
descent, giving a gradient-boosted analogue of the classical model.

### OrdBoost's approach

`OrdBoostClassifier` takes a different route to the same ordinal structure.
Rather than fitting one shared latent score and threshold vector, it trains
`K - 1` **independent** binary classifiers, one per cumulative edge: the
`k`-th classifier estimates $F_k(x) = P(Y <= c_k | x)$ directly, separating
observations below a threshold from those above it. Because these `K - 1`
edge classifiers are fit independently, nothing guarantees their raw outputs
are monotonically non-decreasing (a valid CDF requires
$F_k(x) <= F_{k+1}(x)$ for every sample). OrdBoost enforces this after
fitting, via either a cumulative running maximum or per-sample isotonic
regression (`monotonicity="running_max"` or `"isotonic"`), rather than
building the monotonicity constraint into the estimation itself the way a
classical cumulative link model does.

This trade-off is deliberate: it lets each edge classifier be an arbitrary,
independently-tuned `HistGradientBoostingClassifier` (with its own tree
depth, regularization, and native handling of missing feature values),
rather than requiring a single joint model family across all thresholds.
The cost is that monotonicity is a post-hoc correction rather than a
guarantee of the fitting procedure.

See [`docs/api/models.md`](api/models.md) for the full
`OrdBoostClassifier` reference, and the
[quickstart example](examples/quickstart.md#ordboostclassifier-simple-example)
for a minimal worked example of the `OrdBoostClassifier`.

## The regressor: from bins back to a continuous distribution

`OrdBoostClassifier` alone only produces a discrete probability mass function
(PMF) over bins, which is coarser than the original continuous target.
`OrdBoostRegressor` wraps the classifier and adds a **bin mapper**, which
converts that discrete PMF back into a continuous predictive cumulative
distribution function (CDF) in the target's original physical units.

### `OrdBoostRegressor` is the single source of truth for binning

`OrdBoostRegressor` computes the bin
edges and digitizes the training targets **exactly once**, then passes that
same result to both the classifier and the mapper. If you construct a bin
mapper yourself and pass a pre-configured instance to `mapper=`, and that
instance's own `bin_edges` conflicts with what the regressor resolved, the
regressor raises an error rather than silently using one or the other. The
classifier and the mapper must always agree on what a "bin" means for a
given sample.

```python
from ordboost import OrdBoostRegressor
from ordboost.mappers import QuantileBinMapper

# Recommended: leave bin_edges unset on the mapper and let the regressor
# assign them automatically.
model = OrdBoostRegressor(n_bins=10, mapper=QuantileBinMapper())
```

### `bin_edges` are interior thresholds only

`bin_edges` follows the same convention as `numpy.digitize`: every value you
supply is a real, enforced threshold, and there is no implicit padding.
`K - 1` thresholds define `K` bins with bin edges $[t_0, t_1, ..., t_{K-2}]$ as:

\begin{cases}
    b_0     = (-\infty, t_0) \\
    b_k     = [t_{k-1}, t_k)  &    \text{ for } 0 < k < K-1 \\
    b_{K-1}   = [t_{K-2}, +\infty)\\
\end{cases}

Whether the outermost bins are genuinely unbounded, or the outcome has a
known, finite true limit, is a *separate* question, controlled by
`lower_bound` and `upper_bound`, not encoded in `bin_edges` itself.

### `lower_bound` / `upper_bound`: does the outcome have a true boundary?

- `lower_bound=None` (the default): the first bin is treated as genuinely
  unbounded below, and the mapper anchors the predictive CDF's floor to the
  *observed* minimum of the training targets.
- `lower_bound=<value>`: asserts that the outcome's support has a real,
  known lower limit at that value (e.g. `0` for a non-negative duration).
  The predictive CDF is forced to `0` there for every sample, regardless of
  what any individual training bin happened to observe.

`upper_bound` mirrors this for the top of the range.

### `floor_atom` / `ceiling_atom`: does the outcome have a point mass there?

Even once a boundary is known to be real, does a
meaningful fraction of the population actually take that exact boundary
value (a genuine discrete atom in an otherwise continuous outcome), or does
the outcome merely approach the boundary smoothly? Setting `floor_atom=True`
(which requires `lower_bound` to be set) tells the mapper to estimate, from
training data, what fraction of each relevant bin's mass sits *exactly* at
the boundary, and to represent that as a sharp, near-discontinuous rise in
the predictive CDF rather than smoothing it uniformly across the bin.
`ceiling_atom` mirrors this at the top.

These two properties, *is there a boundary* and *is there an atom at it*,
are independent and should be reasoned about separately for your own
data. A bounded, smoothly-decaying outcome (e.g. a proportion that
approaches but rarely equals 1) should typically use a boundary without an
atom; a zero-inflated outcome (e.g. a cost or duration where a large
subgroup has exactly zero) needs the atom flag as well.

See [Advanced: Bounded data and atoms](examples/advanced.md#bounded-data-and-atoms)
for a worked example contrasting a dataset with atoms against the same data with
atom handling turned off, and
[Evaluation: Model calibration diagnostics](examples/evaluation.md#model-calibration-diagnostics)
for an example on purely continuous data, where no boundary handling is
needed at all.

### Bin mappers

The mapper determines how each bin's probability mass is distributed
across the continuous grid within that bin. `OrdBoost` ships several:

| Mapper | Intra-bin refinement |
|---|---|
| `UniformBinMapper` | None by default (`n_points=0` reduces to a straight line across raw bin edges, the null baseline), or `n_points` evenly-spaced points assuming uniform density within the bin. |
| `EmpiricalMeanBinMapper` | One point at each bin's empirical mean. |
| `EmpiricalMedianBinMapper` | One point at each bin's empirical median. |
| `QuantileBinMapper` | Several points at fixed empirical quantile levels (e.g. the 25th/50th/75th percentiles) within each bin. |
| `ContinuousBinMapper` | A dense grid of points spaced at a fixed resolution, e.g. every achievable integer value for an integer-valued outcome. |

All of these subclass `BaseBinMapper` and only need to implement one method,
`_intra_bin_points`, which returns the interior grid points to add for a
given bin and their associated cumulative weights. Everything else 
(boundary anchoring, atom handling, grid deduplication) is shared,
guaranteeing that a mapper's point estimate (`transform`) can never disagree
with its full predicted distribution (`to_continuous_dist`). However, if a mapper 
needs another attribute to compute the intra bin points, the `__init__` method will
need to be redefined. Optionally, the `_validate_intra_bin_params` method should be
revised to validate any new parameters. 

See [Advanced: Writing a custom mapper](examples/advanced.md#writing-a-custom-mapper)
for a worked example implementing a new `BaseBinMapper` subclass, and
[Quickstart: OrdBoostRegressor simple example](examples/quickstart.md#ordboostregressor-simple-example)
for basic usage of `predict`, `predict_dist().ppf()`, and `predict_dist().cdf()`.

## Evaluation

Once fit, `OrdBoostRegressor` produces a `ContinuousPredictiveDistribution`
per sample. OrdBoost's evaluation follows the framework laid out by
Gneiting, Balabdaoui, and Raftery (2007), which argues that a probabilistic
forecast should be assessed on **calibration** (does the forecast match
reality, statistically) and **sharpness** (how concentrated/informative the
forecast is), and that neither alone is sufficient: a forecast can be
calibrated but uselessly diffuse, or sharp but systematically wrong.

### Accuracy and proper scoring rules

- **MAE**: mean absolute error between a point prediction (e.g.
  `dist.median()`) and the true outcome. Useful, but says nothing about the
  quality of the full predictive distribution.
- **CRPS** (`crps_score`): the Continuous Ranked Probability Score, a
  proper scoring rule that evaluates the entire predicted CDF against the
  true outcome, sensitive to both location and spread.
- **Skill scores** (`crps_skill_score`, `pinball_loss_skill_score`):
  express CRPS or pinball loss relative to a naive, covariate-free
  reference forecast (`baseline_distribution`, built from the unconditional
  empirical distribution of the training targets). A skill score of `0`
  means no better than guessing the population distribution for everyone;
  `1` is a perfect forecast.

See [Evaluation: Preformance evaluation](examples/evaluation.md#performance-evaluation).

### Calibration diagnostics

- **PIT histogram** (`pit_diagnostics`): the probability integral
  transform, `F(y_true)`. Under correct calibration, PIT values should be
  uniformly distributed. OrdBoost uses the exact discontinuity-aware
  construction of Taggart and Bureau of Meteorology Australia (2022) rather
  than an approximate randomized/jittered PIT, so floor and ceiling atoms
  are represented correctly without Monte Carlo sampling. The resulting
  object exposes both a histogram and the alpha score of Renard et al.
  (2010), a summary statistic purpose-built for assessing calibration when
  the predictive distribution has discontinuities.
- **Marginal calibration** (`marginal_calibration_curve`): the difference
  between the empirical CDF of the true outcomes and the *average* predicted
  CDF across all samples, evaluated across the target's range. This and the
  PIT histogram are logically independent checks: marginal calibration is
  an aggregate, population-level comparison, while PIT is evaluated
  per-sample against each sample's own forecast. A forecaster can pass one
  check while failing the other, so both are worth examining together, not
  as substitutes for each other.

See [Evaluation: Model calibration diagnostics](examples/evaluation.md#model-calibration-diagnostics).

### Prediction interval evaluation

- **Coverage** (`interval_coverage_rate`): the proportion of true outcomes
  falling within a predicted central interval at a given significance level
  `alpha` (e.g. `alpha=0.10` for a nominal 90% interval).
- **Sharpness** (`sharpness`): the mean width of that interval. A narrow
  interval is only good news if coverage is also close to nominal; a narrow
  but under-covering interval is overconfident, not informative.
- **Winkler score** (`winkler_score`): a single proper scoring rule that
  jointly penalizes interval width and coverage failures, useful for
  comparing interval quality across models or configurations with one
  number rather than two separate curves.

See [Evaluation: Evaluating prediction intervals](examples/evaluation.md#evaluating-prediction-intervals).

---

**A note on interpreting these together:** no single metric on this page is
sufficient on its own. A model can have excellent CRPS while still being
badly miscalibrated in the tails; a model can have excellent PIT/marginal
calibration while producing wide, uninformative intervals. Evaluating a
fitted `OrdBoostRegressor` well means looking at several of these
diagnostics together, not optimizing for any single one in isolation.

## References

* McCullagh, P. (1980). Regression Models for Ordinal Data. Journal of the Royal Statistical Society: Series B (Methodological), 42(2), 109–127. [https://doi.org/10.1111/j.2517-6161.1980.tb01109.x](https://doi.org/10.1111/j.2517-6161.1980.tb01109.x)
* Sharabani, M., et al. (2025). OGBoost: Gradient Boosting for Ordinal Regression. arXiv:2502.13456. [https://arxiv.org/abs/2502.13456](https://arxiv.org/abs/2502.13456)
* Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic Forecasts, Calibration and Sharpness. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(2), 243–268. [https://doi.org/10.1111/j.1467-9868.2007.00587.x)](https://doi.org/10.1111/j.1467-9868.2007.00587.x)
* Taggart, R., & Bureau of Meteorology Australia (2022). Assessing Calibration When Predictive Distributions Have Discontinuities. Trove. [https://nla.gov.au/nla.obj-30799618620)](https://nla.gov.au/nla.obj-30799618620)
* Renard, B., et al. (2010). Understanding Predictive Uncertainty in Hydrologic Modeling: The Challenge of Identifying Input and Structural Errors. Water Resources Research, 46(5). [https://doi.org/10.1029/2009WR008328](https://doi.org/10.1029/2009WR008328)
