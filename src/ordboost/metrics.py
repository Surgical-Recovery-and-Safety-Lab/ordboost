"""Evaluation metrics for discrete ordinal and continuous probabilistic forecasts."""

from typing import Union

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scores.probability import crps_cdf

from ordboost.distributions import (
    ContinuousPredictiveDistribution,
    DiscretePredictiveDistribution,
)


def baseline_distribution(
    y_train: ArrayLike, n_samples: int, boundary_epsilon: float = 1e-4
) -> ContinuousPredictiveDistribution:
    """Construct a climatological (no-covariate) baseline distribution.

    Builds the unconditional empirical CDF of `y_train` and broadcasts it
    identically across `n_samples` rows, representing the best achievable
    forecast in the complete absence of covariate information -- the
    standard reference forecast against which skill scores are computed.

    Parameters
    ----------
    y_train : ArrayLike of shape (n_train_samples,)
        Training targets defining the unconditional empirical distribution.
        Should be the full, unfiltered training set, even when scoring a
        filtered evaluation subset, so the baseline continues to represent
        "no covariate information" rather than "no information restricted
        to a subgroup" (see `crps_skill_score`).
    n_samples : int
        Number of rows to broadcast the climatological CDF across, e.g.
        the number of samples in the evaluation set this baseline will be
        scored against.
    boundary_epsilon : float, default=1e-4
        Offset used to place a grid point strictly below the observed
        minimum of `y_train`, forced to CDF=0.0. Required because
        `ContinuousPredictiveDistribution` validates that every row's
        first value equals 0.0 within tolerance, and the empirical CDF at
        the observed minimum itself is generally nonzero (typically
        `1/n_train_samples`, not 0).

    Returns
    -------
    ContinuousPredictiveDistribution
        A distribution with `n_samples` identical rows, each equal to the
        empirical CDF of `y_train`.

    Raises
    ------
    ValueError
        If `y_train` is empty, or `n_samples` is not a positive integer.

    """
    y_train_arr = np.asarray(y_train, dtype=float)
    if y_train_arr.size == 0:
        raise ValueError("'y_train' must not be empty.")
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError(f"'n_samples' must be a positive integer, got {n_samples}.")

    unique_y = np.sort(np.unique(y_train_arr))
    unique_cdf = np.array([np.mean(y_train_arr <= v) for v in unique_y])

    grid_y = np.concatenate([[unique_y[0] - boundary_epsilon], unique_y])
    grid_cdf_row = np.concatenate([[0.0], unique_cdf])

    grid_cdf = np.tile(grid_cdf_row, (n_samples, 1))

    return ContinuousPredictiveDistribution(grid_y=grid_y, grid_cdf=grid_cdf)


def crps_score(
    y_true: ArrayLike,
    y_dist: Union[ContinuousPredictiveDistribution, DiscretePredictiveDistribution],
    sample_weight: Union[ArrayLike, None] = None,
) -> float:
    """Compute the Continuous Ranked Probability Score (CRPS).

    For discrete distributions, evaluates the exact squared cumulative
    probability error across threshold classes (a finite sum, not an
    approximation). For continuous predictive distributions, evaluates
    the exact integrated squared distance between the predicted CDF
    `F(y)` and the empirical step function `I(y_true <= y)`, via
    `scores.probability.crps_cdf`. This computes the exact integral for
    a piecewise-linear CDF -- including correctly splitting the grid
    segment containing `y_true` -- rather than approximating it via
    trapezoidal integration over grid points alone, which systematically
    under- or over-estimates depending on where `y_true` falls within a
    segment (worse for wider bins).

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True physical target values.
    y_dist : PredictiveDistribution
        Predicted probability distribution object (discrete or continuous).
    sample_weight : ArrayLike of shape (n_samples,), optional
        Sample weights for weighted mean computation.

    Returns
    -------
    float
        The average CRPS across all samples (lower is better).

    Raises
    ------
    ValueError
        If `y_true` shape or sample count mismatches `y_dist`, or (for
        discrete distributions) `y_true` contains a value not present in
        `y_dist.classes`.

    """
    y_true_arr = np.asarray(y_true, dtype=float)
    if y_true_arr.ndim != 1:
        raise ValueError(
            f"Expected 'y_true' to be a 1D array, got shape {y_true_arr.shape}."
        )

    n_samples = len(y_true_arr)

    if isinstance(y_dist, ContinuousPredictiveDistribution):
        if n_samples != y_dist.grid_cdf.shape[0]:
            raise ValueError(
                f"Sample count mismatch: 'y_true' has {n_samples} samples, but "
                f"'y_dist' has {y_dist.grid_cdf.shape[0]} samples."
            )

        fcst_da = xr.DataArray(
            y_dist.grid_cdf,
            dims=["sample", "threshold"],
            coords={"threshold": y_dist.grid_y},
        )
        obs_da = xr.DataArray(y_true_arr, dims=["sample"])

        result = crps_cdf(
            fcst_da, obs_da, threshold_dim="threshold", preserve_dims=["sample"]
        )
        sample_crps = result.total.values

    else:
        # Handle Discrete PredictiveDistribution
        if n_samples != y_dist.pmf.shape[0]:
            raise ValueError(
                f"Sample count mismatch: 'y_true' has {n_samples} samples, but "
                f"'y_dist' has {y_dist.pmf.shape[0]} samples."
            )
        if not set(y_true_arr).issubset(set(y_dist.classes)):
            missing_classes = set(y_true_arr) - set(y_dist.classes)
            raise ValueError(
                f"y_true contains target values not present in y_dist.classes: "
                f"{missing_classes}"
            )
        true_indicator = (
            y_true_arr[:, np.newaxis] <= y_dist.classes[np.newaxis, :]
        ).astype(float)
        cdf_diff_sq = (y_dist.cdf - true_indicator) ** 2
        sample_crps = np.sum(cdf_diff_sq, axis=1)

    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (n_samples,):
            raise ValueError(
                f"Expected 'sample_weight' shape ({n_samples},), got {weights.shape}."
            )
        return float(np.average(sample_crps, weights=weights))

    return float(np.mean(sample_crps))


def crps_skill_score(
    y_true: ArrayLike,
    dist_model: ContinuousPredictiveDistribution,
    dist_baseline: ContinuousPredictiveDistribution,
    sample_weight: Union[ArrayLike, None] = None,
) -> float:
    """Compute the CRPS skill score relative to a reference forecast.

    Defined as ``CRPSS = 1 - CRPS(model) / CRPS(baseline)``. A value of 0
    indicates no improvement over the baseline, 1 indicates a perfect
    forecast, and negative values indicate performance worse than the
    baseline. `dist_baseline` is typically constructed via
    `baseline_distribution`, but may be any reference forecast.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True physical target values.
    dist_model : ContinuousPredictiveDistribution
        The model's predicted distribution.
    dist_baseline : ContinuousPredictiveDistribution
        The reference forecast distribution to compare against.
    sample_weight : ArrayLike of shape (n_samples,), optional
        Sample weights, applied identically to both CRPS computations.

    Returns
    -------
    float
        The CRPS skill score.

    Raises
    ------
    ValueError
        If `y_true`'s sample count mismatches `dist_model` or
        `dist_baseline` (raised by `crps_score`).

    """
    crps_model = crps_score(y_true, dist_model, sample_weight=sample_weight)
    crps_baseline = crps_score(y_true, dist_baseline, sample_weight=sample_weight)
    return 1.0 - (crps_model / crps_baseline)


def pinball_loss(
    y_true: ArrayLike,
    y_pred_q: ArrayLike,
    q: float,
    sample_weight: Union[ArrayLike, None] = None,
) -> float:
    """Compute the pinball (quantile) loss for a specific quantile level q.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True physical target labels.
    y_pred_q : ArrayLike of shape (n_samples,)
        Predicted target values at quantile level `q`.
    q : float
        Target quantile level in the range (0.0, 1.0).
    sample_weight : ArrayLike of shape (n_samples,), optional
        Sample weights for weighted mean computation.

    Returns
    -------
    float
        The average pinball loss across samples.

    Raises
    ------
    ValueError
        If `q` lies outside (0.0, 1.0) or array shapes do not match.

    """
    if not 0.0 < q < 1.0:
        raise ValueError(
            f"Quantile level 'q' must be strictly between 0.0 and 1.0, got {q}."
        )

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred_q, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: 'y_true' shape {y_true_arr.shape} does not match "
            f"'y_pred_q' shape {y_pred_arr.shape}."
        )

    errors = y_true_arr - y_pred_arr
    loss = np.maximum(q * errors, (q - 1.0) * errors)

    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != y_true_arr.shape:
            raise ValueError(
                f"Expected 'sample_weight' shape {y_true_arr.shape}, got {weights.shape}."
            )
        return float(np.average(loss, weights=weights))

    return float(np.mean(loss))


def pinball_loss_skill_score(
    y_true: ArrayLike,
    dist_model: ContinuousPredictiveDistribution,
    dist_baseline: ContinuousPredictiveDistribution,
    q: float,
    sample_weight: Union[ArrayLike, None] = None,
) -> float:
    """Compute the pinball loss skill score at quantile level `q`.

    Defined as ``PLSS = 1 - pinball_loss(model) / pinball_loss(baseline)``,
    both evaluated at the same quantile level `q` and against the same
    `y_true`. Interpretation mirrors `crps_skill_score`: 0 indicates no
    improvement over the baseline, 1 indicates a perfect forecast at that
    quantile, negative values indicate worse-than-baseline performance.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True physical target values.
    dist_model : ContinuousPredictiveDistribution
        The model's predicted distribution.
    dist_baseline : ContinuousPredictiveDistribution
        The reference forecast distribution to compare against.
    q : float
        Quantile level in (0.0, 1.0) at which to evaluate both forecasts.
    sample_weight : ArrayLike of shape (n_samples,), optional
        Sample weights, applied identically to both pinball loss
        computations.

    Returns
    -------
    float
        The pinball loss skill score at quantile `q`.

    Raises
    ------
    ValueError
        If `q` lies outside (0.0, 1.0), or if shapes mismatch (raised by
        `pinball_loss`).

    """
    y_pred_model = dist_model.ppf(q)
    y_pred_baseline = dist_baseline.ppf(q)
    loss_model = pinball_loss(y_true, y_pred_model, q=q, sample_weight=sample_weight)
    loss_baseline = pinball_loss(
        y_true, y_pred_baseline, q=q, sample_weight=sample_weight
    )
    return 1.0 - (loss_model / loss_baseline)


def interval_coverage_rate(
    y_true: ArrayLike,
    dist: ContinuousPredictiveDistribution,
    alpha: float = 0.10,
    sample_weight: Union[ArrayLike, None] = None,
) -> float:
    """Compute empirical coverage rate for a central prediction interval.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True continuous target values.
    dist : ContinuousPredictiveDistribution
        Predicted continuous distributions.
    alpha : float, default=0.10
        Tail significance level (e.g., alpha=0.10 specifies a 90% interval).
    sample_weight : ArrayLike of shape (n_samples,), optional
        Sample weights for weighted coverage computation.

    Returns
    -------
    float
        Proportion of true observations lying within predicted interval bounds.

    """
    y_true_arr = np.asarray(y_true, dtype=float)
    lower, upper = dist.interval(alpha=alpha)

    covered = ((y_true_arr >= lower) & (y_true_arr <= upper)).astype(float)

    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != y_true_arr.shape:
            raise ValueError(
                f"Expected 'sample_weight' shape {y_true_arr.shape}, got {weights.shape}."
            )
        return float(np.average(covered, weights=weights))

    return float(np.mean(covered))


def winkler_score(
    y_true: ArrayLike,
    dist: ContinuousPredictiveDistribution,
    alpha: float = 0.10,
    sample_weight: Union[ArrayLike, None] = None,
) -> float:
    """Compute mean Winkler score for prediction intervals at significance level alpha.

    Penalizes interval width and asymmetrically penalizes targets that fall
    outside the predicted lower and upper bounds.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True continuous target values.
    dist : ContinuousPredictiveDistribution
        Predicted continuous distributions.
    alpha : float, default=0.10
        Tail significance level in range (0.0, 1.0).
    sample_weight : ArrayLike of shape (n_samples,), optional
        Sample weights for weighted mean computation.

    Returns
    -------
    float
        Mean Winkler score across samples (lower is better).

    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("Significance level 'alpha' must lie within (0.0, 1.0).")

    y_true_arr = np.asarray(y_true, dtype=float)
    lower, upper = dist.interval(alpha=alpha)

    width = upper - lower
    under_penalty = (2.0 / alpha) * (lower - y_true_arr) * (y_true_arr < lower)
    over_penalty = (2.0 / alpha) * (y_true_arr - upper) * (y_true_arr > upper)

    sample_scores = width + under_penalty + over_penalty

    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != y_true_arr.shape:
            raise ValueError(
                f"Expected 'sample_weight' shape {y_true_arr.shape}, got {weights.shape}."
            )
        return float(np.average(sample_scores, weights=weights))

    return float(np.mean(sample_scores))
