"""Evaluation metrics for discrete ordinal and continuous probabilistic forecasts."""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from ordboost.distributions import (
    ContinuousPredictiveDistribution,
    DiscretePredictiveDistribution,
)


def crps_score(
    y_true: ArrayLike,
    y_dist: Union[ContinuousPredictiveDistribution, DiscretePredictiveDistribution],
    sample_weight: Union[ArrayLike, None] = None,
) -> float:
    """Compute the Continuous Ranked Probability Score (CRPS).

    For discrete distributions, evaluates squared cumulative probability error
    across threshold classes. For continuous predictive distributions,
    evaluates integrated squared distance between predicted CDF F(y) and
    the empirical step function I(y_true <= y) via trapezoidal integration.

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
        If `y_true` shape or sample count mismatches `y_dist`.

    """
    y_true_arr = np.asarray(y_true, dtype=float)
    if y_true_arr.ndim != 1:
        raise ValueError(
            f"Expected 'y_true' to be a 1D array, got shape {y_true_arr.shape}."
        )

    n_samples = len(y_true_arr)

    # Handle ContinuousPredictiveDistribution via trapezoidal integration
    if isinstance(y_dist, ContinuousPredictiveDistribution):
        if n_samples != y_dist.grid_cdf.shape[0]:
            raise ValueError(
                f"Sample count mismatch: 'y_true' has {n_samples} samples, but "
                f"'y_dist' has {y_dist.grid_cdf.shape[0]} samples."
            )

        grid_y = y_dist.grid_y
        grid_cdf = y_dist.grid_cdf

        # Empirical step function I(y_true <= grid_y)
        true_indicator = (y_true_arr[:, np.newaxis] <= grid_y[np.newaxis, :]).astype(
            float
        )

        # Integrated squared distance along continuous physical grid dy
        cdf_diff_sq = (grid_cdf - true_indicator) ** 2
        dy = np.diff(grid_y)
        avg_sq_diff = 0.5 * (cdf_diff_sq[:, :-1] + cdf_diff_sq[:, 1:])
        sample_crps = np.sum(avg_sq_diff * dy, axis=1)

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
