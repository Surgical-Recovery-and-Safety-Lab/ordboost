"""Evaluation metrics for discrete ordinal probabilistic forecasts."""

from typing import Union

import numpy as np

from ordboost.distributions import PredictiveDistribution


def crps_score(
    y_true: Union[np.ndarray, list],
    y_dist: PredictiveDistribution,
    sample_weight: Union[np.ndarray, None] = None,
) -> float:
    """Compute the discrete Continuous Ranked Probability Score (CRPS).

    The discrete CRPS measures the distance between the predicted cumulative
    distribution function (CDF) and the empirical step function of the true
    observed ordinal target:

        CRPS(F, y) = sum_{k=0}^{K-1} (F(c_k) - I(y <= c_k))^2

    A lower CRPS score indicates better probabilistic forecast performance,
    balancing both calibration and sharpness.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True physical target labels corresponding to class values in `y_dist.classes`.
    y_dist : PredictiveDistribution
        Predicted probability distribution object containing CDF and class metadata.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights for weighted mean computation.

    Returns
    -------
    float
        The average discrete CRPS across all samples.

    Raises
    ------
    ValueError
        If `y_true` contains class values not present in `y_dist.classes`,
        or if array dimensions do not match `y_dist`.

    """
    y_true_arr = np.asarray(y_true)
    if y_true_arr.ndim != 1:
        raise ValueError(
            f"Expected 'y_true' to be a 1D array, got shape {y_true_arr.shape}."
        )

    n_samples = len(y_true_arr)
    if n_samples != y_dist.pmf.shape[0]:
        raise ValueError(
            f"Sample count mismatch: 'y_true' has {n_samples} samples, but "
            f"'y_dist' has {y_dist.pmf.shape[0]} samples."
        )

    # Validate that all true targets exist in classes
    if not set(y_true_arr).issubset(set(y_dist.classes)):
        missing_classes = set(y_true_arr) - set(y_dist.classes)
        raise ValueError(
            f"y_true contains target values not present in y_dist.classes: "
            f"{missing_classes}"
        )

    # Construct empirical step function I(y_true <= c_k) shape: (n_samples, n_classes)
    # Broadcasting: y_true_arr[:, None] <= classes[None, :]
    true_indicator = (
        y_true_arr[:, np.newaxis] <= y_dist.classes[np.newaxis, :]
    ).astype(float)

    # Compute squared cumulative probability error across all threshold levels
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
    y_true: Union[np.ndarray, list],
    y_pred_q: np.ndarray,
    q: float,
    sample_weight: Union[np.ndarray, None] = None,
) -> float:
    """Compute the pinball (quantile) loss for a specific quantile level q.

    The pinball loss penalizes over- and under-prediction asymmetrically:

        L_q(y, q_pred) = max(q * (y - q_pred), (q - 1) * (y - q_pred))

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True physical target labels.
    y_pred_q : array-like of shape (n_samples,)
        Predicted target values at quantile level `q`.
    q : float
        Target quantile level in the range (0.0, 1.0).
    sample_weight : array-like of shape (n_samples,), optional
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
