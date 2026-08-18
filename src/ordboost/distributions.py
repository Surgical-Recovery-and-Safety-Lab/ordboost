"""Predictive probability distributions for discrete ordinal outcomes."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class PredictiveDistribution(ABC):
    """Abstract base class for all predictive probability distributions.

    Defines a unified interface for extracting point estimates, quantiles,
    prediction intervals, and cumulative probabilities regardless of whether
    the distribution is discrete or continuous.

    Methods
    -------
    mean()
        Calculate expected values across samples.
    median()
        Calculate 50th percentile predictions across samples.
    ppf(q)
        Calculate percent point function (inverse CDF / quantiles).
    interval(alpha=0.10)
        Calculate central prediction bounds for a given significance level.

    """

    @abstractmethod
    def mean(self) -> np.ndarray:
        """Calculate the expected value for each sample."""
        pass

    @abstractmethod
    def ppf(self, q: Union[float, np.ndarray]) -> np.ndarray:
        """Percent Point Function (inverse CDF / quantile calculation)."""
        pass

    def median(self) -> np.ndarray:
        """Calculate the 50th percentile (median) prediction for each sample.

        Returns
        -------
        np.ndarray
            1D array of shape (n_samples,) containing median predictions.

        """
        return self.ppf(0.5)

    def interval(self, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
        """Calculate central prediction bounds for a given significance level.

        Parameters
        ----------
        alpha : float, default=0.10
            Significance level (e.g., alpha=0.10 yields a 90% central interval).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (lower_bounds, upper_bounds), each as a 1D array.

        Raises
        ------
        ValueError
            If `alpha` is not strictly within (0.0, 1.0).

        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("Significance level 'alpha' must be between 0.0 and 1.0.")

        lower_q = alpha / 2.0
        upper_q = 1.0 - (alpha / 2.0)
        bounds = self.ppf(np.array([lower_q, upper_q]))
        return bounds[:, 0], bounds[:, 1]


class DiscretePredictiveDistribution(PredictiveDistribution):
    """Encapsulates a discrete Probability Mass Function (PMF) matrix.

    Provides vectorized utilities for computing cumulative distribution
    functions (CDF), percent point functions (quantiles/PPF), expected values,
    medians, and prediction intervals across samples.

    Parameters
    ----------
    pmf : np.ndarray
        A 2D float array of shape (n_samples, n_classes) representing the
        predicted probability for each discrete target class. Values along
        each row must sum to 1.0.
    classes : np.ndarray
        A 1D array of shape (n_classes,) representing the physical ordinal
        class labels in strictly ascending order.

    Attributes
    ----------
    pmf : np.ndarray
        A 2D float array of shape (n_samples, n_classes) containing predicted
        class probabilities.
    classes : np.ndarray
        A 1D array of shape (n_classes,) containing the ordinal class labels.
    cdf : np.ndarray
        A 2D float array of shape (n_samples, n_classes) containing cumulative
        probabilities computed from `pmf`.

    Methods
    -------
    mean()
        Calculate the expected value for each sample.
    ppf(q)
        Calculate the percent point function (inverse CDF / quantiles).
    median()
        Calculate the 50th percentile prediction for each sample.
    interval(alpha=0.10)
        Calculate central prediction bounds for a given significance level.

    Raises
    ------
    ValueError
        If `pmf` is not a 2D array, `classes` is not a 1D array, or the
        number of columns in `pmf` does not match the length of `classes`.

    """

    def __init__(self, pmf: np.ndarray, classes: np.ndarray) -> None:
        pmf_arr = np.asarray(pmf, dtype=float)
        classes_arr = np.asarray(classes)

        if pmf_arr.ndim != 2:
            raise ValueError(
                f"Expected 'pmf' to be a 2D array of shape (n_samples, n_classes), "
                f"got shape {pmf_arr.shape}."
            )
        if classes_arr.ndim != 1:
            raise ValueError(
                f"Expected 'classes' to be a 1D array, got shape {classes_arr.shape}."
            )
        if pmf_arr.shape[1] != classes_arr.shape[0]:
            raise ValueError(
                f"Mismatch between PMF class dimension ({pmf_arr.shape[1]}) "
                f"and classes array length ({classes_arr.shape[0]})."
            )

        self.pmf = pmf_arr
        self.classes = classes_arr
        self._cdf: np.ndarray | None = None

    @property
    def cdf(self) -> np.ndarray:
        """Compute the Cumulative Distribution Function (CDF) array.

        Returns
        -------
        np.ndarray
            2D array of shape (n_samples, n_classes) containing cumulative probabilities.

        """
        if self._cdf is None:
            self._cdf = np.clip(np.cumsum(self.pmf, axis=1), 0.0, 1.0)
        return self._cdf

    def mean(self) -> np.ndarray:
        """Calculate the expected value (mean) for each sample.

        Returns
        -------
        np.ndarray
            1D array of shape (n_samples,) representing expected values in
            physical class units.

        """
        return np.dot(self.pmf, self.classes)

    def ppf(self, q: Union[float, np.ndarray]) -> np.ndarray:
        """Percent Point Function (inverse CDF / quantile calculation).

        Maps quantile probabilities back to discrete physical class levels.

        Parameters
        ----------
        q : float | np.ndarray
            Quantile level(s) in the range [0.0, 1.0]. Can be a single scalar
            or an array of quantiles.

        Returns
        -------
        np.ndarray
            If `q` is a scalar, returns a 1D array of shape (n_samples,).
            If `q` is 1D array of length `n_quantiles`, returns a 2D array of
            shape (n_samples, n_quantiles).

        Raises
        ------
        ValueError
            If any quantile in `q` lies outside [0.0, 1.0].

        """
        q_arr = np.asarray(q, dtype=float)
        if np.any((q_arr < 0.0) | (q_arr > 1.0)):
            raise ValueError("All quantiles in 'q' must lie within [0.0, 1.0].")

        cdf = self.cdf
        if q_arr.ndim == 0:
            indices = np.argmax(cdf >= q_arr, axis=1)
            return self.classes[indices]

        cdf_expanded = cdf[:, np.newaxis, :]
        q_expanded = q_arr[np.newaxis, :, np.newaxis]
        indices = np.argmax(cdf_expanded >= q_expanded, axis=2)
        return self.classes[indices]


class ContinuousPredictiveDistribution(PredictiveDistribution):
    """Encapsulates a continuous predictive Cumulative Distribution Function (CDF).

    Provides vectorized utilities for computing expected continuous values,
    medians, percent point functions (quantiles/PPF), central prediction
    intervals, and continuous CDF probabilities across samples.

    Parameters
    ----------
    grid_y : np.ndarray
        1D float array of shape (n_grid_points,) representing continuous physical
        target grid values in strictly ascending order.
    grid_cdf : np.ndarray
        2D float array of shape (n_samples, n_grid_points) containing evaluated
        cumulative probabilities across grid points.

    Attributes
    ----------
    grid_y : np.ndarray
        1D float array containing grid values.
    grid_cdf : np.ndarray
        2D float array containing cumulative probabilities bounded in [0.0, 1.0].

    Methods
    -------
    mean()
        Calculate expected continuous values via numerical integration.
    ppf(q)
        Calculate percent point function (inverse CDF / quantiles).
    cdf(y)
        Evaluate continuous CDF probability P(Y <= y) at physical value y.

    Raises
    ------
    ValueError
        If `grid_y` is not 1D, `grid_cdf` is not 2D, or shape dimensions mismatch.

    """

    def __init__(self, grid_y: np.ndarray, grid_cdf: np.ndarray) -> None:
        y_arr = np.asarray(grid_y, dtype=float)
        cdf_arr = np.asarray(grid_cdf, dtype=float)

        if y_arr.ndim != 1 or cdf_arr.ndim != 2:
            raise ValueError("Invalid array dimensions for grid_y or grid_cdf.")
        if cdf_arr.shape[1] != y_arr.shape[0]:
            raise ValueError("Grid CDF column dimension must match grid_y length.")

        self.grid_y = y_arr
        self.grid_cdf = np.clip(cdf_arr, 0.0, 1.0)
        self._n_samples = cdf_arr.shape[0]

    def mean(self) -> np.ndarray:
        """Calculate expected continuous values via numerical integration.

        Returns
        -------
        np.ndarray
            1D array of shape (n_samples,) containing expected physical values.

        """
        dy = np.diff(self.grid_y)
        avg_prob = 1.0 - 0.5 * (self.grid_cdf[:, :-1] + self.grid_cdf[:, 1:])
        return np.sum(avg_prob * dy, axis=1) + self.grid_y[0]

    def ppf(self, q: Union[float, np.ndarray]) -> np.ndarray:
        """Calculate continuous interpolated values at quantile level `q`.

        Parameters
        ----------
        q : float | np.ndarray
            Quantile level(s) strictly in the range [0.0, 1.0].

        Returns
        -------
        np.ndarray
            If `q` is a scalar, returns a 1D array of shape (n_samples,).
            If `q` is a 1D array of length `n_quantiles`, returns a 2D array
            of shape (n_samples, n_quantiles).

        Raises
        ------
        ValueError
            If any quantile in `q` lies outside [0.0, 1.0].

        """
        q_arr = np.asarray(q, dtype=float)
        if np.any((q_arr < 0.0) | (q_arr > 1.0)):
            raise ValueError("All quantiles in 'q' must lie within [0.0, 1.0].")

        if q_arr.ndim == 0:
            result = np.empty(self._n_samples, dtype=float)
            for i in range(self._n_samples):
                result[i] = np.interp(q_arr, self.grid_cdf[i], self.grid_y)
            return result

        result = np.empty((self._n_samples, len(q_arr)), dtype=float)
        for i in range(self._n_samples):
            result[i] = np.interp(q_arr, self.grid_cdf[i], self.grid_y)
        return result

    def cdf(self, y: float) -> np.ndarray:
        """Evaluate continuous CDF probability P(Y <= y) at physical value y.

        Parameters
        ----------
        y : float
            Physical target value at which to evaluate cumulative probability.

        Returns
        -------
        np.ndarray
            1D array of shape (n_samples,) containing evaluated probabilities.

        """
        probs = np.empty(self._n_samples, dtype=float)
        for i in range(self._n_samples):
            probs[i] = np.interp(y, self.grid_y, self.grid_cdf[i])
        return probs
