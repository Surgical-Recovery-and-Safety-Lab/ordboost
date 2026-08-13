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

        cdf = self.cdf  # shape: (n_samples, n_classes)

        if q_arr.ndim == 0:
            # Single quantile scalar -> search along class axis
            indices = np.argmax(cdf >= q_arr, axis=1)
            return self.classes[indices]

        # Array of quantiles -> broadcasting shape:
        # (n_samples, 1, n_classes) vs (1, n_quantiles, 1)
        cdf_expanded = cdf[:, np.newaxis, :]  # (n_samples, 1, n_classes)
        q_expanded = q_arr[np.newaxis, :, np.newaxis]  # (1, n_quantiles, 1)

        indices = np.argmax(
            cdf_expanded >= q_expanded, axis=2
        )  # (n_samples, n_quantiles)
        return self.classes[indices]

    def median(self) -> np.ndarray:
        """Calculate the 50th percentile (median) prediction for each sample.

        Returns
        -------
        np.ndarray
            1D array of shape (n_samples,) containing median class predictions.

        """
        return self.ppf(0.5)

    def interval(self, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
        """Calculate central prediction bounds for a given significance level.

        Parameters
        ----------
        alpha : float, default=0.10
            Significance level. For example, `alpha=0.10` produces a 90%
            central prediction interval [q(0.05), q(0.95)].

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing (lower_bounds, upper_bounds), each as a 1D array
            of shape (n_samples,).

        Raises
        ------
        ValueError
            If `alpha` is not within (0.0, 1.0).

        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("Significance level 'alpha' must be between 0.0 and 1.0.")

        lower_q = alpha / 2.0
        upper_q = 1.0 - (alpha / 2.0)

        bounds = self.ppf(np.array([lower_q, upper_q]))
        return bounds[:, 0], bounds[:, 1]
