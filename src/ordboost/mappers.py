"""Base interfaces for bin-to-continuous target mappers."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ordboost.distributions import ContinuousPredictiveDistribution


class BaseBinMapper(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for all bin-to-continuous target mappers.

    Parameters
    ----------
    bin_edges : array-like of shape (n_bins + 1,)
        Monotonically increasing boundaries defining continuous bin intervals.

    Attributes
    ----------
    bin_edges_ : np.ndarray
        1D float array of shape (n_bins + 1,) containing validated bin edges.
    n_bins_ : int
        Number of discrete bins defined by `bin_edges_`.

    Methods
    -------
    fit(y_continuous, y_binned=None)
        Compute empirical bin statistics from continuous training targets.
    transform(pmf)
        Map discrete PMF probability matrix to continuous expected values.
    to_continuous_dist(pmf)
        Construct a ContinuousPredictiveDistribution from a discrete PMF matrix.

    """

    def __init__(self, bin_edges: Any) -> None:
        self.bin_edges = bin_edges

    def _validate_edges(self) -> np.ndarray:
        """Validate and return bin edges array.

        Returns
        -------
        np.ndarray
            1D float array containing validated bin edges.

        Raises
        ------
        ValueError
            If `bin_edges` has fewer than 2 edges, is not 1D, or is not
            strictly monotonically increasing.

        """
        edges = np.asarray(self.bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError(
                "Expected 'bin_edges' to be a 1D array with at least 2 edges."
            )
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("'bin_edges' must be strictly monotonically increasing.")
        return edges

    @abstractmethod
    def fit(self, y_continuous: Any, y_binned: Any = None) -> "BaseBinMapper":
        """Compute empirical bin statistics from continuous training targets.

        Parameters
        ----------
        y_continuous : array-like of shape (n_samples,)
            Unbinned continuous target values (e.g., exact physical units).
        y_binned : array-like of shape (n_samples,), optional
            Corresponding 0-indexed discrete bin labels. If None, labels are
            computed automatically from `bin_edges`.

        Returns
        -------
        BaseBinMapper
            Fitted mapper instance.

        """
        pass

    @abstractmethod
    def transform(self, pmf: Any) -> np.ndarray:
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

        """
        pass

    @abstractmethod
    def to_continuous_dist(self, pmf: Any) -> ContinuousPredictiveDistribution:
        """Construct a ContinuousPredictiveDistribution from a discrete PMF matrix.

        Parameters
        ----------
        pmf : array-like of shape (n_samples, n_bins)
            Discrete probability mass function matrix where rows sum to 1.0.

        Returns
        -------
        ContinuousPredictiveDistribution
            Continuous distribution evaluated over physical target grid.

        """
        pass


class EmpiricalMeanMapper(BaseBinMapper):
    """Maps discrete bin probabilities using empirical intra-bin means.

    Computes the empirical mean of continuous targets within each bin during
    fitting. Maps discrete probability mass functions (PMF) to continuous
    expected target values using these fitted means. Empty bins are filled by
    interpolating between adjacent non-empty bin means.

    Parameters
    ----------
    bin_edges : array-like of shape (n_bins + 1,)
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

    def __init__(self, bin_edges: ArrayLike) -> None:
        super().__init__(bin_edges=bin_edges)

    def fit(
        self, y_continuous: ArrayLike, y_binned: Union[ArrayLike, None] = None
    ) -> "EmpiricalMeanMapper":
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
        EmpiricalMeanMapper
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
