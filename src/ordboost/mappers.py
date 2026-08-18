"""Base interfaces for bin-to-continuous target mappers."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from ordboost.distributions import ContinuousPredictiveDistribution


class BaseBinMapper(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for all bin-to-continuous target mappers.

    Parameters
    ----------
    bin_edges : array-like of shape (n_bins + 1,)
        Monotonically increasing boundaries defining continuous bin intervals.

    Attributes
    ----------
    bin_edges_ : array-like
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

    def __init__(self, bin_edges: ArrayLike) -> None:
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
    def fit(
        self, y_continuous: ArrayLike, y_binned: Union[ArrayLike, None] = None
    ) -> "BaseBinMapper":
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

        """
        pass

    @abstractmethod
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

        """
        pass


class EmpiricalMeanBinMapper(BaseBinMapper):
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


class EmpiricalMedianBinMapper(BaseBinMapper):
    """Maps discrete bin probabilities using empirical intra-bin medians.

    Computes the empirical median of continuous targets within each bin during
    fitting. Maps discrete probability mass functions (PMF) to continuous
    point estimates using these fitted medians. Empty bins are filled by
    interpolating between adjacent non-empty bin medians.

    Parameters
    ----------
    bin_edges : ArrayLike of shape (n_bins + 1,)
        Monotonically increasing boundaries defining continuous bin intervals.

    Attributes
    ----------
    bin_edges_ : np.ndarray
        1D float array of shape (n_bins + 1,) containing validated bin edges.
    bin_medians_ : np.ndarray
        1D float array of shape (n_bins,) containing empirical medians of
        continuous targets within each bin.
    n_bins_ : int
        Number of discrete bins defined by `bin_edges_`.

    Methods
    -------
    fit(y_continuous, y_binned=None)
        Compute empirical bin medians from continuous training targets.
    transform(pmf)
        Map discrete PMF probability matrix to continuous median estimates.
    to_continuous_dist(pmf)
        Construct a ContinuousPredictiveDistribution from a discrete PMF matrix.

    """

    def __init__(self, bin_edges: ArrayLike) -> None:
        super().__init__(bin_edges=bin_edges)

    def fit(
        self,
        y_continuous: ArrayLike,
        y_binned: Union[ArrayLike, None] = None,
    ) -> "EmpiricalMedianBinMapper":
        """Compute empirical bin medians from continuous training targets.

        Parameters
        ----------
        y_continuous : ArrayLike of shape (n_samples,)
            Unbinned continuous target values (e.g., exact physical units).
        y_binned : ArrayLike of shape (n_samples,), optional
            Corresponding 0-indexed discrete bin labels. If None, labels are
            computed automatically from `bin_edges`.

        Returns
        -------
        EmpiricalMedianBinMapper
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

        self.bin_medians_ = np.empty(self.n_bins_, dtype=float)
        empty_bins = []

        for k in range(self.n_bins_):
            mask = binned == k
            if np.any(mask):
                self.bin_medians_[k] = np.median(y_cont[mask])
            else:
                empty_bins.append(k)

        if empty_bins:
            valid_bins = np.setdiff1d(np.arange(self.n_bins_), empty_bins)
            if len(valid_bins) > 0:
                self.bin_medians_[empty_bins] = np.interp(
                    empty_bins, valid_bins, self.bin_medians_[valid_bins]
                )
            else:
                # Fallback to geometric midpoints if all bins are empty
                self.bin_medians_ = (edges[:-1] + edges[1:]) / 2.0

        return self

    def transform(self, pmf: ArrayLike) -> np.ndarray:
        """Map discrete PMF probability matrix to continuous expected values.

        Parameters
        ----------
        pmf : ArrayLike of shape (n_samples, n_bins)
            Probability mass function matrix where rows sum to 1.0.

        Returns
        -------
        np.ndarray
            1D float array of shape (n_samples,) containing continuous
            point estimates weighted by bin medians.

        Raises
        ------
        NotFittedError
            If the mapper instance has not been fitted prior to calling transform.
        ValueError
            If `pmf` is not a 2D array or column count does not match `n_bins_`.

        """
        check_is_fitted(self, attributes=["bin_edges_", "bin_medians_", "n_bins_"])
        pmf_arr = np.asarray(pmf, dtype=float)

        if pmf_arr.ndim != 2:
            raise ValueError("Expected 'pmf' to be a 2D array.")
        if pmf_arr.shape[1] != self.n_bins_:
            raise ValueError(
                f"PMF column dimension ({pmf_arr.shape[1]}) does not match "
                f"fitted bin count ({self.n_bins_})."
            )

        return np.dot(pmf_arr, self.bin_medians_)

    def to_continuous_dist(self, pmf: ArrayLike) -> ContinuousPredictiveDistribution:
        """Construct a ContinuousPredictiveDistribution from a discrete PMF matrix.

        Parameters
        ----------
        pmf : ArrayLike of shape (n_samples, n_bins)
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
        check_is_fitted(self, attributes=["bin_edges_", "bin_medians_", "n_bins_"])
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


class QuantileBinMapper(BaseBinMapper):
    """Maps discrete bin probabilities using intra-bin empirical quantiles.

    Fits intra-bin empirical quantiles from continuous training targets to build
    a fine-grained sub-grid. Constructing a continuous distribution on this grid
    allows for smoother cumulative distribution function (CDF) curves and more
    accurate continuous quantile predictions in wide or skewed bins.

    Parameters
    ----------
    bin_edges : ArrayLike of shape (n_bins + 1,)
        Monotonically increasing boundaries defining continuous bin intervals.
    quantiles : ArrayLike of shape (n_quantiles,), default=(0.25, 0.50, 0.75)
        Intra-bin quantile levels strictly in the range (0.0, 1.0) used to
        construct the fine-grained continuous grid.

    Attributes
    ----------
    bin_edges_ : np.ndarray
        1D float array of shape (n_bins + 1,) containing validated bin edges.
    quantiles_ : np.ndarray
        1D float array containing validated intra-bin quantile levels.
    grid_y_ : np.ndarray
        1D float array containing sub-grid target values in ascending order.
    grid_cdf_weights_ : np.ndarray
        1D float array containing sub-grid weights along bin threshold indices.
    n_bins_ : int
        Number of discrete bins defined by `bin_edges_`.

    Methods
    -------
    fit(y_continuous, y_binned=None)
        Compute intra-bin empirical quantiles from continuous training targets.
    transform(pmf)
        Map discrete PMF probability matrix to continuous expected values.
    to_continuous_dist(pmf)
        Construct a ContinuousPredictiveDistribution over the fitted sub-grid.

    """

    def __init__(
        self,
        bin_edges: ArrayLike,
        quantiles: ArrayLike = (0.25, 0.50, 0.75),
    ) -> None:
        super().__init__(bin_edges=bin_edges)
        self.quantiles = quantiles

    def fit(
        self,
        y_continuous: ArrayLike,
        y_binned: Union[ArrayLike, None] = None,
    ) -> "QuantileBinMapper":
        """Compute intra-bin empirical quantiles from continuous training targets.

        Parameters
        ----------
        y_continuous : ArrayLike of shape (n_samples,)
            Unbinned continuous target values (e.g., exact physical units).
        y_binned : ArrayLike of shape (n_samples,), optional
            Corresponding 0-indexed discrete bin labels. If None, labels are
            computed automatically from `bin_edges`.

        Returns
        -------
        QuantileBinMapper
            Fitted mapper instance.

        Raises
        ------
        ValueError
            If `bin_edges` is invalid, `quantiles` lie outside (0, 1),
            `y_continuous` is not 1D, or `y_binned` shape mismatches.

        """
        edges = self._validate_edges()
        q_arr = np.sort(np.asarray(self.quantiles, dtype=float))

        if q_arr.ndim != 1 or len(q_arr) == 0:
            raise ValueError("Expected 'quantiles' to be a non-empty 1D array-like.")
        if np.any((q_arr <= 0.0) | (q_arr >= 1.0)):
            raise ValueError(
                "All intra-bin quantiles must lie strictly within (0.0, 1.0)."
            )

        y_cont = np.asarray(y_continuous, dtype=float)
        if y_cont.ndim != 1:
            raise ValueError("Expected 'y_continuous' to be a 1D array.")

        self.bin_edges_ = edges
        self.quantiles_ = q_arr
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

        grid_y = [edges[0]]
        grid_weights = [0.0]

        for k in range(self.n_bins_):
            mask = binned == k
            low, high = edges[k], edges[k + 1]

            if np.any(mask):
                sub_q = np.quantile(y_cont[mask], self.quantiles_)
                sub_q = np.clip(sub_q, low, high)
            else:
                # Interpolate linear fraction if bin k has no training samples
                sub_q = low + (high - low) * self.quantiles_

            grid_y.extend(sub_q)
            grid_weights.extend(k + self.quantiles_)

            grid_y.append(high)
            grid_weights.append(float(k + 1))

        self.grid_y_ = np.array(grid_y, dtype=float)
        self.grid_cdf_weights_ = np.array(grid_weights, dtype=float)

        return self

    def transform(self, pmf: ArrayLike) -> np.ndarray:
        """Map discrete PMF probability matrix to continuous expected values.

        Parameters
        ----------
        pmf : ArrayLike of shape (n_samples, n_bins)
            Probability mass function matrix where rows sum to 1.0.

        Returns
        -------
        np.ndarray
            1D float array of shape (n_samples,) containing continuous
            expected target values evaluated over the fitted sub-grid.

        Raises
        ------
        NotFittedError
            If the mapper instance has not been fitted prior to calling transform.
        ValueError
            If `pmf` is not a 2D array or column count does not match `n_bins_`.

        """
        dist = self.to_continuous_dist(pmf)
        return dist.mean()

    def to_continuous_dist(self, pmf: ArrayLike) -> ContinuousPredictiveDistribution:
        """Construct a ContinuousPredictiveDistribution over the fitted sub-grid.

        Parameters
        ----------
        pmf : ArrayLike of shape (n_samples, n_bins)
            Discrete probability mass function matrix where rows sum to 1.0.

        Returns
        -------
        ContinuousPredictiveDistribution
            Continuous distribution evaluated over sub-grid `grid_y_`.

        Raises
        ------
        NotFittedError
            If the mapper instance has not been fitted prior to calling.
        ValueError
            If `pmf` is not a 2D array or column count does not match `n_bins_`.

        """
        check_is_fitted(
            self,
            attributes=[
                "bin_edges_",
                "quantiles_",
                "grid_y_",
                "grid_cdf_weights_",
                "n_bins_",
            ],
        )
        pmf_arr = np.asarray(pmf, dtype=float)

        if pmf_arr.ndim != 2:
            raise ValueError("Expected 'pmf' to be a 2D array.")
        if pmf_arr.shape[1] != self.n_bins_:
            raise ValueError(
                f"PMF column dimension ({pmf_arr.shape[1]}) does not match "
                f"fitted bin count ({self.n_bins_})."
            )

        n_samples = pmf_arr.shape[0]
        cum_pmf = np.hstack(
            [
                np.zeros((n_samples, 1), dtype=float),
                np.cumsum(pmf_arr, axis=1),
            ]
        )

        x_grid = np.arange(self.n_bins_ + 1, dtype=float)
        n_grid = len(self.grid_y_)
        grid_cdf = np.empty((n_samples, n_grid), dtype=float)

        for i in range(n_samples):
            grid_cdf[i] = np.interp(self.grid_cdf_weights_, x_grid, cum_pmf[i])

        return ContinuousPredictiveDistribution(grid_y=self.grid_y_, grid_cdf=grid_cdf)


class UniformBinMapper(BaseBinMapper):
    """Maps discrete bin probabilities using geometric bin midpoints.

    Uses fixed geometric midpoints between bin edges to convert discrete
    probability mass functions (PMF) into continuous point estimates and piecewise
    linear cumulative distributions. This mapper does not require continuous
    training targets to estimate bin statistics.

    Parameters
    ----------
    bin_edges : ArrayLike of shape (n_bins + 1,)
        Monotonically increasing boundaries defining continuous bin intervals.

    Attributes
    ----------
    bin_edges_ : np.ndarray
        1D float array of shape (n_bins + 1,) containing validated bin edges.
    bin_midpoints_ : np.ndarray
        1D float array of shape (n_bins,) containing geometric midpoints.
    n_bins_ : int
        Number of discrete bins defined by `bin_edges_`.

    Methods
    -------
    fit(y_continuous=None, y_binned=None)
        Compute geometric bin midpoints from bin edges.
    transform(pmf)
        Map discrete PMF probability matrix to continuous midpoint estimates.
    to_continuous_dist(pmf)
        Construct a ContinuousPredictiveDistribution from a discrete PMF matrix.

    """

    def __init__(self, bin_edges: ArrayLike) -> None:
        super().__init__(bin_edges=bin_edges)

    def fit(
        self,
        y_continuous: Union[ArrayLike, None] = None,
        y_binned: Union[ArrayLike, None] = None,
    ) -> "UniformBinMapper":
        """Compute geometric bin midpoints from bin edges.

        Parameters
        ----------
        y_continuous : ArrayLike of shape (n_samples,), optional
            Ignored. Retained for API compatibility with `BaseBinMapper`.
        y_binned : ArrayLike of shape (n_samples,), optional
            Ignored. Retained for API compatibility with `BaseBinMapper`.

        Returns
        -------
        UniformBinMapper
            Fitted mapper instance.

        Raises
        ------
        ValueError
            If `bin_edges` has fewer than 2 edges, is not 1D, or is not
            strictly monotonically increasing.

        """
        edges = self._validate_edges()
        self.bin_edges_ = edges
        self.n_bins_ = len(edges) - 1
        self.bin_midpoints_ = (edges[:-1] + edges[1:]) / 2.0
        return self

    def transform(self, pmf: ArrayLike) -> np.ndarray:
        """Map discrete PMF probability matrix to continuous expected values.

        Parameters
        ----------
        pmf : ArrayLike of shape (n_samples, n_bins)
            Probability mass function matrix where rows sum to 1.0.

        Returns
        -------
        np.ndarray
            1D float array of shape (n_samples,) containing continuous
            point estimates weighted by geometric bin midpoints.

        Raises
        ------
        NotFittedError
            If the mapper instance has not been fitted prior to calling transform.
        ValueError
            If `pmf` is not a 2D array or column count does not match `n_bins_`.

        """
        check_is_fitted(self, attributes=["bin_edges_", "bin_midpoints_", "n_bins_"])
        pmf_arr = np.asarray(pmf, dtype=float)

        if pmf_arr.ndim != 2:
            raise ValueError("Expected 'pmf' to be a 2D array.")
        if pmf_arr.shape[1] != self.n_bins_:
            raise ValueError(
                f"PMF column dimension ({pmf_arr.shape[1]}) does not match "
                f"fitted bin count ({self.n_bins_})."
            )

        return np.dot(pmf_arr, self.bin_midpoints_)

    def to_continuous_dist(self, pmf: ArrayLike) -> ContinuousPredictiveDistribution:
        """Construct a ContinuousPredictiveDistribution from a discrete PMF matrix.

        Parameters
        ----------
        pmf : ArrayLike of shape (n_samples, n_bins)
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
        check_is_fitted(self, attributes=["bin_edges_", "bin_midpoints_", "n_bins_"])
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


class ContinuousBinMapper(BaseBinMapper):
    """Maps discrete bin probabilities to a dense continuous target grid.

    Constructs a fine-grained cumulative predictive distribution over a dense y-grid
    spanning [y_min, y_max]. Intra-bin CDF shapes can either be weighted by the
    empirical target density of training data within each bin or linearly
    interpolated under a uniform intra-bin assumption.

    Parameters
    ----------
    bin_edges : ArrayLike of shape (n_bins + 1,)
        Monotonically increasing boundaries defining continuous bin intervals.
    grid_resolution : int, default=100
        Number of points in the automatically generated dense target grid if
        `grid_y` is None.
    grid_y : ArrayLike of shape (n_grid_points,), optional
        Custom 1D grid of target values spanning [bin_edges[0], bin_edges[-1]].
        If None, a linear grid of size `grid_resolution` is constructed.
    density_weighted : bool, default=True
        If True, intra-bin CDF increments are weighted by the empirical CDF of
        training targets in each bin. If False, uniform intra-bin linear
        interpolation is used.

    Attributes
    ----------
    bin_edges_ : np.ndarray
        1D float array of shape (n_bins + 1,) containing validated bin edges.
    grid_y_ : np.ndarray
        1D float array of shape (n_grid_points,) containing dense target values.
    bin_indices_ : np.ndarray
        1D int array of shape (n_grid_points,) identifying bin index for each grid point.
    intra_bin_cdf_ : np.ndarray
        1D float array of shape (n_grid_points,) containing intra-bin CDF weights in [0, 1].
    n_bins_ : int
        Number of discrete bins defined by `bin_edges_`.

    Methods
    -------
    fit(y_continuous, y_binned=None)
        Compute empirical intra-bin CDF weights across the dense evaluation grid.
    transform(pmf)
        Map discrete PMF probability matrix to continuous expected values.
    to_continuous_dist(pmf)
        Construct a ContinuousPredictiveDistribution over the fitted target grid.

    """

    def __init__(
        self,
        bin_edges: ArrayLike,
        grid_resolution: int = 100,
        grid_y: Union[ArrayLike, None] = None,
        density_weighted: bool = True,
    ) -> None:
        super().__init__(bin_edges=bin_edges)
        self.grid_resolution = grid_resolution
        self.grid_y = grid_y
        self.density_weighted = density_weighted

    def fit(
        self,
        y_continuous: ArrayLike,
        y_binned: Union[ArrayLike, None] = None,
    ) -> "ContinuousBinMapper":
        """Compute empirical intra-bin CDF weights across dense target grid.

        Parameters
        ----------
        y_continuous : ArrayLike of shape (n_samples,)
            Unbinned continuous target values (e.g., exact physical units).
        y_binned : ArrayLike of shape (n_samples,), optional
            Corresponding 0-indexed discrete bin labels. If None, labels are
            computed automatically from `bin_edges`.

        Returns
        -------
        ContinuousBinMapper
            Fitted mapper instance.

        Raises
        ------
        ValueError
            If `bin_edges` is invalid, `y_continuous` is not 1D, or `grid_y` is invalid.

        """
        edges = self._validate_edges()
        y_cont = np.asarray(y_continuous, dtype=float)

        if y_cont.ndim != 1:
            raise ValueError("Expected 'y_continuous' to be a 1D array.")

        self.bin_edges_ = edges
        self.n_bins_ = len(edges) - 1

        # Determine dense evaluation grid
        if self.grid_y is not None:
            g_y = np.sort(np.asarray(self.grid_y, dtype=float))
            if g_y.ndim != 1 or len(g_y) < 2:
                raise ValueError("Expected 'grid_y' to be a 1D array with >= 2 points.")
            if g_y[0] < edges[0] or g_y[-1] > edges[-1]:
                raise ValueError(
                    f"'grid_y' range [{g_y[0]}, {g_y[-1]}] must lie within "
                    f"bin bounds [{edges[0]}, {edges[-1]}]."
                )
            self.grid_y_ = g_y
        else:
            if self.grid_resolution < 2:
                raise ValueError("'grid_resolution' must be at least 2.")
            self.grid_y_ = np.linspace(edges[0], edges[-1], num=self.grid_resolution)

        # Digitize continuous training targets
        if y_binned is None:
            binned = np.digitize(y_cont, edges[1:-1])
        else:
            binned = np.asarray(y_binned, dtype=int)
            if binned.shape != y_cont.shape:
                raise ValueError(
                    f"Shape mismatch: 'y_binned' shape {binned.shape} "
                    f"does not match 'y_continuous' shape {y_cont.shape}."
                )

        # Assign each grid point to a bin index [0, n_bins - 1]
        grid_bins = np.digitize(self.grid_y_, edges[1:-1])
        # Force exact upper boundary edge to belong to final bin
        grid_bins = np.clip(grid_bins, 0, self.n_bins_ - 1)
        self.bin_indices_ = grid_bins

        n_grid = len(self.grid_y_)
        self.intra_bin_cdf_ = np.empty(n_grid, dtype=float)

        # Compute intra-bin CDF weights for each grid point
        for k in range(self.n_bins_):
            grid_mask = self.bin_indices_ == k
            if not np.any(grid_mask):
                continue

            low, high = edges[k], edges[k + 1]
            sub_y = self.grid_y_[grid_mask]

            train_mask = binned == k
            if self.density_weighted and np.any(train_mask):
                y_k = y_cont[train_mask]
                # Compute empirical CDF of training targets in bin k: P(Y <= y | Y in bin k)
                counts = np.searchsorted(np.sort(y_k), sub_y, side="right")
                self.intra_bin_cdf_[grid_mask] = counts / float(len(y_k))
            else:
                # Uniform intra-bin linear fallback
                denom = high - low
                if denom > 0:
                    self.intra_bin_cdf_[grid_mask] = (sub_y - low) / denom
                else:
                    self.intra_bin_cdf_[grid_mask] = 1.0

        return self

    def transform(self, pmf: ArrayLike) -> np.ndarray:
        """Map discrete PMF probability matrix to continuous expected values.

        Parameters
        ----------
        pmf : ArrayLike of shape (n_samples, n_bins)
            Probability mass function matrix where rows sum to 1.0.

        Returns
        -------
        np.ndarray
            1D float array of shape (n_samples,) containing continuous
            expected target values evaluated over the dense target grid.

        """
        dist = self.to_continuous_dist(pmf)
        return dist.mean()

    def to_continuous_dist(self, pmf: ArrayLike) -> ContinuousPredictiveDistribution:
        """Construct a ContinuousPredictiveDistribution over the dense target grid.

        Parameters
        ----------
        pmf : ArrayLike of shape (n_samples, n_bins)
            Discrete probability mass function matrix where rows sum to 1.0.

        Returns
        -------
        ContinuousPredictiveDistribution
            Continuous distribution evaluated over dense `grid_y_`.

        """
        check_is_fitted(
            self,
            attributes=[
                "bin_edges_",
                "grid_y_",
                "bin_indices_",
                "intra_bin_cdf_",
                "n_bins_",
            ],
        )
        pmf_arr = np.asarray(pmf, dtype=float)

        if pmf_arr.ndim != 2:
            raise ValueError("Expected 'pmf' to be a 2D array.")
        if pmf_arr.shape[1] != self.n_bins_:
            raise ValueError(
                f"PMF column dimension ({pmf_arr.shape[1]}) does not match "
                f"fitted bin count ({self.n_bins_})."
            )

        n_samples = pmf_arr.shape[0]

        # Precompute prior cumulative sum C_k for each sample
        cum_pmf = np.hstack(
            [
                np.zeros((n_samples, 1), dtype=float),
                np.cumsum(pmf_arr, axis=1),
            ]
        )

        # F(y|x) = C_k(x) + p_k(x) * intra_bin_cdf(y)
        prior_cdf = cum_pmf[:, self.bin_indices_]  # shape (n_samples, n_grid)
        bin_prob = pmf_arr[:, self.bin_indices_]  # shape (n_samples, n_grid)

        grid_cdf = prior_cdf + bin_prob * self.intra_bin_cdf_
        grid_cdf = np.clip(grid_cdf, 0.0, 1.0)

        return ContinuousPredictiveDistribution(
            grid_y=self.grid_y_,
            grid_cdf=grid_cdf,
        )
