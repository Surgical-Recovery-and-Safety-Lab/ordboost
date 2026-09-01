"""Base interfaces and implementations for bin-to-continuous target mappers."""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from ordboost.distributions import ContinuousPredictiveDistribution


class BaseBinMapper(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for all bin-to-continuous target mappers.

    A bin mapper converts a discrete probability mass function (PMF) over
    ordinal bins into a continuous predictive cumulative distribution
    function (CDF). Subclasses define how each bin's interior is refined
    into additional grid points by implementing `_intra_bin_points` and
    `fit`; `to_continuous_dist` and `transform` are shared and operate
    identically for every subclass once a grid has been fitted, so that a
    subclass's point estimate can never disagree with its full predicted
    distribution.

    Parameters
    ----------
    bin_edges : array-like of shape (n_bins + 1,) or None, default=None
        Monotonically increasing boundaries defining continuous bin
        intervals.
    bounded_below : bool, default=True
        If True, the outcome's support is asserted to have a true lower
        limit, and the predictive CDF is forced to 0 there. The anchor
        value is the observed training minimum when bin 0 contains
        training data, and `bin_edges[0]` otherwise. If False,
        `bin_edges[0]` is treated as an arbitrary or nominal lower bound
        (e.g. an open-ended first bin) rather than the outcome's true
        support boundary; the CDF is still forced to 0 there so the grid
        remains well-defined, but this should be understood as a
        truncation rather than a claim about the outcome itself.
    bounded_above : bool, default=True
        Mirrors `bounded_below` for the upper boundary of the final bin.
    floor_atom : bool, default=False
        If True, the lower boundary of bin 0 is modelled as a probability
        atom: the empirical fraction of bin 0's own training data at or
        below that boundary is estimated and assigned as that grid point's
        cumulative weight, producing an approximated discontinuity there.
        Requires `bounded_below=True`.
    ceiling_atom : bool, default=False
        Mirrors `floor_atom` for the upper boundary of the final bin.
        Requires `bounded_above=True`.
    boundary_epsilon : float, default=1e-4
        Offset used to place two grid points strictly outside/inside the
        outcome's support: the forced CDF=0 anchor at
        ``y_min - boundary_epsilon``, and, when `ceiling_atom` is True,
        the near-ceiling atom point at ``y_max - boundary_epsilon``. Must
        be small relative to the narrowest bin width in `bin_edges`;
        values comparable to or larger than a bin's width can place this
        offset point outside its intended bin or collide with adjacent
        grid points.

    Attributes
    ----------
    bin_edges_ : ndarray of shape (n_bins + 1,)
        Validated bin edges, set during `fit`.
    n_bins_ : int
        Number of discrete bins defined by `bin_edges_`.
    grid_y_ : ndarray of shape (n_grid_points,)
        Fitted sub-grid target values, in ascending order.
    grid_cdf_weights_ : ndarray of shape (n_grid_points,)
        Fitted sub-grid weights in bin-index units, aligned with
        `grid_y_`. A weight of ``k + f`` denotes that a fraction ``f`` of
        bin ``k``'s probability mass lies at or below the corresponding
        grid point.

    Methods
    -------
    fit(y_continuous, y_binned=None)
        Fit the mapper's grid to continuous training targets.
    transform(pmf)
        Map a discrete PMF matrix to continuous point estimates.
    to_continuous_dist(pmf)
        Construct a ContinuousPredictiveDistribution from a discrete PMF matrix.

    """

    def __init__(
        self,
        bin_edges: Union[ArrayLike, None] = None,
        bounded_below: bool = True,
        bounded_above: bool = True,
        floor_atom: bool = False,
        ceiling_atom: bool = False,
        boundary_epsilon: float = 1e-4,
    ) -> None:
        self.bin_edges = bin_edges
        self.bounded_below = bounded_below
        self.bounded_above = bounded_above
        self.floor_atom = floor_atom
        self.ceiling_atom = ceiling_atom
        self.boundary_epsilon = boundary_epsilon

    def _validate_atom_flags(self) -> None:
        """Validate consistency between the atom and boundedness flags.

        Raises
        ------
        ValueError
            If `floor_atom` is True while `bounded_below` is False, or if
            `ceiling_atom` is True while `bounded_above` is False. An atom
            cannot be placed at a boundary that is not itself asserted to
            be the outcome's true support limit.

        """
        if self.floor_atom and not self.bounded_below:
            raise ValueError(
                "'floor_atom=True' requires 'bounded_below=True': an atom "
                "cannot be placed at a boundary that isn't asserted to be "
                "the true lower limit of the outcome's support."
            )
        if self.ceiling_atom and not self.bounded_above:
            raise ValueError(
                "'ceiling_atom=True' requires 'bounded_above=True': an atom "
                "cannot be placed at a boundary that isn't asserted to be "
                "the true upper limit of the outcome's support."
            )

    def _validate_edges(self) -> np.ndarray:
        """Validate and return the mapper's bin edges.

        Returns
        -------
        ndarray of shape (n_bins + 1,)
            Validated bin edges as a 1D float array.

        Raises
        ------
        ValueError
            If `bin_edges` is None, is not 1D, has fewer than 2 edges, or
            is not strictly monotonically increasing.

        """
        if self.bin_edges is None:
            raise ValueError("'bin_edges' must be set on the mapper prior to fitting.")
        edges = np.asarray(self.bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError(
                "Expected 'bin_edges' to be a 1D array with at least 2 edges."
            )
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("'bin_edges' must be strictly monotonically increasing.")
        return edges

    def _validate_intra_bin_params(self) -> None:
        """Validate and set any parameters `_intra_bin_points` depends on.

        Called by `fit` before the grid is constructed. The base
        implementation is a no-op; subclasses that require setup ahead of
        grid construction (e.g. validating quantile levels) should
        override this method rather than `fit` itself.

        """
        return None

    def _digitize(
        self,
        y_cont: np.ndarray,
        edges: np.ndarray,
        y_binned: Union[ArrayLike, None],
    ) -> np.ndarray:
        """Assign each continuous target to a 0-indexed discrete bin.

        Parameters
        ----------
        y_cont : ndarray of shape (n_samples,)
            Continuous target values.
        edges : ndarray of shape (n_bins + 1,)
            Validated bin edges.
        y_binned : array-like of shape (n_samples,) or None
            Pre-computed 0-indexed bin labels. If None, labels are derived
            from `edges` via `numpy.digitize`.

        Returns
        -------
        ndarray of shape (n_samples,)
            Integer bin label for each sample.

        Raises
        ------
        ValueError
            If `y_binned` is provided and its shape does not match
            `y_cont`.

        """
        if y_binned is None:
            return np.digitize(y_cont, edges[1:-1])
        binned = np.asarray(y_binned, dtype=int)
        if binned.shape != y_cont.shape:
            raise ValueError(
                f"Shape mismatch: 'y_binned' shape {binned.shape} "
                f"does not match 'y_continuous' shape {y_cont.shape}."
            )
        return binned

    @staticmethod
    def _boundary_atom_weight(
        bin_data: np.ndarray, boundary: float, side: str
    ) -> float:
        """Compute the empirical fraction of a bin's data at a boundary.

        Parameters
        ----------
        bin_data : ndarray of shape (n_bin_samples,)
            Continuous training targets belonging to the bin being
            evaluated.
        boundary : float
            The boundary value to compare against.
        side : {'lower', 'upper'}
            If ``'lower'``, returns the fraction of `bin_data` at or below
            `boundary` (inclusive comparison, for a floor boundary). If
            ``'upper'``, returns the fraction strictly below `boundary`
            (exclusive comparison, for a ceiling boundary, so the boundary
            value itself is reserved for the forced weight of 1.0 at that
            point).

        Returns
        -------
        float
            The empirical fraction in [0.0, 1.0]. Returns 1.0 if
            `bin_data` is empty, since an empty bin has no interior mass
            to distinguish from its own boundary.

        """
        if len(bin_data) == 0:
            return 1.0
        if side == "upper":
            return float(np.mean(bin_data < boundary))
        return float(np.mean(bin_data <= boundary))

    def _build_grid(
        self, y_continuous: ArrayLike, y_binned: Union[ArrayLike, None] = None
    ) -> None:
        """Construct the fitted CDF grid shared by all mapper subclasses.

        Performs bin assignment, boundary anchoring governed by
        `bounded_below`/`bounded_above`, optional boundary-atom points
        governed by `floor_atom`/`ceiling_atom`, per-bin interior points
        from `_intra_bin_points`, and deduplication of coincident grid
        values (keeping the maximum weight at each unique `y`, so that a
        forced boundary weight is never silently discarded in favour of an
        earlier, lower-weight interior point at the same value).

        Parameters
        ----------
        y_continuous : array-like of shape (n_samples,)
            Unbinned continuous target values.
        y_binned : array-like of shape (n_samples,), optional
            Corresponding 0-indexed discrete bin labels. If None, labels
            are computed automatically from `bin_edges`.

        Raises
        ------
        ValueError
            If `bin_edges` is invalid, `y_continuous` is not 1D, or
            `y_binned` shape mismatches (raised by `_validate_edges` or
            `_digitize`); if `floor_atom`/`ceiling_atom` are set without
            their corresponding `bounded_*` flag (raised by
            `_validate_atom_flags`); or if boundary anchoring together with
            a supplied `y_binned` produces an invalid (non-positive-width)
            bin range -- typically indicating that `y_binned` assigns a
            sample to a bin whose own nominal edges cannot contain that
            sample's value.

        """
        self._validate_atom_flags()
        edges = self._validate_edges()
        y_cont = np.asarray(y_continuous, dtype=float)
        if y_cont.ndim != 1:
            raise ValueError("Expected 'y_continuous' to be a 1D array.")

        n_bins = len(edges) - 1
        binned = self._digitize(y_cont, edges, y_binned)

        y_min, y_max = float(y_cont.min()), float(y_cont.max())
        first_has_data = np.any(binned == 0)
        last_has_data = np.any(binned == n_bins - 1)

        low_bound_0 = y_min if (self.bounded_below and first_has_data) else edges[0]
        high_bound_last = y_max if (self.bounded_above and last_has_data) else edges[-1]

        grid_y = [low_bound_0 - self.boundary_epsilon]
        grid_w = [0.0]

        for k in range(n_bins):
            mask = binned == k
            low = low_bound_0 if k == 0 else edges[k]
            high = high_bound_last if k == n_bins - 1 else edges[k + 1]
            bin_data = y_cont[mask]

            if low > high:
                raise ValueError(
                    f"Bin {k} has an invalid range [low={low}, high={high}] after "
                    f"boundary anchoring. Check that 'y_binned' is consistent "
                    f"with 'y_continuous' and 'bin_edges'."
                )

            if k == 0:
                frac_at_or_below = (
                    self._boundary_atom_weight(bin_data, low, side="lower")
                    if self.floor_atom
                    else 0.0
                )
                grid_y.append(low)
                grid_w.append(k + frac_at_or_below)

            pts, weights = self._intra_bin_points(bin_data, low, high, k)
            grid_y.extend(pts)
            grid_w.extend(weights)

            if k == n_bins - 1 and self.ceiling_atom and last_has_data:
                frac_below = self._boundary_atom_weight(bin_data, high, side="upper")
                grid_y.append(high - self.boundary_epsilon)
                grid_w.append(k + frac_below)

            grid_y.append(high)
            grid_w.append(float(k + 1))

        grid_y_arr = np.array(grid_y, dtype=float)
        grid_w_arr = np.array(grid_w, dtype=float)
        order = np.argsort(grid_y_arr, kind="stable")
        sorted_y, sorted_w = grid_y_arr[order], grid_w_arr[order]
        unique_y, group_start = np.unique(sorted_y, return_index=True)
        max_w = np.maximum.reduceat(sorted_w, group_start)

        # Create fitted attributes
        self.bin_edges_ = edges
        self.n_bins_ = n_bins
        self.grid_y_ = unique_y
        self.grid_cdf_weights_ = max_w

    @abstractmethod
    def _intra_bin_points(
        self, bin_data: np.ndarray, low: float, high: float, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a subclass's interior grid points for one bin.

        Parameters
        ----------
        bin_data : ndarray of shape (n_bin_samples,)
            Continuous training targets belonging to bin `k`.
        low : float
            The effective lower boundary of bin `k` (equal to
            `bin_edges_[k]` for interior bins, or the anchored floor for
            bin 0).
        high : float
            The effective upper boundary of bin `k` (equal to
            `bin_edges_[k + 1]` for interior bins, or the anchored ceiling
            for the final bin).
        k : int
            The 0-indexed bin number.

        Returns
        -------
        points : ndarray
            Interior target values to add to the grid for this bin.
            May be empty for mappers that add no intra-bin refinement.
        weights : ndarray
            Cumulative weights aligned with `points`, in bin-index units
            (i.e. in the range ``[k, k + 1]``).

        """
        ...

    def fit(
        self, y_continuous: ArrayLike, y_binned: Union[ArrayLike, None] = None
    ) -> "BaseBinMapper":
        """Fit the mapper's grid to continuous training targets.

        Shared across all subclasses: validates any subclass-specific
        parameters via `_validate_intra_bin_params`, then constructs the
        grid via `_build_grid`, which sets `bin_edges_`, `n_bins_`,
        `grid_y_`, and `grid_cdf_weights_`.

        Parameters
        ----------
        y_continuous : array-like of shape (n_samples,)
            Unbinned continuous target values (e.g. exact physical units).
        y_binned : array-like of shape (n_samples,), optional
            Corresponding 0-indexed discrete bin labels. If None, labels
            are computed automatically from `bin_edges`.

        Returns
        -------
        BaseBinMapper
            The fitted mapper instance.

        Raises
        ------
        ValueError
            Raised by `_validate_intra_bin_params` or `_build_grid` for
            invalid parameters, edges, or input shapes.

        """
        self._validate_intra_bin_params()
        self._build_grid(y_continuous, y_binned)
        return self

    def to_continuous_dist(self, pmf: ArrayLike) -> ContinuousPredictiveDistribution:
        """Construct a continuous predictive distribution from a PMF.

        Interpolates each sample's cumulative PMF against the fitted
        `grid_cdf_weights_` to obtain the predicted CDF value at each
        `grid_y_` point. Shared across all subclasses; behaviour is fully
        determined by the fitted grid, so this always agrees with
        `transform`.

        Parameters
        ----------
        pmf : array-like of shape (n_samples, n_bins)
            Discrete probability mass function matrix where rows sum to
            1.0.

        Returns
        -------
        ContinuousPredictiveDistribution
            Continuous distribution evaluated over the fitted `grid_y_`.

        Raises
        ------
        NotFittedError
            If the mapper instance has not been fitted prior to calling.
        ValueError
            If `pmf` is not a 2D array or its column count does not match
            `n_bins_`.

        """
        check_is_fitted(
            self, attributes=["bin_edges_", "grid_y_", "grid_cdf_weights_", "n_bins_"]
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
            [np.zeros((n_samples, 1), dtype=float), np.cumsum(pmf_arr, axis=1)]
        )
        x_grid = np.arange(self.n_bins_ + 1, dtype=float)

        grid_cdf = np.empty((n_samples, len(self.grid_y_)), dtype=float)
        for i in range(n_samples):
            grid_cdf[i] = np.interp(self.grid_cdf_weights_, x_grid, cum_pmf[i])

        return ContinuousPredictiveDistribution(grid_y=self.grid_y_, grid_cdf=grid_cdf)

    def transform(self, pmf: ArrayLike) -> np.ndarray:
        """Map a discrete PMF matrix to continuous point estimates.

        Parameters
        ----------
        pmf : array-like of shape (n_samples, n_bins)
            Discrete probability mass function matrix where rows sum to
            1.0.

        Returns
        -------
        ndarray of shape (n_samples,)
            Continuous point estimates, taken as the mean of
            `to_continuous_dist(pmf)`. Subclasses may override this to
            return a different point estimate (e.g. the median).

        Raises
        ------
        NotFittedError
            If the mapper instance has not been fitted prior to calling.
        ValueError
            If `pmf` is not a 2D array or its column count does not match
            `n_bins_`.

        """
        return self.to_continuous_dist(pmf).mean()


class EmpiricalMeanBinMapper(BaseBinMapper):
    """Maps discrete bin probabilities using one empirical-mean point per bin.

    Refines each bin's interior with a single point at the empirical mean
    of that bin's own training data (not the geometric midpoint of the
    bin's edges), weighted by the empirical fraction of that bin's own
    data at or below the mean. This fraction is estimated from data rather
    than assumed to be 0.5, since the mean does not generally split a
    skewed bin's probability mass evenly.

    Parameters
    ----------
    bin_edges : array-like of shape (n_bins + 1,) or None, default=None
        Monotonically increasing boundaries defining continuous bin
        intervals.
    bounded_below : bool, default=True
        See `BaseBinMapper`.
    bounded_above : bool, default=True
        See `BaseBinMapper`.
    floor_atom : bool, default=False
        See `BaseBinMapper`.
    ceiling_atom : bool, default=False
        See `BaseBinMapper`.
    boundary_epsilon : float, default=1e-4
        See `BaseBinMapper`.

    """

    def _intra_bin_points(
        self, bin_data: np.ndarray, low: float, high: float, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one interior point at the bin's empirical mean.

        Parameters
        ----------
        bin_data : ndarray of shape (n_bin_samples,)
            Continuous training targets belonging to bin `k`.
        low : float
            The effective lower boundary of bin `k`.
        high : float
            The effective upper boundary of bin `k`.
        k : int
            The 0-indexed bin number.

        Returns
        -------
        points : ndarray of shape (1,)
            The bin's empirical mean, clipped to ``[low, high]``. Falls
            back to the geometric midpoint if `bin_data` is empty.
        weights : ndarray of shape (1,)
            The empirical fraction of `bin_data` at or below the mean,
            offset into bin-index units (``k + fraction``). Falls back to
            0.5 if `bin_data` is empty.

        """
        if len(bin_data) == 0:
            mean_val = (low + high) / 2.0
            frac_below = 0.5
        else:
            mean_val = float(np.clip(np.mean(bin_data), low, high))
            frac_below = float(np.mean(bin_data <= mean_val))
        return np.array([mean_val]), np.array([k + frac_below])


class EmpiricalMedianBinMapper(BaseBinMapper):
    """Maps discrete bin probabilities using one empirical-median point per bin.

    Refines each bin's interior with a single point at the empirical
    median of that bin's own training data, weighted by the empirical
    fraction of that bin's own data at or below the median. Unlike
    `EmpiricalMeanBinMapper`, this fraction is close to 0.5 by
    construction for even-sized, duplicate-free bins, but is still
    computed empirically rather than assumed, since ties and odd sample
    counts (where the median is itself an observed data point) can shift
    it away from exactly 0.5.

    Parameters
    ----------
    bin_edges : array-like of shape (n_bins + 1,) or None, default=None
        Monotonically increasing boundaries defining continuous bin
        intervals.
    bounded_below : bool, default=True
        See `BaseBinMapper`.
    bounded_above : bool, default=True
        See `BaseBinMapper`.
    floor_atom : bool, default=False
        See `BaseBinMapper`.
    ceiling_atom : bool, default=False
        See `BaseBinMapper`.
    boundary_epsilon : float, default=1e-4
        See `BaseBinMapper`.

    """

    def _intra_bin_points(
        self, bin_data: np.ndarray, low: float, high: float, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one interior point at the bin's empirical median.

        Parameters
        ----------
        bin_data : ndarray of shape (n_bin_samples,)
            Continuous training targets belonging to bin `k`.
        low : float
            The effective lower boundary of bin `k`.
        high : float
            The effective upper boundary of bin `k`.
        k : int
            The 0-indexed bin number.

        Returns
        -------
        points : ndarray of shape (1,)
            The bin's empirical median, clipped to ``[low, high]``. Falls
            back to the geometric midpoint if `bin_data` is empty.
        weights : ndarray of shape (1,)
            The empirical fraction of `bin_data` at or below the median,
            offset into bin-index units (``k + fraction``). Falls back to
            0.5 if `bin_data` is empty.

        """
        if len(bin_data) == 0:
            median_val = (low + high) / 2.0
            frac_below = 0.5
        else:
            median_val = float(np.clip(np.median(bin_data), low, high))
            frac_below = float(np.mean(bin_data <= median_val))
        return np.array([median_val]), np.array([k + frac_below])

    def transform(self, pmf: ArrayLike):
        """Map a discrete PMF matrix to continuous median point estimates.

        Overrides `BaseBinMapper.transform` to return the median of
        `to_continuous_dist(pmf)` rather than the mean, consistent with
        this mapper's use of the empirical median for grid construction.

        Parameters
        ----------
        pmf : array-like of shape (n_samples, n_bins)
            Discrete probability mass function matrix where rows sum to
            1.0.

        Returns
        -------
        ndarray of shape (n_samples,)
            Continuous median point estimates.

        Raises
        ------
        NotFittedError
            If the mapper instance has not been fitted prior to calling.
        ValueError
            If `pmf` is not a 2D array or its column count does not match
            `n_bins_`.

        """
        return self.to_continuous_dist(pmf).median()


class QuantileBinMapper(BaseBinMapper):
    """Maps discrete bin probabilities using intra-bin empirical quantiles.

    Refines each bin's interior using several fixed quantile levels
    computed on that bin's own training data, with each point's weight
    assumed equal to its quantile level (e.g. a bin's own empirical 25th
    percentile is assumed to sit at 25% into that bin's probability
    mass).

    Parameters
    ----------
    bin_edges : array-like of shape (n_bins + 1,) or None, default=None
        Monotonically increasing boundaries defining continuous bin
        intervals.
    quantiles : array-like of shape (n_quantiles,), default=(0.25, 0.50, 0.75)
        Intra-bin quantile levels, strictly within (0.0, 1.0), used to
        construct interior grid points for every bin.
    bounded_below : bool, default=True
        See `BaseBinMapper`.
    bounded_above : bool, default=True
        See `BaseBinMapper`.
    floor_atom : bool, default=False
        See `BaseBinMapper`.
    ceiling_atom : bool, default=False
        See `BaseBinMapper`.
    boundary_epsilon : float, default=1e-4
        See `BaseBinMapper`.

    Attributes
    ----------
    quantiles_ : ndarray of shape (n_quantiles,)
        Validated, sorted quantile levels, set by `_validate_intra_bin_params`
        during `fit`.

    """

    def __init__(
        self,
        bin_edges: Union[ArrayLike, None] = None,
        quantiles: ArrayLike = (0.25, 0.50, 0.75),
        bounded_below: bool = True,
        bounded_above: bool = True,
        floor_atom: bool = False,
        ceiling_atom: bool = False,
        boundary_epsilon: float = 1e-4,
    ) -> None:
        super().__init__(
            bin_edges=bin_edges,
            bounded_below=bounded_below,
            bounded_above=bounded_above,
            floor_atom=floor_atom,
            ceiling_atom=ceiling_atom,
            boundary_epsilon=boundary_epsilon,
        )
        self.quantiles = quantiles

    def _validate_intra_bin_params(self) -> None:
        """Validate `quantiles` and set the fitted `quantiles_` attribute.

        Raises
        ------
        ValueError
            If `quantiles` is empty, not 1D, or contains values outside
            the open interval (0.0, 1.0).

        """
        q_arr = np.sort(np.asarray(self.quantiles, dtype=float))
        if q_arr.ndim != 1 or len(q_arr) == 0:
            raise ValueError("Expected 'quantiles' to be a non-empty 1D array-like.")
        if np.any((q_arr <= 0.0) | (q_arr >= 1.0)):
            raise ValueError(
                "All intra-bin quantiles must lie strictly within (0.0, 1.0)."
            )
        self.quantiles_ = q_arr

    def _intra_bin_points(
        self, bin_data: np.ndarray, low: float, high: float, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return interior points at each fitted quantile level.

        Parameters
        ----------
        bin_data : ndarray of shape (n_bin_samples,)
            Continuous training targets belonging to bin `k`.
        low : float
            The effective lower boundary of bin `k`.
        high : float
            The effective upper boundary of bin `k`.
        k : int
            The 0-indexed bin number.

        Returns
        -------
        points : ndarray of shape (n_quantiles,)
            The bin's empirical values at each level in `quantiles_`,
            clipped to ``[low, high]``. Falls back to linear interpolation
            between `low` and `high` at each quantile level if `bin_data`
            is empty.
        weights : ndarray of shape (n_quantiles,)
            `quantiles_` offset into bin-index units (``k + quantiles_``).

        """
        if len(bin_data) == 0:
            pts = low + (high - low) * self.quantiles_
        else:
            pts = np.clip(np.quantile(bin_data, self.quantiles_), low, high)
        return pts, k + self.quantiles_


class UniformBinMapper(BaseBinMapper):
    """Maps discrete bin probabilities using geometric bin midpoints.

    Uses fixed geometric midpoints between bin edges to convert discrete
    probability mass functions (PMF) into continuous point estimates and piecewise
    linear cumulative distributions. This mapper does not require continuous
    training targets to estimate bin statistics.

    Parameters
    ----------
    bin_edges : ArrayLike of shape (n_bins + 1,) or None, default=None
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

    def __init__(self, bin_edges: Union[ArrayLike, None] = None) -> None:
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
    bin_edges : ArrayLike of shape (n_bins + 1,) or None, default=None
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
        bin_edges: Union[ArrayLike, None] = None,
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
            If `bin_edges` is invalid, `y_continuous` is not 1D,
            or `grid_y` is invalid.

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
            binned = np.digitize(y_cont, edges[:-1])
        else:
            binned = np.asarray(y_binned, dtype=int)
            if binned.shape != y_cont.shape:
                raise ValueError(
                    f"Shape mismatch: 'y_binned' shape {binned.shape} "
                    f"does not match 'y_continuous' shape {y_cont.shape}."
                )

        # Assign each grid point to a bin index [0, n_bins - 1]
        grid_bins = np.digitize(self.grid_y_, edges[:-1])
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
