"""Unit tests for BaseBinMapper in ordboost.debinning."""

import numpy as np
import pytest
from numpy.typing import ArrayLike
from sklearn.exceptions import NotFittedError

from ordboost.distributions import ContinuousPredictiveDistribution
from ordboost.mappers import BaseBinMapper, EmpiricalMeanMapper


class DummyBinMapper(BaseBinMapper):
    """Minimal concrete implementation of BaseBinMapper for unit testing."""

    def fit(
        self, y_continuous: ArrayLike, y_binned: ArrayLike | None = None
    ) -> "DummyBinMapper":
        """Dummy fit method."""
        self.bin_edges_ = self._validate_edges()
        self.n_bins_ = len(self.bin_edges_) - 1
        return self

    def transform(self, pmf: ArrayLike) -> np.ndarray:
        """Dummy transform method."""
        pmf_arr = np.asarray(pmf, dtype=float)
        midpoints = (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2.0
        return np.dot(pmf_arr, midpoints)

    def to_continuous_dist(self, pmf: ArrayLike) -> ContinuousPredictiveDistribution:
        """Dummy to_continuous_dist method."""
        pmf_arr = np.asarray(pmf, dtype=float)
        cum_pmf = np.hstack(
            [
                np.zeros((pmf_arr.shape[0], 1), dtype=float),
                np.cumsum(pmf_arr, axis=1),
            ]
        )
        return ContinuousPredictiveDistribution(
            grid_y=self.bin_edges_, grid_cdf=cum_pmf
        )


class TestBaseBinMapperABC:
    """Tests for abstract base class behavior and instantiation restrictions."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """Verify that instantiating BaseBinMapper directly raises TypeError."""
        with pytest.raises(TypeError):
            BaseBinMapper(bin_edges=[0, 10, 20])  # type: ignore[abstract]


class TestBaseBinMapperEdgeValidation:
    """Tests for _validate_edges method in BaseBinMapper."""

    def test_validate_edges_success(self) -> None:
        """Test validation with valid 1D monotonic edges."""
        mapper = DummyBinMapper(bin_edges=[0.0, 5.0, 10.0, 20.0])
        edges = mapper._validate_edges()
        np.testing.assert_array_equal(edges, np.array([0.0, 5.0, 10.0, 20.0]))

    def test_validate_edges_invalid_ndim(self) -> None:
        """Test that 2D edges array raises ValueError."""
        mapper = DummyBinMapper(bin_edges=np.array([[0, 10], [10, 20]]))
        with pytest.raises(ValueError, match="1D array with at least 2 edges"):
            mapper._validate_edges()

    def test_validate_edges_too_few_edges(self) -> None:
        """Test that fewer than 2 edges raises ValueError."""
        mapper = DummyBinMapper(bin_edges=[10.0])
        with pytest.raises(ValueError, match="at least 2 edges"):
            mapper._validate_edges()

    def test_validate_edges_non_monotonic(self) -> None:
        """Test that decreasing edges raise ValueError."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 5.0, 20.0])
        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            mapper._validate_edges()

    def test_validate_edges_duplicate_edges(self) -> None:
        """Test that duplicate adjacent edges raise ValueError."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 10.0, 20.0])
        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            mapper._validate_edges()


class TestDummyBinMapperConcrete:
    """Tests verifying concrete implementation functionality using DummyBinMapper."""

    def test_fit_and_transform(self) -> None:
        """Test successful fit and transform execution on dummy implementation."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit([2.0, 15.0])
        pmf = np.array([[0.5, 0.5]])
        result = mapper.transform(pmf)
        np.testing.assert_allclose(result, [10.0])

    def test_to_continuous_dist(self) -> None:
        """Test distribution creation on dummy implementation."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit([2.0, 15.0])
        pmf = np.array([[0.5, 0.5]])
        dist = mapper.to_continuous_dist(pmf)
        assert isinstance(dist, ContinuousPredictiveDistribution)
        np.testing.assert_array_equal(dist.grid_y, [0.0, 10.0, 20.0])
        np.testing.assert_allclose(dist.grid_cdf, [[0.0, 0.5, 1.0]])


class TestEmpiricalMeanMapperInit:
    """Tests for EmpiricalMeanMapper initialization."""

    def test_init_stores_bin_edges(self) -> None:
        """Verify __init__ correctly stores bin_edges attribute."""
        edges = [0, 5, 10, 20]
        mapper = EmpiricalMeanMapper(bin_edges=edges)
        assert mapper.bin_edges == edges


class TestEmpiricalMeanMapperFit:
    """Tests for EmpiricalMeanMapper fit method, parameter validation, and fallbacks."""

    def test_fit_success_automatic_binning(self) -> None:
        """Test successful fit with automatic digitization of continuous targets."""
        edges = [0.0, 10.0, 20.0, 30.0]
        y_cont = np.array([2.0, 4.0, 18.0, 19.0, 21.0, 29.0])
        mapper = EmpiricalMeanMapper(bin_edges=edges)

        fitted_mapper = mapper.fit(y_cont)
        assert fitted_mapper is mapper
        assert mapper.n_bins_ == 3
        np.testing.assert_array_equal(mapper.bin_edges_, np.array(edges))

        # Bin 0 ([0, 10)): mean([2, 4]) = 3.0
        # Bin 1 ([10, 20)): mean([18, 19]) = 18.5
        # Bin 2 ([20, 30)): mean([21, 29]) = 25.0
        expected_means = np.array([3.0, 18.5, 25.0])
        np.testing.assert_allclose(mapper.bin_means_, expected_means)

    def test_fit_success_explicit_y_binned(self) -> None:
        """Test successful fit when explicit y_binned is provided."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([1.0, 3.0, 15.0])
        y_binned = np.array([0, 0, 1])

        mapper = EmpiricalMeanMapper(bin_edges=edges)
        mapper.fit(y_cont, y_binned=y_binned)

        assert mapper.bin_means_[0] == pytest.approx(2.0)
        assert mapper.bin_means_[1] == pytest.approx(15.0)

    def test_fit_empty_bin_interpolation(self) -> None:
        """Test that an empty interior bin interpolates adjacent fitted means."""
        edges = [0.0, 10.0, 20.0, 30.0]
        # Bin 1 ([10, 20)) has zero samples
        y_cont = np.array([2.0, 4.0, 22.0, 28.0])

        mapper = EmpiricalMeanMapper(bin_edges=edges)
        mapper.fit(y_cont)

        # Bin 0 mean = 3.0, Bin 2 mean = 25.0
        # Bin 1 interpolated mean = (3.0 + 25.0) / 2 = 14.0
        assert mapper.bin_means_[1] == pytest.approx(14.0)

    def test_fit_all_empty_bins_fallback(self) -> None:
        """Test fallback to geometric midpoints when no training samples exist."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([], dtype=float)

        mapper = EmpiricalMeanMapper(bin_edges=edges)
        mapper.fit(y_cont)

        expected_midpoints = np.array([5.0, 15.0])
        np.testing.assert_allclose(mapper.bin_means_, expected_midpoints)

    def test_fit_invalid_y_continuous_ndim(self) -> None:
        """Test that 2D y_continuous raises ValueError."""
        mapper = EmpiricalMeanMapper(bin_edges=[0, 10, 20])
        with pytest.raises(ValueError, match="1D array"):
            mapper.fit(np.ones((5, 2)))

    def test_fit_y_binned_shape_mismatch(self) -> None:
        """Test error handling when y_binned length differs from y_continuous."""
        mapper = EmpiricalMeanMapper(bin_edges=[0, 10, 20])
        with pytest.raises(ValueError, match="Shape mismatch"):
            mapper.fit(y_continuous=[1, 2, 3], y_binned=[0, 1])


class TestEmpiricalMeanMapperTransform:
    """Tests for EmpiricalMeanMapper transform method and validation."""

    def test_transform_success(self) -> None:
        """Test mapping PMF matrix to continuous expected values."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([2.0, 4.0, 18.0])  # means: bin0=3.0, bin1=18.0
        mapper = EmpiricalMeanMapper(bin_edges=edges).fit(y_cont)

        pmf = np.array([[0.5, 0.5], [1.0, 0.0]])
        expected = mapper.transform(pmf)

        assert expected.shape == (2,)
        assert expected[0] == pytest.approx(10.5)
        assert expected[1] == pytest.approx(3.0)

    def test_transform_not_fitted(self) -> None:
        """Test calling transform on un-fitted instance raises NotFittedError."""
        mapper = EmpiricalMeanMapper(bin_edges=[0, 10, 20])
        with pytest.raises(NotFittedError):
            mapper.transform([[0.5, 0.5]])

    def test_transform_invalid_pmf_ndim(self) -> None:
        """Test that 1D PMF raises ValueError."""
        mapper = EmpiricalMeanMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="2D array"):
            mapper.transform([0.5, 0.5])

    def test_transform_column_dimension_mismatch(self) -> None:
        """Test that PMF column count mismatch with n_bins_ raises ValueError."""
        mapper = EmpiricalMeanMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        invalid_pmf = np.array([[0.3, 0.3, 0.4]])
        with pytest.raises(ValueError, match="PMF column dimension"):
            mapper.transform(invalid_pmf)


class TestEmpiricalMeanMapperToContinuousDist:
    """Tests for EmpiricalMeanMapper to_continuous_dist method and distribution construction."""

    def test_to_continuous_dist_success(self) -> None:
        """Test converting discrete PMF matrix to ContinuousPredictiveDistribution."""
        edges = [0.0, 10.0, 20.0]
        mapper = EmpiricalMeanMapper(bin_edges=edges).fit([2.0, 15.0])

        pmf = np.array([[0.4, 0.6], [0.1, 0.9]])
        dist = mapper.to_continuous_dist(pmf)

        assert isinstance(dist, ContinuousPredictiveDistribution)
        np.testing.assert_array_equal(dist.grid_y, np.array(edges))
        assert dist.grid_cdf.shape == (2, 3)
        np.testing.assert_allclose(dist.grid_cdf[0], [0.0, 0.4, 1.0])
        np.testing.assert_allclose(dist.grid_cdf[1], [0.0, 0.1, 1.0])

    def test_to_continuous_dist_not_fitted(self) -> None:
        """Test calling to_continuous_dist on un-fitted instance raises NotFittedError."""
        mapper = EmpiricalMeanMapper(bin_edges=[0, 10, 20])
        with pytest.raises(NotFittedError):
            mapper.to_continuous_dist([[0.5, 0.5]])

    def test_to_continuous_dist_invalid_pmf_ndim(self) -> None:
        """Test error handling when PMF is not 2D."""
        mapper = EmpiricalMeanMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="2D array"):
            mapper.to_continuous_dist([0.5, 0.5])

    def test_to_continuous_dist_column_dimension_mismatch(self) -> None:
        """Test error handling when PMF columns mismatch fitted bin count."""
        mapper = EmpiricalMeanMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="PMF column dimension"):
            mapper.to_continuous_dist([[0.3, 0.3, 0.4]])


class EmpiricalMedianMapper(BaseBinMapper):
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
    ) -> "EmpiricalMedianMapper":
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
        EmpiricalMedianMapper
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


class TestEmpiricalMedianMapperInit:
    """Tests for EmpiricalMedianMapper initialization."""

    def test_init_stores_bin_edges(self) -> None:
        """Verify __init__ correctly stores bin_edges attribute."""
        edges = [0, 5, 10, 20]
        mapper = EmpiricalMedianMapper(bin_edges=edges)
        assert mapper.bin_edges == edges


class TestEmpiricalMedianMapperFit:
    """Tests for EmpiricalMedianMapper fit method and parameter validation."""

    def test_fit_success_skewed_bins(self) -> None:
        """Test that median calculation differs correctly from mean on skewed data."""
        edges = [0.0, 10.0, 20.0, 30.0]
        # Heavily skewed bin targets where median != mean
        # Bin 0 ([0, 10)): [1.0, 2.0, 9.0] -> median=2.0 (mean=4.0)
        # Bin 1 ([10, 20)): [11.0, 12.0, 19.0] -> median=12.0 (mean=14.0)
        # Bin 2 ([20, 30)): [21.0, 29.0, 29.0] -> median=29.0 (mean=26.33)
        y_cont = np.array([1.0, 2.0, 9.0, 11.0, 12.0, 19.0, 21.0, 29.0, 29.0])
        mapper = EmpiricalMedianMapper(bin_edges=edges)

        fitted_mapper = mapper.fit(y_cont)
        assert fitted_mapper is mapper
        assert mapper.n_bins_ == 3

        expected_medians = np.array([2.0, 12.0, 29.0])
        np.testing.assert_allclose(mapper.bin_medians_, expected_medians)

    def test_fit_success_explicit_y_binned(self) -> None:
        """Test successful fit when explicit y_binned is provided."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([1.0, 3.0, 9.0, 15.0])
        y_binned = np.array([0, 0, 0, 1])

        mapper = EmpiricalMedianMapper(bin_edges=edges)
        mapper.fit(y_cont, y_binned=y_binned)

        assert mapper.bin_medians_[0] == pytest.approx(3.0)
        assert mapper.bin_medians_[1] == pytest.approx(15.0)

    def test_fit_empty_bin_interpolation(self) -> None:
        """Test that an empty interior bin interpolates adjacent fitted medians."""
        edges = [0.0, 10.0, 20.0, 30.0]
        # Bin 1 ([10, 20)) has zero samples
        # Bin 0 median = 2.0, Bin 2 median = 26.0
        y_cont = np.array([1.0, 2.0, 3.0, 25.0, 26.0, 27.0])

        mapper = EmpiricalMedianMapper(bin_edges=edges)
        mapper.fit(y_cont)

        # Bin 1 interpolated median = (2.0 + 26.0) / 2 = 14.0
        assert mapper.bin_medians_[1] == pytest.approx(14.0)

    def test_fit_all_empty_bins_fallback(self) -> None:
        """Test fallback to geometric midpoints when no training samples exist."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([], dtype=float)

        mapper = EmpiricalMedianMapper(bin_edges=edges)
        mapper.fit(y_cont)

        expected_midpoints = np.array([5.0, 15.0])
        np.testing.assert_allclose(mapper.bin_medians_, expected_midpoints)

    def test_fit_invalid_y_continuous_ndim(self) -> None:
        """Test that 2D y_continuous raises ValueError."""
        mapper = EmpiricalMedianMapper(bin_edges=[0, 10, 20])
        with pytest.raises(ValueError, match="1D array"):
            mapper.fit(np.ones((5, 2)))

    def test_fit_y_binned_shape_mismatch(self) -> None:
        """Test error handling when y_binned length differs from y_continuous."""
        mapper = EmpiricalMedianMapper(bin_edges=[0, 10, 20])
        with pytest.raises(ValueError, match="Shape mismatch"):
            mapper.fit(y_continuous=[1, 2, 3], y_binned=[0, 1])


class TestEmpiricalMedianMapperTransform:
    """Tests for EmpiricalMedianMapper transform method and validation."""

    def test_transform_success(self) -> None:
        """Test mapping PMF matrix to continuous values weighted by bin medians."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([1.0, 2.0, 9.0, 18.0])  # medians: bin0=2.0, bin1=18.0
        mapper = EmpiricalMedianMapper(bin_edges=edges).fit(y_cont)

        pmf = np.array([[0.5, 0.5], [1.0, 0.0]])
        expected = mapper.transform(pmf)

        assert expected.shape == (2,)
        assert expected[0] == pytest.approx(10.0)
        assert expected[1] == pytest.approx(2.0)

    def test_transform_not_fitted(self) -> None:
        """Test calling transform on un-fitted instance raises NotFittedError."""
        mapper = EmpiricalMedianMapper(bin_edges=[0, 10, 20])
        with pytest.raises(NotFittedError):
            mapper.transform([[0.5, 0.5]])

    def test_transform_invalid_pmf_ndim(self) -> None:
        """Test that 1D PMF raises ValueError."""
        mapper = EmpiricalMedianMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="2D array"):
            mapper.transform([0.5, 0.5])

    def test_transform_column_dimension_mismatch(self) -> None:
        """Test that PMF column count mismatch with n_bins_ raises ValueError."""
        mapper = EmpiricalMedianMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        invalid_pmf = np.array([[0.3, 0.3, 0.4]])
        with pytest.raises(ValueError, match="PMF column dimension"):
            mapper.transform(invalid_pmf)


class TestEmpiricalMedianMapperToContinuousDist:
    """Tests for EmpiricalMedianMapper to_continuous_dist method."""

    def test_to_continuous_dist_success(self) -> None:
        """Test converting discrete PMF matrix to ContinuousPredictiveDistribution."""
        edges = [0.0, 10.0, 20.0]
        mapper = EmpiricalMedianMapper(bin_edges=edges).fit([2.0, 15.0])

        pmf = np.array([[0.3, 0.7], [0.8, 0.2]])
        dist = mapper.to_continuous_dist(pmf)

        assert isinstance(dist, ContinuousPredictiveDistribution)
        np.testing.assert_array_equal(dist.grid_y, np.array(edges))
        assert dist.grid_cdf.shape == (2, 3)
        np.testing.assert_allclose(dist.grid_cdf[0], [0.0, 0.3, 1.0])
        np.testing.assert_allclose(dist.grid_cdf[1], [0.0, 0.8, 1.0])

    def test_to_continuous_dist_not_fitted(self) -> None:
        """Test calling to_continuous_dist on un-fitted instance raises NotFittedError."""
        mapper = EmpiricalMedianMapper(bin_edges=[0, 10, 20])
        with pytest.raises(NotFittedError):
            mapper.to_continuous_dist([[0.5, 0.5]])

    def test_to_continuous_dist_invalid_pmf_ndim(self) -> None:
        """Test error handling when PMF is not 2D."""
        mapper = EmpiricalMedianMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="2D array"):
            mapper.to_continuous_dist([0.5, 0.5])

    def test_to_continuous_dist_column_dimension_mismatch(self) -> None:
        """Test error handling when PMF columns mismatch fitted bin count."""
        mapper = EmpiricalMedianMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="PMF column dimension"):
            mapper.to_continuous_dist([[0.3, 0.3, 0.4]])
