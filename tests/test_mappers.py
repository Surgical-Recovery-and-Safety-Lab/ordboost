"""Unit tests for BaseBinMapper in ordboost.debinning."""

import numpy as np
import pytest
from numpy.typing import ArrayLike
from sklearn.exceptions import NotFittedError

from ordboost.distributions import ContinuousPredictiveDistribution
from ordboost.mappers import (
    BaseBinMapper,
    EmpiricalMeanBinMapper,
    EmpiricalMedianBinMapper,
    QuantileBinMapper,
)


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


class TestEmpiricalMeanBinMapperInit:
    """Tests for EmpiricalMeanBinMapper initialization."""

    def test_init_stores_bin_edges(self) -> None:
        """Verify __init__ correctly stores bin_edges attribute."""
        edges = [0, 5, 10, 20]
        mapper = EmpiricalMeanBinMapper(bin_edges=edges)
        assert mapper.bin_edges == edges


class TestEmpiricalMeanBinMapperFit:
    """Tests for EmpiricalMeanBinMapper fit method, parameter validation, and fallbacks."""

    def test_fit_success_automatic_binning(self) -> None:
        """Test successful fit with automatic digitization of continuous targets."""
        edges = [0.0, 10.0, 20.0, 30.0]
        y_cont = np.array([2.0, 4.0, 18.0, 19.0, 21.0, 29.0])
        mapper = EmpiricalMeanBinMapper(bin_edges=edges)

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

        mapper = EmpiricalMeanBinMapper(bin_edges=edges)
        mapper.fit(y_cont, y_binned=y_binned)

        assert mapper.bin_means_[0] == pytest.approx(2.0)
        assert mapper.bin_means_[1] == pytest.approx(15.0)

    def test_fit_empty_bin_interpolation(self) -> None:
        """Test that an empty interior bin interpolates adjacent fitted means."""
        edges = [0.0, 10.0, 20.0, 30.0]
        # Bin 1 ([10, 20)) has zero samples
        y_cont = np.array([2.0, 4.0, 22.0, 28.0])

        mapper = EmpiricalMeanBinMapper(bin_edges=edges)
        mapper.fit(y_cont)

        # Bin 0 mean = 3.0, Bin 2 mean = 25.0
        # Bin 1 interpolated mean = (3.0 + 25.0) / 2 = 14.0
        assert mapper.bin_means_[1] == pytest.approx(14.0)

    def test_fit_all_empty_bins_fallback(self) -> None:
        """Test fallback to geometric midpoints when no training samples exist."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([], dtype=float)

        mapper = EmpiricalMeanBinMapper(bin_edges=edges)
        mapper.fit(y_cont)

        expected_midpoints = np.array([5.0, 15.0])
        np.testing.assert_allclose(mapper.bin_means_, expected_midpoints)

    def test_fit_invalid_y_continuous_ndim(self) -> None:
        """Test that 2D y_continuous raises ValueError."""
        mapper = EmpiricalMeanBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(ValueError, match="1D array"):
            mapper.fit(np.ones((5, 2)))

    def test_fit_y_binned_shape_mismatch(self) -> None:
        """Test error handling when y_binned length differs from y_continuous."""
        mapper = EmpiricalMeanBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(ValueError, match="Shape mismatch"):
            mapper.fit(y_continuous=[1, 2, 3], y_binned=[0, 1])


class TestEmpiricalMeanBinMapperTransform:
    """Tests for EmpiricalMeanBinMapper transform method and validation."""

    def test_transform_success(self) -> None:
        """Test mapping PMF matrix to continuous expected values."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([2.0, 4.0, 18.0])  # means: bin0=3.0, bin1=18.0
        mapper = EmpiricalMeanBinMapper(bin_edges=edges).fit(y_cont)

        pmf = np.array([[0.5, 0.5], [1.0, 0.0]])
        expected = mapper.transform(pmf)

        assert expected.shape == (2,)
        assert expected[0] == pytest.approx(10.5)
        assert expected[1] == pytest.approx(3.0)

    def test_transform_not_fitted(self) -> None:
        """Test calling transform on un-fitted instance raises NotFittedError."""
        mapper = EmpiricalMeanBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(NotFittedError):
            mapper.transform([[0.5, 0.5]])

    def test_transform_invalid_pmf_ndim(self) -> None:
        """Test that 1D PMF raises ValueError."""
        mapper = EmpiricalMeanBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="2D array"):
            mapper.transform([0.5, 0.5])

    def test_transform_column_dimension_mismatch(self) -> None:
        """Test that PMF column count mismatch with n_bins_ raises ValueError."""
        mapper = EmpiricalMeanBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        invalid_pmf = np.array([[0.3, 0.3, 0.4]])
        with pytest.raises(ValueError, match="PMF column dimension"):
            mapper.transform(invalid_pmf)


class TestEmpiricalMeanBinMapperToContinuousDist:
    """Tests for EmpiricalMeanBinMapper to_continuous_dist method and distribution construction."""

    def test_to_continuous_dist_success(self) -> None:
        """Test converting discrete PMF matrix to ContinuousPredictiveDistribution."""
        edges = [0.0, 10.0, 20.0]
        mapper = EmpiricalMeanBinMapper(bin_edges=edges).fit([2.0, 15.0])

        pmf = np.array([[0.4, 0.6], [0.1, 0.9]])
        dist = mapper.to_continuous_dist(pmf)

        assert isinstance(dist, ContinuousPredictiveDistribution)
        np.testing.assert_array_equal(dist.grid_y, np.array(edges))
        assert dist.grid_cdf.shape == (2, 3)
        np.testing.assert_allclose(dist.grid_cdf[0], [0.0, 0.4, 1.0])
        np.testing.assert_allclose(dist.grid_cdf[1], [0.0, 0.1, 1.0])

    def test_to_continuous_dist_not_fitted(self) -> None:
        """Test calling to_continuous_dist on un-fitted instance raises NotFittedError."""
        mapper = EmpiricalMeanBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(NotFittedError):
            mapper.to_continuous_dist([[0.5, 0.5]])

    def test_to_continuous_dist_invalid_pmf_ndim(self) -> None:
        """Test error handling when PMF is not 2D."""
        mapper = EmpiricalMeanBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="2D array"):
            mapper.to_continuous_dist([0.5, 0.5])

    def test_to_continuous_dist_column_dimension_mismatch(self) -> None:
        """Test error handling when PMF columns mismatch fitted bin count."""
        mapper = EmpiricalMeanBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="PMF column dimension"):
            mapper.to_continuous_dist([[0.3, 0.3, 0.4]])


class TestEmpiricalMedianBinMapperInit:
    """Tests for EmpiricalMedianBinMapper initialization."""

    def test_init_stores_bin_edges(self) -> None:
        """Verify __init__ correctly stores bin_edges attribute."""
        edges = [0, 5, 10, 20]
        mapper = EmpiricalMedianBinMapper(bin_edges=edges)
        assert mapper.bin_edges == edges


class TestEmpiricalMedianBinMapperFit:
    """Tests for EmpiricalMedianBinMapper fit method and parameter validation."""

    def test_fit_success_skewed_bins(self) -> None:
        """Test that median calculation differs correctly from mean on skewed data."""
        edges = [0.0, 10.0, 20.0, 30.0]
        # Heavily skewed bin targets where median != mean
        # Bin 0 ([0, 10)): [1.0, 2.0, 9.0] -> median=2.0 (mean=4.0)
        # Bin 1 ([10, 20)): [11.0, 12.0, 19.0] -> median=12.0 (mean=14.0)
        # Bin 2 ([20, 30)): [21.0, 29.0, 29.0] -> median=29.0 (mean=26.33)
        y_cont = np.array([1.0, 2.0, 9.0, 11.0, 12.0, 19.0, 21.0, 29.0, 29.0])
        mapper = EmpiricalMedianBinMapper(bin_edges=edges)

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

        mapper = EmpiricalMedianBinMapper(bin_edges=edges)
        mapper.fit(y_cont, y_binned=y_binned)

        assert mapper.bin_medians_[0] == pytest.approx(3.0)
        assert mapper.bin_medians_[1] == pytest.approx(15.0)

    def test_fit_empty_bin_interpolation(self) -> None:
        """Test that an empty interior bin interpolates adjacent fitted medians."""
        edges = [0.0, 10.0, 20.0, 30.0]
        # Bin 1 ([10, 20)) has zero samples
        # Bin 0 median = 2.0, Bin 2 median = 26.0
        y_cont = np.array([1.0, 2.0, 3.0, 25.0, 26.0, 27.0])

        mapper = EmpiricalMedianBinMapper(bin_edges=edges)
        mapper.fit(y_cont)

        # Bin 1 interpolated median = (2.0 + 26.0) / 2 = 14.0
        assert mapper.bin_medians_[1] == pytest.approx(14.0)

    def test_fit_all_empty_bins_fallback(self) -> None:
        """Test fallback to geometric midpoints when no training samples exist."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([], dtype=float)

        mapper = EmpiricalMedianBinMapper(bin_edges=edges)
        mapper.fit(y_cont)

        expected_midpoints = np.array([5.0, 15.0])
        np.testing.assert_allclose(mapper.bin_medians_, expected_midpoints)

    def test_fit_invalid_y_continuous_ndim(self) -> None:
        """Test that 2D y_continuous raises ValueError."""
        mapper = EmpiricalMedianBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(ValueError, match="1D array"):
            mapper.fit(np.ones((5, 2)))

    def test_fit_y_binned_shape_mismatch(self) -> None:
        """Test error handling when y_binned length differs from y_continuous."""
        mapper = EmpiricalMedianBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(ValueError, match="Shape mismatch"):
            mapper.fit(y_continuous=[1, 2, 3], y_binned=[0, 1])


class TestEmpiricalMedianBinMapperTransform:
    """Tests for EmpiricalMedianBinMapper transform method and validation."""

    def test_transform_success(self) -> None:
        """Test mapping PMF matrix to continuous values weighted by bin medians."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([1.0, 2.0, 9.0, 18.0])  # medians: bin0=2.0, bin1=18.0
        mapper = EmpiricalMedianBinMapper(bin_edges=edges).fit(y_cont)

        pmf = np.array([[0.5, 0.5], [1.0, 0.0]])
        expected = mapper.transform(pmf)

        assert expected.shape == (2,)
        assert expected[0] == pytest.approx(10.0)
        assert expected[1] == pytest.approx(2.0)

    def test_transform_not_fitted(self) -> None:
        """Test calling transform on un-fitted instance raises NotFittedError."""
        mapper = EmpiricalMedianBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(NotFittedError):
            mapper.transform([[0.5, 0.5]])

    def test_transform_invalid_pmf_ndim(self) -> None:
        """Test that 1D PMF raises ValueError."""
        mapper = EmpiricalMedianBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="2D array"):
            mapper.transform([0.5, 0.5])

    def test_transform_column_dimension_mismatch(self) -> None:
        """Test that PMF column count mismatch with n_bins_ raises ValueError."""
        mapper = EmpiricalMedianBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        invalid_pmf = np.array([[0.3, 0.3, 0.4]])
        with pytest.raises(ValueError, match="PMF column dimension"):
            mapper.transform(invalid_pmf)


class TestEmpiricalMedianBinMapperToContinuousDist:
    """Tests for EmpiricalMedianBinMapper to_continuous_dist method."""

    def test_to_continuous_dist_success(self) -> None:
        """Test converting discrete PMF matrix to ContinuousPredictiveDistribution."""
        edges = [0.0, 10.0, 20.0]
        mapper = EmpiricalMedianBinMapper(bin_edges=edges).fit([2.0, 15.0])

        pmf = np.array([[0.3, 0.7], [0.8, 0.2]])
        dist = mapper.to_continuous_dist(pmf)

        assert isinstance(dist, ContinuousPredictiveDistribution)
        np.testing.assert_array_equal(dist.grid_y, np.array(edges))
        assert dist.grid_cdf.shape == (2, 3)
        np.testing.assert_allclose(dist.grid_cdf[0], [0.0, 0.3, 1.0])
        np.testing.assert_allclose(dist.grid_cdf[1], [0.0, 0.8, 1.0])

    def test_to_continuous_dist_not_fitted(self) -> None:
        """Test calling to_continuous_dist on un-fitted instance raises NotFittedError."""
        mapper = EmpiricalMedianBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(NotFittedError):
            mapper.to_continuous_dist([[0.5, 0.5]])

    def test_to_continuous_dist_invalid_pmf_ndim(self) -> None:
        """Test error handling when PMF is not 2D."""
        mapper = EmpiricalMedianBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="2D array"):
            mapper.to_continuous_dist([0.5, 0.5])

    def test_to_continuous_dist_column_dimension_mismatch(self) -> None:
        """Test error handling when PMF columns mismatch fitted bin count."""
        mapper = EmpiricalMedianBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="PMF column dimension"):
            mapper.to_continuous_dist([[0.3, 0.3, 0.4]])


class TestQuantileBinMapperInit:
    """Tests for QuantileBinMapper initialization."""

    def test_init_stores_parameters(self) -> None:
        """Verify __init__ correctly stores bin_edges and quantiles attributes."""
        edges = [0, 10, 20]
        q_tuple = (0.1, 0.5, 0.9)
        mapper = QuantileBinMapper(bin_edges=edges, quantiles=q_tuple)
        assert mapper.bin_edges == edges
        assert mapper.quantiles == q_tuple


class TestQuantileBinMapperFit:
    """Tests for QuantileBinMapper fit method, sub-grid logic, and failure modes."""

    def test_fit_success_subgrid_construction(self) -> None:
        """Test successful fit and sub-grid point generation across populated bins."""
        edges = [0.0, 10.0, 20.0]
        # Bin 0 ([0, 10)): [1.0, 2.0, 3.0, 4.0, 5.0]
        # Bin 1 ([10, 20)): [11.0, 12.0, 13.0, 14.0, 15.0]
        y_cont = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        quantiles = (0.25, 0.50, 0.75)
        mapper = QuantileBinMapper(bin_edges=edges, quantiles=quantiles)

        fitted_mapper = mapper.fit(y_cont)
        assert fitted_mapper is mapper
        assert mapper.n_bins_ == 2
        np.testing.assert_array_equal(mapper.quantiles_, np.array([0.25, 0.50, 0.75]))

        # Grid points: 1 (start) + 2 bins * (3 quantiles + 1 upper edge) = 9 points
        assert len(mapper.grid_y_) == 9
        assert len(mapper.grid_cdf_weights_) == 9

        # Verify grid weights: [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0]
        expected_weights = np.array([0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0])
        np.testing.assert_allclose(mapper.grid_cdf_weights_, expected_weights)

    def test_fit_success_explicit_y_binned(self) -> None:
        """Test successful fit when explicit y_binned array is provided."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([1.0, 5.0, 15.0])
        y_binned = np.array([0, 0, 1])

        mapper = QuantileBinMapper(bin_edges=edges)
        mapper.fit(y_cont, y_binned=y_binned)
        assert len(mapper.grid_y_) == 9

    def test_fit_empty_bin_fallback(self) -> None:
        """Test fallback linear fraction sub-grid construction for empty bins."""
        edges = [0.0, 10.0, 20.0]
        # Bin 1 ([10, 20)) has zero samples
        y_cont = np.array([2.0, 4.0, 6.0, 8.0])

        mapper = QuantileBinMapper(bin_edges=edges, quantiles=(0.25, 0.50, 0.75))
        mapper.fit(y_cont)

        # Empty bin 1 quantiles should fall back to 10 + 10 * [0.25, 0.5, 0.75]
        expected_bin1_subgrid = np.array([12.5, 15.0, 17.5])
        np.testing.assert_allclose(mapper.grid_y_[5:8], expected_bin1_subgrid)

    @pytest.mark.parametrize("invalid_q", [[0.0, 0.5], [0.5, 1.0], [-0.1], [1.2]])
    def test_fit_invalid_quantiles_range(self, invalid_q: list[float]) -> None:
        """Test that quantiles outside (0.0, 1.0) raise ValueError."""
        mapper = QuantileBinMapper(bin_edges=[0, 10, 20], quantiles=invalid_q)
        with pytest.raises(ValueError, match="strictly within \\(0.0, 1.0\\)"):
            mapper.fit([1.0, 12.0])

    def test_fit_invalid_quantiles_empty_or_ndim(self) -> None:
        """Test that empty or multi-dimensional quantiles raise ValueError."""
        mapper_empty = QuantileBinMapper(bin_edges=[0, 10, 20], quantiles=[])
        with pytest.raises(ValueError, match="non-empty 1D array-like"):
            mapper_empty.fit([1.0, 12.0])

    def test_fit_invalid_y_continuous_ndim(self) -> None:
        """Test that 2D y_continuous raises ValueError."""
        mapper = QuantileBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(ValueError, match="1D array"):
            mapper.fit(np.ones((5, 2)))

    def test_fit_y_binned_shape_mismatch(self) -> None:
        """Test error handling when y_binned shape mismatches y_continuous."""
        mapper = QuantileBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(ValueError, match="Shape mismatch"):
            mapper.fit(y_continuous=[1, 2, 3], y_binned=[0, 1])


class TestQuantileBinMapperTransform:
    """Tests for QuantileBinMapper transform method and validation."""

    def test_transform_success(self) -> None:
        """Test mapping PMF matrix to continuous expected values over sub-grid."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([1.0, 5.0, 9.0, 11.0, 15.0, 19.0])
        mapper = QuantileBinMapper(bin_edges=edges).fit(y_cont)

        pmf = np.array([[0.5, 0.5], [1.0, 0.0]])
        expected = mapper.transform(pmf)

        assert expected.shape == (2,)
        assert isinstance(expected[0], float)

    def test_transform_not_fitted(self) -> None:
        """Test calling transform on un-fitted instance raises NotFittedError."""
        mapper = QuantileBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(NotFittedError):
            mapper.transform([[0.5, 0.5]])

    def test_transform_invalid_pmf_ndim(self) -> None:
        """Test that 1D PMF raises ValueError."""
        mapper = QuantileBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="2D array"):
            mapper.transform([0.5, 0.5])

    def test_transform_column_dimension_mismatch(self) -> None:
        """Test that PMF column count mismatch with n_bins_ raises ValueError."""
        mapper = QuantileBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        invalid_pmf = np.array([[0.3, 0.3, 0.4]])
        with pytest.raises(ValueError, match="PMF column dimension"):
            mapper.transform(invalid_pmf)


class TestQuantileBinMapperToContinuousDist:
    """Tests for QuantileBinMapper to_continuous_dist method."""

    def test_to_continuous_dist_success(self) -> None:
        """Test constructing ContinuousPredictiveDistribution over sub-grid."""
        edges = [0.0, 10.0, 20.0]
        y_cont = np.array([1.0, 5.0, 9.0, 11.0, 15.0, 19.0])
        mapper = QuantileBinMapper(bin_edges=edges).fit(y_cont)

        pmf = np.array([[0.4, 0.6], [0.8, 0.2]])
        dist = mapper.to_continuous_dist(pmf)

        assert isinstance(dist, ContinuousPredictiveDistribution)
        assert dist.grid_cdf.shape == (2, len(mapper.grid_y_))
        np.testing.assert_array_equal(dist.grid_y, mapper.grid_y_)

        # Start of CDF must be 0.0 and end must be 1.0
        np.testing.assert_allclose(dist.grid_cdf[:, 0], [0.0, 0.0])
        np.testing.assert_allclose(dist.grid_cdf[:, -1], [1.0, 1.0])

    def test_to_continuous_dist_not_fitted(self) -> None:
        """Test calling to_continuous_dist on un-fitted instance raises NotFittedError."""
        mapper = QuantileBinMapper(bin_edges=[0, 10, 20])
        with pytest.raises(NotFittedError):
            mapper.to_continuous_dist([[0.5, 0.5]])

    def test_to_continuous_dist_invalid_pmf_ndim(self) -> None:
        """Test error handling when PMF is not 2D."""
        mapper = QuantileBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="2D array"):
            mapper.to_continuous_dist([0.5, 0.5])

    def test_to_continuous_dist_column_dimension_mismatch(self) -> None:
        """Test error handling when PMF columns mismatch fitted bin count."""
        mapper = QuantileBinMapper(bin_edges=[0, 10, 20]).fit([2, 15])
        with pytest.raises(ValueError, match="PMF column dimension"):
            mapper.to_continuous_dist([[0.3, 0.3, 0.4]])
