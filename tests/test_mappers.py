"""Unit tests for BaseBinMapper in ordboost.debinning."""

from typing import Any

import numpy as np
import pytest

from ordboost.distributions import ContinuousPredictiveDistribution
from ordboost.mappers import BaseBinMapper


class DummyBinMapper(BaseBinMapper):
    """Minimal concrete implementation of BaseBinMapper for unit testing."""

    def fit(self, y_continuous: Any, y_binned: Any = None) -> "DummyBinMapper":
        """Dummy fit method."""
        self.bin_edges_ = self._validate_edges()
        self.n_bins_ = len(self.bin_edges_) - 1
        return self

    def transform(self, pmf: Any) -> np.ndarray:
        """Dummy transform method."""
        pmf_arr = np.asarray(pmf, dtype=float)
        midpoints = (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2.0
        return np.dot(pmf_arr, midpoints)

    def to_continuous_dist(self, pmf: Any) -> ContinuousPredictiveDistribution:
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
