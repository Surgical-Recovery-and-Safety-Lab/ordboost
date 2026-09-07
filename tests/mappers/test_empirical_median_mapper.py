"""Unit tests for EmpiricalMedianBinMapper in ordboost.mappers."""

import numpy as np
import pytest

from ordboost.mappers import EmpiricalMedianBinMapper


class TestIntraBinPoints:
    """Tests for EmpiricalMedianBinMapper._intra_bin_points."""

    def test_returns_empirical_median_not_geometric_midpoint(self) -> None:
        """Test that the returned point is the median of bin_data, not
        (low + high) / 2, for a skewed bin."""
        mapper = EmpiricalMedianBinMapper()
        bin_data = np.array([10.5, 11.0, 12.0, 13.0, 19.0])  # median = 12.0
        points, _ = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=0)
        assert points[0] == pytest.approx(12.0)
        assert points[0] != pytest.approx(15.0)  # would be the midpoint

    def test_odd_length_weight_reflects_median_as_data_point(self) -> None:
        """Test that for an odd-length bin, the weight exceeds 0.5, since
        the median is itself an observed value and is included in the
        at-or-below count -- not a bug, but an inherent property of
        odd-count empirical medians under an inclusive comparison."""
        mapper = EmpiricalMedianBinMapper()
        bin_data = np.array([10.5, 11.0, 12.0, 13.0, 19.0])  # median = 12.0
        # 3 of 5 values (10.5, 11.0, 12.0) are <= 12.0
        _, weights = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=0)
        assert weights[0] == pytest.approx(0.6)

    def test_even_length_distinct_values_gives_weight_of_half(self) -> None:
        """Test that for an even-length bin with no value equal to the
        computed median, the weight lands exactly at 0.5, since the
        median itself is not an observed data point to bias the count."""
        mapper = EmpiricalMedianBinMapper()
        bin_data = np.array([11.0, 13.0, 17.0, 19.0])  # median = 15.0, not in data
        points, weights = mapper._intra_bin_points(
            bin_data,
            low=10.0,
            high=20.0,
            k=0,
        )
        assert points[0] == pytest.approx(15.0)
        assert weights[0] == pytest.approx(0.5)

    def test_weight_offset_by_bin_index(self) -> None:
        """Test that the weight is offset into bin-index units (k + fraction)."""
        mapper = EmpiricalMedianBinMapper()
        bin_data = np.array([11.0, 13.0, 17.0, 19.0])
        _, weights = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=3)
        assert weights[0] == pytest.approx(3.5)

    def test_empty_bin_data_falls_back_to_midpoint(self) -> None:
        """Test that an empty bin falls back to the geometric midpoint
        with weight 0.5, since there is no training data to compute an
        empirical median or fraction from."""
        mapper = EmpiricalMedianBinMapper()
        points, weights = mapper._intra_bin_points(
            np.array([]), low=10.0, high=20.0, k=2
        )
        assert points[0] == pytest.approx(15.0)
        assert weights[0] == pytest.approx(2.5)

    def test_single_value_at_boundary(self) -> None:
        """Test the edge case where all of a bin's data sits exactly at
        its own lower boundary: the median equals the boundary, and the
        weight is 1.0 since all data is at or below it."""
        mapper = EmpiricalMedianBinMapper()
        bin_data = np.array([10.0])
        points, weights = mapper._intra_bin_points(
            bin_data,
            low=10.0,
            high=20.0,
            k=0,
        )
        assert points[0] == pytest.approx(10.0)
        assert weights[0] == pytest.approx(1.0)

    def test_median_is_clipped_to_bin_range(self) -> None:
        """Test that the computed median is clipped to [low, high]."""
        mapper = EmpiricalMedianBinMapper()
        bin_data = np.array([10.0, 10.0, 10.0])
        points, _ = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=0)
        assert points[0] >= 10.0
        assert points[0] <= 20.0

    def test_output_shapes(self) -> None:
        """Test that both returned arrays have exactly one element."""
        mapper = EmpiricalMedianBinMapper()
        points, weights = mapper._intra_bin_points(
            np.array([11.0, 12.0, 19.0]), low=10.0, high=20.0, k=0
        )
        assert points.shape == (1,)
        assert weights.shape == (1,)

    def test_only_returns_interior_point_not_boundaries(self) -> None:
        """Test that the returned points do not include low or high."""
        mapper = EmpiricalMedianBinMapper()
        points, _ = mapper._intra_bin_points(
            np.array([11.0, 12.0, 19.0]), low=10.0, high=20.0, k=0
        )
        assert 10.0 not in points
        assert 20.0 not in points


class TestFitIntegration:
    """Integration tests for EmpiricalMedianBinMapper.fit via the shared
    BaseBinMapper grid-construction pipeline.
    """

    def test_fitted_grid_contains_empirical_median_point(self) -> None:
        """Test that after fit, grid_y_ contains the bin's empirical
        median with the correct empirical weight."""
        mapper = EmpiricalMedianBinMapper(bin_edges=[20.0])
        y_cont = np.array([10.5, 11.0, 12.0, 13.0, 19.0])  # median = 12.0
        mapper.fit(y_cont)

        median_idx = np.searchsorted(mapper.grid_y_, 12.0)
        assert mapper.grid_y_[median_idx] == pytest.approx(12.0)
        assert mapper.grid_cdf_weights_[median_idx] == pytest.approx(0.6)


class TestTransformIntegration:
    """Integration tests confirming transform()'s median override behaves
    correctly and differs meaningfully from the inherited mean.
    """

    def test_transform_matches_to_continuous_dist_median(self) -> None:
        """Test that transform's output equals
        to_continuous_dist(pmf).median(), not .mean()."""
        mapper = EmpiricalMedianBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit(np.array([1.0, 1.0, 1.0, 15.0, 15.0, 19.0]))
        pmf = np.array([[0.2, 0.3, 0.4, 0.1]])
        dist = mapper.to_continuous_dist(pmf)
        np.testing.assert_allclose(mapper.transform(pmf), dist.median())

    def test_transform_differs_from_mean_for_skewed_distribution(self) -> None:
        """Test that the median override actually changes the result
        relative to the inherited mean, for a distribution skewed enough
        that mean and median genuinely disagree. Guards against a
        regression where transform() silently falls back to the base
        class's .mean() implementation."""
        mapper = EmpiricalMedianBinMapper(bin_edges=[0.0, 10.0, 100.0])
        mapper.fit(np.array([1.0, 2.0, 3.0, 95.0, 98.0]))
        pmf = np.array(
            [[0.05, 0.01, 0.04, 0.9]]
        )  # most mass in the wide, skewed upper bin
        dist = mapper.to_continuous_dist(pmf)
        median_result = mapper.transform(pmf)
        mean_result = dist.mean()
        assert not np.isclose(median_result, mean_result)
        np.testing.assert_allclose(median_result, dist.median())
