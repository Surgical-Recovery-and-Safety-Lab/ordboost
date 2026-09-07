"""Unit tests for EmpiricalMeanBinMapper in ordboost.mappers."""

import numpy as np
import pytest

from ordboost.mappers import EmpiricalMeanBinMapper


class TestIntraBinPoints:
    """Tests for EmpiricalMeanBinMapper._intra_bin_points."""

    def test_returns_empirical_mean_not_geometric_midpoint(self) -> None:
        """Test that the returned point is the mean of bin_data, not
        (low + high) / 2, for a skewed bin."""
        mapper = EmpiricalMeanBinMapper()
        bin_data = np.array([11.0, 12.0, 12.0, 19.0])  # mean = 13.5
        points, weights = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=0)
        assert points[0] == pytest.approx(13.5)
        assert points[0] != pytest.approx(15.0)  # would be the midpoint

    def test_weight_is_empirical_fraction_at_or_below_mean(self) -> None:
        """Test that the weight equals the fraction of bin_data at or
        below the mean, not a fixed 0.5, for skewed data."""
        mapper = EmpiricalMeanBinMapper()
        bin_data = np.array([11.0, 12.0, 12.0, 19.0])  # mean = 13.5
        # 3 of 4 values (11, 12, 12) are <= 13.5
        _, weights = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=0)
        assert weights[0] == pytest.approx(0.75)

    def test_weight_offset_by_bin_index(self) -> None:
        """Test that the weight is offset into bin-index units (k + fraction)."""
        mapper = EmpiricalMeanBinMapper()
        bin_data = np.array([11.0, 12.0, 12.0, 19.0])
        _, weights = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=3)
        assert weights[0] == pytest.approx(3.75)

    def test_symmetric_data_gives_weight_near_half(self) -> None:
        """Test that symmetric bin data yields a mean and weight close to
        the geometric-midpoint / 0.5 case, as a sanity check that the
        empirical calculation converges to the naive case when justified."""
        mapper = EmpiricalMeanBinMapper()
        bin_data = np.array([11.0, 13.0, 17.0, 19.0])  # symmetric, mean = 15
        points, weights = mapper._intra_bin_points(
            bin_data,
            low=10.0,
            high=20.0,
            k=0,
        )
        assert points[0] == pytest.approx(15.0)
        assert weights[0] == pytest.approx(0.5)

    def test_empty_bin_data_falls_back_to_midpoint(self) -> None:
        """Test that an empty bin falls back to the geometric midpoint
        with weight 0.5, since there is no training data to compute an
        empirical mean or fraction from."""
        mapper = EmpiricalMeanBinMapper()
        points, weights = mapper._intra_bin_points(
            np.array([]), low=10.0, high=20.0, k=2
        )
        assert points[0] == pytest.approx(15.0)
        assert weights[0] == pytest.approx(2.5)

    def test_single_value_at_boundary(self) -> None:
        """Test the edge case where all of a bin's data sits exactly at
        its own lower boundary: the mean equals the boundary, and the
        weight is 1.0 since all data is at or below it."""
        mapper = EmpiricalMeanBinMapper()
        bin_data = np.array([10.0])
        points, weights = mapper._intra_bin_points(
            bin_data,
            low=10.0,
            high=20.0,
            k=0,
        )
        assert points[0] == pytest.approx(10.0)
        assert weights[0] == pytest.approx(1.0)

    def test_mean_is_clipped_to_bin_range(self) -> None:
        """Test that the computed mean is clipped to [low, high], guarding
        against floating-point drift or anchored boundaries that could
        otherwise place the mean fractionally outside its own bin."""
        mapper = EmpiricalMeanBinMapper()
        bin_data = np.array([10.0, 10.0, 10.0])
        points, _ = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=0)
        assert points[0] >= 10.0
        assert points[0] <= 20.0

    def test_output_shapes(self) -> None:
        """Test that both returned arrays have exactly one element,
        regardless of bin_data size."""
        mapper = EmpiricalMeanBinMapper()
        points, weights = mapper._intra_bin_points(
            np.array([11.0, 12.0, 19.0]), low=10.0, high=20.0, k=0
        )
        assert points.shape == (1,)
        assert weights.shape == (1,)

    def test_only_returns_interior_point_not_boundaries(self) -> None:
        """Test that the returned points do not include low or high,
        since BaseBinMapper._build_grid appends boundary points itself."""
        mapper = EmpiricalMeanBinMapper()
        points, _ = mapper._intra_bin_points(
            np.array([11.0, 12.0, 19.0]), low=10.0, high=20.0, k=0
        )
        assert 10.0 not in points
        assert 20.0 not in points


class TestFitIntegration:
    """Integration tests for EmpiricalMeanBinMapper.fit via the shared
    BaseBinMapper grid-construction pipeline.
    """

    def test_fitted_grid_contains_empirical_mean_point(self) -> None:
        """Test that after fit, grid_y_ contains the bin's empirical mean
        (not the geometric midpoint) with the correct empirical weight."""
        mapper = EmpiricalMeanBinMapper(bin_edges=[20.0])
        y_cont = np.array([11.0, 12.0, 12.0, 19.0])  # mean = 13.5
        mapper.fit(y_cont)

        mean_idx = np.searchsorted(mapper.grid_y_, 13.5)
        assert mapper.grid_y_[mean_idx] == pytest.approx(13.5)
        assert mapper.grid_cdf_weights_[mean_idx] == pytest.approx(0.75)

    def test_multiple_bins_each_get_own_empirical_mean(self) -> None:
        """Test that each bin's interior point reflects that bin's own
        empirical mean, independent of other bins' data."""
        mapper = EmpiricalMeanBinMapper(bin_edges=[0.0, 10.0, 20.0])
        y_cont = np.array(
            [2.0, 4.0, 6.0, 15.0, 15.0, 19.0]
        )  # bin0 mean=4, bin1 mean~16.33
        mapper.fit(y_cont)

        bin0_mean = np.mean([2.0, 4.0, 6.0])
        bin1_mean = np.mean([15.0, 15.0, 19.0])
        assert np.any(np.isclose(mapper.grid_y_, bin0_mean))
        assert np.any(np.isclose(mapper.grid_y_, bin1_mean))


class TestTransformIntegration:
    """Integration test confirming transform() (inherited from
    BaseBinMapper) is consistent with the mean-derived grid.
    """

    def test_transform_matches_to_continuous_dist_mean(self) -> None:
        """Test that transform's output equals to_continuous_dist(pmf).mean(),
        confirming the mapper does not override the inherited default."""
        mapper = EmpiricalMeanBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit(np.array([2.0, 4.0, 6.0, 15.0, 15.0, 19.0]))
        pmf = np.array([[0.2, 0.3, 0.4, 0.1], [0.1, 0.7, 0.1, 0.1]])
        dist = mapper.to_continuous_dist(pmf)
        np.testing.assert_allclose(mapper.transform(pmf), dist.mean())
