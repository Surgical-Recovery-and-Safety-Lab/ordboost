"""Unit tests for QuantileBinMapper in ordboost.mappers."""

import numpy as np
import pytest

from ordboost.mappers import QuantileBinMapper


class TestValidateIntraBinParams:
    """Tests for QuantileBinMapper._validate_intra_bin_params."""

    def test_sets_quantiles_sorted(self) -> None:
        """Test that quantiles_ is set as a sorted array, even when the
        input is given out of order."""
        mapper = QuantileBinMapper(quantiles=(0.75, 0.25, 0.5))
        mapper._validate_intra_bin_params()
        np.testing.assert_array_equal(mapper.quantiles_, np.array([0.25, 0.5, 0.75]))

    def test_default_quantiles(self) -> None:
        """Test that the default quantile levels are (0.25, 0.5, 0.75)."""
        mapper = QuantileBinMapper()
        mapper._validate_intra_bin_params()
        np.testing.assert_array_equal(mapper.quantiles_, np.array([0.25, 0.5, 0.75]))

    def test_empty_quantiles_raises(self) -> None:
        """Test that an empty quantiles sequence raises ValueError."""
        mapper = QuantileBinMapper(quantiles=())
        with pytest.raises(ValueError, match="non-empty 1D array-like"):
            mapper._validate_intra_bin_params()

    def test_quantile_equal_to_zero_raises(self) -> None:
        """Test that a quantile level of exactly 0.0 raises ValueError."""
        mapper = QuantileBinMapper(quantiles=(0.0, 0.5))
        with pytest.raises(ValueError, match="strictly within"):
            mapper._validate_intra_bin_params()

    def test_quantile_equal_to_one_raises(self) -> None:
        """Test that a quantile level of exactly 1.0 raises ValueError."""
        mapper = QuantileBinMapper(quantiles=(0.5, 1.0))
        with pytest.raises(ValueError, match="strictly within"):
            mapper._validate_intra_bin_params()

    def test_quantile_outside_unit_interval_raises(self) -> None:
        """Test that a quantile level outside [0, 1] entirely raises ValueError."""
        mapper = QuantileBinMapper(quantiles=(0.5, 1.5))
        with pytest.raises(ValueError, match="strictly within"):
            mapper._validate_intra_bin_params()


class TestIntraBinPoints:
    """Tests for QuantileBinMapper._intra_bin_points."""

    def test_points_match_numpy_quantile(self) -> None:
        """Test that returned points equal np.quantile(bin_data, quantiles_)."""
        mapper = QuantileBinMapper(quantiles=(0.25, 0.5, 0.75))
        mapper._validate_intra_bin_params()
        bin_data = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        points, _ = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=0)
        expected = np.quantile(bin_data, [0.25, 0.5, 0.75])
        np.testing.assert_allclose(points, expected)

    def test_weights_equal_quantile_levels_offset_by_bin_index(self) -> None:
        """Test that weights are exactly k + quantile_level, not empirically
        derived from the data."""
        mapper = QuantileBinMapper(quantiles=(0.25, 0.5, 0.75))
        mapper._validate_intra_bin_params()
        bin_data = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        _, weights = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=3)
        np.testing.assert_allclose(weights, np.array([3.25, 3.5, 3.75]))

    def test_empty_bin_interpolates_linearly(self) -> None:
        """Test that an empty bin falls back to linear interpolation
        between low and high at each quantile level."""
        mapper = QuantileBinMapper(quantiles=(0.25, 0.5, 0.75))
        mapper._validate_intra_bin_params()
        points, weights = mapper._intra_bin_points(
            np.array([]), low=10.0, high=20.0, k=0
        )
        np.testing.assert_allclose(points, np.array([12.5, 15.0, 17.5]))
        np.testing.assert_allclose(weights, np.array([0.25, 0.5, 0.75]))

    def test_points_clipped_to_bin_range(self) -> None:
        """Test that computed quantile points are clipped to [low, high]."""
        mapper = QuantileBinMapper(quantiles=(0.1, 0.9))
        mapper._validate_intra_bin_params()
        bin_data = np.array([10.0, 10.0, 10.0, 10.0])  # degenerate, all at floor
        points, _ = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=0)
        assert np.all(points >= 10.0)
        assert np.all(points <= 20.0)

    def test_output_length_matches_n_quantiles(self) -> None:
        """Test that the number of returned points/weights matches the
        number of fitted quantile levels."""
        mapper = QuantileBinMapper(quantiles=(0.1, 0.25, 0.5, 0.75, 0.9))
        mapper._validate_intra_bin_params()
        points, weights = mapper._intra_bin_points(
            np.array([11.0, 15.0, 19.0]), low=10.0, high=20.0, k=0
        )
        assert points.shape == (5,)
        assert weights.shape == (5,)

    def test_only_returns_interior_points_not_boundaries(self) -> None:
        """Test that the returned points do not include low or high."""
        mapper = QuantileBinMapper(quantiles=(0.25, 0.5, 0.75))
        mapper._validate_intra_bin_params()
        points, _ = mapper._intra_bin_points(
            np.array([11.0, 15.0, 19.0]), low=10.0, high=20.0, k=0
        )
        assert 10.0 not in points
        assert 20.0 not in points


class TestFitIntegration:
    """Integration tests for QuantileBinMapper.fit via the shared
    BaseBinMapper grid-construction pipeline.
    """

    def test_validates_quantiles_before_building_grid(self) -> None:
        """Test that fit raises for invalid quantiles before attempting
        to build the grid, rather than failing later or silently."""
        mapper = QuantileBinMapper(bin_edges=[0.0, 10.0], quantiles=(0.0, 0.5))
        with pytest.raises(ValueError, match="strictly within"):
            mapper.fit(np.array([1.0, 5.0]))

    def test_fitted_grid_contains_all_quantile_points(self) -> None:
        """Test that after fit, grid_y_ contains a point for every fitted
        quantile level, correctly positioned by empirical value."""
        mapper = QuantileBinMapper(bin_edges=[10.0, 20.0], quantiles=(0.25, 0.5, 0.75))
        bin_data = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        mapper.fit(bin_data)

        expected_points = np.quantile(bin_data, [0.25, 0.5, 0.75])
        for expected in expected_points:
            assert np.any(np.isclose(mapper.grid_y_, expected))

    def test_collapsing_quantiles_deduplicate_to_max_weight(self) -> None:
        """Test that when a bin's quantile points collapse to the same value
        as an adjacent bin's shared boundary, deduplication retains the
        largest colliding weight. Uses three bins so the narrow bin under
        test is neither the first nor last bin, avoiding degenerate
        zero-width boundary anchoring.
        """
        mapper = QuantileBinMapper(
            bin_edges=[0.0, 10.0, 11.0, 20.0], quantiles=(0.25, 0.5, 0.75)
        )
        # bin 1 spans [10, 11) with all its data at the shared edge y=10.0
        mapper.fit(np.array([1.0, 10.0, 10.0, 10.0, 10.0, 15.0]))

        # Colliding weights at y=10.0: bin 0's boundary (1.0) and bin 1's
        # three quantile points (1.25, 1.5, 1.75) -- max should win.
        idx = np.searchsorted(mapper.grid_y_, 10.0)
        assert mapper.grid_y_[idx] == pytest.approx(10.0)
        assert mapper.grid_cdf_weights_[idx] == pytest.approx(1.75)

    def test_zero_width_bin_from_degenerate_data_does_not_raise(self) -> None:
        """Test that a single-bin mapper whose entire training data is one
        repeated value produces a zero-width (low == high) bin without
        raising, collapsing correctly to weight 1.0 at that point.
        """
        mapper = QuantileBinMapper(bin_edges=[10.0, 11.0], quantiles=(0.25, 0.5, 0.75))
        mapper.fit(np.array([10.0, 10.0, 10.0, 10.0]))

        idx = np.searchsorted(mapper.grid_y_, 10.0)
        assert mapper.grid_y_[idx] == pytest.approx(10.0)
        assert mapper.grid_cdf_weights_[idx] == pytest.approx(1.0)


class TestTransformIntegration:
    """Integration test confirming transform() (inherited from
    BaseBinMapper, unmodified) is consistent with the quantile-derived grid.
    """

    def test_transform_matches_to_continuous_dist_mean(self) -> None:
        """Test that transform's output equals to_continuous_dist(pmf).mean(),
        confirming QuantileBinMapper does not override the inherited
        default (unlike EmpiricalMedianBinMapper)."""
        mapper = QuantileBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit(np.array([1.0, 4.0, 6.0, 15.0, 15.0, 19.0]))
        pmf = np.array([[0.7, 0.3], [0.2, 0.8]])
        dist = mapper.to_continuous_dist(pmf)
        np.testing.assert_allclose(mapper.transform(pmf), dist.mean())
