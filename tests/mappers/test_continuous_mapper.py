"""Unit tests for ContinuousBinMapper in ordboost.mappers."""

import numpy as np
import pytest

from ordboost.mappers import ContinuousBinMapper


class TestValidateIntraBinParams:
    """Tests for ContinuousBinMapper._validate_intra_bin_params."""

    def test_sets_resolution_and_resets_counter(self) -> None:
        """Test that resolution_ is set and the point counter starts at 0."""
        mapper = ContinuousBinMapper(resolution=2.0)
        mapper._validate_intra_bin_params()
        assert mapper.resolution_ == 2.0
        assert mapper._n_generated_points == 0

    def test_zero_resolution_raises(self) -> None:
        """Test that resolution=0.0 raises ValueError."""
        mapper = ContinuousBinMapper(resolution=0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            mapper._validate_intra_bin_params()

    def test_negative_resolution_raises(self) -> None:
        """Test that a negative resolution raises ValueError."""
        mapper = ContinuousBinMapper(resolution=-1.0)
        with pytest.raises(ValueError, match="strictly positive"):
            mapper._validate_intra_bin_params()

    def test_non_positive_max_grid_points_raises(self) -> None:
        """Test that max_grid_points=0 raises ValueError."""
        mapper = ContinuousBinMapper(max_grid_points=0)
        with pytest.raises(ValueError, match="positive integer"):
            mapper._validate_intra_bin_params()


class TestIntraBinPoints:
    """Tests for ContinuousBinMapper._intra_bin_points."""

    def test_integer_resolution_enumerates_every_integer(self) -> None:
        """Test that resolution=1.0 returns every integer strictly
        between low and high."""
        mapper = ContinuousBinMapper(resolution=1.0)
        mapper._validate_intra_bin_params()
        points, _ = mapper._intra_bin_points(np.array([]), low=10.0, high=15.0, k=0)
        np.testing.assert_allclose(points, [11.0, 12.0, 13.0, 14.0])

    def test_points_exclude_boundaries(self) -> None:
        """Test that returned points never include low or high, since
        BaseBinMapper._build_grid adds those separately."""
        mapper = ContinuousBinMapper(resolution=1.0)
        mapper._validate_intra_bin_params()
        points, _ = mapper._intra_bin_points(np.array([]), low=10.0, high=15.0, k=0)
        assert 10.0 not in points
        assert 15.0 not in points

    def test_bin_narrower_than_resolution_returns_empty(self) -> None:
        """Test that a bin narrower than resolution_ produces no interior
        points."""
        mapper = ContinuousBinMapper(resolution=5.0)
        mapper._validate_intra_bin_params()
        points, weights = mapper._intra_bin_points(
            np.array([]), low=10.0, high=12.0, k=0
        )
        assert points.shape == (0,)
        assert weights.shape == (0,)

    def test_density_weighted_true_uses_empirical_fraction(self) -> None:
        """Test that density_weighted=True computes each point's weight
        as the empirical fraction of bin_data at or below it."""
        mapper = ContinuousBinMapper(resolution=1.0, density_weighted=True)
        mapper._validate_intra_bin_params()
        bin_data = np.array([10.5, 10.5, 12.5])  # skewed toward the low end
        _, weights = mapper._intra_bin_points(bin_data, low=10.0, high=13.0, k=0)
        # points = [11, 12]; frac(<=11) = 2/3, frac(<=12) = 2/3
        np.testing.assert_allclose(weights, [2.0 / 3.0, 2.0 / 3.0])

    def test_density_weighted_false_uses_uniform_interpolation(self) -> None:
        """Test that density_weighted=False ignores bin_data entirely and
        uses uniform linear interpolation across the bin width."""
        mapper = ContinuousBinMapper(resolution=2.5, density_weighted=False)
        mapper._validate_intra_bin_params()
        bin_data = np.array([10.5, 10.5, 12.5])  # should be ignored
        points, weights = mapper._intra_bin_points(bin_data, low=10.0, high=20.0, k=0)
        expected_weights = (points - 10.0) / 10.0
        np.testing.assert_allclose(weights, expected_weights)

    def test_empty_bin_data_falls_back_to_uniform_even_when_density_weighted(
        self,
    ) -> None:
        """Test that an empty bin falls back to uniform interpolation
        regardless of density_weighted, since there is no data to
        compute an empirical fraction from."""
        mapper = ContinuousBinMapper(resolution=2.5, density_weighted=True)
        mapper._validate_intra_bin_params()
        points, weights = mapper._intra_bin_points(
            np.array([]), low=10.0, high=20.0, k=0
        )
        expected_weights = (points - 10.0) / 10.0
        np.testing.assert_allclose(weights, expected_weights)

    def test_weights_offset_by_bin_index(self) -> None:
        """Test that weights are offset into bin-index units (k + fraction)."""
        mapper = ContinuousBinMapper(resolution=5.0, density_weighted=False)
        mapper._validate_intra_bin_params()
        _, weights = mapper._intra_bin_points(np.array([]), low=10.0, high=20.0, k=3)
        assert np.all(weights >= 3.0)
        assert np.all(weights <= 4.0)

    def test_exceeding_max_grid_points_raises(self) -> None:
        """Test that a resolution generating more points than
        max_grid_points raises ValueError."""
        mapper = ContinuousBinMapper(resolution=0.001, max_grid_points=10)
        mapper._validate_intra_bin_params()
        with pytest.raises(ValueError, match="max_grid_points"):
            mapper._intra_bin_points(np.array([]), low=0.0, high=1.0, k=0)

    def test_max_grid_points_accumulates_across_bins(self) -> None:
        """Test that the point limit is enforced cumulatively across
        multiple calls (i.e. multiple bins), not reset per bin."""
        mapper = ContinuousBinMapper(resolution=1.0, max_grid_points=5)
        mapper._validate_intra_bin_params()
        mapper._intra_bin_points(np.array([]), low=0.0, high=4.0, k=0)  # 3 points, ok
        with pytest.raises(ValueError, match="max_grid_points"):
            mapper._intra_bin_points(
                np.array([]), low=10.0, high=14.0, k=1
            )  # would push to 6


class TestFitIntegration:
    """Integration tests via the shared BaseBinMapper grid-construction
    pipeline.
    """

    def test_fitted_grid_contains_every_integer_in_range(self) -> None:
        """Test that after fit with resolution=1.0, grid_y_ contains
        every achievable integer value, matching the DAOH use case."""
        mapper = ContinuousBinMapper(
            bin_edges=[0.0, 5.0], resolution=1.0, bounded_below=False
        )
        mapper.fit(np.array([1.0, 4.0]))
        for val in range(0, 5):
            assert np.any(np.isclose(mapper.grid_y_, float(val)))

    def test_nominal_edges_included_when_unbounded(self) -> None:
        """Test that with bounded_below=False and bounded_above=False, the
        nominal bin_edges appear in the grid even when observed data doesn't
        reach them."""
        mapper = ContinuousBinMapper(
            bin_edges=[0.0, 5.0],
            resolution=1.0,
            bounded_below=False,
            bounded_above=False,
        )
        mapper.fit(np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.any(np.isclose(mapper.grid_y_, 0.0))
        assert np.any(np.isclose(mapper.grid_y_, 5.0))

    def test_fine_resolution_raises_before_fit_completes(self) -> None:
        """Test that fit itself surfaces the max_grid_points error, not
        just direct calls to _intra_bin_points."""
        mapper = ContinuousBinMapper(
            bin_edges=[0.0, 100.0], resolution=0.0001, max_grid_points=100
        )
        with pytest.raises(ValueError, match="max_grid_points"):
            mapper.fit(np.array([1.0, 50.0]))


class TestTransformIntegration:
    """Integration test confirming transform() (inherited, unmodified) is
    consistent with the maximally fine grid.
    """

    def test_transform_matches_to_continuous_dist_mean(self) -> None:
        """Test that transform's output equals to_continuous_dist(pmf).mean()."""
        mapper = ContinuousBinMapper(bin_edges=[0.0, 10.0, 20.0], resolution=2.0)
        mapper.fit(np.array([2.0, 8.0, 12.0, 18.0]))
        pmf = np.array([[0.6, 0.4]])
        dist = mapper.to_continuous_dist(pmf)
        np.testing.assert_allclose(mapper.transform(pmf), dist.mean())
