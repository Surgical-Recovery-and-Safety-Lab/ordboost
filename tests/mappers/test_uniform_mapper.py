"""Unit tests for UniformBinMapper in ordboost.mappers."""

import numpy as np
import pytest

from ordboost.mappers import UniformBinMapper


class TestInit:
    """Tests for UniformBinMapper.__init__ parameter storage and threading."""

    def test_default_n_points(self) -> None:
        """Test that n_points defaults to 1."""
        mapper = UniformBinMapper()
        assert mapper.n_points == 1

    def test_base_parameters_threaded_through_super(self) -> None:
        """Test that base-class parameters are correctly passed to
        BaseBinMapper.__init__ rather than silently dropped."""
        mapper = UniformBinMapper(
            bin_edges=[0.0, 10.0],
            n_points=3,
            lower_bound=None,
            ceiling_atom=True,
            upper_bound=10.0,
            boundary_epsilon=0.01,
        )
        assert mapper.bin_edges == [0.0, 10.0]
        assert mapper.n_points == 3
        assert mapper.lower_bound is None
        assert mapper.upper_bound is 10.0
        assert mapper.ceiling_atom is True
        assert mapper.boundary_epsilon == 0.01


class TestValidateIntraBinParams:
    """Tests for UniformBinMapper._validate_intra_bin_params."""

    def test_sets_n_points(self) -> None:
        """Test that a valid n_points is copied to n_points_."""
        mapper = UniformBinMapper(n_points=3)
        mapper._validate_intra_bin_params()
        assert mapper.n_points_ == 3

    def test_zero_is_valid(self) -> None:
        """Test that n_points=0 is accepted, for null-mapper behaviour."""
        mapper = UniformBinMapper(n_points=0)
        mapper._validate_intra_bin_params()
        assert mapper.n_points_ == 0

    def test_negative_n_points_raises(self) -> None:
        """Test that a negative n_points raises ValueError."""
        mapper = UniformBinMapper(n_points=-1)
        with pytest.raises(ValueError, match="non-negative"):
            mapper._validate_intra_bin_params()

    def test_float_n_points_raises_type_error(self) -> None:
        """Test that a non-integer n_points raises TypeError."""
        mapper = UniformBinMapper(n_points=1.5)
        with pytest.raises(TypeError, match="Expected 'n_points' to be an int"):
            mapper._validate_intra_bin_params()

    def test_bool_n_points_raises_type_error(self) -> None:
        """Test that a bool n_points is rejected, even though bool is a
        subclass of int in Python, since it is not a meaningful point
        count and is most likely a user error."""
        mapper = UniformBinMapper(n_points=True)
        with pytest.raises(TypeError, match="Expected 'n_points' to be an int"):
            mapper._validate_intra_bin_params()


class TestIntraBinPoints:
    """Tests for UniformBinMapper._intra_bin_points."""

    def test_zero_points_returns_empty_arrays(self) -> None:
        """Test that n_points_=0 returns no interior points, matching
        the null-mapper baseline."""
        mapper = UniformBinMapper(n_points=0)
        mapper._validate_intra_bin_params()
        points, weights = mapper._intra_bin_points(
            np.array([1.0]), low=10.0, high=20.0, k=0
        )
        assert points.shape == (0,)
        assert weights.shape == (0,)

    def test_one_point_at_midpoint_with_weight_half(self) -> None:
        """Test that n_points_=1 places a single point at the geometric
        midpoint with weight k + 0.5."""
        mapper = UniformBinMapper(n_points=1)
        mapper._validate_intra_bin_params()
        points, weights = mapper._intra_bin_points(
            np.array([]), low=10.0, high=20.0, k=0
        )
        np.testing.assert_allclose(points, [15.0])
        np.testing.assert_allclose(weights, [0.5])

    def test_three_points_evenly_spaced_with_quarter_weights(self) -> None:
        """Test that n_points_=3 places points at 1/4, 2/4, 3/4 of the
        bin width, with matching quarter-fraction weights."""
        mapper = UniformBinMapper(n_points=3)
        mapper._validate_intra_bin_params()
        points, weights = mapper._intra_bin_points(
            np.array([]), low=10.0, high=20.0, k=0
        )
        np.testing.assert_allclose(points, [12.5, 15.0, 17.5])
        np.testing.assert_allclose(weights, [0.25, 0.5, 0.75])

    def test_weights_offset_by_bin_index(self) -> None:
        """Test that weights are offset into bin-index units (k + fraction)."""
        mapper = UniformBinMapper(n_points=3)
        mapper._validate_intra_bin_params()
        _, weights = mapper._intra_bin_points(np.array([]), low=10.0, high=20.0, k=4)
        np.testing.assert_allclose(weights, [4.25, 4.5, 4.75])

    def test_independent_of_bin_data(self) -> None:
        """Test that point placement is unaffected by bin_data, unlike
        the empirical mean/median/quantile mappers."""
        mapper = UniformBinMapper(n_points=3)
        mapper._validate_intra_bin_params()
        points_empty, _ = mapper._intra_bin_points(
            np.array([]), low=10.0, high=20.0, k=0
        )
        points_skewed, _ = mapper._intra_bin_points(
            np.array([10.1, 10.2, 19.9]), low=10.0, high=20.0, k=0
        )
        np.testing.assert_allclose(points_empty, points_skewed)

    def test_points_strictly_interior(self) -> None:
        """Test that returned points never equal low or high, for any
        n_points_ >= 1, since fractions are strictly within (0, 1)."""
        mapper = UniformBinMapper(n_points=5)
        mapper._validate_intra_bin_params()
        points, _ = mapper._intra_bin_points(np.array([]), low=10.0, high=20.0, k=0)
        assert np.all(points > 10.0)
        assert np.all(points < 20.0)


class TestFitIntegration:
    """Integration tests for UniformBinMapper.fit via the shared
    BaseBinMapper grid-construction pipeline.
    """

    def test_null_mapper_grid_is_edges_only(self) -> None:
        """Test that n_points=0 produces a grid with no interior
        refinement beyond the boundary/anchor points BaseBinMapper adds
        for every mapper."""
        mapper = UniformBinMapper(bin_edges=[0.0, 10.0, 20.0], n_points=0)
        mapper.fit(np.array([1.0, 5.0, 15.0, 19.0]))
        # Only floor anchor, bin edges (0, 10, 20) -- no interior points
        assert len(mapper.grid_y_) == 4

    def test_fitted_grid_contains_evenly_spaced_points(self) -> None:
        """Test that after fit, grid_y_ contains the expected evenly
        spaced interior points for each bin."""
        mapper = UniformBinMapper(bin_edges=[0.0, 10.0], n_points=1)
        mapper.fit(np.array([2.0, 8.0]))
        assert np.any(np.isclose(mapper.grid_y_, 5.0))


class TestTransformIntegration:
    """Integration test confirming transform() (inherited, unmodified) is
    consistent with the uniformly-refined grid.
    """

    def test_transform_matches_to_continuous_dist_mean(self) -> None:
        """Test that transform's output equals to_continuous_dist(pmf).mean()."""
        mapper = UniformBinMapper(bin_edges=[0.0, 10.0, 20.0], n_points=2)
        mapper.fit(np.array([2.0, 8.0, 12.0, 18.0]))
        pmf = np.array([[0.2, 0.1, 0.2, 0.5]])
        dist = mapper.to_continuous_dist(pmf)
        np.testing.assert_allclose(mapper.transform(pmf), dist.mean())
