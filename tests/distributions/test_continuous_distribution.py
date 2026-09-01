"""Unit tests for ContinuousPredictiveDistribution in ordboost.distributions."""

import numpy as np
import pytest

from ordboost.distributions import ContinuousPredictiveDistribution


class TestInit:
    """Tests for ContinuousPredictiveDistribution.__init__ validation."""

    def test_valid_construction_stores_data(self) -> None:
        """Test that a valid grid_y/grid_cdf pair constructs without
        error and stores the expected values."""
        grid_y = np.array([0.0, 10.0, 20.0])
        grid_cdf = np.array([[0.0, 0.5, 1.0]])
        dist = ContinuousPredictiveDistribution(grid_y=grid_y, grid_cdf=grid_cdf)
        np.testing.assert_array_equal(dist.grid_y, grid_y)
        np.testing.assert_array_equal(dist.grid_cdf, grid_cdf)

    def test_non_1d_grid_y_raises(self) -> None:
        """Test that a 2D grid_y raises ValueError."""
        with pytest.raises(ValueError, match="Invalid array dimensions"):
            ContinuousPredictiveDistribution(
                grid_y=np.ones((3, 1)), grid_cdf=np.array([[0.0, 0.5, 1.0]])
            )

    def test_non_2d_grid_cdf_raises(self) -> None:
        """Test that a 1D grid_cdf raises ValueError."""
        with pytest.raises(ValueError, match="Invalid array dimensions"):
            ContinuousPredictiveDistribution(
                grid_y=np.array([0.0, 10.0, 20.0]), grid_cdf=np.array([0.0, 0.5, 1.0])
            )

    def test_shape_mismatch_raises(self) -> None:
        """Test that mismatched grid_y length and grid_cdf columns raises ValueError."""
        with pytest.raises(ValueError, match="Grid CDF column dimension"):
            ContinuousPredictiveDistribution(
                grid_y=np.array([0.0, 10.0]), grid_cdf=np.array([[0.0, 0.5, 1.0]])
            )

    def test_non_ascending_grid_y_raises(self) -> None:
        """Test that a non-strictly-ascending grid_y raises ValueError."""
        with pytest.raises(ValueError, match="strictly ascending"):
            ContinuousPredictiveDistribution(
                grid_y=np.array([0.0, 20.0, 10.0]), grid_cdf=np.array([[0.0, 0.5, 1.0]])
            )

    def test_repeated_grid_y_values_raises(self) -> None:
        """Test that duplicate (tied) grid_y values raise ValueError, since
        ascending order must be strict, not non-decreasing."""
        with pytest.raises(ValueError, match="strictly ascending"):
            ContinuousPredictiveDistribution(
                grid_y=np.array([0.0, 10.0, 10.0]), grid_cdf=np.array([[0.0, 0.5, 1.0]])
            )

    def test_grid_cdf_not_starting_at_zero_raises(self) -> None:
        """Test that a grid_cdf row not starting at 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="must start at 0.0"):
            ContinuousPredictiveDistribution(
                grid_y=np.array([0.0, 10.0, 20.0]), grid_cdf=np.array([[0.2, 0.5, 1.0]])
            )

    def test_grid_cdf_not_ending_at_one_raises(self) -> None:
        """Test that a grid_cdf row not ending at 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must end at 1.0"):
            ContinuousPredictiveDistribution(
                grid_y=np.array([0.0, 10.0, 20.0]), grid_cdf=np.array([[0.0, 0.5, 0.8]])
            )

    def test_out_of_bound_values_are_clipped_before_boundary_check(self) -> None:
        """Test that grid_cdf values outside [0.0, 1.0] are clipped first,
        so a row like [-0.1, 0.5, 1.2] passes boundary validation after
        clipping to [0.0, 0.5, 1.0]."""
        grid_y = np.array([0.0, 10.0, 20.0])
        grid_cdf = np.array([[-0.1, 0.5, 1.2]])
        dist = ContinuousPredictiveDistribution(grid_y=grid_y, grid_cdf=grid_cdf)
        np.testing.assert_array_equal(dist.grid_cdf, [[0.0, 0.5, 1.0]])

    def test_multi_row_grid_cdf_only_one_bad_row_raises(self) -> None:
        """Test that a single boundary-violating row among otherwise valid
        rows still raises ValueError."""
        grid_y = np.array([0.0, 10.0, 20.0])
        grid_cdf = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 0.9]])
        with pytest.raises(ValueError, match="must end at 1.0"):
            ContinuousPredictiveDistribution(grid_y=grid_y, grid_cdf=grid_cdf)

    def test_grid_y_is_read_only_after_construction(self) -> None:
        """Test that dist.grid_y cannot be mutated in place after construction."""
        dist = ContinuousPredictiveDistribution(
            grid_y=np.array([0.0, 10.0, 20.0]), grid_cdf=np.array([[0.0, 0.5, 1.0]])
        )
        with pytest.raises(ValueError, match="read-only"):
            dist.grid_y[0] = 5.0

    def test_grid_cdf_is_read_only_after_construction(self) -> None:
        """Test that dist.grid_cdf cannot be mutated in place after construction."""
        dist = ContinuousPredictiveDistribution(
            grid_y=np.array([0.0, 10.0, 20.0]), grid_cdf=np.array([[0.0, 0.5, 1.0]])
        )
        with pytest.raises(ValueError, match="read-only"):
            dist.grid_cdf[0, 1] = 0.9

    def test_user_grid_y_is_not_read_only_after_construction(self) -> None:
        """Test that dist.grid_y cannot be mutated in place after construction."""
        grid_y = np.array([0.0, 10.0, 20.0])
        ContinuousPredictiveDistribution(
            grid_y=grid_y, grid_cdf=np.array([[0.0, 0.5, 1.0]])
        )

        grid_y[0] = 10
        assert grid_y[0] == 10

    def test_user_grid_cdf_is_not_read_only_after_construction(self) -> None:
        """Test that dist.grid_cdf cannot be mutated in place after construction."""
        grid_cdf = np.array([[0.0, 0.5, 1.0]])
        ContinuousPredictiveDistribution(
            grid_y=np.array([0.0, 10.0, 20.0]), grid_cdf=grid_cdf
        )

        grid_cdf[0, 1] = 0.9
        assert grid_cdf[0, 1] == 0.9


class TestMean:
    """Tests for ContinuousPredictiveDistribution.mean."""

    @pytest.fixture
    def sample_distribution(self) -> ContinuousPredictiveDistribution:
        """Fixture providing a known 2-sample continuous distribution."""
        grid_y = np.array([0.0, 10.0, 20.0, 30.0])
        grid_cdf = np.array(
            [
                [0.0, 0.4, 0.8, 1.0],
                [0.0, 0.1, 0.9, 1.0],
            ]
        )
        return ContinuousPredictiveDistribution(grid_y=grid_y, grid_cdf=grid_cdf)

    def test_mean_matches_hand_computation(self, sample_distribution) -> None:
        """Test expected value calculation via trapezoidal integration
        against hand-derived expected values."""
        means = sample_distribution.mean()
        expected_means = np.array([13.0, 15.0])
        np.testing.assert_allclose(means, expected_means, atol=1e-6)

    def test_mean_output_shape(self, sample_distribution) -> None:
        """Test that mean() returns one value per sample."""
        assert sample_distribution.mean().shape == (2,)

    def test_uniform_cdf_mean_equals_midpoint(self) -> None:
        """Test that a perfectly uniform CDF across the grid yields a mean
        equal to the midpoint of the range, as a sanity check on the
        integration formula."""
        grid_y = np.array([0.0, 10.0])
        grid_cdf = np.array([[0.0, 1.0]])
        dist = ContinuousPredictiveDistribution(grid_y=grid_y, grid_cdf=grid_cdf)
        np.testing.assert_allclose(dist.mean(), [5.0])


class TestPpf:
    """Tests for ContinuousPredictiveDistribution._ppf, exercised via the
    shared public ppf() (validation itself is covered by
    test_predictive_distribution.py).
    """

    @pytest.fixture
    def sample_distribution(self) -> ContinuousPredictiveDistribution:
        """Fixture providing a known 2-sample continuous distribution."""
        grid_y = np.array([0.0, 10.0, 20.0, 30.0])
        grid_cdf = np.array(
            [
                [0.0, 0.4, 0.8, 1.0],
                [0.0, 0.1, 0.9, 1.0],
            ]
        )
        return ContinuousPredictiveDistribution(grid_y=grid_y, grid_cdf=grid_cdf)

    def test_ppf_exact_grid_match_for_sample_with_matching_cdf_value(
        self, sample_distribution
    ) -> None:
        """Test ppf at a quantile that exactly matches sample 0's CDF grid
        value (grid_cdf[0] hits 0.4 exactly at y=10)."""
        result = sample_distribution.ppf(0.4)
        np.testing.assert_allclose(result[0], 10.0, atol=1e-6)

    def test_ppf_interpolation_for_sample_without_matching_grid_value(
        self, sample_distribution
    ) -> None:
        """Test ppf at a quantile that does not exactly match sample 1's CDF
        grid (0.4 falls strictly between grid_cdf[1]'s 0.1 at y=10 and 0.9
        at y=20), requiring linear interpolation."""
        result = sample_distribution.ppf(0.4)
        # t = (0.4 - 0.1) / (0.9 - 0.1) = 0.375 -> y = 10 + 0.375 * 10 = 13.75
        np.testing.assert_allclose(result[1], 13.75, atol=1e-6)

    def test_ppf_inner_interpolation(self, sample_distribution) -> None:
        """Test ppf at a quantile strictly between two grid CDF values."""
        result = sample_distribution.ppf(0.6)
        # Sample 0: interp between (10, 0.4) and (20, 0.8) -> 15.0
        # Sample 1: interp between (20, 0.9) and (10, 0.1) is not applicable
        #   here since 0.6 falls between (10, 0.1) and (20, 0.9) ->
        #   t=(0.6-0.1)/0.8=0.625 -> 10 + 0.625*10 = 16.25
        np.testing.assert_allclose(result, [15.0, 16.25], atol=1e-6)

    def test_ppf_zero_returns_grid_y_min(self, sample_distribution) -> None:
        """Test that ppf(0.0) returns the first grid_y value for every sample."""
        result = sample_distribution.ppf(0.0)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-6)

    def test_ppf_one_returns_grid_y_max(self, sample_distribution) -> None:
        """Test that ppf(1.0) returns the last grid_y value for every sample."""
        result = sample_distribution.ppf(1.0)
        np.testing.assert_allclose(result, [30.0, 30.0], atol=1e-6)

    def test_ppf_array_output_shape(self, sample_distribution) -> None:
        """Test that an array of quantiles returns a
        (n_samples, n_quantiles) shaped result."""
        result = sample_distribution.ppf(np.array([0.1, 0.5, 0.9]))
        assert result.shape == (2, 3)

    def test_ppf_scalar_matches_corresponding_array_column(
        self, sample_distribution
    ) -> None:
        """Test that ppf(q) as a scalar matches the corresponding column
        of ppf([q, ...]), for consistency between the scalar and array
        branches of _ppf."""
        scalar_result = sample_distribution.ppf(0.6)
        array_result = sample_distribution.ppf(np.array([0.6, 0.9]))
        np.testing.assert_allclose(scalar_result, array_result[:, 0], atol=1e-6)


class TestCdf:
    """Tests for ContinuousPredictiveDistribution.cdf."""

    @pytest.fixture
    def sample_distribution(self) -> ContinuousPredictiveDistribution:
        """Fixture providing a known 2-sample continuous distribution."""
        grid_y = np.array([0.0, 10.0, 20.0, 30.0])
        grid_cdf = np.array(
            [
                [0.0, 0.4, 0.8, 1.0],
                [0.0, 0.1, 0.9, 1.0],
            ]
        )
        return ContinuousPredictiveDistribution(grid_y=grid_y, grid_cdf=grid_cdf)

    def test_scalar_y_exact_grid_lookup(self, sample_distribution) -> None:
        """Test scalar CDF evaluation at a value exactly on the grid,
        broadcast across all samples."""
        probs = sample_distribution.cdf(10.0)
        assert probs.shape == (2,)
        np.testing.assert_allclose(probs, [0.4, 0.1], atol=1e-6)

    def test_scalar_y_inner_interpolation(self, sample_distribution) -> None:
        """Test scalar CDF evaluation at a value strictly between grid points."""
        probs = sample_distribution.cdf(15.0)
        np.testing.assert_allclose(probs, [0.6, 0.5], atol=1e-6)

    def test_scalar_y_extrapolates_to_zero_below_grid(
        self, sample_distribution
    ) -> None:
        """Test that evaluating below grid_y's minimum extrapolates flat to 0.0."""
        probs = sample_distribution.cdf(-5.0)
        np.testing.assert_allclose(probs, [0.0, 0.0])

    def test_scalar_y_extrapolates_to_one_above_grid(self, sample_distribution) -> None:
        """Test that evaluating above grid_y's maximum extrapolates flat to 1.0."""
        probs = sample_distribution.cdf(35.0)
        np.testing.assert_allclose(probs, [1.0, 1.0])

    def test_vectorized_y_evaluates_each_sample_at_its_own_value(
        self, sample_distribution
    ) -> None:
        """Test 1D array CDF evaluation, one y value per sample."""
        y_targets = np.array([10.0, 15.0])
        probs = sample_distribution.cdf(y_targets)
        assert probs.shape == (2,)
        np.testing.assert_allclose(probs, [0.4, 0.5], atol=1e-6)

    def test_vectorized_y_extrapolation(self, sample_distribution) -> None:
        """Test that vectorized evaluation also extrapolates flat at the
        grid boundaries, per-sample."""
        extrap_targets = np.array([-5.0, 35.0])
        probs = sample_distribution.cdf(extrap_targets)
        np.testing.assert_allclose(probs, [0.0, 1.0])

    def test_wrong_length_1d_y_raises(self, sample_distribution) -> None:
        """Test that a 1D y array of the wrong length raises ValueError."""
        with pytest.raises(ValueError, match="scalar or 1D array of length"):
            sample_distribution.cdf(np.array([10.0, 15.0, 20.0]))

    def test_2d_y_raises(self, sample_distribution) -> None:
        """Test that a 2D y array raises ValueError."""
        with pytest.raises(ValueError, match="scalar or 1D array of length"):
            sample_distribution.cdf(np.array([[10.0, 15.0]]))
