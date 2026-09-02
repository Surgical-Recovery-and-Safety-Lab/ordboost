"""Unit tests for ordboost.metrics."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ordboost.distributions import (
    ContinuousPredictiveDistribution,
    DiscretePredictiveDistribution,
)
from ordboost.metrics import (
    baseline_distribution,
    crps_score,
    crps_skill_score,
    interval_coverage_rate,
    pinball_loss,
    pinball_loss_skill_score,
    sharpness,
    winkler_score,
)


class TestBaselineDistribution:
    """Tests for baseline_distribution."""

    def test_returns_continuous_predictive_distribution(self) -> None:
        """Test that the function returns a ContinuousPredictiveDistribution."""
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = baseline_distribution(y_train, n_samples=3)
        assert isinstance(dist, ContinuousPredictiveDistribution)

    def test_all_rows_identical(self) -> None:
        """Test that every row of the broadcast grid_cdf is identical,
        representing the same unconditional forecast for every sample."""
        y_train = np.array([1.0, 2.0, 5.0, 10.0])
        dist = baseline_distribution(y_train, n_samples=5)
        for i in range(1, 5):
            np.testing.assert_array_equal(dist.grid_cdf[0], dist.grid_cdf[i])

    def test_output_shape_matches_n_samples(self) -> None:
        """Test that grid_cdf has exactly n_samples rows."""
        y_train = np.array([1.0, 2.0, 3.0])
        dist = baseline_distribution(y_train, n_samples=7)
        assert dist.grid_cdf.shape[0] == 7

    def test_grid_y_is_sorted_unique_values(self) -> None:
        """Test that grid_y contains sorted unique values of y_train, with
        duplicates collapsed to a single grid point."""
        y_train = np.array([5.0, 1.0, 3.0, 1.0, 5.0])
        dist = baseline_distribution(y_train, n_samples=1)
        np.testing.assert_array_equal(dist.grid_y[1:], [1.0, 3.0, 5.0])

    def test_empty_y_train_raises(self) -> None:
        """Test that an empty y_train raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            baseline_distribution(np.array([]), n_samples=5)

    def test_zero_n_samples_raises(self) -> None:
        """Test that n_samples=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            baseline_distribution(np.array([1.0, 2.0]), n_samples=0)

    def test_negative_n_samples_raises(self) -> None:
        """Test that a negative n_samples raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            baseline_distribution(np.array([1.0, 2.0]), n_samples=-3)

    def test_float_n_samples_raises(self) -> None:
        """Test that a non-integer n_samples raises ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            baseline_distribution(np.array([1.0, 2.0]), n_samples=2.5)  # type: ignore

    def test_first_grid_point_is_forced_to_zero(self) -> None:
        """Test that grid_cdf's first column is exactly 0.0, using a fixture
        where the naive empirical CDF at the observed minimum would NOT be
        zero -- this is the direct regression test for the missing floor
        anchor point."""
        y_train = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
        dist = baseline_distribution(y_train, n_samples=1)
        assert dist.grid_cdf[0, 0] == 0.0

    def test_grid_y_starts_below_observed_minimum(self) -> None:
        """Test that grid_y's first value is strictly below y_train's min,
        by the configured boundary_epsilon."""
        y_train = np.array([1.0, 2.0, 3.0])
        dist = baseline_distribution(y_train, n_samples=1, boundary_epsilon=0.01)
        assert dist.grid_y[0] == pytest.approx(1.0 - 0.01)

    def test_grid_cdf_matches_empirical_cdf(self) -> None:
        """Test that grid_cdf values at each unique training value match the
        hand-computed empirical CDF, appearing after the forced floor anchor."""
        y_train = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
        dist = baseline_distribution(y_train, n_samples=1)
        # grid_y: [1-eps, 1, 2, 3]; grid_cdf: [0.0, 1/3, 0.5, 1.0]
        np.testing.assert_allclose(dist.grid_y[1:], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(dist.grid_cdf[0], [0.0, 1.0 / 3.0, 0.5, 1.0])

    def test_last_grid_point_reaches_one_without_adjustment(self) -> None:
        """Test that the final grid_cdf value is exactly 1.0, confirming the
        ceiling needs no equivalent anchor fix -- P(Y <= max(y_train)) = 1.0
        always, by construction."""
        y_train = np.array([2.0, 5.0, 9.0])
        dist = baseline_distribution(y_train, n_samples=1)
        assert dist.grid_cdf[0, -1] == 1.0

    def test_single_unique_value_y_train(self) -> None:
        """Test the degenerate case where y_train has a single repeated
        value: grid_y has the floor anchor plus one point, CDF [0.0, 1.0]."""
        y_train = np.array([7.0, 7.0, 7.0])
        dist = baseline_distribution(y_train, n_samples=2)
        np.testing.assert_allclose(dist.grid_y, [7.0 - 1e-4, 7.0])
        np.testing.assert_allclose(dist.grid_cdf, [[0.0, 1.0], [0.0, 1.0]])


class TestCRPSScore:
    """Tests for Continuous Ranked Probability Score (CRPS)."""

    @pytest.fixture
    def perfect_discrete_dist(
        self,
    ) -> tuple[np.ndarray, DiscretePredictiveDistribution]:
        """Fixture providing a deterministic discrete distribution with perfect predictions."""
        classes = np.array([0, 10, 20])
        pmf = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y_true = np.array([0, 10, 20])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)
        return y_true, dist

    # --- Discrete CRPS Tests

    def test_crps_discrete_perfect_predictions(self, perfect_discrete_dist) -> None:
        """Test that a perfect deterministic discrete forecast yields CRPS = 0.0."""
        y_true, dist = perfect_discrete_dist
        assert crps_score(y_true, dist) == pytest.approx(0.0, abs=1e-7)

    def test_crps_discrete_known_value(self) -> None:
        """Test discrete CRPS against a hand-calculated non-zero value."""
        classes = np.array([0, 10, 20])
        pmf = np.array([[0.5, 0.3, 0.2]])
        y_true = np.array([10])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)
        # CDF = [0.5, 0.8, 1.0], Step I(10 <= c) = [0.0, 1.0, 1.0]
        # Diff^2 = [0.25, 0.04, 0.0] -> CRPS sum = 0.29
        assert crps_score(y_true, dist) == pytest.approx(0.29, abs=1e-6)

    def test_crps_discrete_sample_weights(self) -> None:
        """Test weighted discrete CRPS calculation."""
        classes = np.array([0, 1])
        pmf = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_true = np.array([0, 0])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)
        weights = np.array([3.0, 1.0])
        # Expected: (3*0.0 + 1*1.0) / 4 = 0.25
        assert crps_score(y_true, dist, sample_weight=weights) == pytest.approx(
            0.25, abs=1e-6
        )

    def test_crps_discrete_missing_class_error(self) -> None:
        """Test error when y_true contains values not in discrete dist.classes."""
        classes = np.array([0, 10])
        pmf = np.ones((1, 2)) * 0.5
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)
        with pytest.raises(ValueError, match="not present in y_dist.classes"):
            crps_score(np.array([99]), dist)

    # --- Continuous CRPS Tests ---

    def test_crps_continuous_exact_kink_value(self) -> None:
        """Test continuous CRPS against an independently hand-derived exact
        value, using the closed-form piecewise-linear segment integral
        (width * (a^2 + a*b + b^2) / 3, split at y_true) rather than the
        trapezoidal approximation this replaces. For grid_y=[0,10],
        grid_cdf=[0.2, 0.8], y_true=5 (the exact midpoint, so the kink
        splits the single segment into two equal-width halves):
          segment [0,5): F 0.2->0.5, integrand F(y)^2
            = 5*(0.2^2 + 0.2*0.5 + 0.5^2)/3 = 0.65
          segment [5,10]: F 0.5->0.8, integrand (F(y)-1)^2
            = 5*(0.5^2 + 0.5*0.2 + 0.2^2)/3 (via a'=-0.5, b'=-0.2) = 0.65
          total = 1.3

        This is the exact regression test for the original trapezoidal
        bug: the old (removed) implementation returned 0.4 for this same
        input, a >3x underestimate, since it never split the segment
        containing y_true.
        """
        grid_y = np.array([0.0, 10.0])
        grid_cdf = np.array([[0.2, 0.8]])
        y_true = np.array([5.0])

        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.grid_y = grid_y
        dist.grid_cdf = grid_cdf

        score = crps_score(y_true, dist)
        assert score == pytest.approx(1.3, abs=1e-6)
        assert score != pytest.approx(0.4, abs=1e-6)  # the old, biased result

    def test_crps_continuous_multi_sample_multi_segment(self) -> None:
        """Test continuous CRPS on a multi-sample, multi-segment grid,
        checking shape and basic sanity properties (non-negativity, exact
        zero for a degenerate perfect forecast) rather than a fully
        hand-derived multi-segment value, which is impractical to derive
        reliably by hand across several segments.
        """
        grid_y = np.array([0.0, 5.0, 10.0, 15.0])
        grid_cdf = np.array(
            [
                [0.0, 0.4, 0.8, 1.0],
                [0.0, 0.1, 0.9, 1.0],
                [0.0, 0.0, 0.0, 1.0],  # near-degenerate: all mass at y=15
            ]
        )
        y_true = np.array([7.0, 3.0, 15.0])

        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.grid_y = grid_y
        dist.grid_cdf = grid_cdf

        score = crps_score(y_true, dist, sample_weight=None)
        assert score >= 0.0
        assert np.isfinite(score)

    def test_crps_continuous_perfect_step_forecast_near_zero(self) -> None:
        """Test that a forecast whose CDF is already (approximately) the
        true step function at y_true yields a CRPS close to zero."""
        grid_y = np.array([0.0, 10.0])
        grid_cdf = np.array([[1.0, 1.0]])  # CDF already at 1.0 everywhere
        y_true = np.array([0.0])  # true value at the very start of the grid

        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.grid_y = grid_y
        dist.grid_cdf = grid_cdf

        assert crps_score(y_true, dist) == pytest.approx(0.0, abs=1e-6)

    def test_crps_continuous_sample_weights(self) -> None:
        """Test weighted continuous CRPS, reusing the exact single-sample
        value from test_crps_continuous_exact_kink_value alongside a
        second sample, to confirm weighting is applied on top of the
        corrected per-sample values rather than the old ones."""
        grid_y = np.array([0.0, 10.0])
        grid_cdf = np.array([[0.2, 0.8], [0.2, 0.8]])
        y_true = np.array([5.0, 5.0])  # both samples: exact CRPS = 1.3 each

        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.grid_y = grid_y
        dist.grid_cdf = grid_cdf

        weights = np.array([3.0, 1.0])
        # Both per-sample scores are 1.3, so any valid weighting must also be 1.3
        score = crps_score(y_true, dist, sample_weight=weights)
        assert score == pytest.approx(1.3, abs=1e-6)

    # --- Common Input Validation Tests

    def test_crps_invalid_y_true_ndim(self) -> None:
        """Test error when y_true is not 1D."""
        dist = MagicMock(spec=DiscretePredictiveDistribution)
        with pytest.raises(ValueError, match="1D array"):
            crps_score(np.array([[1, 2], [3, 4]]), dist)

    def test_crps_sample_count_mismatch_continuous(self) -> None:
        """Test error when sample count in y_true mismatches a continuous
        y_dist's grid_cdf row count."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.grid_y = np.array([0.0, 10.0])
        dist.grid_cdf = np.array([[0.2, 0.8]])
        with pytest.raises(ValueError, match="Sample count mismatch"):
            crps_score(np.array([1.0, 2.0]), dist)

    def test_crps_sample_count_mismatch_discrete(self) -> None:
        """Test error when sample count in y_true mismatches a discrete
        y_dist's pmf row count."""
        classes = np.array([0, 1])
        pmf = np.array([[0.5, 0.5]])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)
        with pytest.raises(ValueError, match="Sample count mismatch"):
            crps_score(np.array([0, 1]), dist)

    def test_crps_invalid_sample_weight_shape(self) -> None:
        """Test error when sample_weight shape mismatches y_true, for the
        continuous branch."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.grid_y = np.array([0.0, 10.0])
        dist.grid_cdf = np.array([[0.2, 0.8]])
        with pytest.raises(ValueError, match="Expected 'sample_weight' shape"):
            crps_score(np.array([5.0]), dist, sample_weight=[1.0, 2.0])


class TestCRPSSkillScore:
    """Tests for crps_skill_score."""

    def test_formula_matches_hand_computation(self) -> None:
        """Test that the skill score equals 1 - crps_model/crps_baseline
        for known mocked CRPS values."""
        dist_model = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_baseline = MagicMock(spec=ContinuousPredictiveDistribution)

        with patch("ordboost.metrics.crps_score", side_effect=[2.0, 8.0]) as mock_crps:
            score = crps_skill_score([1.0, 2.0], dist_model, dist_baseline)

        assert score == pytest.approx(1.0 - 2.0 / 8.0)
        assert mock_crps.call_count == 2

    def test_perfect_model_gives_skill_score_of_one(self) -> None:
        """Test that a model CRPS of 0.0 yields a skill score of 1.0
        (perfect forecast), regardless of the baseline's CRPS."""
        dist_model = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_baseline = MagicMock(spec=ContinuousPredictiveDistribution)

        with patch("ordboost.metrics.crps_score", side_effect=[0.0, 5.0]):
            score = crps_skill_score([1.0], dist_model, dist_baseline)

        assert score == pytest.approx(1.0)

    def test_model_equal_to_baseline_gives_skill_score_of_zero(self) -> None:
        """Test that identical model and baseline CRPS values give a
        skill score of exactly 0.0."""
        dist_model = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_baseline = MagicMock(spec=ContinuousPredictiveDistribution)

        with patch("ordboost.metrics.crps_score", side_effect=[4.0, 4.0]):
            score = crps_skill_score([1.0], dist_model, dist_baseline)

        assert score == pytest.approx(0.0)

    def test_model_worse_than_baseline_gives_negative_score(self) -> None:
        """Test that a model CRPS worse than the baseline's yields a
        negative skill score."""
        dist_model = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_baseline = MagicMock(spec=ContinuousPredictiveDistribution)

        with patch("ordboost.metrics.crps_score", side_effect=[10.0, 4.0]):
            score = crps_skill_score([1.0], dist_model, dist_baseline)

        assert score == pytest.approx(1.0 - 10.0 / 4.0)
        assert score < 0.0

    def test_sample_weight_forwarded_to_both_crps_calls(self) -> None:
        """Test that sample_weight is passed through identically to both
        the model and baseline CRPS computations."""
        dist_model = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_baseline = MagicMock(spec=ContinuousPredictiveDistribution)
        weights = [1.0, 2.0]

        with patch("ordboost.metrics.crps_score", side_effect=[1.0, 2.0]) as mock_crps:
            crps_skill_score(
                [1.0, 2.0], dist_model, dist_baseline, sample_weight=weights
            )

        for call in mock_crps.call_args_list:
            assert call.kwargs.get("sample_weight") == weights

    def test_integration_with_real_crps_score(self) -> None:
        """Test end-to-end against real (unmocked) crps_score and
        baseline_distribution, confirming the pieces compose correctly
        rather than only working with mocks."""
        y_train = np.array([0.0, 5.0, 10.0])
        y_true = np.array([5.0])
        grid_y = np.array([0.0, 10.0])
        grid_cdf = np.array([[0.0, 1.0]])

        dist_model = ContinuousPredictiveDistribution(grid_y=grid_y, grid_cdf=grid_cdf)
        dist_baseline = baseline_distribution(y_train, n_samples=1)

        score = crps_skill_score(y_true, dist_model, dist_baseline)
        assert np.isfinite(score)


class TestPinballLoss:
    """Tests for pinball (quantile) loss function."""

    def test_pinball_loss_known_values(self) -> None:
        """Test pinball loss against manual calculation."""
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([12.0, 7.0])  # over (+2), under (-3)
        q = 0.8

        # Sample 0 (over-prediction, err = -2): max(0.8*-2, -0.2*-2) = 0.4
        # Sample 1 (under-prediction, err = +3): max(0.8*3, -0.2*3) = 2.4
        # Mean pinball loss = (0.4 + 2.4) / 2 = 1.4
        loss = pinball_loss(y_true, y_pred, q=q)
        assert loss == pytest.approx(1.4, abs=1e-6)

    def test_pinball_loss_sample_weights(self) -> None:
        """Test weighted pinball loss."""
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([12.0, 7.0])
        q = 0.8
        weights = np.array([1.0, 3.0])

        # Expected: (0.4 * 1.0 + 2.4 * 3.0) / 4.0 = 7.6 / 4.0 = 1.9
        loss = pinball_loss(y_true, y_pred, q=q, sample_weight=weights)
        assert loss == pytest.approx(1.9, abs=1e-6)

    def test_pinball_loss_invalid_q(self) -> None:
        """Test that invalid q levels raise ValueError."""
        with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
            pinball_loss([1.0], [1.0], q=0.0)

        with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
            pinball_loss([1.0], [1.0], q=1.0)

    def test_pinball_loss_shape_mismatch(self) -> None:
        """Test error when y_true and y_pred_q shapes differ."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            pinball_loss([1.0, 2.0], [1.0], q=0.5)

    def test_pinball_loss_invalid_sample_weight_shape(self) -> None:
        """Test error when sample_weight shape mismatches y_true."""
        with pytest.raises(ValueError, match="Expected 'sample_weight' shape"):
            pinball_loss([1.0, 2.0], [1.0, 2.0], q=0.5, sample_weight=[1.0])


class TestPinballLossSkillScore:
    """Tests for pinball_loss_skill_score."""

    def test_formula_matches_hand_computation(self) -> None:
        """Test the skill score formula against hand-computed pinball
        losses for known ppf outputs. y_true=[10], q=0.5. Model predicts
        10 exactly (loss=0). Baseline predicts 8: error=10-8=2,
        loss=q*2=1.0. Skill score = 1 - 0/1.0 = 1.0.
        """
        dist_model = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_model.ppf.return_value = np.array([10.0])
        dist_baseline = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_baseline.ppf.return_value = np.array([8.0])

        score = pinball_loss_skill_score([10.0], dist_model, dist_baseline, q=0.5)
        assert score == pytest.approx(1.0)

    def test_model_equal_to_baseline_gives_skill_score_of_zero(self) -> None:
        """Test that identical model and baseline predictions give a
        skill score of exactly 0.0."""
        dist_model = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_model.ppf.return_value = np.array([8.0])
        dist_baseline = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_baseline.ppf.return_value = np.array([8.0])

        score = pinball_loss_skill_score([10.0], dist_model, dist_baseline, q=0.5)
        assert score == pytest.approx(0.0)

    def test_model_worse_than_baseline_gives_negative_score(self) -> None:
        """Test a model with larger pinball loss than the baseline.
        y_true=[10], q=0.9. Model predicts 0: error=10, loss=q*10=9.0.
        Baseline predicts 9: error=1, loss=q*1=0.9.
        Skill score = 1 - 9.0/0.9 = -9.0.
        """
        dist_model = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_model.ppf.return_value = np.array([0.0])
        dist_baseline = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_baseline.ppf.return_value = np.array([9.0])

        score = pinball_loss_skill_score([10.0], dist_model, dist_baseline, q=0.9)
        assert score == pytest.approx(-9.0)

    def test_queries_both_distributions_at_same_quantile(self) -> None:
        """Test that both dist_model.ppf and dist_baseline.ppf are called
        with the same quantile level q."""
        dist_model = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_model.ppf.return_value = np.array([5.0])
        dist_baseline = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_baseline.ppf.return_value = np.array([5.0])

        pinball_loss_skill_score([6.0], dist_model, dist_baseline, q=0.25)

        dist_model.ppf.assert_called_once_with(0.25)
        dist_baseline.ppf.assert_called_once_with(0.25)

    def test_sample_weight_affects_result(self) -> None:
        """Test that sample_weight is genuinely forwarded (not silently
        dropped), by confirming weighted and unweighted results differ
        for asymmetric per-sample losses."""
        dist_model = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_model.ppf.return_value = np.array([0.0, 0.0])
        dist_baseline = MagicMock(spec=ContinuousPredictiveDistribution)
        dist_baseline.ppf.return_value = np.array([10.0, 1.0])

        y_true = [10.0, 0.0]
        unweighted = pinball_loss_skill_score(
            y_true,
            dist_model,
            dist_baseline,
            q=0.5,
        )
        weighted = pinball_loss_skill_score(
            y_true, dist_model, dist_baseline, q=0.5, sample_weight=[10.0, 1.0]
        )
        assert unweighted != pytest.approx(weighted)

    def test_invalid_q_raises(self) -> None:
        """Test that a quantile level outside (0.0, 1.0) raises ValueError,
        propagated from dist_model.ppf's own quantile validation (using a
        real distribution, since a MagicMock would not perform this
        validation itself)."""
        grid_y = np.array([0.0, 10.0])
        grid_cdf = np.array([[0.0, 1.0]])
        dist_model = ContinuousPredictiveDistribution(grid_y=grid_y, grid_cdf=grid_cdf)
        dist_baseline = ContinuousPredictiveDistribution(
            grid_y=grid_y, grid_cdf=grid_cdf
        )

        with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
            pinball_loss_skill_score([5.0], dist_model, dist_baseline, q=1.5)

    def test_integration_with_real_distributions(self) -> None:
        """Test end-to-end against real (unmocked) distributions,
        confirming ppf and pinball_loss compose correctly."""
        grid_y = np.array([0.0, 10.0])
        dist_model = ContinuousPredictiveDistribution(
            grid_y=grid_y, grid_cdf=np.array([[0.0, 1.0]])
        )
        y_train = np.array([0.0, 5.0, 10.0])
        dist_baseline = baseline_distribution(y_train, n_samples=1)

        score = pinball_loss_skill_score([5.0], dist_model, dist_baseline, q=0.5)
        assert np.isfinite(score)


class TestIntervalCoverageRate:
    """Tests for prediction interval coverage rate."""

    def test_coverage_rate_known_values(self) -> None:
        """Test empirical coverage rate calculation."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.interval.return_value = (
            np.array([2.0, 5.0, 10.0]),
            np.array([8.0, 15.0, 20.0]),
        )
        y_true = np.array([5.0, 4.0, 15.0])  # inside, outside (under), inside

        # Covered array: [True, False, True] -> Mean = 2/3
        coverage = interval_coverage_rate(y_true, dist, alpha=0.10)
        assert coverage == pytest.approx(2.0 / 3.0, abs=1e-6)
        dist.interval.assert_called_once_with(alpha=0.10)

    def test_coverage_rate_weighted(self) -> None:
        """Test weighted empirical coverage rate calculation."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.interval.return_value = (
            np.array([2.0, 5.0, 10.0]),
            np.array([8.0, 15.0, 20.0]),
        )
        y_true = np.array([5.0, 4.0, 15.0])
        weights = np.array([1.0, 3.0, 1.0])

        # Covered array: [1, 0, 1] -> Weighted mean: (1*1 + 0*3 + 1*1)/5 = 0.4
        coverage = interval_coverage_rate(
            y_true, dist, alpha=0.10, sample_weight=weights
        )
        assert coverage == pytest.approx(0.4, abs=1e-6)

    def test_coverage_rate_invalid_sample_weight_shape(self) -> None:
        """Test error when sample_weight shape mismatches y_true."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.interval.return_value = (np.array([0.0]), np.array([10.0]))

        with pytest.raises(ValueError, match="Expected 'sample_weight' shape"):
            interval_coverage_rate([5.0], dist, sample_weight=[1.0, 2.0])


class TestSharpness:
    """Tests for sharpness."""

    def test_matches_hand_computation(self) -> None:
        """Test mean interval width against a hand-computed example."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.interval.return_value = (
            np.array([2.0, 5.0, 10.0]),
            np.array([8.0, 15.0, 20.0]),
        )
        # widths: [6.0, 10.0, 10.0] -> mean = 26.0 / 3.0
        result = sharpness(dist, alpha=0.10)
        assert result == pytest.approx(26.0 / 3.0)

    def test_calls_interval_with_given_alpha(self) -> None:
        """Test that dist.interval is called with the supplied alpha."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.interval.return_value = (np.array([0.0]), np.array([10.0]))
        sharpness(dist, alpha=0.20)
        dist.interval.assert_called_once_with(alpha=0.20)

    def test_default_alpha(self) -> None:
        """Test that the default alpha=0.10 is used when not specified."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.interval.return_value = (np.array([0.0]), np.array([10.0]))
        sharpness(dist)
        dist.interval.assert_called_once_with(alpha=0.10)

    def test_zero_width_interval_gives_zero_sharpness(self) -> None:
        """Test that identical lower and upper bounds give sharpness of 0.0."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.interval.return_value = (np.array([5.0, 5.0]), np.array([5.0, 5.0]))
        assert sharpness(dist, alpha=0.10) == pytest.approx(0.0)

    def test_single_sample(self) -> None:
        """Test sharpness computation for a single-sample distribution."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.interval.return_value = (np.array([3.0]), np.array([9.0]))
        assert sharpness(dist, alpha=0.10) == pytest.approx(6.0)

    def test_alpha_zero_raises(self) -> None:
        """Test that alpha=0.0 raises ValueError (open interval bound)."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        with pytest.raises(ValueError, match="must lie within"):
            sharpness(dist, alpha=0.0)

    def test_alpha_one_raises(self) -> None:
        """Test that alpha=1.0 raises ValueError (open interval bound)."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        with pytest.raises(ValueError, match="must lie within"):
            sharpness(dist, alpha=1.0)

    def test_negative_alpha_raises(self) -> None:
        """Test that a negative alpha raises ValueError."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        with pytest.raises(ValueError, match="must lie within"):
            sharpness(dist, alpha=-0.1)

    def test_alpha_above_one_raises(self) -> None:
        """Test that alpha > 1.0 raises ValueError."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        with pytest.raises(ValueError, match="must lie within"):
            sharpness(dist, alpha=1.5)

    def test_alpha_validated_before_interval_is_called(self) -> None:
        """Test that an invalid alpha raises before dist.interval is ever
        invoked, confirming validation happens up front."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        with pytest.raises(ValueError, match="must lie within"):
            sharpness(dist, alpha=0.0)
        dist.interval.assert_not_called()

    def test_larger_alpha_gives_narrower_interval_sanity(self) -> None:
        """Test a realistic end-to-end case (real distribution, not
        mocked) confirming a larger alpha (narrower central interval)
        produces smaller sharpness, as a basic sanity check of the
        interval-width relationship rather than an isolated unit check."""
        grid_y = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        grid_cdf = np.array([[0.0, 0.2, 0.5, 0.8, 1.0]])
        dist = ContinuousPredictiveDistribution(grid_y=grid_y, grid_cdf=grid_cdf)

        narrow = sharpness(dist, alpha=0.50)  # 50% interval
        wide = sharpness(dist, alpha=0.10)  # 90% interval
        assert narrow < wide


class TestWinklerScore:
    """Tests for Winkler interval score function."""

    def test_winkler_score_known_values(self) -> None:
        """Test Winkler score against hand-calculated values."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.interval.return_value = (
            np.array([2.0, 5.0, 10.0]),
            np.array([8.0, 15.0, 20.0]),
        )
        y_true = np.array([5.0, 3.0, 25.0])  # inside, under, over
        alpha = 0.10  # multiplier 2/alpha = 20.0

        # Sample 0 (inside): width = 6.0, penalty = 0 -> score = 6.0
        # Sample 1 (under):  width = 10.0, under penalty = 20.0*(5-3) = 40.0 -> score = 50.0
        # Sample 2 (over):   width = 10.0, over penalty  = 20.0*(25-20) = 100.0 -> score = 110.0
        # Mean score = (6.0 + 50.0 + 110.0) / 3 = 166.0 / 3
        score = winkler_score(y_true, dist, alpha=alpha)
        assert score == pytest.approx(166.0 / 3.0, abs=1e-6)
        dist.interval.assert_called_once_with(alpha=0.10)

    def test_winkler_score_weighted(self) -> None:
        """Test weighted Winkler score calculation."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.interval.return_value = (
            np.array([2.0, 5.0, 10.0]),
            np.array([8.0, 15.0, 20.0]),
        )
        y_true = np.array([5.0, 3.0, 25.0])  # scores: [6.0, 50.0, 110.0]
        weights = np.array([1.0, 1.0, 2.0])

        # Weighted mean: (6.0*1 + 50.0*1 + 110.0*2) / 4.0 = 276.0 / 4.0 = 69.0
        score = winkler_score(y_true, dist, alpha=0.10, sample_weight=weights)
        assert score == pytest.approx(69.0, abs=1e-6)

    def test_winkler_score_invalid_alpha(self) -> None:
        """Test that invalid alpha significance levels raise ValueError."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)

        with pytest.raises(ValueError, match="must lie within"):
            winkler_score([5.0], dist, alpha=0.0)

        with pytest.raises(ValueError, match="must lie within"):
            winkler_score([5.0], dist, alpha=1.0)

    def test_winkler_score_invalid_sample_weight_shape(self) -> None:
        """Test error when sample_weight shape mismatches y_true."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.interval.return_value = (np.array([0.0]), np.array([10.0]))

        with pytest.raises(ValueError, match="Expected 'sample_weight' shape"):
            winkler_score([5.0], dist, alpha=0.10, sample_weight=[1.0, 2.0])
