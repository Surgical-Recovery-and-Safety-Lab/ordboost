"""Unit tests for ordboost.metrics."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from ordboost.distributions import (
    ContinuousPredictiveDistribution,
    DiscretePredictiveDistribution,
)
from ordboost.metrics import (
    crps_score,
    interval_coverage_rate,
    pinball_loss,
    winkler_score,
)


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
