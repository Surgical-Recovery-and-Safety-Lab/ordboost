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
        pmf = np.array(
            [
                [1.0, 0.0, 0.0],  # target = 0
                [0.0, 1.0, 0.0],  # target = 10
                [0.0, 0.0, 1.0],  # target = 20
            ]
        )
        y_true = np.array([0, 10, 20])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)
        return y_true, dist

    # --- Discrete CRPS Tests ---

    def test_crps_discrete_perfect_predictions(
        self, perfect_discrete_dist: tuple[np.ndarray, DiscretePredictiveDistribution]
    ) -> None:
        """Test that a perfect deterministic discrete forecast yields CRPS = 0.0."""
        y_true, dist = perfect_discrete_dist
        score = crps_score(y_true, dist)
        assert score == pytest.approx(0.0, abs=1e-7)

    def test_crps_discrete_known_value(self) -> None:
        """Test discrete CRPS against a hand-calculated non-zero value."""
        classes = np.array([0, 10, 20])
        pmf = np.array([[0.5, 0.3, 0.2]])
        y_true = np.array([10])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)

        # CDF = [0.5, 0.8, 1.0], Step I(10 <= c) = [0.0, 1.0, 1.0]
        # Diff^2 = [0.25, 0.04, 0.0] -> CRPS sum = 0.29
        score = crps_score(y_true, dist)
        assert score == pytest.approx(0.29, abs=1e-6)

    def test_crps_discrete_sample_weights(self) -> None:
        """Test weighted discrete CRPS calculation."""
        classes = np.array([0, 1])
        pmf = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_true = np.array([0, 0])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)

        weights = np.array([3.0, 1.0])
        # Expected: (3*0.0 + 1*1.0) / 4 = 0.25
        score = crps_score(y_true, dist, sample_weight=weights)
        assert score == pytest.approx(0.25, abs=1e-6)

    def test_crps_discrete_missing_class_error(self) -> None:
        """Test error when y_true contains values not in discrete dist.classes."""
        classes = np.array([0, 10])
        pmf = np.ones((1, 2)) * 0.5
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)

        with pytest.raises(ValueError, match="not present in y_dist.classes"):
            crps_score(np.array([99]), dist)

    # --- Continuous CRPS Tests ---

    def test_crps_continuous_known_value(self) -> None:
        """Test continuous CRPS calculation via trapezoidal integration against manual value."""
        grid_y = np.array([0.0, 5.0, 10.0])
        grid_cdf = np.array([[0.0, 0.5, 1.0], [0.1, 0.8, 1.0]])
        y_true = np.array([5.0, 0.0])

        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.grid_y = grid_y
        dist.grid_cdf = grid_cdf

        # Sample 0: y_true=5.0 -> Step=[0, 1, 1], cdf=[0, 0.5, 1] -> diff^2=[0, 0.25, 0]
        #   trapz areas: 0.5*(0+0.25)*5 = 0.625; 0.5*(0.25+0)*5 = 0.625 -> sum = 1.25
        # Sample 1: y_true=0.0 -> Step=[1, 1, 1], cdf=[0.1, 0.8, 1] -> diff^2=[0.81, 0.04, 0]
        #   trapz areas: 0.5*(0.81+0.04)*5 = 2.125; 0.5*(0.04+0)*5 = 0.1 -> sum = 2.225
        # Mean CRPS = (1.25 + 2.225) / 2 = 1.7375
        score = crps_score(y_true, dist)
        assert score == pytest.approx(1.7375, abs=1e-6)

    def test_crps_continuous_sample_weights(self) -> None:
        """Test weighted continuous CRPS calculation."""
        grid_y = np.array([0.0, 5.0, 10.0])
        grid_cdf = np.array([[0.0, 0.5, 1.0], [0.1, 0.8, 1.0]])
        y_true = np.array([5.0, 0.0])

        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.grid_y = grid_y
        dist.grid_cdf = grid_cdf

        weights = np.array([3.0, 1.0])
        # Expected: (3 * 1.25 + 1 * 2.225) / 4.0 = 5.975 / 4 = 1.49375
        score = crps_score(y_true, dist, sample_weight=weights)
        assert score == pytest.approx(1.49375, abs=1e-6)

    # --- Common Input Validation Tests ---

    def test_crps_invalid_y_true_ndim(self) -> None:
        """Test error when y_true is not 1D."""
        dist = MagicMock(spec=DiscretePredictiveDistribution)
        with pytest.raises(ValueError, match="1D array"):
            crps_score(np.array([[1, 2], [3, 4]]), dist)

    def test_crps_sample_count_mismatch(self) -> None:
        """Test error when sample count in y_true mismatches y_dist."""
        dist = MagicMock(spec=ContinuousPredictiveDistribution)
        dist.grid_cdf = np.ones((5, 10))
        dist.grid_y = np.linspace(0, 10, 10)

        with pytest.raises(ValueError, match="Sample count mismatch"):
            crps_score(np.array([1, 2, 3]), dist)

    def test_crps_invalid_sample_weight_shape(self) -> None:
        """Test error when sample_weight shape does not match y_true."""
        classes = np.array([0, 1])
        pmf = np.array([[1.0, 0.0]])
        y_true = np.array([0])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)

        with pytest.raises(ValueError, match="Expected 'sample_weight' shape"):
            crps_score(y_true, dist, sample_weight=np.array([1.0, 2.0]))


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
