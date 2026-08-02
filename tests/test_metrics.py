"""Unit tests for ordboost.metrics."""

import numpy as np
import pytest

from ordboost.distributions import PredictiveDistribution
from ordboost.metrics import crps_score, pinball_loss


class TestMetrics:
    """Tests for probabilistic metrics functions."""

    @pytest.fixture
    def perfect_distribution(self) -> tuple[np.ndarray, PredictiveDistribution]:
        """Fixture providing a deterministic distribution with perfect predictions."""
        classes = np.array([0, 10, 20])
        # PMF places 100% weight on true target
        pmf = np.array(
            [
                [1.0, 0.0, 0.0],  # true target = 0
                [0.0, 1.0, 0.0],  # true target = 10
                [0.0, 0.0, 1.0],  # true target = 20
            ]
        )
        y_true = np.array([0, 10, 20])
        dist = PredictiveDistribution(pmf=pmf, classes=classes)
        return y_true, dist

    def test_crps_perfect_predictions(
        self, perfect_distribution: tuple[np.ndarray, PredictiveDistribution]
    ) -> None:
        """Test that a perfect deterministic forecast yields CRPS = 0.0."""
        y_true, dist = perfect_distribution
        score = crps_score(y_true, dist)
        assert score == pytest.approx(0.0, abs=1e-7)

    def test_crps_known_value(self) -> None:
        """Test CRPS against a hand-calculated non-zero value."""
        classes = np.array([0, 10, 20])
        # PMF for 1 sample: [0.5, 0.3, 0.2] -> CDF = [0.5, 0.8, 1.0]
        # True target = 10 -> Empirical step I(10 <= c) = [0.0, 1.0, 1.0]
        # CDF diff = [0.5 - 0.0, 0.8 - 1.0, 1.0 - 1.0] = [0.5, -0.2, 0.0]
        # Squared diff = [0.25, 0.04, 0.0] -> CRPS sum = 0.29
        pmf = np.array([[0.5, 0.3, 0.2]])
        y_true = np.array([10])
        dist = PredictiveDistribution(pmf=pmf, classes=classes)

        score = crps_score(y_true, dist)
        assert score == pytest.approx(0.29, abs=1e-6)

    def test_crps_sample_weights(self) -> None:
        """Test weighted CRPS calculation."""
        classes = np.array([0, 1])
        pmf = np.array(
            [
                [1.0, 0.0],  # perfect -> CRPS = 0.0
                [
                    0.0,
                    1.0,
                ],  # wrong (true target is 0) -> CDF=[0,1], Step=[1,1] -> CRPS = 1.0
            ]
        )
        y_true = np.array([0, 0])
        dist = PredictiveDistribution(pmf=pmf, classes=classes)

        weights = np.array([3.0, 1.0])  # 3:1 weight ratio
        score = crps_score(y_true, dist, sample_weight=weights)
        # Expected: (3*0.0 + 1*1.0) / 4 = 0.25
        assert score == pytest.approx(0.25, abs=1e-6)

    def test_crps_missing_class_error(self) -> None:
        """Test error when y_true contains values not in dist.classes."""
        classes = np.array([0, 10])
        pmf = np.ones((1, 2)) * 0.5
        dist = PredictiveDistribution(pmf=pmf, classes=classes)

        # 99 is not in [0, 10]
        with pytest.raises(ValueError, match="not present in y_dist.classes"):
            crps_score(np.array([99]), dist)

    def test_pinball_loss_known_values(self) -> None:
        """Test pinball loss against manual calculation."""
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([12.0, 7.0])  # over-predicted (+2), under-predicted (-3)
        q = 0.8

        # Sample 0 (over-prediction, err = -2): max(0.8*-2, -0.2*-2) = max(-1.6, 0.4) = 0.4
        # Sample 1 (under-prediction, err = +3): max(0.8*3, -0.2*3) = max(2.4, -0.6) = 2.4
        # Mean pinball loss = (0.4 + 2.4) / 2 = 1.4
        loss = pinball_loss(y_true, y_pred, q=q)
        assert loss == pytest.approx(1.4, abs=1e-6)

    def test_pinball_loss_invalid_q(self) -> None:
        """Test that invalid q levels raise ValueError."""
        with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
            pinball_loss([1], [1], q=0.0)

        with pytest.raises(ValueError, match="strictly between 0.0 and 1.0"):
            pinball_loss([1], [1], q=1.0)
