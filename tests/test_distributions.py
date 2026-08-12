"""Unit tests for ordboost.distributions."""

import numpy as np
import pytest

from ordboost.distributions import PredictiveDistribution


class TestPredictiveDistribution:
    """Tests for the PredictiveDistribution class."""

    @pytest.fixture
    def sample_distribution(self) -> PredictiveDistribution:
        """Fixture providing a known 3-sample, 4-class distribution."""
        # Classes: [0, 10, 20, 30]
        # Sample 0: Heavy left tail
        # Sample 1: Symmetric around 10/20
        # Sample 2: Heavy right tail
        pmf = np.array(
            [
                [0.70, 0.20, 0.10, 0.00],
                [0.10, 0.40, 0.40, 0.10],
                [0.00, 0.05, 0.15, 0.80],
            ]
        )
        classes = np.array([0, 10, 20, 30])
        return PredictiveDistribution(pmf=pmf, classes=classes)

    def test_initialization_invalid_shapes(self) -> None:
        """Test that invalid matrix shapes raise ValueError."""
        # 1D PMF
        with pytest.raises(ValueError, match="2D array"):
            PredictiveDistribution(pmf=np.array([0.5, 0.5]), classes=np.array([0, 1]))

        # 2D classes
        with pytest.raises(ValueError, match="1D array"):
            PredictiveDistribution(pmf=np.ones((2, 2)), classes=np.ones((2, 2)))

        # Mismatched dimensions
        with pytest.raises(ValueError, match="Mismatch"):
            PredictiveDistribution(pmf=np.ones((3, 4)), classes=np.array([0, 1, 2]))

    def test_cdf_property(self, sample_distribution: PredictiveDistribution) -> None:
        """Test cumulative distribution function values."""
        cdf = sample_distribution.cdf
        expected_cdf = np.array(
            [
                [0.70, 0.90, 1.00, 1.00],
                [0.10, 0.50, 0.90, 1.00],
                [0.00, 0.05, 0.20, 1.00],
            ]
        )
        np.testing.assert_allclose(cdf, expected_cdf, atol=1e-6)

    def test_mean(self, sample_distribution: PredictiveDistribution) -> None:
        """Test expected value calculation."""
        means = sample_distribution.mean()
        # Sample 0: 0.7*0 + 0.2*10 + 0.1*20 + 0.0*30 = 4.0
        # Sample 1: 0.1*0 + 0.4*10 + 0.4*20 + 0.1*30 = 15.0
        # Sample 2: 0.0*0 + 0.05*10 + 0.15*20 + 0.8*30 = 27.5
        expected_means = np.array([4.0, 15.0, 27.5])
        np.testing.assert_allclose(means, expected_means, atol=1e-6)

    def test_ppf_scalar_and_array(
        self, sample_distribution: PredictiveDistribution
    ) -> None:
        """Test percent point function (quantiles) for scalar and array inputs."""
        # Single quantile (median)
        medians = sample_distribution.ppf(0.5)
        # Sample 0: CDF reaches 0.5 at class 0 (cdf=0.7)
        # Sample 1: CDF reaches 0.5 at class 10 (cdf=0.5)
        # Sample 2: CDF reaches 0.5 at class 30 (cdf=1.0)
        np.testing.assert_array_equal(medians, np.array([0, 10, 30]))

        # Array of quantiles
        quantiles = np.array([0.1, 0.9])
        results = sample_distribution.ppf(quantiles)
        assert results.shape == (3, 2)

    def test_ppf_invalid_quantiles(
        self, sample_distribution: PredictiveDistribution
    ) -> None:
        """Test that quantiles outside [0, 1] raise ValueError."""
        with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
            sample_distribution.ppf(-0.1)

        with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
            sample_distribution.ppf(1.05)

    def test_interval(self, sample_distribution: PredictiveDistribution) -> None:
        """Test prediction interval calculation."""
        lower, upper = sample_distribution.interval(
            alpha=0.20
        )  # 80% interval (10th to 90th percentile)
        assert lower.shape == (3,)
        assert upper.shape == (3,)
        assert np.all(lower <= upper)
