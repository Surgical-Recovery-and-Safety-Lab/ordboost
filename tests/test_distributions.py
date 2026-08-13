"""Unit tests for ordboost.distributions."""

import numpy as np
import pytest

from ordboost.distributions import (
    DiscretePredictiveDistribution,
    PredictiveDistribution,
)


class DummyPredictiveDistribution(PredictiveDistribution):
    """Dummy implementation of PredictiveDistribution for testing ABC behavior."""

    def mean(self) -> np.ndarray:
        """Return dummy expected value."""
        return np.array([10.0, 20.0])

    def ppf(self, q: float | np.ndarray) -> np.ndarray:
        """Return dummy quantile predictions."""
        q_arr = np.asarray(q, dtype=float)
        if q_arr.ndim == 0:
            return np.array([10.0 * q_arr, 20.0 * q_arr])
        return np.array(
            [
                [10.0 * q_val for q_val in q_arr],
                [20.0 * q_val for q_val in q_arr],
            ]
        )


class TestPredictiveDistributionABC:
    """Tests for the abstract base class PredictiveDistribution."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """Verify that instantiating PredictiveDistribution directly raises TypeError."""
        with pytest.raises(TypeError):
            PredictiveDistribution()  # type: ignore[abstract]

    def test_inherited_median(self) -> None:
        """Verify that inherited median() calls ppf(0.5) correctly."""
        dist = DummyPredictiveDistribution()
        np.testing.assert_allclose(dist.median(), np.array([5.0, 10.0]))

    def test_inherited_interval_success(self) -> None:
        """Verify interval calculation on concrete distribution."""
        dist = DummyPredictiveDistribution()
        lower, upper = dist.interval(alpha=0.20)
        np.testing.assert_allclose(lower, np.array([1.0, 2.0]))
        np.testing.assert_allclose(upper, np.array([9.0, 18.0]))

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_inherited_interval_invalid_alpha(self, alpha: float) -> None:
        """Verify that significance levels outside (0.0, 1.0) raise ValueError."""
        dist = DummyPredictiveDistribution()
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            dist.interval(alpha=alpha)


class TestDiscretePredictiveDistribution:
    """Tests for the DiscretePredictiveDistribution class."""

    @pytest.fixture
    def sample_distribution(self) -> DiscretePredictiveDistribution:
        """Fixture providing a known 3-sample, 4-class discrete distribution."""
        pmf = np.array(
            [
                [0.70, 0.20, 0.10, 0.00],
                [0.10, 0.40, 0.40, 0.10],
                [0.00, 0.05, 0.15, 0.80],
            ]
        )
        classes = np.array([0, 10, 20, 30])
        return DiscretePredictiveDistribution(pmf=pmf, classes=classes)

    def test_initialization_invalid_shapes(self) -> None:
        """Test that invalid matrix shapes raise ValueError."""
        # 1D PMF
        with pytest.raises(ValueError, match="'pmf' to be a 2D"):
            DiscretePredictiveDistribution(
                pmf=np.array([0.5, 0.5]), classes=np.array([0, 1])
            )

        # 2D classes
        with pytest.raises(ValueError, match="'classes' to be a 1D"):
            DiscretePredictiveDistribution(pmf=np.ones((2, 2)), classes=np.ones((2, 2)))

        # Mismatched dimensions
        with pytest.raises(ValueError, match="Mismatch between PMF class"):
            DiscretePredictiveDistribution(
                pmf=np.ones((3, 4)), classes=np.array([0, 1, 2])
            )

    def test_cdf_property(
        self, sample_distribution: DiscretePredictiveDistribution
    ) -> None:
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

    def test_mean(self, sample_distribution: DiscretePredictiveDistribution) -> None:
        """Test expected value calculation in physical class units."""
        means = sample_distribution.mean()
        expected_means = np.array([4.0, 15.0, 27.5])
        np.testing.assert_allclose(means, expected_means, atol=1e-6)

    def test_ppf_scalar_and_array(
        self, sample_distribution: DiscretePredictiveDistribution
    ) -> None:
        """Test percent point function for scalar and array inputs."""
        # Single quantile (median)
        medians = sample_distribution.ppf(0.5)
        np.testing.assert_array_equal(medians, np.array([0, 10, 30]))

        # Array of quantiles
        quantiles = np.array([0.1, 0.9])
        results = sample_distribution.ppf(quantiles)
        assert results.shape == (3, 2)

    def test_ppf_invalid_quantiles(
        self, sample_distribution: DiscretePredictiveDistribution
    ) -> None:
        """Test that quantiles outside [0, 1] raise ValueError."""
        with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
            sample_distribution.ppf(-0.1)

        with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
            sample_distribution.ppf(1.05)

    def test_interval(
        self, sample_distribution: DiscretePredictiveDistribution
    ) -> None:
        """Test prediction interval calculation."""
        lower, upper = sample_distribution.interval(alpha=0.20)
        assert lower.shape == (3,)
        assert upper.shape == (3,)
        assert np.all(lower <= upper)
