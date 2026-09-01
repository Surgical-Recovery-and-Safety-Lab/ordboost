"""Unit tests for the PredictiveDistribution abstract base class."""

import numpy as np
import pytest

from ordboost.distributions import PredictiveDistribution


class DummyDistribution(PredictiveDistribution):
    """Minimal concrete PredictiveDistribution for testing shared base
    class behaviour (ppf validation/delegation, median, interval).

    `_ppf` returns a deterministic, checkable function of `q_arr` (scaled
    by 100) rather than anything statistically meaningful, so tests can
    verify exactly what was passed through from the public `ppf` method.
    """

    def __init__(self, n_samples: int = 3) -> None:
        self.n_samples = n_samples

    def mean(self) -> np.ndarray:
        """Return a fixed, checkable per-sample mean."""
        return np.arange(self.n_samples, dtype=float)

    def _ppf(self, q_arr: np.ndarray) -> np.ndarray:
        """Return q_arr * 100, broadcast across samples, for verification."""
        if q_arr.ndim == 0:
            return np.full(self.n_samples, float(q_arr) * 100.0)
        return np.tile(q_arr * 100.0, (self.n_samples, 1))


class TestPredictiveDistributionABC:
    """Tests for abstract base class instantiation restrictions."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """Test that instantiating PredictiveDistribution directly raises
        TypeError, since mean() and _ppf() are abstract."""
        with pytest.raises(TypeError):
            PredictiveDistribution()  # type: ignore[abstract]


class TestPpf:
    """Tests for PredictiveDistribution.ppf (shared validation + delegation)."""

    def test_scalar_delegates_to_ppf_underscore(self) -> None:
        """Test that a scalar q is passed through to _ppf and its output
        (a 1D array) is returned unmodified."""
        dist = DummyDistribution(n_samples=3)
        result = dist.ppf(0.5)
        np.testing.assert_allclose(result, [50.0, 50.0, 50.0])

    def test_array_delegates_to_ppf_underscore(self) -> None:
        """Test that a 1D array q is passed through to _ppf and its
        output (a 2D array) is returned unmodified."""
        dist = DummyDistribution(n_samples=2)
        result = dist.ppf(np.array([0.1, 0.9]))
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0], [10.0, 90.0])

    def test_negative_quantile_raises(self) -> None:
        """Test that a quantile below 0.0 raises ValueError."""
        dist = DummyDistribution()
        with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
            dist.ppf(-0.1)

    def test_quantile_above_one_raises(self) -> None:
        """Test that a quantile above 1.0 raises ValueError."""
        dist = DummyDistribution()
        with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
            dist.ppf(1.05)

    def test_boundary_values_accepted(self) -> None:
        """Test that q=0.0 and q=1.0 are valid and do not raise."""
        dist = DummyDistribution(n_samples=1)
        np.testing.assert_allclose(dist.ppf(0.0), [0.0])
        np.testing.assert_allclose(dist.ppf(1.0), [100.0])

    def test_one_element_array_within_range_does_not_raise(self) -> None:
        """Test that an array q with any single out-of-range element
        raises, isolating the elementwise range check."""
        dist = DummyDistribution()
        with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
            dist.ppf(np.array([0.1, 0.5, 1.5]))

    def test_2d_q_raises_value_error(self) -> None:
        """Test that a 2D q array raises ValueError before reaching _ppf,
        per the shape contract stated in ppf's docstring."""
        dist = DummyDistribution()
        with pytest.raises(ValueError, match="scalar or 1D array"):
            dist.ppf(np.array([[0.1, 0.5], [0.2, 0.6]]))

    def test_scalar_result_is_ndarray(self) -> None:
        """Test that ppf's return type is an ndarray, not a Python list
        or scalar, even for scalar q."""
        dist = DummyDistribution()
        assert isinstance(dist.ppf(0.5), np.ndarray)


class TestMedian:
    """Tests for PredictiveDistribution.median."""

    def test_median_equals_ppf_at_half(self) -> None:
        """Test that median() returns the same result as ppf(0.5)."""
        dist = DummyDistribution(n_samples=4)
        np.testing.assert_allclose(dist.median(), dist.ppf(0.5))

    def test_median_output_shape(self) -> None:
        """Test that median() returns one value per sample."""
        dist = DummyDistribution(n_samples=5)
        assert dist.median().shape == (5,)


class TestInterval:
    """Tests for PredictiveDistribution.interval."""

    def test_returns_lower_upper_tuple(self) -> None:
        """Test that interval() returns a 2-tuple of arrays."""
        dist = DummyDistribution(n_samples=3)
        result = dist.interval(alpha=0.10)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_uses_correct_quantile_levels(self) -> None:
        """Test that interval(alpha) queries ppf at exactly alpha/2 and
        1 - alpha/2, verified via DummyDistribution's deterministic
        q_arr * 100 output."""
        dist = DummyDistribution(n_samples=1)
        lower, upper = dist.interval(alpha=0.20)
        # alpha=0.20 -> lower_q=0.10, upper_q=0.90 -> *100 -> 10.0, 90.0
        np.testing.assert_allclose(lower, [10.0])
        np.testing.assert_allclose(upper, [90.0])

    def test_default_alpha(self) -> None:
        """Test the default alpha=0.10 produces the expected 5th/95th
        percentile query levels."""
        dist = DummyDistribution(n_samples=1)
        lower, upper = dist.interval()
        np.testing.assert_allclose(lower, [5.0])
        np.testing.assert_allclose(upper, [95.0])

    def test_output_shapes(self) -> None:
        """Test that both bounds have one value per sample."""
        dist = DummyDistribution(n_samples=6)
        lower, upper = dist.interval(alpha=0.10)
        assert lower.shape == (6,)
        assert upper.shape == (6,)

    def test_alpha_zero_raises(self) -> None:
        """Test that alpha=0.0 raises ValueError (open interval bound)."""
        dist = DummyDistribution()
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            dist.interval(alpha=0.0)

    def test_alpha_one_raises(self) -> None:
        """Test that alpha=1.0 raises ValueError (open interval bound)."""
        dist = DummyDistribution()
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            dist.interval(alpha=1.0)

    def test_negative_alpha_raises(self) -> None:
        """Test that a negative alpha raises ValueError."""
        dist = DummyDistribution()
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            dist.interval(alpha=-0.1)

    def test_alpha_above_one_raises(self) -> None:
        """Test that alpha > 1.0 raises ValueError."""
        dist = DummyDistribution()
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            dist.interval(alpha=1.5)


class TestValidateStrictlyAscending:
    """Tests for PredictiveDistribution._validate_strictly_ascending."""

    def test_strictly_ascending_does_not_raise(self) -> None:
        """Test that a strictly increasing array passes validation."""
        PredictiveDistribution._validate_strictly_ascending(
            "test_array", np.array([1.0, 2.0, 5.0, 10.0])
        )

    def test_descending_raises(self) -> None:
        """Test that a strictly decreasing array raises ValueError."""
        with pytest.raises(ValueError, match="'test_array' must be strictly ascending"):
            PredictiveDistribution._validate_strictly_ascending(
                "test_array", np.array([10.0, 5.0, 1.0])
            )

    def test_repeated_consecutive_values_raises(self) -> None:
        """Test that a tie between consecutive values raises ValueError,
        since strictly ascending excludes equal neighbours."""
        with pytest.raises(ValueError, match="strictly ascending"):
            PredictiveDistribution._validate_strictly_ascending(
                "test_array", np.array([1.0, 2.0, 2.0, 3.0])
            )

    def test_single_non_monotonic_pair_raises(self) -> None:
        """Test that a single out-of-order pair amid otherwise ascending
        values still raises ValueError."""
        with pytest.raises(ValueError, match="strictly ascending"):
            PredictiveDistribution._validate_strictly_ascending(
                "test_array", np.array([1.0, 2.0, 1.5, 4.0])
            )

    def test_single_element_array_does_not_raise(self) -> None:
        """Test that a single-element array is trivially valid, since
        there are no consecutive pairs to compare."""
        PredictiveDistribution._validate_strictly_ascending(
            "test_array", np.array([5.0])
        )

    def test_error_message_includes_provided_name(self) -> None:
        """Test that the error message uses the caller-supplied name,
        so validation errors from different subclasses are distinguishable."""
        with pytest.raises(ValueError, match="'my_custom_field'"):
            PredictiveDistribution._validate_strictly_ascending(
                "my_custom_field", np.array([3.0, 1.0])
            )
