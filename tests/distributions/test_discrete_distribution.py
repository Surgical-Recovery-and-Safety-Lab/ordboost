"""Unit tests for DiscretePredictiveDistribution in ordboost.distributions."""

import numpy as np
import pytest

from ordboost.distributions import DiscretePredictiveDistribution


class TestInit:
    """Tests for DiscretePredictiveDistribution.__init__ validation."""

    def test_valid_construction_stores_data(self) -> None:
        """Test that a valid pmf/classes pair constructs without error
        and stores the expected values."""
        pmf = np.array([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]])
        classes = np.array([0, 10, 20])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)
        np.testing.assert_array_equal(dist.pmf, pmf)
        np.testing.assert_array_equal(dist.classes, classes)

    def test_non_2d_pmf_raises(self) -> None:
        """Test that a 1D pmf raises ValueError."""
        with pytest.raises(ValueError, match="'pmf' to be a 2D"):
            DiscretePredictiveDistribution(
                pmf=np.array([0.5, 0.5]), classes=np.array([0, 1])
            )

    def test_non_1d_classes_raises(self) -> None:
        """Test that a 2D classes array raises ValueError."""
        with pytest.raises(ValueError, match="'classes' to be a 1D"):
            DiscretePredictiveDistribution(
                pmf=np.ones((2, 2)) / 2, classes=np.ones((2, 2))
            )

    def test_shape_mismatch_raises(self) -> None:
        """Test that mismatched pmf columns and classes length raises ValueError."""
        pmf = np.full((3, 4), 0.25)
        with pytest.raises(ValueError, match="Mismatch between PMF class"):
            DiscretePredictiveDistribution(pmf=pmf, classes=np.array([0, 1, 2]))

    def test_non_ascending_classes_raises(self) -> None:
        """Test that a non-strictly-ascending classes array raises ValueError."""
        pmf = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="strictly ascending"):
            DiscretePredictiveDistribution(pmf=pmf, classes=np.array([10, 0]))

    def test_repeated_class_values_raises(self) -> None:
        """Test that duplicate (tied) class values raise ValueError, since
        ascending order must be strict, not non-decreasing."""
        pmf = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="strictly ascending"):
            DiscretePredictiveDistribution(pmf=pmf, classes=np.array([10, 10]))

    def test_pmf_row_summing_below_one_raises(self) -> None:
        """Test that a pmf row summing meaningfully below 1.0 raises ValueError."""
        pmf = np.array([[0.5, 0.3]])  # sums to 0.8
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DiscretePredictiveDistribution(pmf=pmf, classes=np.array([0, 1]))

    def test_pmf_row_summing_above_one_raises(self) -> None:
        """Test that a pmf row summing meaningfully above 1.0 raises ValueError."""
        pmf = np.array([[0.7, 0.7]])  # sums to 1.4
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DiscretePredictiveDistribution(pmf=pmf, classes=np.array([0, 1]))

    def test_pmf_row_within_floating_point_tolerance_does_not_raise(self) -> None:
        """Test that a pmf row summing to 1.0 within floating-point error
        (not exactly 1.0 due to accumulation) is accepted."""
        pmf = np.array([[0.1, 0.2, 0.7 + 1e-12]])
        DiscretePredictiveDistribution(pmf=pmf, classes=np.array([0, 1, 2]))

    def test_multi_row_pmf_only_one_bad_row_raises(self) -> None:
        """Test that a single non-normalized row among otherwise valid
        rows still raises ValueError."""
        pmf = np.array([[0.5, 0.5], [0.3, 0.3]])  # second row sums to 0.6
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DiscretePredictiveDistribution(pmf=pmf, classes=np.array([0, 1]))

    def test_pmf_is_read_only_after_construction(self) -> None:
        """Test that dist.pmf cannot be mutated in place after construction."""
        dist = DiscretePredictiveDistribution(
            pmf=np.array([[0.5, 0.5]]), classes=np.array([0, 1])
        )
        with pytest.raises(ValueError, match="read-only"):
            dist.pmf[0, 0] = 0.9

    def test_classes_is_read_only_after_construction(self) -> None:
        """Test that dist.classes cannot be mutated in place after construction."""
        dist = DiscretePredictiveDistribution(
            pmf=np.array([[0.5, 0.5]]), classes=np.array([0, 1])
        )
        with pytest.raises(ValueError, match="read-only"):
            dist.classes[0] = 99

    def test_user_pmf_is_not_read_only_after_construction(self) -> None:
        """Test that dist.pmf cannot be mutated in place after construction."""
        pmf = np.array([[0.5, 0.5]])
        DiscretePredictiveDistribution(pmf=pmf, classes=np.array([0, 1]))
        pmf[0, 0] = 0.2
        assert pmf[0, 0] == 0.2

    def test_user_classes_is_not_read_only_after_construction(self) -> None:
        """Test that dist.classes cannot be mutated in place after construction."""
        classes = np.array([0, 1])
        DiscretePredictiveDistribution(pmf=np.array([[0.5, 0.5]]), classes=classes)

        classes[0] = 1
        assert classes[0] == 1


class TestCdf:
    """Tests for DiscretePredictiveDistribution.cdf."""

    @pytest.fixture
    def sample_distribution(self) -> DiscretePredictiveDistribution:
        """Fixture providing a known 3-sample discrete distribution."""
        pmf = np.array(
            [
                [0.70, 0.20, 0.10, 0.00],
                [0.10, 0.40, 0.40, 0.10],
                [0.00, 0.05, 0.15, 0.80],
            ]
        )
        classes = np.array([0, 10, 20, 30])
        return DiscretePredictiveDistribution(pmf=pmf, classes=classes)

    def test_cdf_values(self, sample_distribution) -> None:
        """Test cumulative distribution function values match a hand
        computation of the cumulative sum."""
        expected_cdf = np.array(
            [
                [0.70, 0.90, 1.00, 1.00],
                [0.10, 0.50, 0.90, 1.00],
                [0.00, 0.05, 0.20, 1.00],
            ]
        )
        np.testing.assert_allclose(sample_distribution.cdf, expected_cdf, atol=1e-6)

    def test_cdf_is_cached(self, sample_distribution) -> None:
        """Test that repeated access returns the same cached array object,
        not a freshly recomputed one."""
        first = sample_distribution.cdf
        second = sample_distribution.cdf
        assert first is second

    def test_final_column_forced_to_exactly_one(self) -> None:
        """Test that the final CDF column is exactly 1.0 even when the
        pmf row's floating-point sum falls fractionally short, guarding
        against the ppf(1.0) wrong-class edge case."""
        # Construct a row that sums to 1.0 within tolerance but not exactly,
        # so cumsum's last entry could land at e.g. 0.999999999999.
        pmf = np.array([[0.1, 0.2, 0.7 - 1e-13]])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=np.array([0, 1, 2]))
        assert dist.cdf[0, -1] == 1.0


class TestMean:
    """Tests for DiscretePredictiveDistribution.mean."""

    def test_mean_matches_hand_computation(self) -> None:
        """Test expected value calculation in physical class units."""
        pmf = np.array(
            [
                [0.70, 0.20, 0.10, 0.00],
                [0.10, 0.40, 0.40, 0.10],
                [0.00, 0.05, 0.15, 0.80],
            ]
        )
        classes = np.array([0, 10, 20, 30])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)
        expected_means = np.array([4.0, 15.0, 27.5])
        np.testing.assert_allclose(dist.mean(), expected_means, atol=1e-6)

    def test_mean_output_shape(self) -> None:
        """Test that mean() returns one value per sample."""
        pmf = np.full((5, 2), 0.5)
        classes = np.array([0, 1])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)
        assert dist.mean().shape == (5,)


class TestPpf:
    """Tests for DiscretePredictiveDistribution._ppf, exercised via the
    shared public ppf() (validation is covered by
    test_predictive_distribution.py; these focus on the discrete lookup
    logic itself).
    """

    @pytest.fixture
    def sample_distribution(self) -> DiscretePredictiveDistribution:
        """Fixture providing a known 3-sample discrete distribution."""
        pmf = np.array(
            [
                [0.70, 0.20, 0.10, 0.00],
                [0.10, 0.40, 0.40, 0.10],
                [0.00, 0.05, 0.15, 0.80],
            ]
        )
        classes = np.array([0, 10, 20, 30])
        return DiscretePredictiveDistribution(pmf=pmf, classes=classes)

    def test_scalar_quantile_median(self, sample_distribution) -> None:
        """Test ppf at a scalar quantile (the median)."""
        medians = sample_distribution.ppf(0.5)
        np.testing.assert_array_equal(medians, np.array([0, 10, 30]))

    def test_array_quantile_output_shape(self, sample_distribution) -> None:
        """Test that an array of quantiles returns a
        (n_samples, n_quantiles) shaped result."""
        results = sample_distribution.ppf(np.array([0.1, 0.9]))
        assert results.shape == (3, 2)

    def test_ppf_at_zero_returns_lowest_class(self) -> None:
        """Test that ppf(0.0) returns the first class whose cumulative
        probability reaches 0.0, i.e. the first class."""
        pmf = np.array([[0.0, 0.0, 1.0]])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=np.array([0, 1, 2]))
        assert dist.ppf(0.0)[0] == 0

    def test_ppf_at_one_returns_highest_class(self) -> None:
        """Test that ppf(1.0) returns the highest class, including for a
        pmf row whose floating-point cumsum falls fractionally short of
        1.0 -- the exact scenario the forced cdf[:, -1] = 1.0 fix guards
        against."""
        pmf = np.array([[0.1, 0.2, 0.7 - 1e-13]])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=np.array([0, 1, 2]))
        assert dist.ppf(1.0)[0] == 2

    def test_ppf_exact_grid_match(self, sample_distribution) -> None:
        """Test ppf at a quantile that exactly matches a CDF breakpoint."""
        # Sample 0's CDF is [0.70, 0.90, 1.00, 1.00]; q=0.70 should return
        # class 0 (the first class whose CDF >= 0.70).
        result = sample_distribution.ppf(0.70)
        assert result[0] == 0

    def test_ppf_scalar_result_matches_array_result_at_same_level(
        self, sample_distribution
    ) -> None:
        """Test that ppf(q) as a scalar matches column q of ppf([q, ...])
        for consistency between the two branches of _ppf."""
        scalar_result = sample_distribution.ppf(0.5)
        array_result = sample_distribution.ppf(np.array([0.5, 0.9]))
        np.testing.assert_array_equal(scalar_result, array_result[:, 0])


class TestMedian:
    """Integration tests for the inherited PredictiveDistribution.median()
    against a real DiscretePredictiveDistribution (the base class's own
    tests only exercise median() via an abstract Dummy)."""

    def test_median_matches_ppf_at_half(self) -> None:
        """Test that median() equals ppf(0.5) for a real discrete
        distribution, consistent with the cdf fixture used elsewhere in
        this file (sample 0's cdf reaches 0.5 at class 0, sample 1 at
        class 10, sample 2 at class 30)."""
        pmf = np.array(
            [
                [0.70, 0.20, 0.10, 0.00],
                [0.10, 0.40, 0.40, 0.10],
                [0.00, 0.05, 0.15, 0.80],
            ]
        )
        classes = np.array([0, 10, 20, 30])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)
        np.testing.assert_array_equal(dist.median(), np.array([0, 10, 30]))
        np.testing.assert_array_equal(dist.median(), dist.ppf(0.5))


class TestInterval:
    """Integration tests for the inherited PredictiveDistribution.interval()
    against a real DiscretePredictiveDistribution (the base class's own
    tests only exercise interval() via an abstract Dummy)."""

    def test_interval_bounds_match_expected_classes(self) -> None:
        """Test that interval() returns the expected lower/upper class
        bounds for a real discrete distribution (10th/90th percentile
        classes at alpha=0.20), and that they correctly bracket the
        median for every sample."""
        pmf = np.array(
            [
                [0.70, 0.20, 0.10, 0.00],
                [0.10, 0.40, 0.40, 0.10],
                [0.00, 0.05, 0.15, 0.80],
            ]
        )
        classes = np.array([0, 10, 20, 30])
        dist = DiscretePredictiveDistribution(pmf=pmf, classes=classes)

        lower, upper = dist.interval(alpha=0.20)
        np.testing.assert_array_equal(lower, np.array([0, 0, 20]))
        np.testing.assert_array_equal(upper, np.array([20, 20, 30]))
        assert np.all(lower <= dist.median())
        assert np.all(dist.median() <= upper)
