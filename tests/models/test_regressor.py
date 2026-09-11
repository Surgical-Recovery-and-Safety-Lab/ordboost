"""Unit tests for OrdBoostRegressor in ordboost.models."""

from typing import cast

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV

from ordboost.distributions import ContinuousPredictiveDistribution
from ordboost.mappers import (
    BaseBinMapper,
    ContinuousBinMapper,
    EmpiricalMeanBinMapper,
    EmpiricalMedianBinMapper,
    QuantileBinMapper,
    UniformBinMapper,
)
from ordboost.models import OrdBoostRegressor


class TestInit:
    """Tests for OrdBoostRegressor.__init__ default parameter assignment."""

    def test_default_params(self) -> None:
        """Test default hyperparameter assignment during initialization."""
        reg = OrdBoostRegressor()
        assert reg.n_bins == 20
        assert reg.bin_edges is None
        assert reg.bin_strategy == "quantile"
        assert reg.mapper == "median"
        assert reg.mapper_kwargs is None
        assert reg.learning_rate == 0.1
        assert reg.max_iter == 100

    def test_kwargs_stored_separately(self) -> None:
        """Test that unrecognized keyword arguments are captured in self.kwargs
        rather than raising at construction time."""
        reg = OrdBoostRegressor(max_leaf_nodes=15, early_stopping=False)
        assert reg.kwargs == {"max_leaf_nodes": 15, "early_stopping": False}


class TestComputeBinEdges:
    """Tests for OrdBoostRegressor._compute_bin_edges."""

    def test_custom_bin_edges_valid(self) -> None:
        """Test that explicit valid bin_edges are returned unmodified."""
        edges = [0.0, 10.0, 20.0, 50.0]
        reg = OrdBoostRegressor(bin_edges=edges)
        resolved = reg._compute_bin_edges(np.array([1.0, 15.0, 40.0]))
        np.testing.assert_array_equal(resolved, edges)

    def test_custom_bin_edges_ignores_y(self) -> None:
        """Test that when bin_edges is explicitly set, y's values do not
        influence the resolved edges at all."""
        edges = [0.0, 10.0, 20.0]
        reg = OrdBoostRegressor(bin_edges=edges)
        resolved_a = reg._compute_bin_edges(np.array([100.0, 200.0]))
        resolved_b = reg._compute_bin_edges(np.array([-50.0]))
        np.testing.assert_array_equal(resolved_a, resolved_b)

    def test_custom_bin_edges_invalid_ndim_raises(self) -> None:
        """Test that a 2D bin_edges raises ValueError."""
        reg = OrdBoostRegressor(bin_edges=np.array([[0, 10], [10, 20]]))
        with pytest.raises(ValueError, match="1D array with >= 2 edges"):
            reg._compute_bin_edges(np.array([1.0, 2.0]))

    def test_custom_bin_edges_too_few_raises(self) -> None:
        """Test that fewer than 2 edges raises ValueError."""
        reg = OrdBoostRegressor(bin_edges=[10.0])
        with pytest.raises(ValueError, match=">= 2 edges"):
            reg._compute_bin_edges(np.array([1.0, 2.0]))

    def test_custom_bin_edges_non_monotonic_raises(self) -> None:
        """Test that non-monotonically increasing edges raise ValueError."""
        reg = OrdBoostRegressor(bin_edges=[0.0, 10.0, 5.0])
        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            reg._compute_bin_edges(np.array([1.0, 2.0]))

    def test_n_bins_too_small_raises(self) -> None:
        """Test that n_bins < 2 raises ValueError."""
        reg = OrdBoostRegressor(n_bins=1)
        with pytest.raises(ValueError, match="Parameter 'n_bins' must be >= 2"):
            reg._compute_bin_edges(np.array([1.0, 2.0]))

    def test_bin_strategy_quantile(self) -> None:
        """Test quantile bin strategy: interior edges are empirical quantiles
        of y at n_bins + 1 evenly spaced probability levels, with the two
        outer levels (0.0 and 1.0) dropped since they are not interior
        thresholds -- mirroring bin_strategy="uniform", which drops the
        corresponding [y.min(), y.max()] endpoints via its own [1:-1] slice."""
        reg = OrdBoostRegressor(n_bins=4, bin_strategy="quantile")
        y = np.linspace(0.0, 100.0, 101)
        edges = reg._compute_bin_edges(y)
        assert len(edges) == 3
        np.testing.assert_allclose(edges, [25.0, 50.0, 75.0])

    def test_bin_strategy_quantile_deduplicates_dense_regions(self) -> None:
        """Test that duplicate quantile edges (from heavily repeated y values)
        are deduplicated rather than producing zero-width bins."""
        reg = OrdBoostRegressor(n_bins=10, bin_strategy="quantile")
        y = np.concatenate([np.zeros(90), np.linspace(1.0, 10.0, 10)])
        edges = reg._compute_bin_edges(y)
        assert len(edges) == len(np.unique(edges))

    def test_bin_strategy_uniform(self) -> None:
        """Test uniform bin strategy: edges are evenly spaced across
        [y.min(), y.max()]."""
        reg = OrdBoostRegressor(n_bins=4, bin_strategy="uniform")
        y = np.array([0.0, 100.0])
        edges = reg._compute_bin_edges(y)
        assert len(edges) == 3
        np.testing.assert_allclose(edges, [25.0, 50.0, 75.0])

    def test_bin_strategy_invalid_raises(self) -> None:
        """Test that an unrecognized bin_strategy raises ValueError."""
        reg = OrdBoostRegressor(bin_strategy="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Invalid bin_strategy"):
            reg._compute_bin_edges(np.array([1.0, 2.0]))


class TestResolveMapper:
    """Tests for OrdBoostRegressor._resolve_mapper."""

    @pytest.mark.parametrize(
        ("shortcut", "expected_cls"),
        [
            ("median", EmpiricalMedianBinMapper),
            ("mean", EmpiricalMeanBinMapper),
            ("quantile", QuantileBinMapper),
            ("uniform", UniformBinMapper),
            ("continuous", ContinuousBinMapper),
        ],
    )
    def test_string_shortcuts_resolve_correctly(
        self, shortcut: str, expected_cls: type[BaseBinMapper]
    ) -> None:
        """Test that valid string shortcuts resolve to the correct mapper
        class, with bin_edges_ assigned."""
        reg = OrdBoostRegressor(mapper=shortcut)  # type: ignore[arg-type]
        reg.bin_edges_ = np.array([0.0, 10.0, 20.0])
        resolved = reg._resolve_mapper()

        assert isinstance(resolved, expected_cls)
        np.testing.assert_array_equal(resolved.bin_edges, reg.bin_edges_)

    def test_none_defaults_to_median(self) -> None:
        """Test that mapper=None resolves identically to mapper='median'."""
        reg = OrdBoostRegressor(mapper=None)
        reg.bin_edges_ = np.array([0.0, 10.0, 20.0])
        resolved = reg._resolve_mapper()
        assert isinstance(resolved, EmpiricalMedianBinMapper)

    def test_mapper_kwargs_forwarded_to_string_shortcut(self) -> None:
        """Test that mapper_kwargs are passed through when instantiating a
        string-shortcut mapper."""
        reg = OrdBoostRegressor(mapper="continuous", mapper_kwargs={"resolution": 2.0})
        reg.bin_edges_ = np.array([0.0, 10.0, 20.0])
        resolved = cast(ContinuousBinMapper, reg._resolve_mapper())
        assert resolved.resolution == 2.0

    def test_unconfigured_custom_instance_is_cloned_and_assigned_edges(self) -> None:
        """Test that a BaseBinMapper instance with no bin_edges of its own
        is cloned (not mutated in place) and assigned bin_edges_."""
        custom_mapper = EmpiricalMeanBinMapper(bin_edges=None)
        reg = OrdBoostRegressor(mapper=custom_mapper)
        reg.bin_edges_ = np.array([0.0, 5.0, 10.0])
        resolved = reg._resolve_mapper()

        assert isinstance(resolved, EmpiricalMeanBinMapper)
        assert resolved is not custom_mapper
        assert custom_mapper.bin_edges is None  # original left untouched
        np.testing.assert_array_equal(resolved.bin_edges, reg.bin_edges_)

    def test_custom_instance_with_matching_edges_does_not_raise(self) -> None:
        """Test that a mapper instance pre-configured with bin_edges
        matching bin_edges_ exactly is accepted without error."""
        matching_edges = np.array([0.0, 5.0, 10.0])
        custom_mapper = QuantileBinMapper(bin_edges=matching_edges.tolist())
        reg = OrdBoostRegressor(mapper=custom_mapper)
        reg.bin_edges_ = matching_edges
        resolved = reg._resolve_mapper()
        np.testing.assert_array_equal(resolved.bin_edges, matching_edges)

    def test_custom_instance_with_conflicting_edges_raises(self) -> None:
        """Test that a mapper instance pre-configured with bin_edges that
        differ from bin_edges_ raises ValueError, since OrdBoostRegressor
        is the single source of truth for bin edges."""
        custom_mapper = QuantileBinMapper(bin_edges=[0.0, 3.0, 6.0, 9.0])
        reg = OrdBoostRegressor(mapper=custom_mapper)
        reg.bin_edges_ = np.array([0.0, 5.0, 10.0])
        with pytest.raises(ValueError, match="already has 'bin_edges' set"):
            reg._resolve_mapper()

    def test_unknown_string_shortcut_raises(self) -> None:
        """Test that an unrecognized string shortcut raises ValueError."""
        reg = OrdBoostRegressor(mapper="unknown_shortcut")  # type: ignore[arg-type]
        reg.bin_edges_ = np.array([0.0, 10.0, 20.0])
        with pytest.raises(ValueError, match="Unknown mapper shortcut"):
            reg._resolve_mapper()

    def test_invalid_type_raises(self) -> None:
        """Test that a mapper value that is neither a string nor a
        BaseBinMapper instance raises ValueError."""
        reg = OrdBoostRegressor(mapper=12345)  # type: ignore[arg-type]
        reg.bin_edges_ = np.array([0.0, 10.0, 20.0])
        with pytest.raises(ValueError, match="Expected 'mapper' to be a valid string"):
            reg._resolve_mapper()


class TestFit:
    """Tests for OrdBoostRegressor.fit."""

    @pytest.fixture
    def synthetic_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixture providing a synthetic feature matrix and continuous target."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        y = X[:, 0] * 10.0 + rng.standard_normal(50)
        return X, y

    def test_fit_returns_self(self, synthetic_data) -> None:
        """Test that fit returns the estimator instance for chaining."""
        X, y = synthetic_data
        reg = OrdBoostRegressor(n_bins=5, max_iter=5, random_state=42)
        result = reg.fit(X, y)
        assert result is reg

    def test_fit_sets_expected_attributes(self, synthetic_data) -> None:
        """Test that fit sets classifier_, mapper_, bin_edges_, n_features_in_."""
        X, y = synthetic_data
        reg = OrdBoostRegressor(n_bins=5, max_iter=5, random_state=42)
        reg.fit(X, y)

        assert hasattr(reg, "classifier_")
        assert hasattr(reg, "mapper_")
        assert hasattr(reg, "bin_edges_")
        assert isinstance(reg.mapper_, EmpiricalMedianBinMapper)
        assert reg.n_features_in_ == 3
        assert len(reg.bin_edges_) >= 2

    @pytest.mark.parametrize(
        "custom_mapper",
        [
            EmpiricalMeanBinMapper(),
            QuantileBinMapper(),
            UniformBinMapper(),
            ContinuousBinMapper(),
        ],
    )
    def test_fit_with_unconfigured_custom_mapper_instances(
        self, synthetic_data, custom_mapper
    ) -> None:
        """Test fitting with various unconfigured mapper instances passed
        directly, rather than string shortcuts."""
        X, y = synthetic_data
        reg = OrdBoostRegressor(
            n_bins=5, mapper=custom_mapper, max_iter=5, random_state=42
        )
        reg.fit(X, y)
        assert isinstance(reg.mapper_, type(custom_mapper))

    def test_fit_1d_X_raises(self, synthetic_data) -> None:
        """Test that passing a 1D X feature matrix raises ValueError."""
        _, y = synthetic_data
        reg = OrdBoostRegressor(n_bins=5, max_iter=5)
        with pytest.raises(ValueError, match="Expected 2D array"):
            reg.fit(np.array([1.0, 2.0, 3.0]), y[:3])

    def test_fit_nan_in_y_raises(self, synthetic_data) -> None:
        """Test that NaN values in y raise ValueError, distinguishing this
        from the permissive NaN handling applied to X."""
        X, y = synthetic_data
        y = y.copy()
        y[0] = np.nan
        reg = OrdBoostRegressor(n_bins=5, max_iter=5)
        with pytest.raises(ValueError, match="contains NaN"):
            reg.fit(X, y)

    def test_fit_inf_in_y_raises(self, synthetic_data) -> None:
        """Test that infinite values in y raise ValueError."""
        X, y = synthetic_data
        y = y.copy()
        y[0] = np.inf
        reg = OrdBoostRegressor(n_bins=5, max_iter=5)
        with pytest.raises(ValueError, match="contains infinity"):
            reg.fit(X, y)

    def test_fit_nan_in_X_is_permitted(self, synthetic_data) -> None:
        """Test that NaN values in X do not raise, since
        HistGradientBoostingClassifier handles missing features natively."""
        X, y = synthetic_data
        X = X.copy()
        X[0, 0] = np.nan
        reg = OrdBoostRegressor(n_bins=5, max_iter=5, random_state=42)
        reg.fit(X, y)  # should not raise
        assert hasattr(reg, "classifier_")

    def test_classifier_and_mapper_agree_on_bin_count(self, synthetic_data) -> None:
        """Test that the fitted classifier's number of classes matches the
        mapper's n_bins_, confirming both were fit against the same
        single-source-of-truth y_binned rather than independently
        digitized values that could disagree."""
        X, y = synthetic_data
        reg = OrdBoostRegressor(n_bins=5, max_iter=5, random_state=42)
        reg.fit(X, y)
        assert len(reg.classifier_.classes_) == reg.mapper_.n_bins_

    def test_predictions_only_use_bins_within_expected_range(
        self, synthetic_data
    ) -> None:
        """Test that fitted y_binned values (indirectly, via classifier
        classes_) never exceed n_bins - 1, guarding against a digitize
        off-by-one producing an extra out-of-range bin."""
        X, y = synthetic_data
        reg = OrdBoostRegressor(n_bins=5, max_iter=5, random_state=42)
        reg.fit(X, y)
        assert reg.classifier_.classes_.max() <= len(reg.bin_edges_)  # n_bins - 1

    def test_fit_handles_empty_intermediate_bin_via_explicit_bin_edges(self) -> None:
        """Test that fit succeeds, and produces the full expected bin
        count, even when explicit bin_edges create empty intermediate
        bins (no y values fall inside them). This is the scenario that
        makes OrdBoostRegressor.fit's classes=np.arange(n_bins) call to
        the underlying OrdBoostClassifier.fit necessary: without it,
        np.unique(y_binned) alone would silently drop the empty bins
        from classes_, leaving classifier_ and mapper_ disagreeing on
        n_bins."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 2))
        # Two well-separated clusters; nothing falls in the [25, 75) span,
        # so bins 1 and 2 of bin_edges=[25, 50, 75] receive zero samples.
        y = np.where(
            np.arange(60) < 30, rng.uniform(0, 10, 60), rng.uniform(90, 100, 60)
        )
        reg = OrdBoostRegressor(
            bin_edges=[25.0, 50.0, 75.0], max_iter=10, random_state=0
        )
        reg.fit(X, y)

        assert reg.mapper_.n_bins_ == 4
        np.testing.assert_array_equal(
            reg.classifier_.classes_, np.array([0, 1, 2, 3])
        )
        pmf = reg.classifier_.predict_proba(X)
        assert pmf.shape[1] == 4
        preds = reg.predict(X)
        assert preds.shape == (60,)
        assert np.all(np.isfinite(preds))


class TestPredictDist:
    """Tests for OrdBoostRegressor.predict_dist."""

    @pytest.fixture
    def fitted_model(self) -> tuple[OrdBoostRegressor, np.ndarray, np.ndarray]:
        """Fixture providing a fitted regressor instance."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        y = X[:, 0] * 5.0 + 10.0
        reg = OrdBoostRegressor(n_bins=4, max_iter=5, random_state=42)
        reg.fit(X, y)
        return reg, X, y

    def test_returns_continuous_predictive_distribution(self, fitted_model) -> None:
        """Test that predict_dist returns a ContinuousPredictiveDistribution
        with one row per sample."""
        reg, X, _ = fitted_model
        dist = reg.predict_dist(X)
        assert isinstance(dist, ContinuousPredictiveDistribution)
        assert dist.grid_cdf.shape[0] == 30

    def test_single_sample_evaluations(self, fitted_model) -> None:
        """Test that distribution methods work correctly for a single-row
        prediction."""
        reg, X, _ = fitted_model
        dist = reg.predict_dist(X[[0]])
        assert dist.mean().shape == (1,)
        assert dist.median().shape == (1,)
        assert dist.cdf(10.0).shape == (1,)
        lower, upper = dist.interval(alpha=0.10)
        assert lower.shape == (1,)
        assert upper.shape == (1,)

    def test_not_fitted_raises(self) -> None:
        """Test that calling predict_dist before fit raises NotFittedError."""
        reg = OrdBoostRegressor()
        with pytest.raises(NotFittedError):
            reg.predict_dist(np.ones((2, 2)))


class TestPredict:
    """Tests for OrdBoostRegressor.predict."""

    @pytest.fixture
    def fitted_model(self) -> tuple[OrdBoostRegressor, np.ndarray, np.ndarray]:
        """Fixture providing a fitted regressor instance."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        y = X[:, 0] * 5.0 + 10.0
        reg = OrdBoostRegressor(n_bins=4, max_iter=5, random_state=42)
        reg.fit(X, y)
        return reg, X, y

    def test_default_method_is_mean(self, fitted_model) -> None:
        """Test that predict with no explicit method matches method='mean'."""
        reg, X, _ = fitted_model
        np.testing.assert_allclose(reg.predict(X), reg.predict(X, method="mean"))

    def test_median_method(self, fitted_model) -> None:
        """Test that method='median' returns the distribution's median,
        not its mean."""
        reg, X, _ = fitted_model
        dist = reg.predict_dist(X)
        np.testing.assert_allclose(reg.predict(X, method="median"), dist.median())

    def test_output_shape(self, fitted_model) -> None:
        """Test that predict returns one value per sample."""
        reg, X, _ = fitted_model
        assert reg.predict(X).shape == (30,)

    def test_single_sample_shape(self, fitted_model) -> None:
        """Test that predict on a single-row X returns a length-1 array."""
        reg, X, _ = fitted_model
        assert reg.predict(X[[0]]).shape == (1,)

    def test_1d_single_sample_raises(self, fitted_model) -> None:
        """Test that passing a 1D array (rather than a single-row 2D
        array) raises ValueError."""
        reg, X, _ = fitted_model
        with pytest.raises(ValueError, match="Expected 2D array"):
            reg.predict(X[0])

    def test_not_fitted_raises(self) -> None:
        """Test that calling predict before fit raises NotFittedError."""
        reg = OrdBoostRegressor()
        with pytest.raises(NotFittedError):
            reg.predict(np.ones((2, 2)))

    def test_invalid_method_raises(self, fitted_model) -> None:
        """Test that an unrecognized method raises ValueError."""
        reg, X, _ = fitted_model
        with pytest.raises(ValueError, match="Invalid.*method"):
            reg.predict(X, method="invalid")  # type: ignore[arg-type]


class TestGetSetParams:
    """Tests for OrdBoostRegressor get_params/set_params and scikit-learn
    estimator compatibility.
    """

    def test_get_params_merges_kwargs_at_top_level(self) -> None:
        """Test that get_params exposes both explicit fields and pass-through
        kwargs at the top level, with no raw 'kwargs' key visible."""
        reg = OrdBoostRegressor(
            learning_rate=0.05, max_leaf_nodes=15, early_stopping=False
        )
        params = reg.get_params()
        assert params["learning_rate"] == 0.05
        assert params["max_leaf_nodes"] == 15
        assert params["early_stopping"] is False
        assert "kwargs" not in params

    def test_set_params_updates_explicit_and_kwargs_fields(self) -> None:
        """Test that set_params correctly routes known fields to attributes
        and unknown fields into self.kwargs."""
        reg = OrdBoostRegressor(learning_rate=0.1, max_leaf_nodes=31)
        reg.set_params(learning_rate=0.01, max_leaf_nodes=15, min_samples_leaf=10)

        assert reg.learning_rate == 0.01
        assert reg.min_samples_leaf == 10
        assert reg.kwargs["max_leaf_nodes"] == 15

    def test_set_params_with_no_arguments_returns_self(self) -> None:
        """Test that calling set_params with no arguments is a no-op that
        still returns self."""
        reg = OrdBoostRegressor()
        result = reg.set_params()
        assert result is reg

    def test_clone_compatibility(self) -> None:
        """Test that sklearn.base.clone produces an independent, correctly
        parameterized copy, including pass-through kwargs."""
        reg = OrdBoostRegressor(max_iter=20, max_bins=64, random_state=42)
        cloned = clone(reg)

        assert cloned.max_iter == 20  # type: ignore[attr-defined]
        assert cloned.kwargs.get("max_bins") == 64  # type: ignore[attr-defined]
        assert cloned.random_state == 42  # type: ignore[attr-defined]
        assert cloned is not reg

    def test_grid_search_cv_compatibility(self) -> None:
        """Test that GridSearchCV can tune both explicit and pass-through
        hyperparameters without error."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 2))
        y = X[:, 0] * 3.0 + rng.standard_normal(40)

        reg = OrdBoostRegressor(n_bins=3, max_iter=5, random_state=0)
        search = GridSearchCV(
            reg, param_grid={"learning_rate": [0.05, 0.1]}, cv=2, error_score="raise"
        )
        search.fit(X, y)
        assert hasattr(search, "best_params_")
