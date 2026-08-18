"""Comprehensive unit tests for OrdBoostClassifier."""

from typing import Literal, cast

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV

from ordboost.distributions import (
    ContinuousPredictiveDistribution,
    DiscretePredictiveDistribution,
)
from ordboost.mappers import (
    BaseBinMapper,
    ContinuousBinMapper,
    EmpiricalMeanBinMapper,
    EmpiricalMedianBinMapper,
    QuantileBinMapper,
    UniformBinMapper,
)
from ordboost.models import OrdBoostClassifier, OrdBoostRegressor


class TestOrdBoostClassifier:
    """Tests for OrdBoostClassifier fitting, predictions, and edge cases."""

    @pytest.fixture
    def synthetic_ordinal_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fixture generating synthetic 4-class ordinal dataset."""
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 4)

        # Ordinal target with physical gaps: [0, 5, 10, 30]
        latent = X[:, 0] * 1.5 + X[:, 1] * 0.8 + np.random.randn(n_samples) * 0.5
        y = np.select(
            [latent < -1.0, latent < 0.0, latent < 1.0],
            [0, 5, 10],
            default=30,
        )
        return X[:150], y[:150], X[150:], y[150:]

    def test_fit_and_predict_basic(
        self,
        synthetic_ordinal_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test standard fit, predict, predict_proba, and predict_dist execution."""
        X_train, y_train, X_test, _ = synthetic_ordinal_data

        model = OrdBoostClassifier(max_iter=20, min_samples_leaf=5, random_state=42)
        model.fit(X_train, y_train)

        # Check fitted attributes
        assert hasattr(model, "classes_")
        assert hasattr(model, "estimators_")
        assert model.estimators_
        assert len(model.classes_) == 4
        assert len(model.estimators_) == 3  # K-1 edge models
        np.testing.assert_array_equal(model.classes_, np.array([0, 5, 10, 30]))

        # Check predict point estimate
        y_pred = model.predict(X_test, method="median")
        assert y_pred.shape == (len(X_test),)
        assert set(y_pred).issubset(set(model.classes_))

        # Check predict_proba PMF matrix
        pmf = model.predict_proba(X_test)
        assert pmf.shape == (len(X_test), 4)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(pmf >= 0.0)

        # Check predict_dist
        dist = model.predict_dist(X_test)
        assert isinstance(dist, DiscretePredictiveDistribution)
        np.testing.assert_array_equal(dist.classes, model.classes_)

    def test_kwargs_pass_through(
        self,
        synthetic_ordinal_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test that additional kwargs are passed to HistGradientBoostingClassifier."""
        X_train, y_train, _, _ = synthetic_ordinal_data

        # Pass specific HistGradientBoostingClassifier parameters via kwargs
        model = OrdBoostClassifier(
            max_iter=10,
            max_leaf_nodes=15,
            max_bins=128,
            early_stopping=False,
            random_state=42,
        )
        model.fit(X_train, y_train)

        assert model.estimators_ is not None
        for est in model.estimators_:
            assert est.max_leaf_nodes == 15
            assert est.max_bins == 128
            assert est.early_stopping is False

    def test_get_params_returns_merged_kwargs(self) -> None:
        """Verify get_params returns explicit attributes and extra kwargs at top level."""
        model = OrdBoostClassifier(
            learning_rate=0.05,
            max_iter=50,
            max_leaf_nodes=15,  # pass-through kwarg
            early_stopping=False,  # pass-through kwarg
        )
        params = model.get_params()

        # Check explicit parameters
        assert params["learning_rate"] == 0.05
        assert params["max_iter"] == 50

        # Check pass-through kwargs
        assert params["max_leaf_nodes"] == 15
        assert params["early_stopping"] is False

        # Ensure raw 'kwargs' container dict isn't exposed directly
        assert "kwargs" not in params

    def test_set_params_updates_both_explicit_and_kwargs(self) -> None:
        """Verify set_params updates both standard init fields and extra kwargs."""
        model = OrdBoostClassifier(learning_rate=0.1, max_leaf_nodes=31)

        model.set_params(learning_rate=0.01, max_leaf_nodes=15, min_samples_leaf=10)

        # Check explicit params
        assert model.learning_rate == 0.01
        assert model.min_samples_leaf == 10

        # Check kwarg param
        assert model.kwargs["max_leaf_nodes"] == 15

        # Verify get_params reflects changes
        updated_params = model.get_params()
        assert updated_params["learning_rate"] == 0.01
        assert updated_params["min_samples_leaf"] == 10
        assert updated_params["max_leaf_nodes"] == 15

    def test_clone_compatibility(self) -> None:
        """Verify sklearn.base.clone works seamlessly with kwargs."""
        model = OrdBoostClassifier(
            max_iter=20,
            max_bins=64,
            random_state=42,
        )

        cloned_model = clone(model)

        assert cloned_model.max_iter == 20  # type: ignore
        assert cloned_model.kwargs.get("max_bins") == 64  # type: ignore
        assert cloned_model.random_state == 42  # type: ignore
        # Ensure it's a fresh instance, not a reference copy
        assert cloned_model is not model

    def test_grid_search_cv_compatibility(self) -> None:
        """Test that GridSearchCV can manipulate both explicit and kwarg hyperparameters."""
        X = np.random.randn(50, 3)
        y = np.random.choice([0, 1, 2], size=50)

        param_grid = {
            "learning_rate": [0.01, 0.1],
            "max_leaf_nodes": [10, 20],  # kwarg tuning
        }

        grid = GridSearchCV(
            estimator=OrdBoostClassifier(max_iter=5),
            param_grid=param_grid,
            cv=2,
        )

        # Should fit without throwing TypeError or KeyError
        grid.fit(X, y)
        assert grid.best_estimator_ is not None
        assert "max_leaf_nodes" in grid.best_params_

    def test_get_set_params_sklearn_compatibility(self) -> None:
        """Test scikit-learn get_params and set_params with explicit kwargs."""
        model = OrdBoostClassifier(learning_rate=0.05, max_leaf_nodes=20, max_bins=128)

        params = model.get_params()

        # Check explicit and kwarg params are exposed at the top level
        assert params["learning_rate"] == 0.05
        assert params["max_leaf_nodes"] == 20
        assert params["max_bins"] == 128
        assert "kwargs" not in params

        # Test set_params modifying both standard and kwarg params
        model.set_params(learning_rate=0.2, max_leaf_nodes=50, max_bins=64)

        assert model.learning_rate == 0.2
        assert model.kwargs["max_leaf_nodes"] == 50
        assert model.kwargs["max_bins"] == 64

        # Verify get_params reflects changes after set_params
        new_params = model.get_params()
        assert new_params["learning_rate"] == 0.2
        assert new_params["max_leaf_nodes"] == 50
        assert new_params["max_bins"] == 64

    def test_not_fitted_error(self) -> None:
        """Test that calling prediction methods before fit raises NotFittedError."""
        model = OrdBoostClassifier()
        X_dummy = np.ones((5, 2))

        with pytest.raises(NotFittedError):
            model.predict(X_dummy)

        with pytest.raises(NotFittedError):
            model.predict_proba(X_dummy)

        with pytest.raises(NotFittedError):
            model.predict_dist(X_dummy)

    def test_single_class_error(self) -> None:
        """Test that fitting on dataset with < 2 unique classes raises ValueError."""
        X = np.ones((10, 2))
        y = np.zeros(10, dtype=int)  # Single class

        model = OrdBoostClassifier()
        with pytest.raises(ValueError, match="at least 2 unique classes"):
            model.fit(X, y)

    def test_invalid_monotonicity_method(
        self,
        synthetic_ordinal_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test that invalid monotonicity parameter raises ValueError."""
        X_train, y_train, _, _ = synthetic_ordinal_data
        model = OrdBoostClassifier(monotonicity="invalid_method")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Invalid monotonicity method"):
            model.fit(X_train, y_train)

    @pytest.mark.parametrize("mono_method", ["running_max", "isotonic"])
    def test_monotonicity_methods(
        self,
        mono_method: Literal["running_max", "isotonic"],
        synthetic_ordinal_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test both running_max and isotonic monotonicity methods."""
        X_train, y_train, X_test, _ = synthetic_ordinal_data

        model = OrdBoostClassifier(
            monotonicity=mono_method, max_iter=15, min_samples_leaf=5
        )
        model.fit(X_train, y_train)

        pmf = model.predict_proba(X_test)
        assert pmf.shape == (len(X_test), 4)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(pmf >= 0.0)

    def test_negative_and_non_consecutive_targets(self) -> None:
        """Test edge case with negative and widely spaced ordinal class labels."""
        X = np.random.randn(100, 3)
        # Ordinal target spanning negative to positive non-consecutive integers
        y = np.random.choice([-10, -2, 0, 50, 100], size=100)

        model = OrdBoostClassifier(max_iter=10)
        model.fit(X, y)

        np.testing.assert_array_equal(model.classes_, np.array([-10, -2, 0, 50, 100]))
        pmf = model.predict_proba(X[:10])
        assert pmf.shape == (10, 5)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_dimension_mismatch(
        self,
        synthetic_ordinal_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test error handling when test data feature count does not match fit data."""
        X_train, y_train, _, _ = synthetic_ordinal_data
        model = OrdBoostClassifier(max_iter=10)
        model.fit(X_train, y_train)

        # Pass 2 features instead of 4
        X_invalid = np.random.randn(10, 2)
        with pytest.raises(ValueError):
            model.predict(X_invalid)

    def test_predict_mean_vs_median(
        self,
        synthetic_ordinal_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test point predictions using both mean and median methods."""
        X_train, y_train, X_test, _ = synthetic_ordinal_data

        model = OrdBoostClassifier(max_iter=20, min_samples_leaf=5)
        model.fit(X_train, y_train)

        y_median = model.predict(X_test, method="median")
        y_mean = model.predict(X_test, method="mean")

        assert y_median.shape == (len(X_test),)
        assert y_mean.shape == (len(X_test),)
        # Median predictions must strictly equal one of the discrete class levels
        assert set(y_median).issubset(set(model.classes_))

    def test_invalid_predict_method(
        self,
        synthetic_ordinal_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Test that passing an invalid point prediction method raises ValueError."""
        X_train, y_train, X_test, _ = synthetic_ordinal_data

        model = OrdBoostClassifier(max_iter=10)
        model.fit(X_train, y_train)

        with pytest.raises(ValueError, match="Invalid prediction method"):
            model.predict(X_test, method="invalid_method")  # type: ignore[arg-type]


class TestOrdBoostRegressorInit:
    """Tests for OrdBoostRegressor initialization."""

    def test_init_default_params(self) -> None:
        """Verify default hyperparameter assignment during initialization."""
        reg = OrdBoostRegressor()
        assert reg.n_bins == 20
        assert reg.bin_edges is None
        assert reg.bin_strategy == "quantile"
        assert reg.mapper == "median"
        assert reg.mapper_kwargs is None
        assert reg.learning_rate == 0.1
        assert reg.max_iter == 100


class TestOrdBoostRegressorComputeBinEdges:
    """Tests for _compute_bin_edges private method."""

    def test_custom_bin_edges_valid(self) -> None:
        """Test explicit valid bin_edges array."""
        edges = [0.0, 10.0, 20.0, 50.0]
        reg = OrdBoostRegressor(bin_edges=edges)
        y = np.array([1.0, 15.0, 40.0])
        resolved_edges = reg._compute_bin_edges(y)
        np.testing.assert_array_equal(resolved_edges, edges)

    def test_custom_bin_edges_invalid_ndim(self) -> None:
        """Test that 2D bin_edges raises ValueError."""
        reg = OrdBoostRegressor(bin_edges=np.array([[0, 10], [10, 20]]))
        with pytest.raises(ValueError, match="1D array with >= 2 edges"):
            reg._compute_bin_edges(np.array([1.0, 2.0]))

    def test_custom_bin_edges_too_few_edges(self) -> None:
        """Test that fewer than 2 edges raises ValueError."""
        reg = OrdBoostRegressor(bin_edges=[10.0])
        with pytest.raises(ValueError, match=">= 2 edges"):
            reg._compute_bin_edges(np.array([1.0, 2.0]))

    def test_custom_bin_edges_non_monotonic(self) -> None:
        """Test non-monotonically increasing edges raise ValueError."""
        reg = OrdBoostRegressor(bin_edges=[0.0, 10.0, 5.0])
        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            reg._compute_bin_edges(np.array([1.0, 2.0]))

    def test_n_bins_too_small(self) -> None:
        """Test that n_bins < 2 raises ValueError."""
        reg = OrdBoostRegressor(n_bins=1)
        with pytest.raises(ValueError, match="Parameter 'n_bins' must be >= 2"):
            reg._compute_bin_edges(np.array([1.0, 2.0]))

    def test_bin_strategy_quantile(self) -> None:
        """Test quantile bin strategy computation."""
        reg = OrdBoostRegressor(n_bins=4, bin_strategy="quantile")
        y = np.linspace(0.0, 100.0, 101)
        edges = reg._compute_bin_edges(y)
        assert len(edges) == 5
        np.testing.assert_allclose(edges, [0.0, 25.0, 50.0, 75.0, 100.0])

    def test_bin_strategy_uniform(self) -> None:
        """Test uniform bin strategy computation."""
        reg = OrdBoostRegressor(n_bins=4, bin_strategy="uniform")
        y = np.array([0.0, 100.0])
        edges = reg._compute_bin_edges(y)
        assert len(edges) == 5
        np.testing.assert_allclose(edges, [0.0, 25.0, 50.0, 75.0, 100.0])

    def test_bin_strategy_invalid(self) -> None:
        """Test that invalid bin strategy raises ValueError."""
        reg = OrdBoostRegressor(bin_strategy="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Invalid bin_strategy"):
            reg._compute_bin_edges(np.array([1.0, 2.0]))


class TestOrdBoostRegressorResolveMapper:
    """Tests for _resolve_mapper private method and string shortcut handling."""

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
    def test_resolve_mapper_string_shortcuts(
        self, shortcut: str, expected_cls: type[BaseBinMapper]
    ) -> None:
        """Test that valid string shortcuts resolve to correct mapper instances."""
        reg = OrdBoostRegressor(mapper=shortcut)  # type: ignore[arg-type]
        reg.bin_edges_ = np.array([0.0, 10.0, 20.0])
        resolved = reg._resolve_mapper()

        assert isinstance(resolved, expected_cls)
        np.testing.assert_array_equal(resolved.bin_edges, reg.bin_edges_)

    def test_resolve_mapper_none_defaults_to_median(self) -> None:
        """Test that mapper=None defaults to EmpiricalMedianBinMapper."""
        reg = OrdBoostRegressor(mapper=None)
        reg.bin_edges_ = np.array([0.0, 10.0, 20.0])
        resolved = reg._resolve_mapper()

        assert isinstance(resolved, EmpiricalMedianBinMapper)

    def test_resolve_mapper_passes_mapper_kwargs(self) -> None:
        """Test that mapper_kwargs are forwarded during string mapper instantiation."""
        reg = OrdBoostRegressor(
            mapper="continuous",
            mapper_kwargs={"grid_resolution": 50},
        )
        reg.bin_edges_ = np.array([0.0, 10.0, 20.0])
        resolved = cast(ContinuousBinMapper, reg._resolve_mapper())

        assert isinstance(resolved, ContinuousBinMapper)
        assert resolved.grid_resolution == 50

    def test_resolve_mapper_custom_instance_cloned(self) -> None:
        """Test that pre-instantiated BaseBinMapper object is cloned and assigned edges."""
        custom_mapper = EmpiricalMeanBinMapper(bin_edges=None)
        reg = OrdBoostRegressor(mapper=custom_mapper)
        reg.bin_edges_ = np.array([0.0, 5.0, 10.0])
        resolved = reg._resolve_mapper()

        assert isinstance(resolved, EmpiricalMeanBinMapper)
        assert resolved is not custom_mapper  # Must be a cloned instance
        np.testing.assert_array_equal(resolved.bin_edges, reg.bin_edges_)

    def test_resolve_mapper_invalid_shortcut_raises_error(self) -> None:
        """Test that unknown string shortcut raises descriptive ValueError."""
        reg = OrdBoostRegressor(mapper="unknown_shortcut")  # type: ignore[arg-type]
        reg.bin_edges_ = np.array([0.0, 10.0, 20.0])

        with pytest.raises(ValueError, match="Unknown mapper shortcut"):
            reg._resolve_mapper()

    def test_resolve_mapper_invalid_type_raises_error(self) -> None:
        """Test that non-string and non-mapper object raises ValueError."""
        reg = OrdBoostRegressor(mapper=12345)  # type: ignore[arg-type]
        reg.bin_edges_ = np.array([0.0, 10.0, 20.0])

        with pytest.raises(ValueError, match="Expected 'mapper' to be a valid string"):
            reg._resolve_mapper()


class TestOrdBoostRegressorFit:
    """Tests for OrdBoostRegressor fit method."""

    @pytest.fixture
    def synthetic_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixture providing synthetic 2D feature matrix and continuous target."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        y = X[:, 0] * 10.0 + rng.standard_normal(50)
        return X, y

    def test_fit_success_default_mapper(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test fitting model with default settings (median mapper shortcut)."""
        X, y = synthetic_data
        reg = OrdBoostRegressor(n_bins=5, max_iter=5, random_state=42)
        fitted_reg = reg.fit(X, y)

        assert fitted_reg is reg
        assert hasattr(reg, "classifier_")
        assert hasattr(reg, "mapper_")
        assert hasattr(reg, "bin_edges_")
        assert isinstance(reg.mapper_, EmpiricalMedianBinMapper)
        assert reg.n_features_in_ == 3
        assert len(reg.bin_edges_) >= 2

    def test_fit_with_custom_mean_mapper_instance(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test fitting model with unconfigured EmpiricalMeanBinMapper instance."""
        X, y = synthetic_data
        custom_mapper = EmpiricalMeanBinMapper()
        reg = OrdBoostRegressor(
            n_bins=5, mapper=custom_mapper, max_iter=5, random_state=42
        )
        reg.fit(X, y)

        assert isinstance(reg.mapper_, EmpiricalMeanBinMapper)

    def test_fit_with_custom_quantile_mapper_instance(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test fitting model with unconfigured QuantileBinMapper instance."""
        X, y = synthetic_data
        custom_mapper = QuantileBinMapper()
        reg = OrdBoostRegressor(
            n_bins=5, mapper=custom_mapper, max_iter=5, random_state=42
        )
        reg.fit(X, y)

        assert isinstance(reg.mapper_, QuantileBinMapper)

    def test_fit_1d_X_raises_error(self) -> None:
        """Test that passing 1D X feature matrix raises ValueError."""
        reg = OrdBoostRegressor(max_iter=5)
        with pytest.raises(ValueError):
            reg.fit(X=np.array([1.0, 2.0, 3.0]), y=np.array([1.0, 2.0, 3.0]))

    def test_fit_shape_mismatch(self) -> None:
        """Test error handling when X and y sample counts mismatch."""
        reg = OrdBoostRegressor(max_iter=5)
        X = np.ones((10, 2))
        y = np.ones(5)
        with pytest.raises(ValueError):
            reg.fit(X, y)

    def test_fit_invalid_mapper_raises_error(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test fitting with invalid mapper shortcut raises ValueError."""
        X, y = synthetic_data
        reg = OrdBoostRegressor(max_iter=5, mapper="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown mapper shortcut"):
            reg.fit(X, y)


class TestOrdBoostRegressorPredict:
    """Tests for OrdBoostRegressor predict method."""

    @pytest.fixture
    def fitted_model(self) -> tuple[OrdBoostRegressor, np.ndarray, np.ndarray]:
        """Fixture providing fitted regressor with synthetic dataset."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        y = X[:, 0] * 5.0 + 10.0
        reg = OrdBoostRegressor(n_bins=4, max_iter=5, random_state=42)
        reg.fit(X, y)
        return reg, X, y

    def test_predict_mean_and_median(
        self, fitted_model: tuple[OrdBoostRegressor, np.ndarray, np.ndarray]
    ) -> None:
        """Test predict with mean and median point estimation methods."""
        reg, X, _ = fitted_model

        preds_mean = reg.predict(X, method="mean")
        assert preds_mean.shape == (30,)
        assert np.issubdtype(preds_mean.dtype, np.floating)

        preds_median = reg.predict(X, method="median")
        assert preds_median.shape == (30,)

    def test_predict_single_sample_2d(
        self, fitted_model: tuple[OrdBoostRegressor, np.ndarray, np.ndarray]
    ) -> None:
        """Test predicting on single 2D sample array of shape (1, n_features)."""
        reg, X, _ = fitted_model
        single_sample = X[[0]]  # Shape: (1, 2)

        pred = reg.predict(single_sample)
        assert pred.shape == (1,)

    def test_predict_single_sample_1d_raises_error(
        self, fitted_model: tuple[OrdBoostRegressor, np.ndarray, np.ndarray]
    ) -> None:
        """Test that passing 1D single sample array raises ValueError."""
        reg, X, _ = fitted_model
        single_sample_1d = X[0]  # Shape: (2,)

        with pytest.raises(ValueError, match="Expected 2D array"):
            reg.predict(single_sample_1d)

    def test_predict_not_fitted_raises_error(self) -> None:
        """Test calling predict on unfitted estimator raises NotFittedError."""
        reg = OrdBoostRegressor()
        with pytest.raises(NotFittedError):
            reg.predict(np.ones((2, 2)))

    def test_predict_invalid_method(
        self, fitted_model: tuple[OrdBoostRegressor, np.ndarray, np.ndarray]
    ) -> None:
        """Test passing invalid point prediction strategy raises ValueError."""
        reg, X, _ = fitted_model
        with pytest.raises(ValueError, match="Invalid method"):
            reg.predict(X, method="invalid")  # type: ignore[arg-type]


class TestOrdBoostRegressorPredictDist:
    """Tests for OrdBoostRegressor predict_dist method."""

    @pytest.fixture
    def fitted_model(self) -> tuple[OrdBoostRegressor, np.ndarray, np.ndarray]:
        """Fixture providing fitted regressor instance."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        y = X[:, 0] * 5.0 + 10.0
        reg = OrdBoostRegressor(n_bins=4, max_iter=5, random_state=42)
        reg.fit(X, y)
        return reg, X, y

    def test_predict_dist_returns_continuous_dist(
        self, fitted_model: tuple[OrdBoostRegressor, np.ndarray, np.ndarray]
    ) -> None:
        """Test predict_dist produces ContinuousPredictiveDistribution."""
        reg, X, _ = fitted_model
        dist = reg.predict_dist(X)

        assert isinstance(dist, ContinuousPredictiveDistribution)
        assert dist.grid_cdf.shape[0] == 30

    def test_predict_dist_single_sample_evaluations(
        self, fitted_model: tuple[OrdBoostRegressor, np.ndarray, np.ndarray]
    ) -> None:
        """Test evaluating distribution methods on single sample forecast."""
        reg, X, _ = fitted_model
        single_sample = X[[0]]

        dist = reg.predict_dist(single_sample)
        assert dist.mean().shape == (1,)
        assert dist.median().shape == (1,)
        assert dist.cdf(10.0).shape == (1,)

        lower, upper = dist.interval(alpha=0.10)
        assert lower.shape == (1,)
        assert upper.shape == (1,)

    def test_predict_dist_not_fitted_raises_error(self) -> None:
        """Test calling predict_dist prior to fit raises NotFittedError."""
        reg = OrdBoostRegressor()
        with pytest.raises(NotFittedError):
            reg.predict_dist(np.ones((2, 2)))


class TestOrdBoostRegressorGetSetParams:
    """Tests for OrdBoostRegressor get_params, set_params, and scikit-learn compatibility."""

    def test_get_params_returns_merged_kwargs(self) -> None:
        """Verify get_params returns explicit attributes and extra kwargs at top level."""
        reg = OrdBoostRegressor(
            learning_rate=0.05,
            max_iter=50,
            max_leaf_nodes=15,  # pass-through kwarg
            early_stopping=False,  # pass-through kwarg
        )
        params = reg.get_params()

        # Check explicit parameters
        assert params["learning_rate"] == 0.05
        assert params["max_iter"] == 50

        # Check pass-through kwargs
        assert params["max_leaf_nodes"] == 15
        assert params["early_stopping"] is False

        # Ensure raw 'kwargs' container dict isn't exposed directly
        assert "kwargs" not in params

    def test_set_params_updates_both_explicit_and_kwargs(self) -> None:
        """Verify set_params updates both standard init fields and extra kwargs."""
        reg = OrdBoostRegressor(learning_rate=0.1, max_leaf_nodes=31)

        reg.set_params(learning_rate=0.01, max_leaf_nodes=15, min_samples_leaf=10)

        # Check explicit params
        assert reg.learning_rate == 0.01
        assert reg.min_samples_leaf == 10

        # Check kwarg param
        assert reg.kwargs["max_leaf_nodes"] == 15

        # Verify get_params reflects changes
        updated_params = reg.get_params()
        assert updated_params["learning_rate"] == 0.01
        assert updated_params["min_samples_leaf"] == 10
        assert updated_params["max_leaf_nodes"] == 15

    def test_clone_compatibility(self) -> None:
        """Verify sklearn.base.clone works seamlessly with extra kwargs."""
        reg = OrdBoostRegressor(
            max_iter=20,
            max_bins=64,
            random_state=42,
        )

        cloned_reg = clone(reg)

        assert cloned_reg.max_iter == 20  # type: ignore[attr-defined]
        assert cloned_reg.kwargs.get("max_bins") == 64  # type: ignore[attr-defined]
        assert cloned_reg.random_state == 42  # type: ignore[attr-defined]
        assert cloned_reg is not reg

    def test_grid_search_cv_compatibility(self) -> None:
        """Test that GridSearchCV can manipulate explicit and kwarg hyperparameters."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50) * 10.0

        param_grid = {
            "learning_rate": [0.01, 0.1],
            "max_leaf_nodes": [10, 20],  # kwarg tuning
        }

        grid = GridSearchCV(
            estimator=OrdBoostRegressor(max_iter=5, n_bins=3),
            param_grid=param_grid,
            cv=2,
        )

        grid.fit(X, y)
        assert grid.best_estimator_ is not None
        assert "max_leaf_nodes" in grid.best_params_
