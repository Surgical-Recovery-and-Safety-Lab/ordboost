"""Comprehensive unit tests for OrdBoostClassifier."""

from typing import Literal

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV

from ordboost.distributions import PredictiveDistribution
from ordboost.models import OrdBoostClassifier


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
        assert isinstance(dist, PredictiveDistribution)
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

        assert cloned_model.max_iter == 20
        assert cloned_model.kwargs.get("max_bins") == 64
        assert cloned_model.random_state == 42
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
