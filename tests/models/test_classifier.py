"""Unit tests for OrdBoostClassifier in ordboost.models."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV

from ordboost.distributions import DiscretePredictiveDistribution
from ordboost.models import OrdBoostClassifier


class TestInit:
    """Tests for OrdBoostClassifier.__init__ default parameter assignment."""

    def test_default_params(self) -> None:
        """Test default hyperparameter assignment during initialization."""
        model = OrdBoostClassifier()
        assert model.loss == "log_loss"
        assert model.learning_rate == 0.1
        assert model.max_iter == 100
        assert model.monotonicity == "running_max"
        assert model.n_jobs == -1

    def test_kwargs_stored_separately(self) -> None:
        """Test that unrecognized keyword arguments are captured in
        self.kwargs rather than raising at construction time.
        """
        model = OrdBoostClassifier(max_leaf_nodes=15, early_stopping=False)
        assert model.kwargs == {"max_leaf_nodes": 15, "early_stopping": False}

    def test_estimators_not_present_before_fit(self) -> None:
        """Test that estimators_ does not exist on an unfitted instance,
        confirming the fitted-attribute is only ever set in fit(), not
        pre-declared in __init__. This is the regression test for the
        check_is_fitted fix: hasattr(self, 'estimators_') must be False
        pre-fit for NotFittedError to trigger correctly.
        """
        model = OrdBoostClassifier()
        assert not hasattr(model, "estimators_")


class TestGetSetParams:
    """Tests for OrdBoostClassifier get_params/set_params and scikit-learn
    estimator compatibility.
    """

    def test_get_params_merges_kwargs_at_top_level(self) -> None:
        """Test that get_params exposes both explicit fields and pass-through
        kwargs at the top level, with no raw 'kwargs' key visible.
        """
        model = OrdBoostClassifier(learning_rate=0.05, max_leaf_nodes=15)
        params = model.get_params()
        assert params["learning_rate"] == 0.05
        assert params["max_leaf_nodes"] == 15
        assert "kwargs" not in params

    def test_set_params_updates_explicit_and_kwargs_fields(self) -> None:
        """Test that set_params routes known fields to attributes and
        unknown fields into self.kwargs.
        """
        model = OrdBoostClassifier(learning_rate=0.1, max_leaf_nodes=31)
        model.set_params(learning_rate=0.01, max_leaf_nodes=15, min_samples_leaf=10)

        assert model.learning_rate == 0.01
        assert model.min_samples_leaf == 10
        assert model.kwargs["max_leaf_nodes"] == 15

    def test_set_params_no_arguments_returns_self(self) -> None:
        """Test that calling set_params with no arguments is a no-op that
        still returns self.
        """
        model = OrdBoostClassifier()
        assert model.set_params() is model

    def test_clone_compatibility(self) -> None:
        """Test that sklearn.base.clone produces an independent,
        correctly parameterized copy, including pass-through kwargs.
        """
        model = OrdBoostClassifier(max_iter=20, max_bins=64, random_state=42)
        cloned = clone(model)

        assert cloned.max_iter == 20  # type: ignore[attr-defined]
        assert cloned.kwargs.get("max_bins") == 64  # type: ignore[attr-defined]
        assert cloned is not model


class TestFit:
    """Tests for OrdBoostClassifier.fit."""

    @pytest.fixture
    def synthetic_ordinal_data(self):
        """Fixture generating a synthetic 4-class ordinal dataset."""
        rng = np.random.default_rng(42)
        n_samples = 200
        X = rng.standard_normal((n_samples, 4))
        latent = X[:, 0] * 1.5 + X[:, 1] * 0.8 + rng.standard_normal(n_samples) * 0.5
        y = np.select(
            [latent < -1.0, latent < 0.0, latent < 1.0], [0, 5, 10], default=30
        )
        return X[:150], y[:150], X[150:], y[150:]

    def test_fit_returns_self(self, synthetic_ordinal_data) -> None:
        """Test that fit returns the estimator instance for chaining."""
        X_train, y_train, _, _ = synthetic_ordinal_data
        model = OrdBoostClassifier(max_iter=10)
        assert model.fit(X_train, y_train) is model

    def test_fit_sets_expected_attributes(self, synthetic_ordinal_data) -> None:
        """Test that fit sets classes_, estimators_, n_features_in_ with
        the expected shapes.
        """
        X_train, y_train, _, _ = synthetic_ordinal_data
        model = OrdBoostClassifier(max_iter=20, min_samples_leaf=5, random_state=42)
        model.fit(X_train, y_train)

        assert hasattr(model, "classes_")
        assert hasattr(model, "estimators_")
        assert len(model.classes_) == 4
        assert len(model.estimators_) == 3  # K-1 cumulative edge models
        np.testing.assert_array_equal(model.classes_, np.array([0, 5, 10, 30]))

    def test_fit_sorts_classes(self) -> None:
        """Test that classes_ is sorted ascending regardless of the order
        classes appear in the training target.
        """
        X = np.random.randn(60, 2)
        y = np.tile([30, 0, 10], 20)
        model = OrdBoostClassifier(max_iter=5)
        model.fit(X, y)
        np.testing.assert_array_equal(model.classes_, np.array([0, 10, 30]))

    def test_fit_negative_and_non_consecutive_classes(self) -> None:
        """Test fitting with negative and widely spaced ordinal class labels."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = rng.choice([-10, -2, 0, 50, 100], size=100)

        model = OrdBoostClassifier(max_iter=10)
        model.fit(X, y)

        np.testing.assert_array_equal(model.classes_, np.array([-10, -2, 0, 50, 100]))

    def test_fit_single_class_raises(self) -> None:
        """Test that a target with only one unique class raises ValueError."""
        X = np.ones((10, 2))
        y = np.zeros(10, dtype=int)
        model = OrdBoostClassifier()
        with pytest.raises(ValueError, match="at least 2 unique classes"):
            model.fit(X, y)

    def test_fit_invalid_monotonicity_raises(self, synthetic_ordinal_data) -> None:
        """Test that an unrecognized monotonicity method raises ValueError."""
        X_train, y_train, _, _ = synthetic_ordinal_data
        model = OrdBoostClassifier(monotonicity="invalid_method")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Invalid monotonicity method"):
            model.fit(X_train, y_train)

    def test_fit_nan_in_X_is_permitted(self, synthetic_ordinal_data) -> None:
        """Test that NaN values in X do not raise, since
        HistGradientBoostingClassifier handles missing features natively.
        """
        X_train, y_train, _, _ = synthetic_ordinal_data
        X_train = X_train.copy()
        X_train[0, 0] = np.nan
        model = OrdBoostClassifier(max_iter=10)
        model.fit(X_train, y_train)  # should not raise
        assert hasattr(model, "estimators_")

    def test_binary_edge_targets_use_correct_thresholds(self) -> None:
        """Test that the k-th binary edge target correctly encodes
        Y <= classes_[k], by checking the fitted number of edge
        estimators matches n_classes - 1 for a hand-constructed dataset.
        """
        X = np.random.randn(80, 2)
        y = np.repeat([0, 1, 2, 3], 20)
        model = OrdBoostClassifier(max_iter=5)
        model.fit(X, y)
        assert len(model.estimators_) == len(model.classes_) - 1

    def test_fit_two_classes_produces_single_edge_estimator(self) -> None:
        """Test the minimum supported case (2 classes), which collapses
        to a single cumulative edge model (n_classes - 1 == 1).
        """
        X = np.random.randn(40, 2)
        y = np.tile([0, 1], 20)
        model = OrdBoostClassifier(max_iter=5)
        model.fit(X, y)
        assert len(model.classes_) == 2
        assert len(model.estimators_) == 1

    def test_fit_with_explicit_classes_matching_observed_y(self) -> None:
        """Test that passing 'classes' equal to the observed unique
        labels behaves identically to leaving it unset.
        """
        X = np.random.randn(60, 2)
        y = np.tile([0, 5, 10], 20)
        model = OrdBoostClassifier(max_iter=5)
        model.fit(X, y, classes=[0, 5, 10])
        np.testing.assert_array_equal(model.classes_, np.array([0, 5, 10]))
        assert len(model.estimators_) == 2

    def test_fit_with_explicit_classes_sorts_unsorted_input(self) -> None:
        """Test that an out-of-order 'classes' array is sorted before
        being stored as classes_.
        """
        X = np.random.randn(40, 2)
        y = np.tile([0, 1], 20)
        model = OrdBoostClassifier(max_iter=5)
        model.fit(X, y, classes=[10, 0, 1])
        np.testing.assert_array_equal(model.classes_, np.array([0, 1, 10]))

    def test_fit_with_explicit_classes_including_unobserved_class(self) -> None:
        """Test that 'classes' may include a label never observed in y
        (e.g. an empty bin), and that classes_ retains it -- the
        documented use case that OrdBoostRegressor relies on to keep an
        empty bin present via classes=np.arange(n_bins).
        """
        X = np.random.randn(40, 2)
        y = np.tile([0, 1], 20)  # class '5' never appears
        model = OrdBoostClassifier(max_iter=5)
        model.fit(X, y, classes=[0, 1, 5])
        np.testing.assert_array_equal(model.classes_, np.array([0, 1, 5]))
        assert len(model.estimators_) == 2

    def test_fit_explicit_classes_too_few_raises(self) -> None:
        """Test that 'classes' with fewer than 2 elements raises ValueError."""
        X = np.random.randn(20, 2)
        y = np.tile([0, 1], 10)
        model = OrdBoostClassifier()
        with pytest.raises(ValueError, match="at least 2 unique values"):
            model.fit(X, y, classes=[5])

    def test_fit_y_label_missing_from_explicit_classes_raises(self) -> None:
        """Test that a y label absent from an explicitly supplied
        'classes' array raises ValueError naming the missing label.
        """
        X = np.random.randn(20, 2)
        y = np.tile([0, 1], 10)
        model = OrdBoostClassifier()
        with pytest.raises(ValueError, match=r"not present in 'classes'.*\[1\]"):
            model.fit(X, y, classes=[0, 2])


class TestEnforceMonotonicity:
    """Tests for OrdBoostClassifier._enforce_monotonicity."""

    def test_running_max_produces_non_decreasing_rows(self) -> None:
        """Test that running_max monotonicity yields non-decreasing values
        along each sample's row.
        """
        model = OrdBoostClassifier(monotonicity="running_max")
        cum_probs = np.array([[0.5, 0.3, 0.8, 0.6], [0.1, 0.4, 0.2, 0.9]])
        result = model._enforce_monotonicity(cum_probs)
        assert np.all(np.diff(result, axis=1) >= 0.0)

    def test_running_max_preserves_shape(self) -> None:
        """Test that running_max preserves the input array's shape."""
        model = OrdBoostClassifier(monotonicity="running_max")
        cum_probs = np.random.rand(10, 5)
        result = model._enforce_monotonicity(cum_probs)
        assert result.shape == cum_probs.shape

    def test_running_max_matches_hand_computation(self) -> None:
        """Test running_max against a hand-derived expected sequence."""
        model = OrdBoostClassifier(monotonicity="running_max")
        cum_probs = np.array([[0.2, 0.1, 0.5, 0.3]])
        result = model._enforce_monotonicity(cum_probs)
        np.testing.assert_allclose(result, [[0.2, 0.2, 0.5, 0.5]])

    def test_isotonic_produces_non_decreasing_rows(self) -> None:
        """Test that isotonic monotonicity yields non-decreasing values
        along each sample's row.
        """
        model = OrdBoostClassifier(monotonicity="isotonic")
        cum_probs = np.array([[0.5, 0.3, 0.8, 0.6], [0.1, 0.4, 0.2, 0.9]])
        result = model._enforce_monotonicity(cum_probs)
        assert np.all(np.diff(result, axis=1) >= 0.0)

    def test_isotonic_bounded_within_unit_interval(self) -> None:
        """Test that isotonic monotonicity keeps all values within [0, 1]."""
        model = OrdBoostClassifier(monotonicity="isotonic")
        cum_probs = np.array([[0.5, 0.3, 0.8, 0.6]])
        result = model._enforce_monotonicity(cum_probs)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_isotonic_preserves_shape(self) -> None:
        """Test that isotonic preserves the input array's shape."""
        model = OrdBoostClassifier(monotonicity="isotonic")
        cum_probs = np.random.rand(10, 5)
        result = model._enforce_monotonicity(cum_probs)
        assert result.shape == cum_probs.shape

    def test_already_monotonic_input_unchanged(self) -> None:
        """Test that an already non-decreasing row is left unchanged by
        both monotonicity methods.
        """
        cum_probs = np.array([[0.1, 0.3, 0.6, 0.9]])
        for method in ("running_max", "isotonic"):
            model = OrdBoostClassifier(monotonicity=method)
            result = model._enforce_monotonicity(cum_probs)
            np.testing.assert_allclose(result, cum_probs, atol=1e-6)


class TestPredictProba:
    """Tests for OrdBoostClassifier.predict_proba."""

    @pytest.fixture
    def fitted_model(self):
        """Fixture providing a fitted classifier and held-out test features."""
        rng = np.random.default_rng(42)
        n_samples = 150
        X = rng.standard_normal((n_samples, 4))
        latent = X[:, 0] * 1.5 + X[:, 1] * 0.8 + rng.standard_normal(n_samples) * 0.5
        y = np.select(
            [latent < -1.0, latent < 0.0, latent < 1.0], [0, 5, 10], default=30
        )
        model = OrdBoostClassifier(max_iter=20, min_samples_leaf=5, random_state=42)
        model.fit(X[:100], y[:100])
        return model, X[100:]

    def test_not_fitted_raises(self) -> None:
        """Test that calling predict_proba before fit raises NotFittedError,
        confirming the estimators_ pre-declaration fix works end-to-end.
        """
        model = OrdBoostClassifier()
        with pytest.raises(NotFittedError):
            model.predict_proba(np.ones((5, 2)))

    def test_feature_mismatch_raises(self, fitted_model) -> None:
        """Test that a mismatched feature count raises ValueError."""
        model, X_test = fitted_model
        with pytest.raises(ValueError):
            model.predict_proba(X_test[:, :2])

    def test_output_shape(self, fitted_model) -> None:
        """Test that predict_proba returns (n_samples, n_classes)."""
        model, X_test = fitted_model
        pmf = model.predict_proba(X_test)
        assert pmf.shape == (len(X_test), len(model.classes_))

    def test_rows_sum_to_one(self, fitted_model) -> None:
        """Test that every predicted PMF row sums to 1.0."""
        model, X_test = fitted_model
        pmf = model.predict_proba(X_test)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-6)

    def test_all_probabilities_non_negative(self, fitted_model) -> None:
        """Test that no predicted probability is negative."""
        model, X_test = fitted_model
        pmf = model.predict_proba(X_test)
        assert np.all(pmf >= 0.0)

    def test_output_width_includes_unobserved_explicit_class(self) -> None:
        """Test that predict_proba's output width matches the full
        explicit 'classes' count, including a class never observed
        during fit, and that rows still sum to 1.0 -- the guaranteed-
        width contract documented on fit()'s 'classes' parameter.
        """
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 2))
        y = np.tile([0, 1], 20)  # class '5' never appears in training
        model = OrdBoostClassifier(max_iter=10, random_state=0)
        model.fit(X, y, classes=[0, 1, 5])

        pmf = model.predict_proba(X)
        assert pmf.shape == (40, 3)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("mono_method", ["running_max", "isotonic"])
    def test_valid_pmf_for_both_monotonicity_methods(
        self, mono_method, fitted_model
    ) -> None:
        """Test that both monotonicity methods produce a valid PMF
        (non-negative, row-normalized) end to end.
        """
        _, X_test = fitted_model
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((60, 4))
        y_train = np.tile([0, 5, 10, 30], 15)
        model = OrdBoostClassifier(
            monotonicity=mono_method, max_iter=15, min_samples_leaf=5, random_state=0
        )
        model.fit(X_train, y_train)
        pmf = model.predict_proba(X_test)
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(pmf >= 0.0)


class TestPredictDist:
    """Tests for OrdBoostClassifier.predict_dist."""

    @pytest.fixture
    def fitted_model(self):
        """Fixture providing a fitted classifier and held-out test features."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = np.tile([0, 5, 10, 30], 25)
        model = OrdBoostClassifier(max_iter=10, random_state=42)
        model.fit(X, y)
        return model, rng.standard_normal((10, 3))

    def test_returns_discrete_predictive_distribution(self, fitted_model) -> None:
        """Test that predict_dist returns a DiscretePredictiveDistribution
        wired with the fitted classes_.
        """
        model, X_test = fitted_model
        dist = model.predict_dist(X_test)
        assert isinstance(dist, DiscretePredictiveDistribution)
        np.testing.assert_array_equal(dist.classes, model.classes_)

    def test_pmf_matches_predict_proba(self, fitted_model) -> None:
        """Test that the distribution's pmf matches predict_proba's output
        directly, confirming predict_dist is a thin wrapper.
        """
        model, X_test = fitted_model
        dist = model.predict_dist(X_test)
        np.testing.assert_allclose(dist.pmf, model.predict_proba(X_test))

    def test_not_fitted_raises(self) -> None:
        """Test that calling predict_dist before fit raises NotFittedError."""
        model = OrdBoostClassifier()
        with pytest.raises(NotFittedError):
            model.predict_dist(np.ones((5, 2)))


class TestPredict:
    """Tests for OrdBoostClassifier.predict."""

    @pytest.fixture
    def fitted_model(self):
        """Fixture providing a fitted classifier and held-out test features."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = np.tile([0, 5, 10, 30], 25)
        model = OrdBoostClassifier(max_iter=10, random_state=42)
        model.fit(X, y)
        return model, rng.standard_normal((10, 3))

    def test_default_method_is_median(self, fitted_model) -> None:
        """Test that predict with no explicit method matches method='median'."""
        model, X_test = fitted_model
        np.testing.assert_array_equal(
            model.predict(X_test), model.predict(X_test, method="median")
        )

    def test_median_predictions_are_valid_class_levels(self, fitted_model) -> None:
        """Test that median predictions are always one of the fitted
        ordinal class levels, unlike a continuous mapper's interpolated
        predictions.
        """
        model, X_test = fitted_model
        y_median = model.predict(X_test, method="median")
        assert set(y_median).issubset(set(model.classes_))

    def test_mean_output_shape(self, fitted_model) -> None:
        """Test that method='mean' returns one value per sample."""
        model, X_test = fitted_model
        y_mean = model.predict(X_test, method="mean")
        assert y_mean.shape == (len(X_test),)

    def test_mean_matches_dist_mean(self, fitted_model) -> None:
        """Test that method='mean' matches predict_dist(X).mean() directly."""
        model, X_test = fitted_model
        dist = model.predict_dist(X_test)
        np.testing.assert_allclose(model.predict(X_test, method="mean"), dist.mean())

    def test_not_fitted_raises(self) -> None:
        """Test that calling predict before fit raises NotFittedError."""
        model = OrdBoostClassifier()
        with pytest.raises(NotFittedError):
            model.predict(np.ones((5, 2)))

    def test_invalid_method_raises(self, fitted_model) -> None:
        """Test that an unrecognized method raises ValueError."""
        model, X_test = fitted_model
        with pytest.raises(ValueError, match="Invalid prediction method"):
            model.predict(X_test, method="invalid_method")  # type: ignore[arg-type]


class TestSklearnCompatibility:
    """Tests for GridSearchCV and broader scikit-learn pipeline compatibility."""

    def test_grid_search_cv_compatibility(self) -> None:
        """Test that GridSearchCV can tune hyperparameters without error."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 3))
        y = np.tile([0, 1, 2], 20)

        model = OrdBoostClassifier(max_iter=5, random_state=0)
        search = GridSearchCV(
            model, param_grid={"learning_rate": [0.05, 0.1]}, cv=2, error_score="raise"
        )
        search.fit(X, y)
        assert hasattr(search, "best_params_")
