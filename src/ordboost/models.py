"""Ordinal Gradient Boosting Classifier compatible with scikit-learn."""

from typing import Any, Literal, cast

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ordboost.distributions import PredictiveDistribution


class OrdBoostClassifier(BaseEstimator, ClassifierMixin):
    """Ordinal Gradient Boosting Classifier based on cumulative binary edge models.

    Parameters
    ----------
    loss : str, default="log_loss"
        The loss function to use in the binary base estimator.
    learning_rate : float, default=0.1
        The learning rate for gradient boosting.
    max_iter : int, default=100
        The maximum number of iterations (trees) for each binary classifier.
    max_depth : int | None, default=None
        The maximum depth of each tree.
    min_samples_leaf : int, default=20
        The minimum number of samples per leaf in binary trees.
    l2_regularization : float, default=0.0
        L2 regularization parameter for binary trees.
    monotonicity : {"running_max", "isotonic"}, default="running_max"
        Method used to enforce monotonicity across cumulative edge probabilities.
    n_jobs : int, default=-1
        Number of parallel jobs to run when fitting binary edge classifiers.
    random_state : int | None, default=None
        Pseudo-random number generator seed for reproducibility.
    **kwargs : dict[str, Any]
        Additional keyword arguments passed directly to `HistGradientBoostingClassifier`
        (e.g., `categorical_features`, `early_stopping`, `interaction_cst`).

    Attributes
    ----------
    classes_ : np.ndarray
        A 1D array containing sorted unique ordinal class labels.
    estimators_ : list of HistGradientBoostingClassifier
        List containing fitted binary edge estimators.
    n_features_in_ : int
        Number of features seen during `fit`.

    Methods
    -------
    get_param(deep)
        Get parameters for this estimator, including dynamically passed kwargs.
    set_param(**param)
        Set the parameters of this estimator.
    fit(X, y)
        Fit the ordinal gradient boosting model.
    predict_proba(X)
        Predict class probability mass functions (PMF) for X.
    predict_dist(X)
        Predict probability mass distributions wrapped in a `PredictiveDistribution`.
    predict(X)
        Predict point estimates (median or expected value) for X.
    """

    def __init__(
        self,
        loss: str = "log_loss",
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        monotonicity: Literal["running_max", "isotonic"] = "running_max",
        n_jobs: int = -1,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.monotonicity = monotonicity
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs

        self.estimators_: list[HistGradientBoostingClassifier] | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator, including dynamically passed kwargs.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        # Fetch standard explicit parameters from BaseEstimator
        params = super().get_params(deep=deep)

        # Remove raw 'kwargs' dictionary entry if BaseEstimator captured it
        params.pop("kwargs", None)

        # Merge extra kwargs directly into top-level parameter dictionary
        if hasattr(self, "kwargs") and isinstance(self.kwargs, dict):
            params.update(self.kwargs)

        return params

    def set_params(self, **params: Any) -> "OrdBoostClassifier":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : OrdBoostClassifier
            Estimator instance.
        """
        if not params:
            return self

        # Separate explicit init fields from additional kwargs
        valid_params = self._get_param_names()

        if not hasattr(self, "kwargs") or self.kwargs is None:
            self.kwargs = {}

        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        return self

    def _fit_single_edge(
        self,
        base_estimator: HistGradientBoostingClassifier,
        X: np.ndarray,
        y_binary: np.ndarray,
    ) -> HistGradientBoostingClassifier:
        """Fit a cloned binary edge estimator for a specific threshold P(Y <= c_k).

        Parameters
        ----------
        base_estimator : HistGradientBoostingClassifier
            The un-fitted base estimator template to clone and fit.
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features).
        y_binary : np.ndarray
            Binary target array of shape (n_samples,) indicating whether y <= c_k.

        Returns
        -------
        HistGradientBoostingClassifier
            Fitted binary classifier instance for the specified threshold.
        """
        estimator = cast(
            HistGradientBoostingClassifier,
            clone(base_estimator),
        )
        estimator.fit(X, y_binary)
        return estimator

    def fit(self, X: Any, y: Any) -> "OrdBoostClassifier":
        """Fit the ordinal gradient boosting model on training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector data.
        y : array-like of shape (n_samples,)
            Target values (ordinal class labels).

        Returns
        -------
        OrdBoostClassifier
            The fitted estimator instance.

        """
        X_arr, y_arr = check_X_y(X, y, ensure_2d=True)
        self.n_features_in_ = X_arr.shape[1]

        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            raise ValueError(
                "OrdBoostClassifier requires at least 2 unique classes in y."
            )

        self.classes_ = np.sort(unique_classes)
        n_classes = len(self.classes_)

        if self.monotonicity not in ("running_max", "isotonic"):
            raise ValueError(
                f"Invalid monotonicity method '{self.monotonicity}'. "
                f"Must be 'running_max' or 'isotonic'."
            )

        # Merge explicit hyperparameters with additional kwargs
        base_params = {
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "random_state": self.random_state,
            **self.kwargs,
        }

        base_estimator = HistGradientBoostingClassifier(**base_params)

        binary_targets = [
            (y_arr <= self.classes_[k]).astype(int) for k in range(n_classes - 1)
        ]

        fitted_estimators = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_edge)(base_estimator, X_arr, y_binary)
            for y_binary in binary_targets
        )

        self.estimators_ = list(fitted_estimators)
        return self

    def _enforce_monotonicity(self, cum_probs: np.ndarray) -> np.ndarray:
        """Enforce non-decreasing cumulative probabilities along edge thresholds.

        Parameters
        ----------
        cum_probs : np.ndarray
            2D array of shape (n_samples, n_edges) containing raw, unadjusted
            cumulative edge probability predictions.

        Returns
        -------
        np.ndarray
            2D array of shape (n_samples, n_edges) with monotonic non-decreasing
            cumulative probabilities across columns.
        """
        if self.monotonicity == "running_max":
            return np.maximum.accumulate(cum_probs, axis=1)

        from sklearn.isotonic import IsotonicRegression

        n_samples, n_edges = cum_probs.shape
        monotonic_probs = np.empty_like(cum_probs)
        x_grid = np.arange(n_edges)

        for i in range(n_samples):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
            monotonic_probs[i] = iso.fit_transform(x_grid, cum_probs[i])

        return monotonic_probs

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probability mass function (PMF) for each sample.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        np.ndarray
            2D float array of shape (n_samples, n_classes) containing class probabilities.
        """
        check_is_fitted(self, attributes=["classes_", "estimators_", "n_features_in_"])
        X_arr = check_array(X, ensure_2d=True)

        if self.estimators_ is None:
            raise NotFittedError("The estimator instance is not fitted yet.")

        n_samples = X_arr.shape[0]
        n_classes = len(self.classes_)
        n_edges = n_classes - 1

        cum_probs = np.empty((n_samples, n_edges), dtype=float)

        for k, estimator in enumerate(self.estimators_):
            prob_le = estimator.predict_proba(X_arr)[:, 1]
            cum_probs[:, k] = prob_le

        cum_probs_mono = self._enforce_monotonicity(cum_probs)

        full_cdf = np.hstack([cum_probs_mono, np.ones((n_samples, 1), dtype=float)])
        full_cdf = np.clip(full_cdf, 0.0, 1.0)

        pmf = np.empty((n_samples, n_classes), dtype=float)
        pmf[:, 0] = full_cdf[:, 0]
        pmf[:, 1:] = np.diff(full_cdf, axis=1)

        pmf = np.clip(pmf, 0.0, None)
        sums = pmf.sum(axis=1, keepdims=True)
        sums[sums == 0.0] = 1.0
        pmf = pmf / sums

        return pmf

    def predict_dist(self, X: Any) -> PredictiveDistribution:
        """Predict probability distribution wrapped in a `PredictiveDistribution`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        PredictiveDistribution
            Distribution object encapsulating predicted PMFs and class labels.
        """
        pmf = self.predict_proba(X)
        return PredictiveDistribution(pmf=pmf, classes=self.classes_)

    def predict(
        self, X: Any, method: Literal["median", "mean"] = "median"
    ) -> np.ndarray:
        """Predict target class point estimates for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features.
        method : {"median", "mean"}, default="median"
            Point prediction strategy:
            - "median": Returns 50th percentile ordinal class level.
            - "mean": Returns the expected value of the distribution.

        Returns
        -------
        np.ndarray
            1D array of predicted values in physical target units.
        """
        dist = self.predict_dist(X)
        if method == "median":
            return dist.median()
        elif method == "mean":
            return dist.mean()
        else:
            raise ValueError(
                f"Invalid prediction method '{method}'. Must be 'median' or 'mean'."
            )
