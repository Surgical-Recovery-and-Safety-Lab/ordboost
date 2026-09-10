"""Ordinal Gradient Boosting Classifier compatible with scikit-learn."""

from typing import Any, Literal, Union, cast

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ordboost.distributions import (
    ContinuousPredictiveDistribution,
    DiscretePredictiveDistribution,
)
from ordboost.mappers import BaseBinMapper


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
    get_params(deep)
        Get parameters for this estimator, including dynamically passed kwargs.
    set_params(**param)
        Set the parameters of this estimator.
    fit(X, y)
        Fit the ordinal gradient boosting model.
    predict_proba(X)
        Predict class probability mass functions (PMF) for X.
    predict_dist(X)
        Predict probability mass distributions wrapped in a `DiscretePredictiveDistribution`.
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

    def fit(
        self, X: ArrayLike, y: ArrayLike, classes: Union[ArrayLike, None] = None
    ) -> "OrdBoostClassifier":
        """Fit the ordinal gradient boosting model on training data.

        Trains one binary `HistGradientBoostingClassifier` per cumulative
        edge threshold `P(Y <= c_k)` for `k = 0, ..., n_classes - 2`, then
        stores the fitted edge estimators for later monotonicity enforcement
        and PMF construction in `predict_proba`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector data. May contain NaN values, which are handled
            natively by the underlying `HistGradientBoostingClassifier`.
        y : array-like of shape (n_samples,)
            Target values (ordinal class labels).
        classes : array-like, optional
            The full set of ordinal class labels the model should support,
            including labels that may not appear in this particular `y`
            (e.g. an empty bin in a specific training sample). If None
            (default), classes are inferred from `np.unique(y)`. If
            provided, `classes_` is set to this full sorted set, guaranteeing
            `predict_proba`'s output width matches the caller's expected
            class count regardless of which classes are actually observed --
            the same convention scikit-learn uses in e.g.
            `SGDClassifier.partial_fit(classes=...)`.

        Returns
        -------
        OrdBoostClassifier
            The fitted estimator instance.

        Raises
        ------
        ValueError
            If `y` contains fewer than 2 unique classes when `classes` is
            not provided, if `classes` has fewer than 2 elements, if `y`
            contains a label not present in `classes`, or if `monotonicity`
            is not `"running_max"` or `"isotonic"`.

        """
        X_arr, y_arr = check_X_y(
            X, y, ensure_2d=True, ensure_all_finite=False, accept_sparse=False
        )
        self.n_features_in_ = X_arr.shape[1]

        unique_classes = np.unique(y_arr)

        if classes is not None:
            classes_arr = np.sort(np.asarray(classes))
            if len(classes_arr) < 2:
                raise ValueError("'classes' must contain at least 2 unique values.")
            if not np.all(np.isin(unique_classes, classes_arr)):
                missing = np.setdiff1d(unique_classes, classes_arr)
                raise ValueError(
                    f"'y' contains class labels not present in 'classes': {missing}."
                )
            self.classes_ = classes_arr
        else:
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

        self.estimators_ = cast(
            list[HistGradientBoostingClassifier], list(fitted_estimators)
        )
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

        from sklearn.isotonic import isotonic_regression

        n_samples, n_edges = cum_probs.shape
        monotonic_probs = np.empty_like(cum_probs)
        x_grid = np.arange(n_edges)

        for i in range(n_samples):
            monotonic_probs[i] = isotonic_regression(
                cum_probs[i], y_min=0.0, y_max=1.0, increasing=True
            )

        return monotonic_probs

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict probability mass function (PMF) for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features. May contain NaN values, which are handled
            natively by the underlying `HistGradientBoostingClassifier`.

        Returns
        -------
        np.ndarray
            2D float array of shape (n_samples, n_classes) containing class
            probabilities, guaranteed non-negative and row-normalized to sum
            to 1.0.

        Raises
        ------
        NotFittedError
            If called before `fit`.
        ValueError
            If `X`'s feature count does not match `n_features_in_`.

        """
        check_is_fitted(self, attributes=["classes_", "estimators_", "n_features_in_"])
        X_arr = check_array(X, ensure_2d=True, ensure_all_finite=False)

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

    def predict_dist(self, X: ArrayLike) -> DiscretePredictiveDistribution:
        """Predict probability distribution wrapped in a
        `DiscretePredictiveDistribution`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        DiscretePredictiveDistribution
            Distribution object encapsulating predicted PMFs and class labels.

        Raises
        ------
        NotFittedError
            If called before `fit` (raised by `predict_proba`).
        ValueError
            If `X`'s feature count does not match `n_features_in_` (raised
            by `predict_proba`).

        """
        pmf = self.predict_proba(X)
        return DiscretePredictiveDistribution(pmf=pmf, classes=self.classes_)

    def predict(
        self, X: ArrayLike, method: Literal["median", "mean"] = "median"
    ) -> np.ndarray:
        """Predict target class point estimates for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        method : {"median", "mean"}, default="median"
            Point prediction strategy:
            - "median": Returns 50th percentile ordinal class level.
            - "mean": Returns the expected value of the distribution.

        Returns
        -------
        np.ndarray
            1D array of predicted values in physical target units.

        Raises
        ------
        NotFittedError
            If called before `fit` (raised by `predict_dist`).
        ValueError
            If `X`'s feature count does not match `n_features_in_` (raised
            by `predict_dist`), or if `method` is not `"median"` or `"mean"`.

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


class OrdBoostRegressor(BaseEstimator, RegressorMixin):
    """Ordinal Gradient Boosting Regressor for continuous target outcomes.

    Discretizes a continuous target into ordinal bins, fits an underlying
    cumulative binary `OrdBoostClassifier` on the binned target, and maps
    predicted probability distributions back to continuous target space
    using a fitted bin mapper. `OrdBoostRegressor` is the single source
    of truth for bin edges and digitization: both the classifier and the
    mapper are fitted against the same `bin_edges_`/`y_binned` computed
    once in `fit`, so they cannot disagree about bin membership.

    Parameters
    ----------
    n_bins : int, default=20
        Number of discrete bins to construct if `bin_edges` is None.
    bin_edges : ArrayLike of shape (n_bins + 1,), optional
        Monotonically increasing boundaries defining continuous bin
        intervals. If provided, takes precedence over `n_bins`/`bin_strategy`.
    bin_strategy : {"quantile", "uniform"}, default="quantile"
        Strategy used to define automatic bin boundaries when `bin_edges`
        is None.
    mapper : BaseBinMapper, {"median", "mean", "quantile", "uniform", "continuous"} or None, default="median"
        Bin mapping strategy used to convert predicted PMFs back to
        continuous predictions. A string selects the corresponding
        `BaseBinMapper` subclass, constructed automatically with
        `bin_edges_` and any `mapper_kwargs`. A `BaseBinMapper` instance
        is cloned and fitted with `bin_edges_`. `None` is equivalent to
        `"median"`.
    mapper_kwargs : dict[str, Any] | None, default=None
        Optional keyword arguments passed when instantiating a
        string-shortcut mapper (e.g. `{"quantiles": (0.1, 0.5, 0.9)}` for
        `mapper="quantile"`). Ignored if `mapper` is a `BaseBinMapper`
        instance.
    learning_rate : float, default=0.1
        Learning rate for gradient boosting.
    max_iter : int, default=100
        Maximum number of iterations (trees) per cumulative edge model.
    max_depth : int | None, default=None
        Maximum depth of each tree.
    min_samples_leaf : int, default=20
        Minimum number of samples per leaf.
    l2_regularization : float, default=0.0
        L2 regularization parameter.
    monotonicity : {"running_max", "isotonic"}, default="running_max"
        Cumulative probability monotonicity enforcement method.
    n_jobs : int, default=-1
        Number of parallel jobs to run when fitting edge classifiers.
    random_state : int | None, default=None
        Random state seed.
    **kwargs : dict[str, Any]
        Additional arguments passed to the underlying
        `HistGradientBoostingClassifier` (e.g. `categorical_features`,
        `early_stopping`).

    Attributes
    ----------
    bin_edges_ : np.ndarray
        1D float array of shape (n_bins + 1,) containing resolved bin
        edges, set during `fit`.
    classifier_ : OrdBoostClassifier
        Fitted underlying ordinal gradient boosting classifier.
    mapper_ : BaseBinMapper
        Fitted bin mapper instance.
    n_features_in_ : int
        Number of features seen during `fit`.

    Methods
    -------
    get_params(deep)
        Get parameters for this estimator, including dynamically passed kwargs.
    set_params(**param)
        Set the parameters of this estimator.
    fit(X, y)
        Fit the continuous ordinal gradient boosting regressor.
    predict_dist(X)
        Predict continuous cumulative distribution functions wrapped in a distribution.
    predict(X, method="mean")
        Predict continuous target point estimates.

    """

    def __init__(
        self,
        n_bins: int = 20,
        bin_edges: Union[ArrayLike, None] = None,
        bin_strategy: Literal["quantile", "uniform"] = "quantile",
        mapper: Union[
            Literal["median", "mean", "quantile", "uniform", "continuous"],
            BaseBinMapper,
            None,
        ] = "median",
        mapper_kwargs: Union[dict[str, Any], None] = None,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_depth: Union[int, None] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        monotonicity: Literal["running_max", "isotonic"] = "running_max",
        n_jobs: int = -1,
        random_state: Union[int, None] = None,
        **kwargs: Any,
    ) -> None:
        self.n_bins = n_bins
        self.bin_edges = bin_edges
        self.bin_strategy = bin_strategy
        self.mapper = mapper
        self.mapper_kwargs = mapper_kwargs
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.monotonicity: Literal["running_max", "isotonic"] = monotonicity
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs

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

    def set_params(self, **params: Any) -> "OrdBoostRegressor":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : OrdBoostRegressor
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

    def _compute_bin_edges(self, y: np.ndarray) -> np.ndarray:
        """Compute or validate continuous target bin boundary edges.

        Parameters
        ----------
        y : np.ndarray
            1D float array of continuous target values used to compute automatic
            bin edges when `bin_edges` is None.

        Returns
        -------
        np.ndarray
            1D float array containing strictly monotonically increasing bin edge
            thresholds.

        Raises
        ------
        ValueError
            If `bin_edges` is not 1D, has fewer than 2 elements, or is not strictly
            monotonically increasing.
            If `n_bins` is less than 2.
            If `bin_strategy` is invalid.

        """
        if self.bin_edges is not None:
            edges = np.asarray(self.bin_edges, dtype=float)
            if edges.ndim != 1 or len(edges) < 2:
                raise ValueError(
                    "Expected 'bin_edges' to be a 1D array with >= 2 edges."
                )
            if np.any(np.diff(edges) <= 0.0):
                raise ValueError(
                    "'bin_edges' must be strictly monotonically increasing."
                )
            return edges

        if self.n_bins < 2:
            raise ValueError("Parameter 'n_bins' must be >= 2.")

        if self.bin_strategy == "quantile":
            quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
            edges = np.quantile(y, quantiles)
            # Ensure unique edges if duplicates occur in dense regions
            edges = np.unique(edges)
            if len(edges) < 1:
                edges = np.linspace(np.min(y), np.max(y), self.n_bins + 1)[1:-1]
        elif self.bin_strategy == "uniform":
            edges = np.linspace(np.min(y), np.max(y), self.n_bins + 1)[1:-1]
        else:
            raise ValueError(f"Invalid bin_strategy '{self.bin_strategy}'.")

        return edges

    def _resolve_mapper(self) -> BaseBinMapper:
        """Resolve the `mapper` parameter into a fitted-ready mapper instance.

        A string shortcut is instantiated as the corresponding `BaseBinMapper`
        subclass, constructed with `bin_edges_` and any `mapper_kwargs`.
        `None` resolves identically to `"median"`. A `BaseBinMapper` instance
        is cloned (never mutated in place) and assigned `bin_edges_`.

        Returns
        -------
        BaseBinMapper
            An unfitted mapper instance with `bin_edges` set to `bin_edges_`,
            ready to be fit by the caller.

        Raises
        ------
        ValueError
            If `mapper` is a string that does not match a known shortcut; if
            `mapper` is a `BaseBinMapper` instance whose own `bin_edges` is
            already set and does not match `bin_edges_`; or if `mapper`
            is neither a recognized string nor a `BaseBinMapper` instance.

        """
        from ordboost.mappers import (
            ContinuousBinMapper,
            EmpiricalMeanBinMapper,
            EmpiricalMedianBinMapper,
            QuantileBinMapper,
            UniformBinMapper,
        )

        mapper_map: dict[str, type[BaseBinMapper]] = {
            "median": EmpiricalMedianBinMapper,
            "mean": EmpiricalMeanBinMapper,
            "quantile": QuantileBinMapper,
            "uniform": UniformBinMapper,
            "continuous": ContinuousBinMapper,
        }

        extra_kwargs = self.mapper_kwargs or {}

        if self.mapper is None or self.mapper == "median":
            mapper_obj = EmpiricalMedianBinMapper(
                bin_edges=self.bin_edges_, **extra_kwargs
            )
        elif isinstance(self.mapper, str):
            if self.mapper not in mapper_map:
                raise ValueError(
                    f"Unknown mapper shortcut '{self.mapper}'. "
                    f"Supported options are: {list(mapper_map.keys())}"
                )
            mapper_cls = mapper_map[self.mapper]
            mapper_obj = mapper_cls(bin_edges=self.bin_edges_, **extra_kwargs)
        elif isinstance(self.mapper, BaseBinMapper):
            if self.mapper.bin_edges is not None:
                supplied_edges = np.asarray(self.mapper.bin_edges, dtype=float)
                if not np.array_equal(supplied_edges, self.bin_edges_):
                    raise ValueError(
                        "The mapper instance passed to 'mapper' already has "
                        "'bin_edges' set, and they do not match the bin edges. "
                        "Construct the mapper without 'bin_edges' to have it "
                        "set automatically, or pass edges that match."
                    )
            mapper_obj = cast(BaseBinMapper, clone(self.mapper))
            mapper_obj.bin_edges = self.bin_edges_
        else:
            raise ValueError(
                "Expected 'mapper' to be a valid string shortcut or an instance "
                "of BaseBinMapper."
            )

        return mapper_obj

    def fit(self, X: ArrayLike, y: ArrayLike) -> "OrdBoostRegressor":
        """Fit the ordinal boosting regressor on continuous targets.

        Discretizes `y` into bins using `bin_edges_`, fits the underlying
        `OrdBoostClassifier`, and fits the resolved `mapper_` strategy.
        `OrdBoostRegressor` is the single source of truth for both bin
        edges and digitization: `y_binned` is computed once here via
        `BaseBinMapper.digitize` and passed explicitly to both the
        classifier and the mapper, so the two components can never disagree
        about which bin a sample belongs to.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training feature matrix. May contain NaN values, which are
            handled natively by the underlying `HistGradientBoostingClassifier`.
        y : ArrayLike of shape (n_samples,)
            Continuous target vector. Must not contain NaN or infinite
            values.

        Returns
        -------
        OrdBoostRegressor
            Fitted estimator instance.

        Raises
        ------
        ValueError
            If `y` contains NaN or infinite values, or if `bin_edges`/`n_bins`/
            `bin_strategy`/`mapper` are invalid (raised by `_compute_bin_edges`
            or `_resolve_mapper`).

        """
        X_arr, y_arr = check_X_y(
            X,
            y,
            ensure_2d=True,
            dtype="numeric",
            ensure_all_finite="allow-nan",  # type: ignore
            accept_sparse=False,
        )

        self.n_features_in_ = X_arr.shape[1]

        self.bin_edges_ = self._compute_bin_edges(y_arr)
        y_binned = BaseBinMapper.digitize(y_arr, self.bin_edges_, y_binned=None)
        n_bins = len(self.bin_edges_) + 1

        self.classifier_ = OrdBoostClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            monotonicity=self.monotonicity,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.classifier_.fit(X_arr, y_binned, classes=np.arange(n_bins))

        # Resolve and fit bin mapper
        self.mapper_ = self._resolve_mapper()
        self.mapper_.fit(y_continuous=y_arr, y_binned=y_binned)

        return self

    def predict_dist(self, X: ArrayLike) -> ContinuousPredictiveDistribution:
        """Predict probability distribution wrapped in ContinuousPredictiveDistribution.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        ContinuousPredictiveDistribution
            Predicted continuous cumulative distribution object.

        Raises
        ------
        NotFittedError
            If called before `fit`.

        """
        check_is_fitted(self, attributes=["bin_edges_", "classifier_", "mapper_"])
        X_arr = check_array(X, ensure_2d=True, ensure_all_finite=False)
        pmf = self.classifier_.predict_proba(X_arr)
        return self.mapper_.to_continuous_dist(pmf)

    def predict(
        self, X: ArrayLike, method: Literal["mean", "median"] = "mean"
    ) -> np.ndarray:
        """Predict continuous target point estimates.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input feature matrix.
        method : {"mean", "median"}, default="mean"
            Point prediction calculation method.

        Returns
        -------
        np.ndarray
            1D float array of predicted target values.

        Raises
        ------
        NotFittedError
            If called before `fit` (raised by `predict_dist`).
        ValueError
            If `method` is not `"mean"` or `"median"`.

        """
        dist = self.predict_dist(X)
        if method == "mean":
            return dist.mean()
        elif method == "median":
            return dist.median()
        else:
            raise ValueError(f"Invalid method '{method}'. Must be 'mean' or 'median'.")
