
### Quickstart examples
#### 1. Discrete ordinal ranking and CRPS evaluation
Use `OrdBoostClassifier` for discrete ordinal problems (like Likert survey responses or stage rankings) and score probabilistic performance via discrete CRPS.

``` python linenums="1", title="OrdBoost classification example"
{%
    include-markdown "../../examples/quickstart_classification.py"
%}
```

#### 2. Continuous predictions with prediction intervals

Use `OrdBoostRegressor` for continuous problems and estimate prediction intervals for the outcome.

``` python linenums="1", title="OrdBoost regression example"
{%
    include-markdown "../../examples/quickstart_regression.py"
%}
```

---


