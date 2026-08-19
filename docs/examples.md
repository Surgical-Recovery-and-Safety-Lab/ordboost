## Examples

This page provides end-to-end examples demonstrating advanced features, custom extensions, and diagnostic workflows with **ordboost**.

---

### Quickstart examples
#### 1. Discrete ordinal ranking and CRPS evaluation
Use `OrdBoostClassifier` for discrete ordinal problems (like Likert survey responses or stage rankings) and score probabilistic performance via discrete CRPS.

``` python linenums="1", title="OrdBoost classification example"
{%
    include-markdown "../examples/quickstart_classification.py"
%}
```

#### 2. Continuous predictions with prediction intervals

Use `OrdBoostRegressor` for continuous problems and estimate prediction intervals for the outcome.

``` python linenums="1", title="OrdBoost regression example"
{%
    include-markdown "../examples/quickstart_regression.py"
%}
```

---

### Evaluating model quality

#### 3. Model diagnostics with PIT histograms

The Probability Integral Transform (PIT) measures how well-calibrated a probabilistic regression model is.
If predictions are properly calibrated, $U_i = F_i(y_i)$ will follow a Uniform distribution $\mathcal{U}(0, 1)$.

``` python linenums="1", title="Plotting the PIT histogram"
{%
    include "../examples/pit_histogram.py"
%}
```

#### 4. Evaluating interval coverage and Winkler Scores

Assess prediction interval calibration and tightness across multiple confidence levels using `interval_coverage_rate` and `winkler_score`.

``` python linenums="1", title="Interval coverage and Winkler scores"
{%
    include "../examples/model_evaluation.py"
%}
```

---

### Customising modelling strategies
#### 5. Custom Bin Mapper (Inheriting from `BaseBinMapper`)

`OrdBoostRegressor` accepts any custom mapper that inherits from `BaseBinMapper`. Below is an example of creating a custom version of the EmpiricalMeanBinMapper:

``` python linenums="1", title="Custom Bin Mapper example"
{%
    include "../examples/custom_mapper.py"
%}
```

---
