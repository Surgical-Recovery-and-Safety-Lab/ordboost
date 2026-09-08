#### Performance evaluation
This example demonstrates how to assess a fitted model's predictive accuracy using
ordboost.metrics. Three metrics are used: 
a plain point-prediction error (MAE), a proper scoring
rule over the full predicted distribution (CRPS), and a skill score
comparing the model against a naive, covariate-free baseline (CRPSS).

First let's create some synthetic data. The data is split into a train and testing
set. 

```python linenums="1"
{%
    include "../../examples/performance_evaluation.py"
    end="# --- Fit"
%}
```

Now we can create the `OrdBoostRegressor` and fit the model. We opted for showing
a model creation with most default settings. Hence, the model uses the 
`EmpiricalMedianBinMapper` as a mapping strategy and the bins are created using
the quantiles as well as the min and max values of `y_train` for a total of 11
bin edges. 

```python linenums="1"
{%
    include "../../examples/performance_evaluation.py"
    start="# --- Fit"
    end="# --- Predict"
%}
```

Once the model has been fitted on the train data, it can be evaluated on the test
data. We are going to measure three different metrics to evaluate the accuracy
of the model. The first is the mean absolute error (MAE) between the ground truth
`y_test` outcomes and the model median predictions.

```python linenums="1"
{%
    include "../../examples/performance_evaluation.py"
    start="# --- Predict"
    end="# --- CRPS ---"
%}
```

The continuous ranked probability score (CRPS) is a generalisation of the MAE to
distributional predictions, such as the ones generated with `OrdBoostRegressor`. 
The `crps_score` function therefore needs the predictive distributions and not the
point-estimate predictions. 

```python linenums="1"
{%
    include "../../examples/performance_evaluation.py"
    start="# --- CRPS ---"
    end="# --- CRPS skill score"
%}
```

The CRPS skill score (CRPSS) compares the CRPS from a naive baseline with the CRPS
from the model. The baseline is built from the full training set, sized to match
the evaluation set (`y_test`), and represents the best achievable forecast
with zero patient/sample-specific information. Every sample gets the
same unconditional distribution. A CRPSS of 0 means no better than this
baseline; 1 is a perfect forecast; negative values are worse than the
baseline.

```python linenums="1"
{%
    include "../../examples/performance_evaluation.py"
    start="# --- CRPS skill score"
%}
```


The results should be:
```
MAE:   3.824
CRPS:  2.813
CRPSS: 0.514
```

Full example:

```python linenums="1", title="Evaluating performances with MAE, CRPS, and CRPSS"
{%
    include "../../examples/performance_evaluation.py"
%}
```

#### Model diagnostics with PIT histograms

The Probability Integral Transform (PIT) measures how well-calibrated a probabilistic regression model is.
If predictions are properly calibrated, $U_i = F_i(y_i)$ will follow a Uniform distribution $\mathcal{U}(0, 1)$.

``` python linenums="1", title="Plotting the PIT histogram"
{%
    include "../../examples/pit_histogram.py"
%}
```

#### Evaluating interval coverage and Winkler Scores

Assess prediction interval calibration and tightness across multiple confidence levels using `interval_coverage_rate` and `winkler_score`.

``` python linenums="1", title="Interval coverage and Winkler scores"
{%
    include "../../examples/model_evaluation.py"
%}
```

---

