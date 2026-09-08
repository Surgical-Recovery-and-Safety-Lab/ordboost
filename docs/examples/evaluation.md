### Evaluating model quality

#### 3. Model diagnostics with PIT histograms

The Probability Integral Transform (PIT) measures how well-calibrated a probabilistic regression model is.
If predictions are properly calibrated, $U_i = F_i(y_i)$ will follow a Uniform distribution $\mathcal{U}(0, 1)$.

``` python linenums="1", title="Plotting the PIT histogram"
{%
    include "../../examples/pit_histogram.py"
%}
```

#### 4. Evaluating interval coverage and Winkler Scores

Assess prediction interval calibration and tightness across multiple confidence levels using `interval_coverage_rate` and `winkler_score`.

``` python linenums="1", title="Interval coverage and Winkler scores"
{%
    include "../../examples/model_evaluation.py"
%}
```

---

