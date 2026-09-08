### Quickstart examples
These examples are minimal examples for the OrdBoostClassifier and the OrdBoostRegressor models.

#### 1. OrdBoostClassifier simple example
Use `OrdBoostClassifier` for discrete ordinal problems (like Likert survey responses or stage rankings) and score probabilistic performance via discrete CRPS.

First let us generate some synthetic ordinal data to train and test the model.

``` python linenums="1"
{%
    include-markdown "../../examples/quickstart_classification.py"
    end="# --- Fit model"
%}
```

Now that the data has been generated, we can very easily fit the classifier. We impose monotonicity with the "isotonic" method and fit the model on the train set. 

``` python linenums="1"
{%
    include-markdown "../../examples/quickstart_classification.py"
    start="# --- Fit model"
    end="# --- Predict"
%}
```

Finally, we can make predictions with the model and predict the median value. Three different ways to achieve this are shown in the example. The `predict()` function with the "median" method is the most direct way. After generating the predictive distributions for the test samples, we can either use the `median()` method, or the `ppf` for quantile q = 0.5. 

``` python linenums="1"
{%
    include-markdown "../../examples/quickstart_classification.py"
    start="# --- Predict"
%}
```

The results should be:
```
Sample 1 median from predict(): 2
Sample 1 median from median(): 2
Sample 1 median from ppf(0.5): 2
```

``` python linenums="1", title="OrdBoost classification example"
{%
    include-markdown "../../examples/quickstart_classification.py"
%}
```

#### 2. OrdBoostRegressor simple example

Use `OrdBoostRegressor` for continuous problems and estimate prediction intervals for the outcome.

``` python linenums="1", title="OrdBoost regression example"
{%
    include-markdown "../../examples/quickstart_regression.py"
%}
```


---
