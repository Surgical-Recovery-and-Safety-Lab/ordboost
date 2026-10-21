### Writing a custom mapper

`OrdBoostRegressor` accepts any custom mapper that inherits from `BaseBinMapper`.
In this example, we will show how to create a new mapper. A minimal mapper only
needs to implement the `_intra_bin_points` method that defines which points to add
between the bin edges and their associated weight.

Below is the simplest mapper to implement: the `NullBinMapper`. It does not add any
points to the grid. Therefore, the CDFs will only be evaluated at the bin edges. 

**NOTE**: This mapper is equivalent to using the UniformBinMapper with `n_points=0`.

``` python linenums="1"
{%
    include "../../examples/custom_mapper.py"
    start="# --- Mapper definition"
    end="# --- Usage"
%}
```

Once the new mapper has been defined, we can now use it when creating an `OrdBoostRegressor`:

``` python linenums="1"
{%
    include "../../examples/custom_mapper.py"
    start="# --- Usage"
%}
```

Running this example should produce the following output:

```
Sample 0 median [95% PI]
  2.76 [1.50 -- 3.10]
Sample 1 median [95% PI]
  3.60 [3.12 -- 6.76]
Sample 2 median [95% PI]
  4.44 [1.78 -- 5.15]
Sample 3 median [95% PI]
  1.96 [1.41 -- 2.49]
Sample 4 median [95% PI]
  2.98 [2.24 -- 3.92]
```

Full example:

``` python linenums="1", title="Writing a custom mapper"
{%
    include "../../examples/custom_mapper.py"
%}
```

### Bounded data and atoms

**NOTE**: This example plots its results and needs `matplotlib`. Install it with
`pip install matplotlib`, or install the package with the development extras
(`pip install -e ".[dev]"`).

This example demonstrates the use of the `lower_bound` and `upper_bound` attributes
with and whitout atoms. We will look at the effects these have on the model.
Let's start by generating some synthetic data. For this example, we want to include
a floor atom at y=0 (zero-inflation) and a ceiling bound at y=10 with no atom.
This means that the target values y can be 0, and they approach 10 without reaching.
We are creating 10,000 samples with 8 features. We then modify the target so that
it is bound between 0 and 10, and zero-inflated (roughly 50% of data at 0).

``` python linenums="1"
{%
    include "../../examples/skewed_distribution.py"
    end="# --- Create and fit models"
%}
```

The code should generate the following figure showing the target test distribution:

![y_test distribution](../assets/target_distribution.png)

**NOTE:** the target has a large amount of data at 0 and never reaches 10.

Now that the data has been generated, we can fit our models. First, we define some
bin edges that cover the 0 to 10 range. We then create two models, one with the 
`mapper_kwargs` and one without. The models have identical hyperparameters and use
the quantile bin mapper. The first model defines four `mapper_kwargs`, the 
`lower_bound` and `upper_bound` values are floats corresponding to the limits we
have imposed on the data. The `floor_atom` is set to `True` because there is data
at 0, however, the `ceiling_atom` is set to `False` because y never reaches 10.

Once the models have been created, we can fit them with the train data and generate
the predictions and distributions. 

``` python linenums="1"
{%
    include "../../examples/skewed_distribution.py"
    start="# --- Create and fit models"
    end="# --- CRPS and skill score"
%}
```

Now let us evaluate the model to see the effects of the bounds and atoms. First, let
us look at the accuracy and skill scores with the MAE, CRPS, and CRPSS for both 
models. 

``` python linenums="1"
{%
    include "../../examples/skewed_distribution.py"
    start="# --- CRPS and skill score"
    end="# --- Prediction interval evaluation"
%}
```

The code will generate the following table:

```
        Bounded | Boundless
MAE:     1.712  |    1.846
CRPS:    0.993  |    1.126
CRPSS:   0.313  |    0.221
```

The bounded model performs better than the boundless model, with a lower CRPS and
MAE and a higher CRPSS. Now
let us take a look at the prediction intervals. We compute the interval coverage 
rate, the sharpness, and the Winkler score at difference significance levels and 
plot the results.

``` python linenums="1"
{%
    include "../../examples/skewed_distribution.py"
    start="# --- Prediction interval evaluation"
    end="# --- Calibration diagnostics"
%}
```

The code will generate the following results: 

![prediction_intervals](../assets/atom_intervals.png)

Now the difference between the bounded and boundless model becomes more apparent. 
The boundless model has too little coverage: it cannot place mass at zero, so its
intervals start above zero and miss the zero targets. The bounded model has too much
coverage.

!!! note
    The bounded model's over-coverage is expected and cannot be removed by a better
    model. About half of the targets are exactly zero, so any interval that contains
    zero already covers about 50% of them, and the coverage cannot fall much below that
    at low nominal levels. The central interval also stretches from the floor up to
    the upper quantile whenever the zero probability exceeds the tail level, which makes
    it conservative. Coverage is only meaningful at the higher nominal levels, where the
    bounded model gets close to nominal.

Finally, let's look at the calibration. We will compute the PIT histograms and the
marginal calibration curves for both models and plot the results.

``` python linenums="1"
{%
    include "../../examples/skewed_distribution.py"
    start="# --- Calibration diagnostics"
%}
```

The code will generate the following results:

![marginal_calibration](../assets/atom_marginal_calibration.png)

The marginal calibration curves of the two models are identical. Both models share
the same classifier and the same grid points, which fall on the bin edges, where the
average predicted CDF depends only on the predicted bin probabilities. The curve is
also exactly zero at the floor, since the left-hand limit of the CDF is used there.
The marginal calibration therefore cannot separate the two models on this example,
and the difference at the floor atom only shows up in the PIT histogram.

![PIT](../assets/atom_pit_histogram.png)

The PIT histograms show the effect of the floor atom. The boundless model cannot
place mass at zero, so the zero targets pile up in the first PIT bin, which holds
about half of the observations, and the $\alpha$ score is 0.189. In contrast, the
bounded model has a PIT histogram that is near uniform and an $\alpha$ score of
0.011, about 17 times smaller.

These results clearly show the effects of using the atoms and bounds to constrain
the models.

Full example:

``` python linenums="1", title="Bounds and atoms"
{%
    include "../../examples/skewed_distribution.py"
%}
```

---

