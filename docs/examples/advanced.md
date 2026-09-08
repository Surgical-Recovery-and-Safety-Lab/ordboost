### Customising modelling strategies
#### 5. Custom Bin Mapper (Inheriting from `BaseBinMapper`)

`OrdBoostRegressor` accepts any custom mapper that inherits from `BaseBinMapper`. In this example, we will show how to create a new mapper. A minimal mapper only needs to implement the `_intra_bin_points` method that defines which points to add between the bin edges and their associated weight.

Below is the simplest mapper to implement: the `NullBinMapper`. It does not add any points to the grid. Therefore, the CDFs will only be evaluated at the bin edges. 

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

``` python linenums="1", title="Creating a custom bin mapper"
{%
    include "../../examples/custom_mapper.py"
%}
```

---

