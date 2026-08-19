# Welcome to OrdBoost

**ordboost** is a Python library for non-parametric discrete ordinal gradient boosting and continuous regression. 

By reframing continuous targets as discrete ordinal binning problems, **ordboost** models complex, skewed, or multimodal target distributions without forcing restrictive parametric assumptions.

---

## Key Features

* **Scikit-Learn API Compatibility**: Fits seamlessly into standard ML workflows using `fit`, `predict`, and `predict_dist`.
* **Non-Parametric Probabilistic Output**: Obtains complete predictive distribution objects capable of extracting probability mass functions (PMF), cumulative distribution functions (CDF), percentiles (`ppf`), and dynamic prediction intervals.
* **Flexible Continuous Target Mapping**: Maps continuous values to discrete target spaces using configurable binning strategies (`QuantileBinMapper`, `UniformBinMapper`, `EmpiricalMeanBinMapper`, `EmpiricalMedianBinMapper`, `ContinuousBinMapper`).
* **Monotonic Ordinal Constraints**: Supports constrained ordinal boosting (e.g., isotonic constraints) across sequential boundaries.
* **Built-in Probabilistic Evaluation**: Evaluates probabilistic predictions directly using CRPS (`crps_score`), quantile loss (`pinball_loss`), prediction interval coverage (`interval_coverage_rate`), and Winkler scores (`winkler_score`).

---

## Quickstart example

Get started with continuous probabilistic regression in just a few lines of code:

``` py linenums="1", title="OrdBoost regression example"
{%
    include "../examples/quickstart_regression.py"
%}
```

## Installation

### Standard Installation

Install the published package directly from PyPI:

```bash
pip install ordboost

```

### Installing from source (Developer setup)

To set up a local development environment and contribute to **ordboost**:

1. Clone the GitHub repository:

```bash
git clone [https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost.git](https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost.git)
cd ordboost

```

2. Install the package in editable mode with development dependencies:
```bash
pip install -e ".[dev]"

```

---

## Explore examples

For step-by-step code walkthroughs and diagnostic workflows, visit the **[Examples Page](examples.md)**:

* [Quickstart examples](examples.md#quickstart-examples): Basic discrete ordinal ranking and continuous prediction interval workflows.
* [Evaluating model quality](examples.md#evaluating-model-quality): Model calibration diagnostics using PIT histograms, interval coverage, and Winkler scores.
* [Customising Modelling Strategies](examples.md#customising-modelling-strategies): Creating custom target mappers by extending `BaseBinMapper`.

---

## Contributing

We welcome contributions from the community! Whether you are fixing bugs, improving documentation, or proposing new features:

1. Feel free to open an issue or start a discussion on our **[GitHub Repository](https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost)**.

2. Submit Pull Requests targeting the `main` branch.
3. Ensure all unit tests pass before submitting (`pytest`).

---

## License

This project is licensed under the Apache-2.0 License. Developed and maintained by the **[Surgical Recovery and Safety Lab](https://github.com/Surgical-Recovery-and-Safety-Lab)**.

