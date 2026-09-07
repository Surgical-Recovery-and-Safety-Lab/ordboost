# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning ([SemVer](https://semver.org/spec/v2.0.0.html)).

## [0.3.0.dev0] 2026-09-19

### Added
* **BREAKING** UniformBinMapper accepts n_points to define the number of midpoints to add.
* **BREAKING** ContinousBinMapper accepts a resolution parameter to determine the number of points.
* Github workflow to publish package on PyPI when a new released is published.
* Example scripts for each example in the MkDocs documentation.
* Package for including the example files directly into the docs.
* The `scores` packages is now installed and listed in dependencies.
* The `numba` packages is now installed and listed in dependencies.
* New metric functions: CRPS and pinball loss skill scores, marginal calibration, sharpness, and PIT diagnostics.
* One test file for each mapper class in the mappers/ test folder.
* One test file for each distribution class in the distributions/ test folder.
* New tests for the new metrics.

### Changed
* **BREAKING** Updated the structure of the BaseBinMapper to define all functions common to the mappers.
* **BREAKING** Updated the specific mapper classes to fit the new BaseBinMapper structure.
* **BREAKING** Changed the init parameters for the mappers.
* **BREAKING** ContinuousBinMapper no longer accepts an arbitrary number of grid points.
* **BREAKING** PredictiveDistribution refactored and now does more validation to avoid duplications.
* **BREAKING** DiscretePredictiveDistribution refactored to fit the new PredictiveDistribution class structure.
* **BREAKING** ContinuousPredictiveDistribution refactored to fit the new PredictiveDistribution class structure.
* Allowing Nan values to be present in train and test data.
* Isotonic regression monotonicity calling isotonic_regression now.
* The crps_score function now relies on the scores package.
* Updated tests for the metric functions.
* Replaced the example code in the documentation with the example scripts.
* The interval_coverage_rate metric now raises ValueError if alpha is not in the correct range.
* Version bumped to 0.3.0.dev0.

### Fixed
* Fixed checks in the models to correspond to the docstrings.

## [0.2.1] 2026-08-19

### Added
* Documentation deployed with MkDocs.

### Changed
* Updated the README.md sections and content.
* Updated the OrdBoostRegressor example using string mapping for the mapper creation.
* Added new badges to the README.
* Optimised the PPF and CDF functions to use vectorisation rather than for loops in the ContinuousPredictiveDistribution class.
* ContinuousPredictiveDistribution CDF function now accepts a 1D array of y values.
* Updated test cases for the ContinuousPredictiveDistribution class.

### Fixed
* Minor typo in the OrdBoostClassifier example.

## [0.2.0] 2026-08-18

### Changed
* **BREAKING** Renamed PredictiveDistribution class to DiscretePredictiveDistribution.
* Updated OrdBoostClassifier to use DiscretePredictiveDistribution.
* Updated tests for distributions module.
* Renamed example.py to classifier_example.py.
* The CRPS function accepts a Continuous or Discrete predictive distribution.
* Updated test suite for the existing CRPS and pinball loss metric functions.
* Updated typing for the OrdBoostClassifier methods.
* Updated the package __init__.py

### Added
* Abstract PredictiveDistribution class base class for predictive probability distrubtions.
* ContinousPredictiveDistribution class encapsulating continous CDF.
* Tests for ContinousPredictiveDistribution class.
* A mappers module for bin mapping classes.
* BaseBinMapper class base class for bin mappers.
* EmpiricalMeanBinMapper class fits the mapping with the bin means.
* EmpiricalMedianBinMapper class fits the mapping with the bin medians.
* QuantileBinMapper class fits the mapping using intra bin quantiles.
* UniformBinMapper class fits the mapping using geometric bin midpoints.
* ContinuousBinMapper class fits the mapping using a fine resolution grid.
* Tests for all mapper classes.
* OrdBoostRegressor class for continuous target outcomes.
* Tests for OrdBoostRegressor class.
* New regressor_example.py example in the examples/
* A function to compute the Winkler score.
* A function to compute the interval coverage.
* Tests for the Winkler score and interval converage metrics.

## [0.1.1] - 2026-08-13
### Fixed
* Fixed README badges.

## [0.1.0] - 2026-08-13
### Added
* OrdBoostClassifier model class based on cumulative binary edge models.
* PredictiveDistribution class that encapsulated a discrete PMF.
* Two evaluation metrics: CRPS and pinball loss.
* Apache 2.0 license
* Simple README.md
* Unit tests
* Github workflow to run tests on push and pull request

[Unreleased]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/compare/v0.3.0.dev0...HEAD
[0.3.0.dev0]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/compare/v0.2.1...v0.3.0.dev0
[0.2.1]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/releases/tag/v0.1.0
