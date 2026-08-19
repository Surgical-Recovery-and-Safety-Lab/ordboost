# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning ([SemVer](https://semver.org/spec/v2.0.0.html)).

## [Unreleased]

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

[Unreleased]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/releases/tag/v0.1.0
