# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to Semantic Versioning ([SemVer](https://semver.org/spec/v2.0.0.html)).

## Unreleased

### Changed
* **BREAKING** Renamed PredictiveDistribution class to DiscretePredictiveDistribution.
* Updated OrdBoostClassifier to use DiscretePredictiveDistribution.
* Updated tests for distributions module.
* Renamed example.py to classifier_example.py

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
* Tests for all mapper classes.
* OrdBoostRegressor class for continuous target outcomes.
* Tests for OrdBoostRegressor class.
* New regressor_example.py example in the examples/

## [0.1.1] - 2026-08-13
### Fixes
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

[0.1.1]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Surgical-Recovery-and-Safety-Lab/ordboost/releases/tag/v0.1.0
