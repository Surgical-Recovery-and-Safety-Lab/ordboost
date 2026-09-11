"""Non-parametric ordinal gradient boosting and probabilistic calibration."""

from ordboost.distributions import (
    ContinuousPredictiveDistribution,
    DiscretePredictiveDistribution,
    PredictiveDistribution,
)
from ordboost.mappers import (
    BaseBinMapper,
    ContinuousBinMapper,
    EmpiricalMeanBinMapper,
    EmpiricalMedianBinMapper,
    QuantileBinMapper,
    UniformBinMapper,
)
from ordboost.metrics import (
    baseline_distribution,
    crps_score,
    crps_skill_score,
    interval_coverage_rate,
    marginal_calibration_curve,
    pinball_loss,
    pinball_loss_skill_score,
    pit_diagnostics,
    pit_ks_test,
    sharpness,
    winkler_score,
)
from ordboost.models import OrdBoostClassifier, OrdBoostRegressor

__version__ = "0.3.0"

__all__ = [
    # Models
    "OrdBoostClassifier",
    "OrdBoostRegressor",
    # Mappers
    "BaseBinMapper",
    "ContinuousBinMapper",
    "EmpiricalMeanBinMapper",
    "EmpiricalMedianBinMapper",
    "QuantileBinMapper",
    "UniformBinMapper",
    # Distributions
    "PredictiveDistribution",
    "DiscretePredictiveDistribution",
    "ContinuousPredictiveDistribution",
    # Metrics
    "baseline_distribution",
    "crps_score",
    "crps_skill_score",
    "interval_coverage_rate",
    "marginal_calibration_curve",
    "sharpness",
    "pinball_loss",
    "pinball_loss_skill_score",
    "pit_diagnostics",
    "pit_ks_test",
    "winkler_score",
]
