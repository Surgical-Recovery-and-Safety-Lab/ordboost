"""OrdBoost: Non-parametric discrete ordinal-binning gradient boosting and continuous regression."""

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
    crps_score,
    interval_coverage_rate,
    pinball_loss,
    winkler_score,
)
from ordboost.models import OrdBoostClassifier, OrdBoostRegressor

__version__ = "0.2.1"

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
    "crps_score",
    "interval_coverage_rate",
    "pinball_loss",
    "winkler_score",
]
