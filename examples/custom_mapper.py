import numpy as np

from ordboost.mappers import BaseBinMapper
from ordboost.models import OrdBoostRegressor


# --- Mapper definition
class NullBinMapper(BaseBinMapper):
    """Maps discrete bin probabilities using raw bin edges only, with no
    intra-bin refinement.

    This is the minimal possible BaseBinMapper subclass: _intra_bin_points
    returns no interior points, so the fitted CDF is a straight
    piecewise-linear interpolation directly across bin_edges (equivalent
    to assuming uniform probability density within each bin). This is
    functionally identical to UniformBinMapper(n_points=0) -- the package
    already ships this exact null mapper -- and is included here purely
    to show the minimal _intra_bin_points implementation a custom mapper
    can have. See :doc:`../concepts` for why a null mapper is a useful
    baseline when evaluating whether a more elaborate mapper is actually
    adding value.
    """

    def _intra_bin_points(self, bin_data: np.ndarray, low: float, high: float, k: int):
        """Return no interior points -- boundary points from
        BaseBinMapper._build_grid alone define the grid.

        Parameters
        ----------
        bin_data : ndarray of shape (n_bin_samples,)
            Continuous training targets belonging to bin `k`. Unused
        low : float
            The effective lower boundary of bin `k`. Unused
        high : float
            The effective upper boundary of bin `k`. Unused
        k : int
            The 0-indexed bin number. Unused

        Returns
        -------
        points : ndarray of shape (1,)
            An empty array of points that are added to the grid.
        weights : ndarray of shape (1,)
            An empty of weights that are added to the grid CDF weights.

        """
        return np.array([]), np.array([])


# --- Usage
# Plug the custom mapper into OrdBoostRegressor

# Generate some synthetic data
rng = np.random.default_rng(42)
n_samples = 500
X = rng.standard_normal((n_samples, 3))
# Positive, right-skewed target (e.g. a duration or cost outcome)
y = np.exp(1.0 + 0.5 * X[:, 0] + rng.standard_normal(n_samples) * 0.3)

# Pass an *unconfigured* mapper instance (no bin_edges set)
# OrdBoostRegressor is the single source of truth for bin_edges
# and assigns them automatically.
custom_mapper = NullBinMapper()

model = OrdBoostRegressor(n_bins=8, mapper=custom_mapper, random_state=42)
model.fit(X, y)

n = 5  # Get the n first CDFs
dist = model.predict_dist(X[:n])
medians = dist.median()
pis = dist.interval(alpha=0.05)  # 95% prediction interval
for i in range(n):
    print(f"Sample {i} median [95% PI]")
    print(f"  {medians[i]:.2f} [{pis[0][i]:.2f} -- {pis[1][i]:.2f}]")
