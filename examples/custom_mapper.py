import numpy as np

from ordboost.mappers import BaseBinMapper
from ordboost.models import OrdBoostRegressor


class GeometricMeanBinMapper(BaseBinMapper):
    """Maps discrete bin probabilities using one geometric-mean point per bin.

    A custom mapper needs only to implement `_intra_bin_points`, which
    defines the interior grid point(s) added within each bin and the
    cumulative weight (in bin-index units) assigned to each. Everything
    else is inherited from `BaseBinMapper`.

    The geometric mean is only meaningful for strictly positive data;
    this mapper is a reasonable choice for a right-skewed, positive-
    valued outcome where the arithmetic mean would be pulled too far
    toward the tail (e.g. cost or duration data).
    """

    def _intra_bin_points(self, bin_data, low, high, k):
        """Return one interior point at the bin's empirical geometric mean.

        Parameters
        ----------
        bin_data : ndarray of shape (n_bin_samples,)
            Continuous training targets belonging to bin `k`.
        low : float
            The effective lower boundary of bin `k`.
        high : float
            The effective upper boundary of bin `k`.
        k : int
            The 0-indexed bin number.

        Returns
        -------
        points : ndarray of shape (1,)
            The bin's empirical geometric mean, clipped to [low, high].
            Falls back to the geometric midpoint if `bin_data` is empty
            or contains non-positive values (geometric mean undefined).
        weights : ndarray of shape (1,)
            The empirical fraction of `bin_data` at or below the
            geometric mean, offset into bin-index units (k + fraction).

        """
        positive_data = bin_data[bin_data > 0] if len(bin_data) > 0 else bin_data

        if len(positive_data) == 0:
            geo_mean = (low + high) / 2.0
            frac_below = 0.5
        else:
            geo_mean = float(
                np.clip(
                    np.exp(np.mean(np.log(positive_data))),
                    low,
                    high,
                )
            )
            frac_below = float(np.mean(bin_data <= geo_mean))

        return np.array([geo_mean]), np.array([k + frac_below])


# --- Usage: plug the custom mapper into OrdBoostRegressor ---

# Generate some synthetic data
rng = np.random.default_rng(42)
n_samples = 500
X = rng.standard_normal((n_samples, 3))
# Positive, right-skewed target (e.g. a duration or cost outcome)
y = np.exp(1.0 + 0.5 * X[:, 0] + rng.standard_normal(n_samples) * 0.3)

# Pass an *unconfigured* mapper instance (no bin_edges set)
# OrdBoostRegressor is the single source of truth for bin_edges
# and assigns them automatically.
custom_mapper = GeometricMeanBinMapper()

model = OrdBoostRegressor(n_bins=8, mapper=custom_mapper, random_state=42)
model.fit(X, y)

n = 5  # Get the n first CDFs
dist = model.predict_dist(X[:n])
medians = dist.median()
pis = dist.interval(alpha=0.05)  # 95% prediction interval
for i in range(n):
    print(f"Sample {i} median [95% PI]")
    print(f"  {medians[i]:.2f} [{pis[0][i]:.2f} -- {pis[1][i]:.2f}]")
