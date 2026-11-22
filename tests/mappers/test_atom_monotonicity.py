"""Regression tests: boundary atoms must keep every mapper's grid monotone.

Mappers that assign interior weights by assumption (uniform interpolation,
or the empty-bin fallbacks) previously ignored the floor/ceiling atom mass,
producing interior weights below the floor atom weight or above the
ceiling atom weight, and hence a non-monotone CDF.
"""

import numpy as np
import pytest

from ordboost.mappers import (
    ContinuousBinMapper,
    EmpiricalMeanBinMapper,
    EmpiricalMedianBinMapper,
    QuantileBinMapper,
    UniformBinMapper,
)

ATOM_KWARGS = dict(
    lower_bound=0.0, upper_bound=89.0, floor_atom=True, ceiling_atom=True
)

MAPPER_FACTORIES = [
    lambda edges: UniformBinMapper(bin_edges=edges, n_points=1, **ATOM_KWARGS),
    lambda edges: UniformBinMapper(bin_edges=edges, n_points=4, **ATOM_KWARGS),
    lambda edges: ContinuousBinMapper(
        bin_edges=edges, density_weighted=False, **ATOM_KWARGS
    ),
    lambda edges: ContinuousBinMapper(
        bin_edges=edges, density_weighted=True, **ATOM_KWARGS
    ),
    lambda edges: QuantileBinMapper(bin_edges=edges, **ATOM_KWARGS),
    lambda edges: EmpiricalMeanBinMapper(bin_edges=edges, **ATOM_KWARGS),
    lambda edges: EmpiricalMedianBinMapper(bin_edges=edges, **ATOM_KWARGS),
]
MAPPER_IDS = [
    "uniform-1",
    "uniform-4",
    "continuous-uniform",
    "continuous-density",
    "quantile",
    "mean",
    "median",
]


@pytest.fixture
def atom_data() -> np.ndarray:
    """Integer outcome with large point masses at 0 and 89."""
    rng = np.random.default_rng(0)
    return np.concatenate(
        [np.zeros(200), rng.integers(1, 89, 800), np.full(100, 89)]
    ).astype(float)


@pytest.mark.parametrize("make_mapper", MAPPER_FACTORIES, ids=MAPPER_IDS)
def test_grid_weights_monotone_with_atoms(make_mapper, atom_data) -> None:
    """Test that grid weights are non-decreasing with both atoms active."""
    mapper = make_mapper([1.0, 10.0, 20.0, 40.0, 60.0, 80.0, 88.0]).fit(atom_data)
    assert np.all(np.diff(mapper.grid_cdf_weights_) >= 0.0)


@pytest.mark.parametrize("make_mapper", MAPPER_FACTORIES, ids=MAPPER_IDS)
def test_grid_weights_monotone_with_partial_atoms(make_mapper, atom_data) -> None:
    """Test monotonicity when the atom bins also contain interior mass."""
    mapper = make_mapper([5.0, 20.0, 40.0, 60.0, 80.0, 85.0]).fit(atom_data)
    assert np.all(np.diff(mapper.grid_cdf_weights_) >= 0.0)


@pytest.mark.parametrize("make_mapper", MAPPER_FACTORIES, ids=MAPPER_IDS)
def test_grid_weights_monotone_with_empty_first_bin(make_mapper, atom_data) -> None:
    """Test monotonicity when bin 0 is empty (floor atom takes full weight)."""
    y = atom_data[atom_data >= 5.0]
    mapper = make_mapper([5.0, 20.0, 40.0, 60.0, 80.0, 85.0]).fit(y)
    assert np.all(np.diff(mapper.grid_cdf_weights_) >= 0.0)


def test_uniform_weights_spread_non_atom_mass() -> None:
    """Test that UniformBinMapper spreads only the non-atom mass uniformly."""
    # Bin 0 = [0, 10): half its data sits on the floor atom.
    # Bin 1 = [10, 20]: a quarter of its data sits on the ceiling atom.
    y = np.array([0.0, 0.0, 4.0, 6.0, 12.0, 14.0, 16.0, 20.0])
    mapper = UniformBinMapper(
        bin_edges=[10.0],
        n_points=1,
        lower_bound=0.0,
        upper_bound=20.0,
        floor_atom=True,
        ceiling_atom=True,
    ).fit(y)
    lookup = dict(zip(mapper.grid_y_, mapper.grid_cdf_weights_))
    assert lookup[0.0] == pytest.approx(0.5)
    assert lookup[5.0] == pytest.approx(0.75)  # 0.5 + 0.5 * (1 - 0.5)
    assert lookup[15.0] == pytest.approx(1.375)  # 1 + 0.5 * 0.75
    assert lookup[20.0 - 1e-4] == pytest.approx(1.75)
    assert lookup[20.0] == pytest.approx(2.0)


def test_uniform_weights_unchanged_without_atoms() -> None:
    """Test that the plain uniform behaviour is unaffected without atoms."""
    y = np.array([0.0, 0.0, 4.0, 6.0, 12.0, 14.0, 16.0, 20.0])
    mapper = UniformBinMapper(
        bin_edges=[10.0], n_points=1, lower_bound=0.0, upper_bound=20.0
    ).fit(y)
    lookup = dict(zip(mapper.grid_y_, mapper.grid_cdf_weights_))
    assert lookup[5.0] == pytest.approx(0.5)
    assert lookup[15.0] == pytest.approx(1.5)
