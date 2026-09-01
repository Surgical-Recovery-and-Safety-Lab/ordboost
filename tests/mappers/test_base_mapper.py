"""Unit tests for BaseBinMapper in ordboost.mappers."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from ordboost.distributions import ContinuousPredictiveDistribution
from ordboost.mappers import BaseBinMapper


class DummyBinMapper(BaseBinMapper):
    """Minimal concrete BaseBinMapper with no intra-bin refinement."""

    def _intra_bin_points(self, bin_data, low, high, k):
        """Return no interior points."""
        return np.array([]), np.array([])


class MidpointBinMapper(BaseBinMapper):
    """Concrete BaseBinMapper adding one midpoint per bin at weight k + 0.5."""

    def _intra_bin_points(self, bin_data, low, high, k):
        """Return the bin's geometric midpoint at cumulative weight k + 0.5."""
        return np.array([(low + high) / 2.0]), np.array([k + 0.5])


class ParamDependentBinMapper(BaseBinMapper):
    """Concrete BaseBinMapper whose _intra_bin_points depends on a value set
    by _validate_intra_bin_params, used to confirm that `fit` validates
    parameters before building the grid.
    """

    def _validate_intra_bin_params(self):
        """Set the fraction used to place each bin's interior point."""
        self.midpoint_frac_ = 0.5

    def _intra_bin_points(self, bin_data, low, high, k):
        """Return one interior point using `midpoint_frac_`."""
        midpoint = (low + high) / 2.0
        return np.array([midpoint]), np.array([k + self.midpoint_frac_])


class TestBaseBinMapperABC:
    """Tests for abstract base class instantiation restrictions."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        """Test that instantiating BaseBinMapper directly raises TypeError."""
        with pytest.raises(TypeError):
            BaseBinMapper(bin_edges=[0, 10, 20])  # type: ignore[abstract]


class TestBaseBinMapperInit:
    """Tests for BaseBinMapper.__init__ parameter storage."""

    def test_default_parameters(self) -> None:
        """Test that default constructor arguments are stored unmodified."""
        mapper = DummyBinMapper()
        assert mapper.bin_edges is None
        assert mapper.bounded_below is True
        assert mapper.bounded_above is True
        assert mapper.floor_atom is False
        assert mapper.ceiling_atom is False

    def test_custom_parameters(self) -> None:
        """Test that custom constructor arguments are stored unmodified."""
        mapper = DummyBinMapper(
            bin_edges=[0.0, 10.0],
            bounded_below=False,
            bounded_above=False,
        )
        assert mapper.bin_edges == [0.0, 10.0]
        assert mapper.bounded_below is False
        assert mapper.bounded_above is False


class TestValidateAtomFlags:
    """Tests for BaseBinMapper._validate_atom_flags."""

    def test_floor_atom_without_bounded_below_raises(self) -> None:
        """Test that floor_atom=True with bounded_below=False raises ValueError."""
        mapper = DummyBinMapper(
            bin_edges=[0.0, 10.0], bounded_below=False, floor_atom=True
        )
        with pytest.raises(ValueError, match="'floor_atom=True' requires"):
            mapper._validate_atom_flags()

    def test_ceiling_atom_without_bounded_above_raises(self) -> None:
        """Test that ceiling_atom=True with bounded_above=False raises ValueError."""
        mapper = DummyBinMapper(
            bin_edges=[0.0, 10.0], bounded_above=False, ceiling_atom=True
        )
        with pytest.raises(ValueError, match="'ceiling_atom=True' requires"):
            mapper._validate_atom_flags()

    def test_valid_combinations_do_not_raise(self) -> None:
        """Test that consistent atom/boundedness flag combinations pass validation."""
        mapper = DummyBinMapper(
            bin_edges=[0.0, 10.0], floor_atom=True, ceiling_atom=True
        )
        mapper._validate_atom_flags()

    def test_atoms_false_with_bounds_false_does_not_raise(self) -> None:
        """Test that both atom flags False never raises, regardless of boundedness."""
        mapper = DummyBinMapper(
            bin_edges=[0.0, 10.0], bounded_below=False, bounded_above=False
        )
        mapper._validate_atom_flags()


class TestValidateEdges:
    """Tests for BaseBinMapper._validate_edges."""

    def test_none_bin_edges_raises(self) -> None:
        """Test that bin_edges=None raises ValueError."""
        mapper = DummyBinMapper(bin_edges=None)
        with pytest.raises(ValueError, match="'bin_edges' must be set"):
            mapper._validate_edges()

    def test_valid_edges_returned_as_float_array(self) -> None:
        """Test that valid monotonic edges are returned as a float ndarray."""
        mapper = DummyBinMapper(bin_edges=[0, 5, 10, 20])
        edges = mapper._validate_edges()
        np.testing.assert_array_equal(edges, np.array([0.0, 5.0, 10.0, 20.0]))
        assert edges.dtype == np.float64

    def test_non_1d_edges_raises(self) -> None:
        """Test that a 2D bin_edges array raises ValueError."""
        mapper = DummyBinMapper(bin_edges=[[0, 10], [10, 20]])
        with pytest.raises(ValueError, match="1D array"):
            mapper._validate_edges()

    def test_fewer_than_two_edges_raises(self) -> None:
        """Test that a single-edge array raises ValueError."""
        mapper = DummyBinMapper(bin_edges=[0.0])
        with pytest.raises(ValueError, match="at least 2 edges"):
            mapper._validate_edges()

    def test_non_monotonic_edges_raises(self) -> None:
        """Test that non-strictly-increasing edges raise ValueError."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 5.0])
        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            mapper._validate_edges()

    def test_duplicate_edges_raises(self) -> None:
        """Test that repeated edge values raise ValueError (zero-width bin)."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 10.0, 20.0])
        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            mapper._validate_edges()


class TestDigitize:
    """Tests for BaseBinMapper.digitize."""

    def test_default_digitization_matches_numpy_digitize(self) -> None:
        """Test that automatic binning matches np.digitize on interior edges."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0, 30.0])
        edges = mapper._validate_edges()
        y_cont = np.array([1.0, 11.0, 25.0])
        binned = mapper.digitize(y_cont, edges, y_binned=None)
        np.testing.assert_array_equal(binned, np.array([0, 1, 2]))

    def test_explicit_y_binned_returned_as_int_array(self) -> None:
        """Test that a supplied y_binned array is cast to int and returned."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        edges = mapper._validate_edges()
        y_cont = np.array([1.0, 15.0])
        binned = mapper.digitize(y_cont, edges, y_binned=[0, 1])
        np.testing.assert_array_equal(binned, np.array([0, 1]))
        assert binned.dtype.kind == "i"

    def test_y_binned_shape_mismatch_raises(self) -> None:
        """Test that a y_binned array of mismatched shape raises ValueError."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        edges = mapper._validate_edges()
        y_cont = np.array([1.0, 15.0, 8.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            mapper.digitize(y_cont, edges, y_binned=[0, 1])


class TestBoundaryAtomWeight:
    """Tests for BaseBinMapper._boundary_atom_weight."""

    def test_lower_side_is_inclusive(self) -> None:
        """Test that side='lower' includes values exactly at the boundary."""
        bin_data = np.array([0.0, 0.0, 0.0, 5.0])
        weight = BaseBinMapper._boundary_atom_weight(
            bin_data, boundary=0.0, side="lower"
        )
        assert weight == pytest.approx(0.75)

    def test_upper_side_is_exclusive(self) -> None:
        """Test that side='upper' excludes values exactly at the boundary."""
        bin_data = np.array([8.0, 8.0, 9.0, 9.0, 9.0])
        weight = BaseBinMapper._boundary_atom_weight(
            bin_data, boundary=9.0, side="upper"
        )
        assert weight == pytest.approx(0.4)

    def test_empty_bin_data_returns_one(self) -> None:
        """Test that an empty bin returns weight 1.0 regardless of side."""
        empty = np.array([])
        assert BaseBinMapper._boundary_atom_weight(empty, 5.0, side="lower") == 1.0
        assert BaseBinMapper._boundary_atom_weight(empty, 5.0, side="upper") == 1.0


class TestValidateIntraBinParams:
    """Tests for the default (base-class) BaseBinMapper._validate_intra_bin_params."""

    def test_default_is_a_no_op(self) -> None:
        """Test that the base implementation does not raise or set attributes."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0])
        assert mapper._validate_intra_bin_params() is None


class TestBuildGrid:
    """Tests for BaseBinMapper._build_grid."""

    def test_sets_fitted_attributes(self) -> None:
        """Test that _build_grid sets bin_edges_, n_bins_, grid_y_, grid_cdf_weights_."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper._build_grid(np.array([1.0, 5.0, 15.0]))
        assert hasattr(mapper, "bin_edges_")
        assert hasattr(mapper, "n_bins_")
        assert hasattr(mapper, "grid_y_")
        assert hasattr(mapper, "grid_cdf_weights_")
        assert mapper.n_bins_ == 2

    def test_grid_endpoints_map_to_zero_and_n_bins(self) -> None:
        """Test that the fitted grid's CDF weights span exactly [0, n_bins]."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper._build_grid(np.array([1.0, 5.0, 15.0, 19.0]))
        assert mapper.grid_cdf_weights_[0] == 0.0
        assert mapper.grid_cdf_weights_[-1] == float(mapper.n_bins_)

    def test_grid_y_is_sorted_and_deduplicated(self) -> None:
        """Test that grid_y_ is strictly ascending with no repeated values."""
        mapper = MidpointBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper._build_grid(np.array([1.0, 5.0, 15.0, 19.0]))
        assert np.all(np.diff(mapper.grid_y_) > 0)

    def test_bounded_below_anchors_to_observed_minimum(self) -> None:
        """Test that bounded_below=True anchors the floor to y_continuous.min()."""
        mapper = DummyBinMapper(bin_edges=[1.0, 10.0, 20.0], bounded_below=True)
        y_cont = np.array([0.0, 3.0, 15.0])  # observed min (0.0) below edges[0] (1.0)
        mapper._build_grid(y_cont)
        assert mapper.grid_y_[1] == pytest.approx(0.0)

    def test_bounded_below_false_uses_nominal_edge(self) -> None:
        """Test that bounded_below=False anchors the floor to bin_edges[0]
        even when the observed minimum lies below it."""
        mapper = DummyBinMapper(bin_edges=[1.0, 10.0, 20.0], bounded_below=False)
        y_cont = np.array([0.0, 3.0, 15.0])
        mapper._build_grid(y_cont)
        assert mapper.grid_y_[1] == pytest.approx(1.0)

    def test_bounded_above_anchors_to_observed_maximum(self) -> None:
        """Test that bounded_above=True anchors the ceiling to y_continuous.max()."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0], bounded_above=True)
        y_cont = np.array(
            [1.0, 5.0, 18.0]
        )  # observed max (18.0) below edges[-1] (20.0)
        mapper._build_grid(y_cont)
        assert mapper.grid_y_[-1] == pytest.approx(18.0)

    def test_bounded_above_false_uses_nominal_edge(self) -> None:
        """Test that bounded_above=False anchors the ceiling to bin_edges[-1]
        even when the observed maximum lies below it."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0], bounded_above=False)
        y_cont = np.array([1.0, 5.0, 18.0])
        mapper._build_grid(y_cont)
        assert mapper.grid_y_[-1] == pytest.approx(20.0)

    def test_empty_terminal_bin_falls_back_to_nominal_edge(self) -> None:
        """Test that an empty final bin falls back to the nominal upper edge
        even when bounded_above=True, avoiding an inverted/degenerate bin."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0], bounded_above=True)
        y_cont = np.array([1.0, 5.0, 8.0])  # nothing in bin 1 ([10, 20))
        mapper._build_grid(y_cont)
        assert mapper.grid_y_[-1] == pytest.approx(20.0)

    def test_floor_atom_assigns_empirical_weight(self) -> None:
        """Test that floor_atom=True assigns the empirical at-or-below fraction
        as the floor point's cumulative weight, rather than 0."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0], floor_atom=True)
        y_cont = np.array([0.0, 0.0, 0.0, 5.0])  # 3 of 4 values at the floor
        mapper._build_grid(y_cont)
        floor_idx = np.searchsorted(mapper.grid_y_, 0.0)
        assert mapper.grid_cdf_weights_[floor_idx] == pytest.approx(0.75)

    def test_floor_atom_false_assigns_zero_weight(self) -> None:
        """Test that floor_atom=False assigns weight 0 at the floor point
        regardless of how much data sits there."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0], floor_atom=False)
        y_cont = np.array([0.0, 0.0, 0.0, 5.0])
        mapper._build_grid(y_cont)
        floor_idx = np.searchsorted(mapper.grid_y_, 0.0)
        assert mapper.grid_cdf_weights_[floor_idx] == 0.0

    def test_ceiling_atom_inserts_epsilon_offset_point(self) -> None:
        """Test that ceiling_atom=True inserts a point just below the ceiling,
        weighted by the empirical strictly-below fraction."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0], ceiling_atom=True)
        y_cont = np.array([2.0, 10.0, 10.0, 10.0])  # 3 of 4 values at the ceiling
        mapper._build_grid(y_cont)
        assert mapper.grid_y_[-2] == pytest.approx(10.0 - 1e-4)
        assert mapper.grid_cdf_weights_[-2] == pytest.approx(0.25)

    def test_dedup_keeps_maximum_weight_at_collision(self) -> None:
        """Test that when an interior point collides with the bin boundary,
        deduplication retains the larger (boundary) weight rather than the
        first-encountered (interior) weight."""

        class CollidingBinMapper(BaseBinMapper):
            """Dummy mapper whose interior point always equals the bin's
            upper edge, forcing a collision with the boundary point."""

            def _intra_bin_points(self, bin_data, low, high, k):
                return np.array([high]), np.array([k + 0.5])

        mapper = CollidingBinMapper(bin_edges=[0.0, 10.0])
        mapper._build_grid(np.array([1.0, 5.0]))  # bounded_above -> high = 5.0
        boundary_idx = np.searchsorted(mapper.grid_y_, 5.0)
        assert mapper.grid_cdf_weights_[boundary_idx] == pytest.approx(1.0)

    def test_invalid_atom_flags_raise_before_building(self) -> None:
        """Test that inconsistent atom/boundedness flags raise before any
        grid construction is attempted."""
        mapper = DummyBinMapper(
            bin_edges=[0.0, 10.0], bounded_below=False, floor_atom=True
        )
        with pytest.raises(ValueError, match="'floor_atom=True' requires"):
            mapper._build_grid(np.array([1.0, 5.0]))

    def test_non_1d_y_continuous_raises(self) -> None:
        """Test that a non-1D y_continuous array raises ValueError."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0])
        with pytest.raises(ValueError, match="1D array"):
            mapper._build_grid(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_boundary_epsilon_default_value(self) -> None:
        """Test that the default boundary_epsilon of 1e-4 is used when
        not explicitly set."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0])
        mapper._build_grid(np.array([2.0, 8.0]))
        assert mapper.grid_y_[0] == pytest.approx(2.0 - 1e-4)

    def test_boundary_epsilon_custom_value_affects_floor_anchor(self) -> None:
        """Test that a custom boundary_epsilon changes the floor anchor
        point's offset from y_min."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0], boundary_epsilon=0.5)
        mapper._build_grid(np.array([2.0, 8.0]))
        assert mapper.grid_y_[0] == pytest.approx(2.0 - 0.5)

    def test_boundary_epsilon_custom_value_affects_ceiling_atom_point(self) -> None:
        """Test that a custom boundary_epsilon changes the ceiling-atom
        offset point's position, not just the floor anchor."""
        mapper = DummyBinMapper(
            bin_edges=[0.0, 10.0], ceiling_atom=True, boundary_epsilon=0.5
        )
        y_cont = np.array([2.0, 10.0, 10.0, 10.0])
        mapper._build_grid(y_cont)
        assert mapper.grid_y_[-2] == pytest.approx(10.0 - 0.5)

    def test_ceiling_atom_false_adds_no_extra_point(self) -> None:
        """Test that ceiling_atom=False produces no epsilon-offset point,
        even with data concentrated at the ceiling. Uses an absolute
        (non-relative) tolerance to avoid np.isclose's default rtol
        coincidentally matching the epsilon magnitude being tested for."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0], ceiling_atom=False)
        y_cont = np.array([2.0, 10.0, 10.0, 10.0])
        mapper._build_grid(y_cont)
        assert not np.any(
            np.abs(mapper.grid_y_ - (10.0 - mapper.boundary_epsilon)) < 1e-9
        )

    def test_y_binned_inconsistent_with_value_raises(self) -> None:
        """Test that forcing a sample into a bin whose own (possibly
        anchored) edges cannot contain its value raises ValueError rather
        than silently producing a non-monotonic grid."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        y_cont = np.array([15.0])
        with pytest.raises(ValueError, match="invalid range"):
            mapper._build_grid(y_cont, y_binned=np.array([0]))

    def test_zero_width_bin_does_not_raise(self) -> None:
        """Test that low == high (a genuinely degenerate, not inverted, bin)
        is permitted and resolves via max-weight dedup, distinguishing this
        from the strictly-invalid low > high case."""
        mapper = DummyBinMapper(bin_edges=[10.0, 11.0])
        mapper._build_grid(np.array([10.0, 10.0]))
        idx = np.searchsorted(mapper.grid_y_, 10.0)
        assert mapper.grid_cdf_weights_[idx] == pytest.approx(1.0)


class TestFit:
    """Tests for BaseBinMapper.fit."""

    def test_returns_self(self) -> None:
        """Test that fit returns the mapper instance for method chaining."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        result = mapper.fit(np.array([1.0, 15.0]))
        assert result is mapper

    def test_sets_fitted_attributes(self) -> None:
        """Test that fit results in the same fitted attributes as _build_grid."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit(np.array([1.0, 15.0]))
        assert hasattr(mapper, "grid_y_")
        assert hasattr(mapper, "grid_cdf_weights_")

    def test_validates_params_before_building_grid(self) -> None:
        """Test that fit calls _validate_intra_bin_params before _build_grid,
        so subclass parameters set during validation are available to
        _intra_bin_points during grid construction."""
        mapper = ParamDependentBinMapper(bin_edges=[0.0, 10.0])
        mapper.fit(np.array([1.0, 5.0]))  # raises AttributeError if mis-ordered
        assert mapper.midpoint_frac_ == 0.5


class TestToContinuousDist:
    """Tests for BaseBinMapper.to_continuous_dist."""

    def test_not_fitted_raises(self) -> None:
        """Test that calling before fit raises NotFittedError."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        with pytest.raises(NotFittedError):
            mapper.to_continuous_dist([[0.5, 0.5]])

    def test_non_2d_pmf_raises(self) -> None:
        """Test that a 1D pmf array raises ValueError."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit(np.array([1.0, 15.0]))
        with pytest.raises(ValueError, match="2D array"):
            mapper.to_continuous_dist([0.5, 0.5])

    def test_wrong_column_count_raises(self) -> None:
        """Test that a pmf with the wrong number of bin columns raises ValueError."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit(np.array([1.0, 15.0]))
        with pytest.raises(ValueError, match="does not match"):
            mapper.to_continuous_dist([[0.3, 0.3, 0.4]])

    def test_returns_continuous_predictive_distribution(self) -> None:
        """Test that the return type and shape match the fitted grid and
        sample count."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit(np.array([1.0, 5.0, 15.0]))
        pmf = np.array([[0.6, 0.4], [0.2, 0.8]])
        dist = mapper.to_continuous_dist(pmf)
        assert isinstance(dist, ContinuousPredictiveDistribution)
        assert dist.grid_cdf.shape == (2, len(mapper.grid_y_))

    def test_cdf_spans_zero_to_one(self) -> None:
        """Test that the CDF's first and last grid values are 0.0 and 1.0."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit(np.array([1.0, 5.0, 15.0]))
        pmf = np.array([[0.6, 0.4]])
        dist = mapper.to_continuous_dist(pmf)
        assert dist.grid_cdf[0, 0] == pytest.approx(0.0)
        assert dist.grid_cdf[0, -1] == pytest.approx(1.0)


class TestTransform:
    """Tests for BaseBinMapper.transform."""

    def test_not_fitted_raises(self) -> None:
        """Test that calling before fit raises NotFittedError (propagated
        from to_continuous_dist)."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        with pytest.raises(NotFittedError):
            mapper.transform([[0.5, 0.5]])

    def test_matches_mean_of_continuous_dist(self) -> None:
        """Test that transform's default behaviour equals
        to_continuous_dist(pmf).mean()."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit(np.array([1.0, 5.0, 15.0]))
        pmf = np.array([[0.6, 0.4], [0.2, 0.8]])
        dist = mapper.to_continuous_dist(pmf)
        np.testing.assert_allclose(mapper.transform(pmf), dist.mean())

    def test_output_shape(self) -> None:
        """Test that transform returns one value per sample row."""
        mapper = DummyBinMapper(bin_edges=[0.0, 10.0, 20.0])
        mapper.fit(np.array([1.0, 5.0, 15.0]))
        pmf = np.array([[0.6, 0.4], [0.2, 0.8], [0.5, 0.5]])
        assert mapper.transform(pmf).shape == (3,)
