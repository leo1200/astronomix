"""
First-order flux correction for a barotropic gas (hydrodynamics): the density is the only criterion.

An isothermal gas has no pressure that could turn negative, and in one dimension at a Courant number below one
half the HLL flux with a limited linear reconstruction cannot empty a cell: the receding streams that break the
ideal-gas schemes (``test_fv_flux_correction.py``) stay positive here at every stream speed. A cell empties
where the faces carry out more than the cell holds:

* the split scheme (``MUSCL_HANCOCK``) at a Courant number of 0.8 in the receding streams;
* ``VL2`` in a flow that leaves a corner along all three axes at once, at its usual Courant number of 0.4;
* a stage of ``RK2_SSP`` at a Courant number of 0.8 in the receding streams (the average with the old state hides
  the negative stage density at the end of the step, so the plain run is told apart by its result only).

The flux correction replaces the fluxes at the faces of such a cell by donor-cell fluxes and the density stays
positive. In a smooth flow it finds nothing to replace.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_barotropic_flux_correction.py
"""

# testing
import pytest

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# astronomix constants
from astronomix import FINITE_VOLUME, OPEN_BOUNDARY, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import HLL, ISOTHERMAL, MUSCL_HANCOCK, RK2_SSP, SUPERBEE, VL2

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)

pytestmark = pytest.mark.x64

SOUND_SPEED = 0.1
STREAM_SPEED = 20.0
INTEGRATORS = (("RK2_SSP", RK2_SSP), ("MUSCL_HANCOCK", MUSCL_HANCOCK), ("VL2", VL2))


def _final_density(time_integrator, boundary, flux_correction, density, velocities, t_end, courant_number):
    """Final density of an isothermal run with HLL and superbee; the dimension follows from the density."""
    side = BoundarySettings1D(boundary, boundary)
    config = SimulationConfig(
        dimensionality=density.ndim,
        box_size=1.0,
        boundary_settings=side if density.ndim == 1 else BoundarySettings(side, side, side),
        solver_mode=FINITE_VOLUME,
        equation_of_state=ISOTHERMAL,
        solver_config=FVSolverConfig(
            time_integrator=time_integrator,
            riemann_solver=HLL,
            limiter=SUPERBEE,
            first_order_flux_correction=flux_correction,
        ),
        mhd=False,
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    registered_variables = get_registered_variables(config)
    velocity_fields = {f"velocity_{axis}": velocity for axis, velocity in zip("xyz", velocities)}
    initial_state = construct_primitive_state(config, registered_variables, density=density, **velocity_fields)
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=t_end, C_cfl=courant_number, isothermal_sound_speed=SOUND_SPEED)
    result = time_integration(initial_state, config, params, registered_variables)
    final_state = np.asarray(result.final_state)
    assert np.all(np.isfinite(final_state))
    return final_state[registered_variables.density_index]


def _receding_streams(time_integrator, flux_correction, courant_number, dimensionality=1, cells=100):
    """Streams leaving the centre of the box along every axis, from a uniform density."""
    centres = (jnp.arange(cells) + 0.5) / cells
    grids = jnp.meshgrid(*([centres] * dimensionality), indexing="ij")
    return _final_density(
        time_integrator,
        OPEN_BOUNDARY,
        flux_correction,
        density=jnp.ones((cells,) * dimensionality),
        velocities=[jnp.where(grid < 0.5, -STREAM_SPEED, STREAM_SPEED) for grid in grids],
        t_end=(0.1 if dimensionality == 1 else 0.15) / STREAM_SPEED,
        courant_number=courant_number,
    )


def test_receding_streams_stay_positive_at_a_courant_number_below_one_half():
    """No scheme needs the correction here: the plain runs stay positive and the option changes nothing."""
    for name, time_integrator in INTEGRATORS:
        plain = _receding_streams(time_integrator, False, 0.4)
        corrected = _receding_streams(time_integrator, True, 0.4)
        print(f"{name}, Courant number 0.4, Mach {STREAM_SPEED / SOUND_SPEED:g}: smallest density {plain.min():.3e}")
        assert plain.min() > 0.0
        assert np.array_equal(plain, corrected)


def test_flux_correction_keeps_the_density_positive():
    """Where the plain update empties a cell, the corrected run stays positive."""
    # name, time integrator, Courant number, dimensionality, cells per axis
    for name, time_integrator, courant_number, dimensionality, cells in (
        ("MUSCL_HANCOCK", MUSCL_HANCOCK, 0.8, 1, 100),
        ("VL2", VL2, 0.4, 3, 16),
    ):
        plain = _receding_streams(time_integrator, False, courant_number, dimensionality, cells)
        corrected = _receding_streams(time_integrator, True, courant_number, dimensionality, cells)
        print(
            f"{name}, {dimensionality}D, Courant number {courant_number}: smallest density {plain.min():.3e} "
            f"(plain), {corrected.min():.3e} (corrected)"
        )
        assert plain.min() < 0.0
        assert corrected.min() > 0.0


def test_flux_correction_acts_on_a_failing_stage_of_rk2_ssp():
    """A stage of the plain run empties a cell: the correction changes the result and keeps it positive."""
    plain = _receding_streams(RK2_SSP, False, 0.8)
    corrected = _receding_streams(RK2_SSP, True, 0.8)
    difference = float(np.abs(corrected - plain).max())
    print(
        f"RK2_SSP, Courant number 0.8: smallest density {plain.min():.3e} (plain), {corrected.min():.3e} "
        f"(corrected), maximum difference {difference:.3e}"
    )
    assert difference > 1e-6
    assert corrected.min() > 0.0


def test_flux_correction_leaves_smooth_flow_unchanged():
    """A smooth sound wave gives the same result with and without the correction."""
    cells = 100
    wave = 1e-3 * jnp.sin(2.0 * jnp.pi * (jnp.arange(cells) + 0.5) / cells)
    for name, time_integrator in INTEGRATORS:
        plain, corrected = (
            _final_density(time_integrator, PERIODIC_BOUNDARY, option, 1.0 + wave, [SOUND_SPEED * wave], 2.0, 0.4)
            for option in (False, True)
        )
        difference = float(np.abs(corrected - plain).max())
        print(f"{name}: maximum difference with against without the correction: {difference:.3e}")
        assert difference < 1e-12


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_receding_streams_stay_positive_at_a_courant_number_below_one_half()
        test_flux_correction_keeps_the_density_positive()
        test_flux_correction_acts_on_a_failing_stage_of_rk2_ssp()
        test_flux_correction_leaves_smooth_flow_unchanged()
