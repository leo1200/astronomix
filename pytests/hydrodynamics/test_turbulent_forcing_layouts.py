"""
Turbulent driving on every state layout (CPU, seconds).

The forcing field is generated on the physical grid and normalised so that one
kick injects exactly ``Edot * dt`` of kinetic energy into the box. Finite
volume runs with non-periodic-roll boundaries carry a ghost-cell halo; a field
generated on that padded grid would have its wavenumbers off by the factor
``(N + 2 n_ghost) / N``, would not be periodic across the box seam, and would
count the ghost cells towards the injected energy.

For every layout (finite difference and the periodic-roll VL2 schemes without
ghost cells, the classic finite-volume scheme with them) this checks that one
kick
- injects exactly ``Edot * dt`` into the physical cells,
- applies the same field as every other layout (same PRNG key), and
- leaves the halo holding the periodic continuation of the physical cells.
"""

# ==== GPU selection ====
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax

# The energy check asserts the injected energy to rtol = 1e-10, which needs
# double precision. It is switched on before astronomix is imported, so that
# arrays created at import time are double precision as well.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# numerics
import numpy as np

# testing
import pytest

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLLC,
    HLLD,
    MINMOD,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    PERIODIC_ROLL,
    RK2_SSP,
    VL2,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FVSolverConfig,
    SimulationConfig,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing import _apply_forcing
from astronomix.time_stepping._utils import (
    _pad,
    _unpad,
)

NUM_CELLS = 16
ENERGY_INJECTION_RATE = 1.0
TIME_STEP = 0.01

LAYOUTS = {
    "finite_difference": dict(solver_mode=FINITE_DIFFERENCE, mhd=False),
    "finite_volume_rk2_ghost_cells": dict(
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(
            time_integrator=RK2_SSP,
            limiter=MINMOD,
            riemann_solver=HLLC,
        ),
        mhd=False,
    ),
    "finite_volume_vl2_hydro": dict(
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(time_integrator=VL2, riemann_solver=HLLC),
        mhd=False,
    ),
    "finite_volume_vl2_mhd": dict(
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(time_integrator=VL2, riemann_solver=HLLD),
        mhd=True,
    ),
}


def _kicked_state(layout):
    """
    A uniform gas at rest after one forcing kick, on the layout's state grid.

    Args:
        layout: The key of the layout in ``LAYOUTS``.

    Returns:
        The finalized configuration, the registered variables, the kicked state
        on the layout's (possibly padded) grid, and the density and velocity
        (stacked components) of the physical cells.
    """
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        dimensionality=3,
        box_size=1.0,
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=True),
        backend_config=BackendConfig(backend=NATIVE_JAX),
        **LAYOUTS[layout],
    )
    registered_variables = get_registered_variables(config)
    ones = jnp.ones((NUM_CELLS,) * 3)
    if config.mhd:
        field = dict(
            magnetic_field_x=0.1 * ones,
            magnetic_field_y=0 * ones,
            magnetic_field_z=0 * ones,
        )
    else:
        field = {}
    state = construct_primitive_state(
        config,
        registered_variables,
        density=ones,
        velocity_x=0 * ones,
        velocity_y=0 * ones,
        velocity_z=0 * ones,
        gas_pressure=ones,
        **field,
    )
    config = finalize_config(config, state.shape)
    forcing_params = TurbulentForcingParams(energy_injection_rate=ENERGY_INJECTION_RATE)

    has_halo = config.boundary_handling != PERIODIC_ROLL
    padded_state = _pad(state, config) if has_halo else state
    _, kicked = _apply_forcing(
        jax.random.key(0),
        padded_state,
        TIME_STEP,
        forcing_params,
        config,
        registered_variables,
    )
    physical_cells = _unpad(kicked, config) if has_halo else kicked
    velocity_indices = tuple(registered_variables.velocity_index)[:3]
    velocity = np.stack([np.asarray(physical_cells[index]) for index in velocity_indices])
    density = np.asarray(physical_cells[registered_variables.density_index])
    return config, registered_variables, kicked, density, velocity


def test_the_ghost_cell_layout_is_exercised():
    """The classic finite-volume layout really carries a ghost-cell halo."""
    config, *_ = _kicked_state("finite_volume_rk2_ghost_cells")
    assert config.boundary_handling != PERIODIC_ROLL and config.num_ghost_cells > 0


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_one_kick_injects_the_prescribed_energy(layout):
    """One kick injects exactly Edot * dt of kinetic energy into the physical cells."""
    _, _, _, density, velocity = _kicked_state(layout)
    injected = 0.5 * np.sum(density * np.sum(velocity**2, axis=0)) / NUM_CELLS**3
    np.testing.assert_allclose(injected, ENERGY_INJECTION_RATE * TIME_STEP, rtol=1e-10)


@pytest.mark.parametrize("layout", [name for name in LAYOUTS if name != "finite_difference"])
def test_every_layout_applies_the_same_field(layout):
    """Every layout applies the same forcing field as the finite-difference one."""
    *_, reference_velocity = _kicked_state("finite_difference")
    *_, velocity = _kicked_state(layout)
    np.testing.assert_allclose(
        velocity,
        reference_velocity,
        rtol=0,
        atol=1e-12 * np.max(np.abs(reference_velocity)),
    )


def test_the_halo_is_the_periodic_continuation():
    """After the kick the halo holds the periodic continuation of the physical cells."""
    config, registered_variables, kicked, _, _ = _kicked_state("finite_volume_rk2_ghost_cells")
    ghosts = config.num_ghost_cells
    for index in tuple(registered_variables.velocity_index)[:3]:
        component = np.asarray(kicked[index])
        physical = component[ghosts:-ghosts, ghosts:-ghosts, ghosts:-ghosts]
        np.testing.assert_array_equal(component, np.pad(physical, ghosts, mode="wrap"))

