"""
Positivity-preserving WENO pytest.

Two stress problems where the finite-difference scheme needs its robustness
options, plus a smooth-flow check that the options do not cost order of
accuracy:

* a cold dense slab rammed at Mach ~800 into tenuous gas. With the legacy
  enthalpy-averaged interface basis (``weno_admissible_face_state=False``) the
  interface sound speed goes negative there and the run blows up within a few
  steps; the default admissible face state and the positivity-preserving limiter
  (``weno_positivity_preserving``) must both hold;
* a strong double rarefaction close to vacuum (positivity of density and
  pressure);
* an advected entropy wave (fifth-order convergence with the limiter on).
"""

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import (
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix import (
    BoundarySettings1D,
    FDSolverConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.test_setups.reference_solutions.riemann_solver import (
    _exact_riemann_ideal_gas,
)

pytestmark = pytest.mark.x64


def _run_riemann(left, right, gamma, t_end, num_cells, **options):
    """
    Run a 1D Riemann problem on [0, 1] with the diaphragm at x = 0.5.

    Args:
        left: The left state as (density, velocity, pressure).
        right: The right state as (density, velocity, pressure).
        gamma: The adiabatic index.
        t_end: The end time.
        num_cells: The number of cells.
        **options: FDSolverConfig options (the robustness switches).

    Returns:
        The final primitive state, the registered variables and the exact
        density at ``t_end``.
    """
    config = SimulationConfig(
        dimensionality=1,
        num_cells=num_cells,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY,
            right_boundary=OPEN_BOUNDARY,
        ),
        solver_config=FDSolverConfig(**options),
    )
    params = SimulationParams(t_end=t_end, gamma=gamma, C_cfl=0.4)
    registered_variables = get_registered_variables(config)
    x = get_helper_data(config).geometric_centers
    is_left = x < 0.5
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.where(is_left, left[0], right[0]),
        velocity_x=jnp.where(is_left, left[1], right[1]),
        gas_pressure=jnp.where(is_left, left[2], right[2]),
    )
    config = finalize_config(config, state.shape)
    final = np.asarray(time_integration(state, config, params, registered_variables))
    exact_density, _, _ = _exact_riemann_ideal_gas(*left, *right, gamma, x, t_end, 0.5)
    return final, registered_variables, np.asarray(exact_density)


def test_cold_dense_ram():
    """Mach ~800 cold slab into tenuous gas: finite, positive, near exact."""
    left, right = (100.0, 1.0, 1e-4), (1.0, 0.0, 1e-4)
    robustness_options = (
        # The default: admissible face state, no positivity-preserving limiter.
        dict(weno_admissible_face_state=True, weno_positivity_preserving=False),
        dict(weno_positivity_preserving=True),
    )
    for options in robustness_options:
        final, registered_variables, exact = _run_riemann(
            left,
            right,
            5.0 / 3.0,
            0.3,
            400,
            **options,
        )
        density = final[registered_variables.density_index]
        pressure = final[registered_variables.pressure_index]
        assert np.all(np.isfinite(final)), options
        assert density.min() > 0.0 and pressure.min() > 0.0, options
        # The error is dominated by the smeared density-400 contact, hence the
        # loose bound.
        assert np.mean(np.abs(density - exact)) < 3.0, options


def test_near_vacuum_double_rarefaction():
    """Double rarefaction with u = -+3.5 (vacuum at u = -+3.74): positive."""
    left, right = (1.0, -3.5, 0.4), (1.0, 3.5, 0.4)
    final, registered_variables, exact = _run_riemann(
        left,
        right,
        1.4,
        0.1,
        400,
        weno_positivity_preserving=True,
    )
    density = final[registered_variables.density_index]
    pressure = final[registered_variables.pressure_index]
    assert density.min() > 0.0 and pressure.min() > 0.0
    assert np.mean(np.abs(density - exact)) < 1e-2


def test_smooth_fifth_order():
    """An advected entropy wave stays fifth order with the limiter on."""
    errors = []
    for num_cells in (32, 64):
        # finalize_config selects the periodic roll (no ghost cells) for the
        # fully periodic 1D boundaries.
        config = SimulationConfig(
            dimensionality=1,
            num_cells=num_cells,
            boundary_settings=BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY,
                right_boundary=PERIODIC_BOUNDARY,
            ),
            solver_config=FDSolverConfig(weno_positivity_preserving=True),
        )
        params = SimulationParams(t_end=1.0, gamma=5.0 / 3.0, C_cfl=0.4)
        registered_variables = get_registered_variables(config)
        x = get_helper_data(config).geometric_centers
        density = 1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * x)
        state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=density,
            velocity_x=jnp.ones_like(x),
            gas_pressure=jnp.ones_like(x),
        )
        config = finalize_config(config, state.shape)
        final = np.asarray(time_integration(state, config, params, registered_variables))
        final_density = final[registered_variables.density_index]
        errors.append(np.mean(np.abs(final_density - np.asarray(density))))
    order = np.log2(errors[0] / errors[1])
    assert order > 4.5, f"observed order {order:.2f}"


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_cold_dense_ram()
        test_near_vacuum_double_rarefaction()
        test_smooth_fifth_order()
