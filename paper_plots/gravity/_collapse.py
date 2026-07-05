"""Shared Evrard-collapse simulation setup for the collapse figures.

Both ``energy_conservation_comparison.py`` and ``radial_profiles_comparison.py``
drive the same finite-difference Evrard collapse, differing only in the end
time and which diagnostics they keep. The FD configuration follows
``tests/self_gravity_tests/simulate_collapse_fd.py`` and runs on the Pallas
backend for speed.
"""

import jax
import jax.numpy as jnp

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    HLLC,
    MINMOD,
    NATIVE_JAX,
    PALLAS,
    RK2_SSP,
    UNSPLIT,
    BoundarySettings,
    BoundarySettings1D,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    FORWARDS,
    GravityConfig,
    PERIODIC_BOUNDARY,
    PositivityConfig,
    SnapshotSettings,
    finalize_config,
)

from _common import pallas_config_kwargs

GAMMA = 5 / 3
BOX_SIZE = 4.0
NUM_SNAPSHOTS = 60


def _backend_kwargs(backend):
    # Pallas (~10x faster) for the default float32 runs; native JAX for float64,
    # where the Pallas/Triton kernel does some math in reduced precision.
    if backend == PALLAS:
        return pallas_config_kwargs()
    return dict(backend=NATIVE_JAX)


def collapse_config(
    num_cells, self_gravity_version, want_states, backend=PALLAS,
    solver_mode=FINITE_DIFFERENCE,
):
    common = dict(
        runtime_debugging=False,
        progress_bar=False,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=self_gravity_version,
            poisson_manual_open_boundaries=True,
        ),
        mhd=False,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        differentiation_mode=FORWARDS,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(
            return_states=want_states,
            return_final_state=True,
            return_total_energy=True,
            return_internal_energy=True,
            return_kinetic_energy=True,
            return_gravitational_energy=True,
        ),
        num_snapshots=NUM_SNAPSHOTS,
    )

    if solver_mode == FINITE_VOLUME:
        # Finite-volume MUSCL-Hancock-style scheme (follows the FV baseline in
        # tests/self_gravity_tests/simulate_collapse_fd.py). The Pallas WENO
        # kernel is finite-difference only, so FV always runs on native JAX.
        return SimulationConfig(
            solver_mode=FINITE_VOLUME,
            first_order_fallback=False,
            split=UNSPLIT,
            limiter=MINMOD,
            time_integrator=RK2_SSP,
            riemann_solver=HLLC,
            backend=NATIVE_JAX,
            **common,
        )

    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        positivity_config=PositivityConfig(default_positivity_protection=False),
        **_backend_kwargs(backend),
        **common,
    )


def run_collapse(
    num_cells, self_gravity_version, t_end, want_states=False, backend=PALLAS,
    solver_mode=FINITE_DIFFERENCE, initial_energy=0.05,
):
    """Run the Evrard collapse and return ``(snapshots, helper_data, regvars)``.

    ``initial_energy`` is the initial thermal energy per unit mass; the standard
    cold Evrard test uses 0.05. A warmer value (e.g. 0.20) gives a *milder*
    collapse that stays finite for the conservative flux schemes down to 32^3.
    """
    config = collapse_config(
        num_cells, self_gravity_version, want_states, backend=backend,
        solver_mode=solver_mode,
    )

    params = SimulationParams(
        t_end=t_end,
        C_cfl=0.4,
        dt_max=jnp.inf,
        minimum_density=1e-5,
        minimum_pressure=3e-6,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    R = 1.0
    M = 1.0
    dx = config.box_size / (config.num_cells - 1)

    rho = jnp.where(
        helper_data.r <= R, M / (2 * jnp.pi * R**2 * helper_data.r), 1e-4
    )
    total_injected_mass = jnp.sum(jnp.where(helper_data.r <= R, rho, 0)) * dx**3
    print(f"    injected mass: {float(total_injected_mass):.6f}", flush=True)

    v = jnp.zeros_like(rho)
    e = initial_energy  # initial thermal energy per unit mass (0.05 = cold Evrard)
    p = (GAMMA - 1) * rho * e
    p = jnp.where(p < params.minimum_pressure, params.minimum_pressure, p)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=v,
        velocity_y=v,
        velocity_z=v,
        gas_pressure=p,
    )
    config = finalize_config(config, initial_state.shape)

    snapshots = jax.block_until_ready(
        time_integration(initial_state, config, params, registered_variables)
    )
    return snapshots, helper_data, registered_variables
