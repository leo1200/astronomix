"""
Regression test of the VL2 self-gravity coupling against Athena++.

Two small Athena++ runs (VL2 + PLM + HLLC with FFT self-gravity, discrete-
Laplacian kernel, periodic: a nonlinear Jeans wave on 16 x 8 x 8 cells and the
advected equilibrium slab on 16^3 cells, one period each) are stored in
``pytests/self_gravity/data/athenapp_vl2_gravity`` (written by
``examples/scripts/validation/athenapp_self_gravity/compare_to_athenapp.py
--regression-data``). Every case is re-run here from the same initial state with
astronomix's ``ATHENA_SECOND_ORDER`` coupling: it has to take the same number of
cycles and agree with Athena++ to round-off. The tolerance (1e-9 of the
perturbation's L1 norm) is orders of magnitude above the observed differences
(1e-15 to 1e-13) and below the effect of any change to the coupling (the
spectral Poisson kernel alone changes the result by ~1e-4).

Two conservation tests follow on a generic periodic state: with the
discrete-Laplacian kernel the momentum source conserves total momentum, and
with the time-centred energy source of Mullen, Hanawa & Gammie (2021) total
energy (including the gravitational energy) is conserved to round-off.
"""

# general
import os

# numerics
import numpy as np

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    HLLC,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    VL2,
)
from astronomix.option_classes.simulation_config import ATHENA_SECOND_ORDER

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FVSolverConfig,
    GravityConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)
from astronomix.option_classes.simulation_config import (
    StaticFloatVector,
    StaticIntVector,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

pytestmark = pytest.mark.x64

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "athenapp_vl2_gravity")

REGRESSION_CASES = ("jeans_16", "slab_16")

#: Largest accepted L1 difference to Athena++, relative to the L1 norm of the
#: perturbation (density, pressure) or of the field (velocities).
RELATIVE_TOLERANCE = 1e-9

#: Largest accepted relative drift of the conserved totals.
CONSERVATION_TOLERANCE = 1e-12


def vl2_gravity_config(num_cells, box_size, time_centred_energy_source=False, **extra_options):
    """
    The periodic VL2 configuration with Athena++'s coupling and Poisson kernel.

    Args:
        num_cells: The cells per axis.
        box_size: The box size per axis.
        time_centred_energy_source: Whether to use the time-centred energy
            source of Mullen, Hanawa & Gammie (2021).
        **extra_options: Further ``SimulationConfig`` options.

    Returns:
        The (not yet finalized) simulation configuration.
    """
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    return SimulationConfig(
        backend_config=BackendConfig(backend=NATIVE_JAX),
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(time_integrator=VL2, riemann_solver=HLLC),
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=ATHENA_SECOND_ORDER,
            poisson_discrete_laplacian=True,
            time_centred_energy_source=time_centred_energy_source,
        ),
        mhd=False,
        dimensionality=3,
        box_size=StaticFloatVector(*[float(size) for size in box_size]),
        num_cells=StaticIntVector(*[int(cells) for cells in num_cells]),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        progress_bar=False,
        return_snapshots=True,
        **extra_options,
    )


@pytest.mark.parametrize("case_name", REGRESSION_CASES)
def test_vl2_gravity_matches_athenapp(case_name):
    reference = np.load(os.path.join(DATA_DIRECTORY, f"{case_name}.npz"))
    initial_state = jnp.asarray(reference["initial_state"])

    config = vl2_gravity_config(
        initial_state.shape[1:],
        reference["box_size"],
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    params = SimulationParams(
        C_cfl=float(reference["courant_number"]),
        t_end=float(reference["t_end"]),
        gamma=float(reference["gamma"]),
        gravitational_constant=float(reference["gravitational_constant"]),
    )
    registered_variables = get_registered_variables(config)
    config = finalize_config(config, initial_state.shape)
    result = time_integration(initial_state, config, params, registered_variables)

    final_state = np.asarray(result.final_state)
    reference_state = reference["final_state"]
    initial = np.asarray(initial_state)
    relative_l1 = []
    for index in range(5):
        scale_field = initial[index]
        if index in (registered_variables.density_index, registered_variables.pressure_index):
            scale_field = scale_field - scale_field.mean()
        relative_l1.append(
            np.sum(np.abs(final_state[index] - reference_state[index])) / np.sum(np.abs(scale_field))
        )

    assert int(result.num_iterations) == int(reference["cycles"])
    assert max(relative_l1) < RELATIVE_TOLERANCE, relative_l1


def _generic_periodic_run(time_centred_energy_source):
    """
    Twenty fixed steps of a self-gravitating periodic state with large density,
    velocity and pressure variations along all axes.

    Args:
        time_centred_energy_source: Whether to use the time-centred energy
            source.

    Returns:
        ``(initial_state, final_state, total_energy)`` with the total energy
        (thermal, kinetic and gravitational) at the start and the end.
    """
    config = vl2_gravity_config(
        (16, 16, 16),
        (1.0, 1.0, 1.0),
        time_centred_energy_source=time_centred_energy_source,
        num_snapshots=2,
        fixed_timestep=True,
        num_timesteps=20,
        snapshot_settings=SnapshotSettings(return_final_state=True, return_total_energy=True),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    x, y, z = jnp.transpose(helper_data.geometric_centers, (3, 0, 1, 2))
    phase = 2 * jnp.pi * (x + 2 * y + z)
    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=1.0 + 0.3 * jnp.sin(phase) + 0.1 * jnp.cos(2 * jnp.pi * y),
        velocity_x=0.3 * jnp.cos(2 * jnp.pi * z),
        velocity_y=0.2 * jnp.sin(phase),
        velocity_z=0.1 + 0.0 * x,
        gas_pressure=1.0 + 0.2 * jnp.cos(phase),
    )
    params = SimulationParams(t_end=0.05, gamma=5 / 3, gravitational_constant=10.0)
    config = finalize_config(config, initial_state.shape)
    result = time_integration(initial_state, config, params, registered_variables)
    return np.asarray(initial_state), np.asarray(result.final_state), np.asarray(result.total_energy)


def test_vl2_gravity_conserves_momentum():
    initial_state, final_state, _ = _generic_periodic_run(time_centred_energy_source=False)
    for velocity_index in (1, 2, 3):
        initial_momentum = np.sum(initial_state[0] * initial_state[velocity_index])
        final_momentum = np.sum(final_state[0] * final_state[velocity_index])
        momentum_scale = np.sum(np.abs(initial_state[0] * initial_state[velocity_index]))
        assert abs(final_momentum - initial_momentum) / momentum_scale < CONSERVATION_TOLERANCE


def test_time_centred_energy_source_conserves_energy():
    _, _, total_energy = _generic_periodic_run(time_centred_energy_source=True)
    assert abs(total_energy[-1] - total_energy[0]) / abs(total_energy[0]) < CONSERVATION_TOLERANCE
