"""
Sink particles in the Evrard collapse pytest.

Runs the Evrard collapse of ``examples/scripts/forward/self_gravity/_collapse.py``
with sink particles on, and checks that the run is finite and that gas plus
sink mass and energy are conserved. The energy check is expected to fail until
the gravity of the sinks is implemented: accretion removes the thermal and
gravitational energy of the accreted gas, and the gravitational energy of the
sinks is not computed.

The runs use the second-order conservative self-gravity coupling, which keeps
the total energy constant to float32 rounding, and end before the core bounce,
since that coupling becomes unstable later in the collapse at these
resolutions.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# typing
from typing import NamedTuple

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    FINITE_DIFFERENCE,
    FORWARDS,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
)
from astronomix.option_classes.simulation_config import SECOND_ORDER_CONSERVATIVE

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    SimulationConfig,
    SimulationParams,
    SinkParticleConfig,
    SnapshotSettings,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    finalize_state,
    get_helper_data,
    get_registered_variables,
    time_integration,
)


GAMMA = 5 / 3
BOX_SIZE = 4.0
NUM_SNAPSHOTS = 10


class CollapseCase(NamedTuple):
    """One run of the collapse: resolution, backend, end time and the
    tolerance of the energy check."""

    num_cells: int
    backend: int
    t_end: float
    # The larger of twice the largest relative drift of the total energy over
    # the snapshots in a run of the same case without sinks, and 1e-5, which
    # stays above the float32 rounding of the total energy on any hardware.
    energy_tolerance: float


CASES = {
    # Measured drift with sinks off: 9.60e-7 (GPU A100 80GB PCIe, float32, native JAX).
    "cpu": CollapseCase(32, NATIVE_JAX, 0.1, 1e-5),
    # Measured drift with sinks off: 9.60e-7 (GPU A100 80GB PCIe, float32, Pallas).
    "gpu": CollapseCase(64, PALLAS, 0.2, 1e-5),
}


def _run_collapse(num_cells, t_end, backend):
    """Run the Evrard collapse (setup of
    ``examples/scripts/forward/self_gravity/_collapse.py``).

    Args:
        num_cells: Number of cells per dimension of the cubic grid.
        t_end: The end time of the integration.
        backend: The compute backend (``NATIVE_JAX`` or ``PALLAS``).

    Returns:
        The snapshots of the run.
    """
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        backend_config=BackendConfig(backend=backend),
        runtime_debugging=False,
        progress_bar=False,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=SECOND_ORDER_CONSERVATIVE,
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
            return_final_state=True,
            return_total_mass=True,
            return_total_energy=True,
        ),
        num_snapshots=NUM_SNAPSHOTS,
        state_struct=True,
        sink_particle_config=SinkParticleConfig(sink_particles=True),
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

    # Evrard sphere: rho = M / (2 pi R^2 r) inside R, at rest, with thermal
    # energy per unit mass e = 0.05.
    R = 1.0
    M = 1.0
    rho = jnp.where(
        helper_data.r <= R, M / (2 * jnp.pi * R**2 * helper_data.r), 1e-4
    )
    v = jnp.zeros_like(rho)
    e = 0.05
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
    initial_state = finalize_state(config, initial_state)

    return jax.block_until_ready(
        time_integration(initial_state, config, params, registered_variables)
    )


@pytest.fixture(
    scope="module",
    params=[
        "cpu",
        pytest.param(
            "gpu",
            marks=pytest.mark.skipif(
                jax.default_backend() != "gpu", reason="needs a GPU"
            ),
        ),
    ],
)
def collapse_run(request):
    """The snapshots of the collapse with sinks on, and the case they were
    run with. Each case runs once and is shared by the tests below."""
    case = CASES[request.param]
    snapshots = _run_collapse(case.num_cells, case.t_end, case.backend)
    return snapshots, case


def test_collapse_finite(collapse_run):
    """The final gas state, the mass and energy diagnostics and the sink
    particles contain no NaN or inf.

    Args:
        collapse_run: The snapshots of the run and its case.
    """
    snapshots, _ = collapse_run
    sinks = snapshots.sink_particles
    for name, values in [
        ("final state", snapshots.final_state),
        ("total mass", snapshots.total_mass),
        ("total energy", snapshots.total_energy),
        ("sink mass", sinks.mass),
        ("sink position", sinks.position),
        ("sink velocity", sinks.velocity),
    ]:
        assert jnp.all(jnp.isfinite(values)), f"{name} is not finite"


def test_collapse_mass_conservation(collapse_run):
    """Gas plus sink mass at every snapshot equals the initial gas mass.

    Args:
        collapse_run: The snapshots of the run and its case.
    """
    snapshots, _ = collapse_run
    sink_mass = snapshots.sink_particles.mass.sum(axis=1)

    # Without accretion the check below would pass trivially.
    assert sink_mass[-1] > 0.0

    total_mass = snapshots.total_mass + sink_mass
    assert jnp.allclose(total_mass, snapshots.total_mass[0], rtol=1e-4, atol=0.0)


def test_collapse_energy_conservation(collapse_run):
    """Gas energy plus sink kinetic energy at every snapshot stays within the
    energy drift of the scheme of the initial gas energy.

    Args:
        collapse_run: The snapshots of the run and its case.
    """
    snapshots, case = collapse_run
    sinks = snapshots.sink_particles
    sink_kinetic_energy = 0.5 * jnp.sum(
        sinks.mass * jnp.sum(sinks.velocity**2, axis=-1), axis=1
    )
    total_energy = snapshots.total_energy + sink_kinetic_energy

    relative_drift = jnp.max(
        jnp.abs(total_energy - total_energy[0]) / jnp.abs(total_energy[0])
    )
    assert relative_drift <= case.energy_tolerance, (
        f"relative energy drift {float(relative_drift):.3e} exceeds "
        f"{case.energy_tolerance:.3e}"
    )
