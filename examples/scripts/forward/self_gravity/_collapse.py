"""Shared Evrard-collapse setup for the self-gravity figures.

The energy budget (``energy_conservation_comparison.py``), the radial profiles
(``radial_profiles_comparison.py``), the energy error against the resolution
(``evrard_energy_convergence.py``) and against a fixed time step
(``evrard_timestep_convergence.py``) all run the same finite-difference Evrard
collapse: a sphere of radius 1 and mass 1 with ``rho ~ 1 / r`` at rest in a
tenuous ambient medium, ``G = 1``, isolated (open-boundary) Poisson solve.
They differ in the resolution, the end time, the initial thermal energy and the
diagnostics they keep.

The scheme is the finite-difference baseline: WENO-Z weights, the admissible
face state and positivity-preserving WENO (Zhang & Shu 2012, inside the
reconstruction and independent of the time step), with the gravitational work
optionally flux-corrected (``GravityConfig.work_flux_correction``). The
128^3 comparison adds the second-order finite-volume scheme (VL2, PLM, HLLC,
C_cfl = 0.3) with Athena++'s coupling, optionally with the time-centred energy
source of Mullen, Hanawa & Gammie (2021). The finite-volume runs need
AthenaPK's density and pressure floors (``MINIMUM_DENSITY``,
``MINIMUM_PRESSURE``): the gravitational work charged to the cold ambient
cells at the sphere's edge exceeds their internal energy in the first steps.
"""

# typing
from typing import NamedTuple, Optional

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    FORWARDS,
    HLLC,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    VL2,
)
from astronomix.option_classes.simulation_config import (
    ATHENA_SECOND_ORDER,
    FOURTH_ORDER_CONSERVATIVE,
    POSITIVITY_HARD_FLOOR,
)

# astronomix containers
from astronomix import (
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    SnapshotSettings,
    BackendConfig,
    FVSolverConfig,
    FDSolverConfig,
)

# astronomix functions
from astronomix import (
    get_helper_data,
    get_registered_variables,
    time_integration,
    construct_primitive_state,
    finalize_config,
)

# figure helpers
from _common import FINITE_VOLUME_COURANT_NUMBER


GAMMA = 5 / 3
BOX_SIZE = 4.0
SPHERE_RADIUS = 1.0
SPHERE_MASS = 1.0
AMBIENT_DENSITY = 1e-4

#: Initial thermal energy per unit mass of the standard (cold) Evrard collapse
#: and of the warmer, milder variant used for the convergence figures.
COLD_INITIAL_ENERGY = 0.05
MILD_INITIAL_ENERGY = 0.20

#: The floors of the setup. With positivity-preserving WENO they only enter
#: the admissibility test of the reconstruction and the wave-speed estimates;
#: the evolved state is never clamped. The finite-volume scheme applies them
#: as AthenaPK's dfloor and pfloor.
MINIMUM_DENSITY = 1e-5
MINIMUM_PRESSURE = 3e-6

#: The Courant number of the CFL-stepped runs: the bound up to which the
#: positivity-preserving reconstruction is provable with SSPRK4.
COURANT_NUMBER = 0.75

DEFAULT_NUM_SNAPSHOTS = 60


class CollapseScheme(NamedTuple):
    """The scheme options of one collapse run.

    The defaults are the finite-difference baseline of the paper: WENO-Z,
    positivity-preserving WENO, the fourth-order conservative (corrected
    flux-based) coupling and no flux-corrected work. The
    ``ATHENA_SECOND_ORDER`` coupling selects the finite-volume VL2 scheme
    (the WENO options are then not read).
    """

    self_gravity_version: int = FOURTH_ORDER_CONSERVATIVE
    weno_z: bool = True
    positivity_preserving: bool = True
    work_flux_correction: bool = False
    time_centred_energy_source: bool = False

    @property
    def finite_volume(self) -> bool:
        """Whether the run uses the finite-volume VL2 scheme."""
        return self.self_gravity_version == ATHENA_SECOND_ORDER


def collapse_config(
    num_cells: int,
    scheme: CollapseScheme,
    backend: int = PALLAS,
    num_snapshots: int = DEFAULT_NUM_SNAPSHOTS,
    want_states: bool = False,
    fixed_num_timesteps: Optional[int] = None,
) -> SimulationConfig:
    """Build the (not yet finalized) ``SimulationConfig`` of the collapse.

    Args:
        num_cells: Number of cells per dimension of the cubic grid.
        scheme: The scheme options.
        backend: The compute backend (``PALLAS`` or ``NATIVE_JAX``) of the
            finite-difference scheme; the finite-volume gravity coupling runs
            natively.
        num_snapshots: Number of energy (and optionally state) snapshots.
        want_states: Whether the snapshots also keep the full states.
        fixed_num_timesteps: If given, integrate with this many equal time
            steps ``dt = t_end / fixed_num_timesteps`` instead of the CFL step.

    Returns:
        The ``SimulationConfig`` of the collapse run.
    """
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    if scheme.finite_volume:
        solver_options = dict(
            backend_config=BackendConfig(backend=NATIVE_JAX),
            solver_mode=FINITE_VOLUME,
            solver_config=FVSolverConfig(time_integrator=VL2, riemann_solver=HLLC),
            positivity_config=PositivityConfig(per_step_mode=POSITIVITY_HARD_FLOOR),
        )
    else:
        solver_options = dict(
            backend_config=BackendConfig(backend=backend),
            solver_config=FDSolverConfig(
                weno_z=scheme.weno_z,
                weno_positivity_preserving=scheme.positivity_preserving,
            ),
        )
    config = SimulationConfig(
        **solver_options,
        runtime_debugging=False,
        progress_bar=False,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=scheme.self_gravity_version,
            poisson_manual_open_boundaries=True,
            work_flux_correction=scheme.work_flux_correction,
            time_centred_energy_source=scheme.time_centred_energy_source,
        ),
        mhd=False,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        differentiation_mode=FORWARDS,
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        return_snapshots=True,
        num_snapshots=num_snapshots,
        snapshot_settings=SnapshotSettings(
            return_states=want_states,
            return_final_state=True,
            return_total_energy=True,
            return_internal_energy=True,
            return_kinetic_energy=True,
            return_gravitational_energy=True,
        ),
    )
    if fixed_num_timesteps is not None:
        config = config._replace(
            fixed_timestep=True,
            num_timesteps=int(fixed_num_timesteps),
        )
    return config


def collapse_params(
    t_end: float,
    courant_number: float = COURANT_NUMBER,
    minimum_density: float = MINIMUM_DENSITY,
    minimum_pressure: float = MINIMUM_PRESSURE,
) -> SimulationParams:
    """The ``SimulationParams`` of the collapse.

    Args:
        t_end: The end time of the integration.
        courant_number: The Courant number of the CFL step (unused by
            fixed-step runs).
        minimum_density: The density floor of the admissibility test and the
            wave-speed estimates.
        minimum_pressure: The pressure floor of the same.

    Returns:
        The simulation parameters.
    """
    return SimulationParams(
        t_end=t_end,
        C_cfl=courant_number,
        dt_max=jnp.inf,
        minimum_density=minimum_density,
        minimum_pressure=minimum_pressure,
    )


def collapse_initial_state(config, params, initial_energy, helper_data, registered_variables):
    """The Evrard initial condition: ``rho = M / (2 pi R^2 r)`` at rest.

    Args:
        config: The simulation configuration.
        params: The simulation parameters (provide the pressure floor).
        initial_energy: The initial thermal energy per unit mass of the
            sphere (and of the ambient medium).
        helper_data: The helper data (provides the cell-centre radius).
        registered_variables: The registered variables.

    Returns:
        The primitive initial state.
    """
    radius = helper_data.r
    density = jnp.where(
        radius <= SPHERE_RADIUS,
        SPHERE_MASS / (2 * jnp.pi * SPHERE_RADIUS**2 * radius),
        AMBIENT_DENSITY,
    )
    pressure = jnp.maximum((GAMMA - 1) * density * initial_energy, params.minimum_pressure)
    velocity = jnp.zeros_like(density)
    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity,
        velocity_y=velocity,
        velocity_z=velocity,
        gas_pressure=pressure,
    )


def run_collapse(
    num_cells: int,
    scheme: CollapseScheme,
    t_end: float,
    initial_energy: float = COLD_INITIAL_ENERGY,
    backend: int = PALLAS,
    num_snapshots: int = DEFAULT_NUM_SNAPSHOTS,
    want_states: bool = False,
    fixed_num_timesteps: Optional[int] = None,
    courant_number: Optional[float] = None,
    minimum_density: float = MINIMUM_DENSITY,
    minimum_pressure: float = MINIMUM_PRESSURE,
):
    """Run the Evrard collapse.

    Args:
        num_cells: Number of cells per dimension of the cubic grid.
        scheme: The scheme options.
        t_end: The end time of the integration.
        initial_energy: Initial thermal energy per unit mass (0.05 = the
            standard cold Evrard collapse, 0.20 = the mild collapse).
        backend: The compute backend.
        num_snapshots: Number of energy (and optionally state) snapshots.
        want_states: Whether the snapshots also keep the full states.
        fixed_num_timesteps: If given, the number of equal time steps;
            otherwise the CFL step with ``courant_number``.
        courant_number: The Courant number of the CFL step (default:
            ``COURANT_NUMBER``, or ``FINITE_VOLUME_COURANT_NUMBER`` for the
            finite-volume scheme).
        minimum_density: The density floor (see ``collapse_params``).
        minimum_pressure: The pressure floor (see ``collapse_params``).

    Returns:
        ``(snapshots, helper_data, registered_variables)`` of the completed run.
    """
    if courant_number is None:
        courant_number = FINITE_VOLUME_COURANT_NUMBER if scheme.finite_volume else COURANT_NUMBER
    config = collapse_config(
        num_cells,
        scheme,
        backend=backend,
        num_snapshots=num_snapshots,
        want_states=want_states,
        fixed_num_timesteps=fixed_num_timesteps,
    )
    params = collapse_params(
        t_end,
        courant_number,
        minimum_density=minimum_density,
        minimum_pressure=minimum_pressure,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    initial_state = collapse_initial_state(
        config,
        params,
        initial_energy,
        helper_data,
        registered_variables,
    )
    config = finalize_config(config, initial_state.shape)

    snapshots = jax.block_until_ready(
        time_integration(initial_state, config, params, registered_variables)
    )
    return snapshots, helper_data, registered_variables


def relative_energy_error(snapshots):
    """The relative total-energy error ``|E - E_0| / |E_0|`` at every snapshot.

    Args:
        snapshots: The snapshot data of a collapse run.

    Returns:
        The relative error per snapshot.
    """
    total_energy = snapshots.total_energy
    return jnp.abs(total_energy - total_energy[0]) / jnp.abs(total_energy[0])
