"""
Static simulation configuration.

Defines :class:`SimulationConfig` — the bundle of options that, unlike the
simulation parameters, necessitate recompilation when changed — together with
the integer-coded enumerations they reference (backends, solver/boundary/Riemann
modes, positivity modes, ...), the small geometry vector helpers, the sub-configs
for gravity and positivity, and the ``finalize_config`` pass that fills in
derived fields and validates the configuration.
"""

# general
import math

# typing
from types import NoneType
from typing import NamedTuple, Tuple, Union
from jaxtyping import Array, Float

# jax
import jax

# astronomix containers
from astronomix._modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
)
from astronomix._modules._cooling.cooling_options import CoolingConfig
from astronomix._modules._cosmic_rays.cosmic_ray_options import CosmicRayConfig
from astronomix._modules._neural_net_force._neural_net_force_options import (
    NeuralNetForceConfig,
)

from jaxtyping import Array, Float, Bool, Int
from astronomix._modules._stellar_wind.stellar_wind_options import WindConfig
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig

# ===================== constant definition =====================

# backends (very limited support currently)
NATIVE_JAX = 0
PALLAS = 1

# positivity-enforcement modes (used by ``PositivityConfig.per_stage_mode`` /
# ``per_step_mode``).  HARD_FLOOR clamps density (and, for ideal
# gas, pressure) pointwise — cheap, non-conservative, matches the *adiabatic*
# HOW-MHD ``prot.f``.  REDISTRIBUTE neighbour-averages density+momentum (and
# energy) over the valid 3x3x3 neighbourhood of sub-threshold cells — much
# gentler at strong shocks than a hard floor (no sharp floored cell), matches
# the *isothermal* HOW-MHD ``prot.f`` (not strictly mass-conserving: like
# ``prot.f`` it copies neighbour values without debiting the donors).
POSITIVITY_NONE = 0
POSITIVITY_HARD_FLOOR = 1
POSITIVITY_REDISTRIBUTE = 2
#: CONSERVATIVE: enforce internal-energy positivity by an antisymmetric
#: face-flux diffusion that pulls internal energy into (near-)negative-pressure
#: cells from their hotter neighbours (exact total-energy conservation), plus a
#: density floor / vacuum-rest for voids and a minimal residual pressure floor
#: as the unconditional guarantee. The smooth, conservative cousin of HARD_FLOOR:
#: it keeps the energy-conserving self-gravity scheme stable on violent collapse
#: without the 100%+ energy injection a bare floor causes.
POSITIVITY_CONSERVATIVE = 3

# solver modes
FINITE_VOLUME = 0
FINITE_DIFFERENCE = 1

# differentiation modes
FORWARDS = 0
BACKWARDS = 1

# limiter types
MINMOD = 0
OSHER = 1
DOUBLE_MINMOD = 2
SUPERBEE = 3
VAN_ALBADA = 4
VAN_ALBADA_PP = 5

# splitting modes
UNSPLIT = 0
SPLIT = 1

# Riemann solvers
HLL = 0
HLLC = 1
HLLC_LM = 2
LAX_FRIEDRICHS = 3
HYBRID_HLLC = 4
AM_HLLC = 5

# time integrators
# currently only for finite volume
RK2_SSP = 0
MUSCL = 1
# currently only for finite difference
RK4_SSP = 2
RK4_LSRK = 3

# boundary conditions
OPEN_BOUNDARY = 0
REFLECTIVE_BOUNDARY = 1
PERIODIC_BOUNDARY = 2
FIXED_BOUNDARY = 3
MHD_JET_BOUNDARY = 4
FIXED_BOUNDARY_OPEN_MOMENTUM = 5

PRIMITIVE_GAS_STATE = 0
CONSERVATIVE_GAS_STATE = 1
VELOCITY_ONLY = 2
MAGNETIC_FIELD_ONLY = 3

# geometry types
CARTESIAN = 0
CYLINDRICAL = 1
SPHERICAL = 2

# axes
VARAXIS = 0
XAXIS = 1
YAXIS = 2
ZAXIS = 3

# boundary handling modes
GHOST_CELLS = 0
PERIODIC_ROLL = 1
# OPEN_SHIFT = 2

# self-gravity coupling schemes (FD):
#   SIMPLE_SOURCE              - rho * v * a energy source (non-conservative)
#   SECOND_ORDER_CONSERVATIVE  - flux-based energy source (2nd-order accurate)
#   FOURTH_ORDER_CONSERVATIVE  - corrected flux-based energy source (4th-order,
#                                the energy-conserving high-order scheme)
SIMPLE_SOURCE = 0
SECOND_ORDER_CONSERVATIVE = 1
FOURTH_ORDER_CONSERVATIVE = 2

# Magnetic part integrators for split MHD
IMPLICIT_MIDPOINT = 0
IMPLICIT_EULER = 1

# Numerical precision
SINGLE_PRECISION = 0
DOUBLE_PRECISION = 1

# Viscosity types
KINEMATIC_VISCOSITY = 0
DYNAMIC_VISCOSITY = 1

# Equation of state
IDEAL_GAS = 0
ISOTHERMAL = 1

# Snapshot storage modes
ON_DEVICE = 0
TO_DISK = 1

# ============================================================

# ===================== type definitions =====================

class StaticIntVector(NamedTuple):
    """A static (compile-time) per-axis integer triple (e.g. cells per axis)."""

    x: int = -1
    y: int = -1
    z: int = -1


class StaticFloatVector(NamedTuple):
    """A static (compile-time) per-axis float triple (e.g. box size per axis)."""

    x: float = -1.0
    y: float = -1.0
    z: float = -1.0

    def __truediv__(self, other: StaticIntVector) -> "StaticFloatVector":
        """Divide component-wise by a :class:`StaticIntVector` (e.g. box / cells)."""
        if not isinstance(other, StaticIntVector):
            return NotImplemented
        return StaticFloatVector(
            x=self.x / other.x,
            y=self.y / other.y,
            z=self.z / other.z,
        )

STATE_TYPE = Union[
    Float[Array, "num_vars num_cells_x"],
    Float[Array, "num_vars num_cells_x num_cells_y"],
    Float[Array, "num_vars num_cells_x num_cells_y num_cells_z"],
]

STATE_TYPE_ALTERED = Union[
    Float[Array, "num_vars num_cells_a"],
    Float[Array, "num_vars num_cells_a num_cells_b"],
    Float[Array, "num_vars num_cells_a num_cells_b num_cells_c"],
]

FIELD_TYPE = Union[
    Float[Array, "num_cells_x"],
    Float[Array, "num_cells_x num_cells_y"],
    Float[Array, "num_cells_x num_cells_y num_cells_z"],
]

BOOL_FIELD_TYPE = Union[
    Bool[Array, "num_cells_x"],
    Bool[Array, "num_cells_x num_cells_y"],
    Bool[Array, "num_cells_x num_cells_y num_cells_z"],
]

INT_FIELD_TYPE = Union[
    Int[Array, "num_cells_x"],
    Int[Array, "num_cells_x num_cells_y"],
    Int[Array, "num_cells_x num_cells_y num_cells_z"],
]

GEOMETRY_TYPE = Union[
    Float[Array, "num_cells_x 2"],
    Float[Array, "num_cells_x num_cells_y 2"],
    Float[Array, "num_cells_x num_cells_y num_cells_z 2"],
]

class SnapshotSettings(NamedTuple):
    """Settings for the snapshot output of the simulation."""

    #: Whether to record the full primitive state at every checkpoint.
    #: This is the single biggest snapshot allocation
    #: (``num_snapshots × num_vars × num_cells^d``); it is **opt-in**.
    #: Set to ``True`` if you actually need the per-snapshot states; for
    #: the common case of only wanting a final state plus integrated
    #: diagnostics (energies, total mass, runtime, num_iterations), the
    #: default ``False`` skips the per-snapshot state allocation entirely.
    return_states: bool = False

    #: Whether to return the final state of the simulation.
    return_final_state: bool = False

    #: Whether to return the total mass at the times the snapshots were taken.
    return_total_mass: bool = False

    #: Whether to return the total energy at the times the snapshots were taken.
    return_total_energy: bool = False

    #: Whether to return internal energy
    return_internal_energy: bool = False

    #: Whether to return kinetic energy
    return_kinetic_energy: bool = False

    #: Whether to return gravitational energy
    return_gravitational_energy: bool = False

    #: Whether to return radial momentum
    return_radial_momentum: bool = False

    #: Whether to return the kinetic energy spectrum
    return_kinetic_energy_spectrum: bool = False

    #: Whether to return the magnetic energy spectrum
    return_magnetic_energy_spectrum: bool = False

    #: Whether to return the helicity spectrum
    return_helicity_spectrum: bool = False

    #: Whether to return the magnetic field divergence
    #: NOTE: currently only implemented for finite difference MHD
    return_magnetic_divergence: bool = False

    #: Whether to return the temperature PDF (dV/dlogT)
    return_temperature_pdf: bool = False
    num_temperature_bins: int = 100
    temperature_pdf_min: float = 1e-10
    temperature_pdf_max: float = 1e10


class BoundarySettings1D(NamedTuple):
    """The boundary-condition type at the left and right end of a single axis."""

    left_boundary: int = OPEN_BOUNDARY
    right_boundary: int = OPEN_BOUNDARY


class BoundarySettings(NamedTuple):
    """Per-axis boundary settings for the simulation."""

    x: BoundarySettings1D = BoundarySettings1D()
    y: BoundarySettings1D = BoundarySettings1D()
    z: BoundarySettings1D = BoundarySettings1D()


class GravityConfig(NamedTuple):
    """Self-gravity and external-potential configuration."""

    #: Self-gravity switch (currently only for periodic / manual-open boundaries).
    self_gravity: bool = False

    #: Coupling of the self-gravity source to the hydrodynamics. One of
    #: ``SIMPLE_SOURCE`` / ``SECOND_ORDER_CONSERVATIVE`` /
    #: ``FOURTH_ORDER_CONSERVATIVE``.
    self_gravity_version: int = FOURTH_ORDER_CONSERVATIVE

    #: Enable an external, static gravitational potential provided via
    #: ``params.gravitational_potential``. It is added to the self-gravity
    #: potential (if any) in ``_compute_total_potential``.
    external_potential: bool = False

    #: Manual open boundary conditions in the Poisson solver.
    poisson_manual_open_boundaries: bool = False

    #: Master gravity switch. Set automatically in ``finalize_config`` to
    #: ``self_gravity or external_potential``; gates the gravity source-term
    #: machinery so an external potential works without self-gravity. Not set
    #: by the user directly.
    gravity: bool = False


class PositivityConfig(NamedTuple):
    """
    Density/pressure positivity-enforcement configuration.
    """

    #: Casual on/off switch for the per-stage / per-step STATE floors. Default
    #: False (no flooring). When True, finalize_config sets per_stage_mode and
    #: per_step_mode to HARD_FLOOR unless explicitly overridden. Does NOT affect
    #: the read-only ``clamp_in_estimates`` (always respected).
    default_positivity_protection: bool = False

    #: Positivity enforcement applied inside every SSPRK/LSRK stage (on the
    #: conserved state — the CFL lever for strong shocks). One of
    #: ``POSITIVITY_{NONE,HARD_FLOOR,REDISTRIBUTE,CONSERVATIVE}``. Default NONE;
    #: set to HARD_FLOOR by finalize when ``default_positivity_protection``.
    per_stage_mode: int = POSITIVITY_NONE

    #: Positivity enforcement applied once per step before the evolve (on the
    #: primitive state). With turbulent forcing + ``vacuum_protection`` the
    #: conservative ``prot`` redistribution already runs once per step, so a
    #: per-step REDISTRIBUTE here is redundant and is auto-skipped.
    per_step_mode: int = POSITIVITY_NONE

    #: Read-only density/pressure clamp in the flux / eigenvalue / timestep
    #: estimates (NaN-safety; does NOT modify the evolved state). This is the
    #: role the old ``enforce_positivity`` bool played in those estimators.
    #: DECOUPLED from ``default_positivity_protection`` and ON by default --
    #: cheap insurance that never touches the conserved solution.
    clamp_in_estimates: bool = True

    #: Vacuum-rest velocity recovery: zero the momentum in below-floor (vacuum)
    #: cells so the recovered velocity is 0 rather than ``momentum/rho_floored``
    #: (which spikes and drives high-Mach blow-up); lets ``minimum_density`` be
    #: lowered by orders of magnitude without instability.
    vacuum_rest: bool = False

    #: NaN/inf backstop: reset non-finite conserved entries to zero before the
    #: density/pressure floors so they become a valid floored state.
    nan_safe: bool = False

    #: POSITIVITY_CONSERVATIVE-mode parameters (conservative internal-energy
    #: redistribution): per-axis diffusion coefficient (stability needs
    #: < 1/(2*dim)), number of Jacobi passes, and the activation margin in units
    #: of the internal-energy floor (keep ~1 -- genuine near-violations only).
    cons_coeff: float = 0.15
    cons_passes: int = 16
    cons_activate: float = 1.0

    #: Deep-void first-order flux blending (FOFC-style): blend the WENO interface
    #: flux toward LLF in cells near the density floor; the weight ramps from 1
    #: at the floor to 0 at ``deepvoid_blend_factor * minimum_density``.
    deepvoid_blend: bool = False
    deepvoid_blend_factor: float = 8.0

    #: Positivity-preserving (Hu-Adams-Shu / Zalesak FCT) flux limiter: blend the
    #: WENO flux toward LLF by the largest weight keeping the LF-updated density
    #: AND pressure above their floors. Shares the unified flux-blending
    #: infrastructure with ``deepvoid_blend`` (different activation path; both may
    #: be on, the stronger blend wins). Forces the non-fused WENO+divergence path.
    preserving_flux: bool = False


class SimulationConfig(NamedTuple):
    """
    Configuration object for the simulation.
    The simulation configuration are parameters defining
    the simulation where changes necessitate recompilation.
    """

    # Static simulation parameters

    #: Backend
    backend: int = NATIVE_JAX
    pallas_block_shape: Tuple[int, int, int] = (4, 4, 8)
    pallas_use_triton: bool = True
    pallas_interpret: bool = False
    pallas_num_warps: int = 4
    #: Toggle for the Pallas constrained-transport helpers
    #: (``update_cell_center_fields``, ``constrained_transport_rhs``).
    #: Disabled by default: the staged Pallas-CT pipeline gives a clear
    #: memory win at small grids (~65% temp at N=16 on alfven_wave3D)
    #: but only marginal savings at production scale (~2% temp at N=64)
    #: while adding ~25s of one-time compile cost.  Flip to True if the
    #: small-N memory profile matters; the rest of the Pallas backend
    #: stays on regardless.
    pallas_ct: bool = False

    #: Basic solver mode, either finite volume or finite difference.
    #: FINITE_DIFFERENCE is for now only planned for the HOW_MHD
    #: scheme (Jeongbhin Seo, Dongsu Ryu, 2023).
    solver_mode: int = FINITE_VOLUME

    #: Precision mode.
    numerical_precision: int = SINGLE_PRECISION

    #: Debug runtime errors, throws exceptions
    #: on e.g. negative pressure or density.
    #: Significantly reduces performance.
    runtime_debugging: bool = False

    #: Donate the state arrays to the time integration function
    #: to reduce memory allocations. If activated, the
    #: initial state arrays will be invalid after
    #: the simulation.
    donate_state: bool = False

    #: Memory analysis of the main time integration
    #: function
    memory_analysis: bool = False

    #: Build the simulation helper data on the host (CPU) and
    #: only move the fields that are actually needed by the
    #: enabled subsystems onto the accelerator. Useful in
    #: production runs where a large meshgrid like
    #: ``geometric_centers`` is not required on device and the
    #: per-field memory footprint matters.
    host_helper_data: bool = False

    #: Print the elapsed time of the simulation
    print_elapsed_time: bool = False

    #: Activate progress bar
    progress_bar: bool = False

    #: The number of dimensions of the simulation.
    dimensionality: int = 1

    #: Use a struct for the state.
    state_struct: bool = False

    #: The geometry of the simulation.
    geometry: int = CARTESIAN

    #: The random seed for any stochastic processes
    #: in the simulation, e.g. turbulent forcing.
    random_seed: int = 42

    #: The equation of state for the simulation.
    #: NOTE: CURRENTLY ONLY IMPLEMENTED FOR 
    #: FINITE DIFFERENCE MODE.
    equation_of_state: int = IDEAL_GAS

    #: Magnetohydrodynamics switch.
    mhd: bool = False

    #: Integrator used for the magnetic part in the FV MHD scheme.
    fv_magnetic_integrator: int = IMPLICIT_MIDPOINT

    #: Density/pressure positivity-enforcement configuration (see PositivityConfig).
    positivity_config: PositivityConfig = PositivityConfig()

    #: Self-gravity / external-potential configuration (see GravityConfig).
    gravity_config: GravityConfig = GravityConfig()

    #: Explicit diffusion term 
    #: (currently only for finite difference mode)
    diffusion: bool = False

    #: Viscosity type - either kinematic or dynamic viscosity.
    viscosity_type: int = DYNAMIC_VISCOSITY

    #: Explicit thermal conduction term div(kappa grad T) in the energy
    #: equation (constant conductivity params.thermal_conductivity,
    #: explicit integration). Currently only for finite difference mode.
    thermal_conduction: bool = False

    #: The size of the simulation box.
    box_size: Union[float, StaticFloatVector] = 1.0

    #: The number of cells in the simulation.
    num_cells: Union[int, StaticIntVector] = 400

    #: The reconstruction order is the number of
    #: cells on each side of the cell of interest
    #: used to calculate the gradients for the
    #: reconstruction at the interfaces.
    reconstruction_order: int = 1

    #: The limiter for the reconstruction.
    #: Only affects finite volume mode.
    limiter: int = MINMOD

    #: The Riemann solver used
    #: Only for finite volume mode.
    riemann_solver: int = HLL

    #: Dimensional splitting / unsplit mode.
    #: Note that the UNSPLIT scheme currently
    #: interferes with energy conservation in settings
    #: with self-gravity.
    split: int = UNSPLIT

    #: Time integration method.
    time_integrator: int = RK2_SSP

    # Explanation of the ghost cells
    #                                |---------|
    #                           |---------|
    # stencil              |---------|
    # cells            || 1g | 2g | 3c | 4g | 5g ||
    # reconstructions        |L  R|L  R|L  R|    |
    # fluxes                     -->  -->
    # update                      | 3c'|
    # --> all others are ghost cells

    #: The number of ghost cells.
    num_ghost_cells: int = reconstruction_order + 1

    #: Grid spacing.
    grid_spacing: float = box_size / num_cells

    #: Explicit boundary handling mode.
    boundary_handling: int = GHOST_CELLS

    #: Boundary settings for the simulation.
    boundary_settings: Union[NoneType, BoundarySettings1D, BoundarySettings] = None

    #: Enables a fixed timestep for the simulation
    #: based on the specified number of timesteps.
    fixed_timestep: bool = False

    #: Exactly reach the end time. In adaptive timestepping,
    #: one might otherwise overshoot.
    exact_end_time: bool = True

    #: Adds the sources with the current timestep to
    #: a hypothetical state to estimate the actual timestep.
    #: Useful for time-dependent sources, but additional
    #: computational overhead.
    source_term_aware_timestep: bool = False

    #: The number of timesteps for the fixed timestep mode.
    num_timesteps: int = 1000

    #: Use a maximum timestep in adaptive timestep mode.
    use_max_adaptive_timestep: bool = True

    #: The differentiation mode one whats to use
    #: the solver in (forwards or backwards).
    differentiation_mode: int = FORWARDS

    #: The number of checkpoints used in the setup
    #: with backwards differetiability and adaptive
    #: time stepping.
    num_checkpoints: int = 100

    #: Return intermediate snapshots of the time evolution
    #: instead of only the final fluid state.
    return_snapshots: bool = False

    #: Snapshot settings
    snapshot_settings: SnapshotSettings = SnapshotSettings()

    #: Where the snapshots are stored. ``ON_DEVICE`` (default) keeps the
    #: snapshot diagnostics in preallocated device buffers and returns them
    #: at the end (the classic behaviour). ``TO_DISK`` instead streams each
    #: snapshot to disk via Orbax: the run is split into segments between the
    #: snapshot times, and the loop carry (primitive state, PRNG key, OU
    #: forcing field) plus the time is written to ``snapshot_storage_path``
    #: after each segment. Each device writes its own shard, so this scales
    #: to multiple devices / nodes. TO_DISK is forward-mode only.
    snapshot_storage_mode: int = ON_DEVICE

    #: Directory the Orbax checkpoints are written to / read from when
    #: ``snapshot_storage_mode == TO_DISK``. Required in that mode.
    snapshot_storage_path: Union[str, NoneType] = None

    #: Call a user given function on the snapshot data,
    #: e.g. for saving or plotting. Must have signature
    #: callback(time, state, registered_variables).
    activate_snapshot_callback: bool = False

    #: Return snapshots at specific time points.
    use_specific_snapshot_timepoints: bool = False

    #: The number of snapshots to return.
    num_snapshots: int = 10

    #: Fallback to the first order Godunov scheme.
    first_order_fallback: bool = False

    # physical modules

    #: Turbulent forcing configuration.
    turbulent_forcing_config: TurbulentForcingConfig = TurbulentForcingConfig()

    #: The configuration for the stellar wind module.
    wind_config: WindConfig = WindConfig()

    #: Cosmic rays
    cosmic_ray_config: CosmicRayConfig = CosmicRayConfig()

    #: The configuration for the cooling module.
    cooling_config: CoolingConfig = CoolingConfig()

    #: Frame tracking in z-direction
    #: shifting the frame to follow a
    #: turbulent radiative mixing layer
    frame_tracking: bool = False

    #: Configuration of the neural network force module.
    neural_net_force_config: NeuralNetForceConfig = NeuralNetForceConfig()

    #: Configuration of the CNN MHD corrector module.
    cnn_mhd_corrector_config: CNNMHDconfig = CNNMHDconfig()


def finalize_config(config: SimulationConfig, state_shape) -> SimulationConfig:
    """Fill in derived configuration fields and validate the configuration.

    Resolves the values that depend on the actual state shape or on
    cross-field consistency: the positivity-protection defaults, the number
    of cells per axis, the grid spacing, the geometry- and solver-specific
    overrides, the master gravity switch, the boundary defaults, and the
    disk-snapshot requirements.

    Args:
        config: The user-supplied simulation configuration.
        state_shape: The shape of the (unpadded) primitive state array, used
            to derive ``num_cells`` per axis.

    Returns:
        The finalized simulation configuration.
    """

    # ``default_positivity_protection`` is a casual on/off switch for the STATE
    # floors only: the default ``False`` is a clean slate (no per-stage /
    # per-step flooring). When set, turn the floors on (HARD_FLOOR) unless the
    # user explicitly chose a mode. The read-only clamps (clamp_in_estimates)
    # are decoupled and left untouched (default on), as are the feature toggles
    # (deepvoid_blend, preserving_flux, conservative redistribution,
    # vacuum_rest, nan_safe).
    positivity_config = config.positivity_config
    if positivity_config.default_positivity_protection:
        config = config._replace(positivity_config=positivity_config._replace(
            per_stage_mode=(POSITIVITY_HARD_FLOOR
                            if positivity_config.per_stage_mode == POSITIVITY_NONE
                            else positivity_config.per_stage_mode),
            per_step_mode=(POSITIVITY_HARD_FLOOR
                           if positivity_config.per_step_mode == POSITIVITY_NONE
                           else positivity_config.per_step_mode),
        ))

    if jax.config.jax_enable_x64:
        config._replace(numerical_precision=DOUBLE_PRECISION)
    else:
        config._replace(numerical_precision=SINGLE_PRECISION)

    # set the number of cells
    if config.dimensionality == 1:
        num_cells_x = state_shape[-1]
        config = config._replace(num_cells=StaticIntVector(num_cells_x, -1, -1))
    if config.dimensionality == 2:
        num_cells_x, num_cells_y = state_shape[-2:]
        config = config._replace(num_cells=StaticIntVector(num_cells_x, num_cells_y, -1))
    elif config.dimensionality == 3:
        num_cells_x, num_cells_y, num_cells_z = state_shape[-3:]
        config = config._replace(num_cells=StaticIntVector(num_cells_x, num_cells_y, num_cells_z))

    if isinstance(config.box_size, float):
        config = config._replace(
            box_size=StaticFloatVector(
                config.box_size,
                config.box_size,
                config.box_size
            )
        )

    # For now we assume the grid spacing is the same in all dimensions, so the
    # scalar ``grid_spacing`` is taken from the x-axis and the other axes are
    # only checked for consistency below. This restriction can be lifted once
    # the solver accepts a per-axis grid-spacing vector.
    grid_spacing_vec = config.box_size / config.num_cells

    if config.dimensionality == 1:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
    elif config.dimensionality == 2:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
        if not math.isclose(grid_spacing_vec.x, grid_spacing_vec.y):
            raise ValueError(
                "For now, we assume the grid spacing is the same in all dimensions. "
                f"Got grid spacing {grid_spacing_vec}."
            )
    elif config.dimensionality == 3:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
        if not (math.isclose(grid_spacing_vec.x, grid_spacing_vec.y) and math.isclose(grid_spacing_vec.x, grid_spacing_vec.z)):
            raise ValueError(
                "For now, we assume the grid spacing is the same in all dimensions. "
                f"Got grid spacing {grid_spacing_vec}."
            )

    if config.geometry == SPHERICAL:
        print(
            "For spherical geometry, only HLL is currently supported. Also, only the unsplit mode has been tested."
        )
        # SPHERICAL is intrinsically 1D in this code; pick the x component
        # so grid_spacing stays a scalar (otherwise CFL divisions blow up
        # because StaticFloatVector can't be divided by a scalar wave speed).
        config = config._replace(grid_spacing=(config.box_size / config.num_cells).x)

        if config.riemann_solver != HLL:
            print("Setting HLL Riemann solver for spherical geometry.")
            config = config._replace(riemann_solver=HLL)

        if config.split != SPLIT:
            print("Setting unsplit mode for spherical geometry")
            config = config._replace(split=SPLIT)

        if config.limiter == VAN_ALBADA or config.limiter == VAN_ALBADA_PP:
            print("Setting minmod limiter for spherical geometry")
            config = config._replace(limiter=MINMOD)

        if config.time_integrator != MUSCL:
            print("Setting MUSCL time integrator for spherical geometry")
            config = config._replace(time_integrator=MUSCL)

    # master gravity switch: active if self-gravity and/or an external
    # potential is used. This gates the (shared) gravity source-term machinery.
    config = config._replace(gravity_config=config.gravity_config._replace(
        gravity=config.gravity_config.self_gravity
        or config.gravity_config.external_potential
    ))

    if config.gravity_config.gravity and (config.limiter != MINMOD):
        print(
            "Curiously, in self-gravitating systems, the VAN_ALBADA limiters seem to cause crashes."
        )
        print("Setting MINMOD limiter for gravity.")
        config = config._replace(limiter=MINMOD)

    # Finite-difference-specific checks.
    if config.solver_mode == FINITE_DIFFERENCE:

        if config.dimensionality == 3 and config.boundary_settings == BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ):
            # Fully periodic boundaries are enforced more cheaply by rolling the
            # arrays (PERIODIC_ROLL) than by maintaining explicit ghost cells.
            print(
                "For 3D simulations with periodic boundaries, setting boundary handling to " \
                "PERIODIC_ROLL and num_ghost_cells to 0 for better performance."
            )
            config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)
        else:
            if config.dimensionality == 3:
                config = config._replace(boundary_handling=GHOST_CELLS, num_ghost_cells=4)

        if config.dimensionality == 2 and config.boundary_settings == BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ):
            # Fully periodic boundaries are enforced more cheaply by rolling the
            # arrays (PERIODIC_ROLL) than by maintaining explicit ghost cells.
            print(
                "For 2D simulations with periodic boundaries, setting boundary handling to " \
                "PERIODIC_ROLL and num_ghost_cells to 0 for better performance."
            )
            config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)
        else:
            if config.dimensionality == 2:
                config = config._replace(boundary_handling=GHOST_CELLS, num_ghost_cells=4)

        # The FD scheme has two supported time integrators: the SSPRK4
        # Spiteri-Ruuth 3-register scheme (default) and the Carpenter-Kennedy
        # 2N-storage LSRK4 ("RK4_LSRK") which trades CFL margin for one fewer
        # full-state buffer.  Anything else falls back to SSPRK4.
        if config.time_integrator not in (RK4_SSP, RK4_LSRK):
            print(
                "Setting time integrator to RK4_SSP for finite difference solver mode."
            )
            config = config._replace(time_integrator=RK4_SSP)

        if config.boundary_handling == PERIODIC_ROLL:
            config = config._replace(num_ghost_cells=0)

        if config.boundary_handling == GHOST_CELLS and (config.diffusion or config.thermal_conduction):
            config = config._replace(num_ghost_cells=max(config.num_ghost_cells, 6))

    # Pick sensible default boundary conditions when the user left them unset.
    if config.boundary_settings is None:
        if config.geometry == CARTESIAN:
            print("Automatically setting open boundaries for Cartesian geometry.")
            if config.dimensionality == 1:
                config = config._replace(
                    boundary_settings=BoundarySettings1D(
                        left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
                    )
                )
            else:
                config = config._replace(boundary_settings=BoundarySettings())
        elif config.geometry == SPHERICAL and config.dimensionality == 1:
            print(
                "Automatically setting reflective left and open right boundary for spherical geometry."
            )
            config = config._replace(
                boundary_settings=BoundarySettings1D(
                    left_boundary=REFLECTIVE_BOUNDARY, right_boundary=OPEN_BOUNDARY
                )
            )

    if config.wind_config.stellar_wind:
        print(
            "For stellar wind simulations, we need source term aware timesteps, turning on."
        )
        config = config._replace(source_term_aware_timestep=True)

    # Disk-snapshot (Orbax) mode requirements.
    if config.snapshot_storage_mode == TO_DISK:
        if not config.snapshot_storage_path:
            raise ValueError(
                "snapshot_storage_mode == TO_DISK requires a non-empty "
                "snapshot_storage_path (the directory the Orbax checkpoints "
                "are written to)."
            )
        if config.differentiation_mode != FORWARDS:
            raise ValueError(
                "snapshot_storage_mode == TO_DISK is forward-mode only; "
                "set differentiation_mode = FORWARDS."
            )

    return config


def riemann_solver_to_string(riemann_solver: int) -> str:
    """Return the human-readable name of a Riemann-solver constant."""
    if riemann_solver == HLL:
        return "HLL"
    elif riemann_solver == HLLC:
        return "HLLC"
    elif riemann_solver == HLLC_LM:
        return "HLLC_LM"
    elif riemann_solver == LAX_FRIEDRICHS:
        return "Lax-Friedrichs"
    elif riemann_solver == HYBRID_HLLC:
        return "Hybrid HLLC"
    elif riemann_solver == AM_HLLC:
        return "AM HLLC"


def limiter_to_string(limiter: int) -> str:
    """Return the human-readable name of a slope-limiter constant."""
    if limiter == MINMOD:
        return "Minmod"
    elif limiter == SUPERBEE:
        return "Superbee"
    elif limiter == OSHER:
        return "Osher"
    elif limiter == DOUBLE_MINMOD:
        return "Double Minmod"
    elif limiter == VAN_ALBADA:
        return "Van Albada"
    elif limiter == VAN_ALBADA_PP:
        return "Van Albada PP"


def solver_mode_to_string(solver_mode: int) -> str:
    """Return the short label (``"FV"`` / ``"FD"``) of a solver-mode constant."""
    if solver_mode == FINITE_VOLUME:
        return "FV"
    elif solver_mode == FINITE_DIFFERENCE:
        return "FD"


def config_to_string(config: SimulationConfig) -> str:
    """Return a compact one-line description of the solver configuration."""
    if config.solver_mode == FINITE_VOLUME:
        return f"FV, {riemann_solver_to_string(config.riemann_solver)}, {limiter_to_string(config.limiter)}, {config.num_cells.x} cells"
    elif config.solver_mode == FINITE_DIFFERENCE:
        return f"FD, {config.num_cells.x} cells"