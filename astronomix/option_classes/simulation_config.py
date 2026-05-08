import math
from types import NoneType
from typing import NamedTuple, Union

import jax

from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
)
from astronomix._physics_modules._cooling.cooling_options import CoolingConfig
from astronomix._physics_modules._cosmic_rays.cosmic_ray_options import CosmicRayConfig
from astronomix._physics_modules._neural_net_force._neural_net_force_options import (
    NeuralNetForceConfig,
)
from astronomix._physics_modules._stellar_wind.stellar_wind_options import WindConfig

from jaxtyping import Array, Float

from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig

# ===================== constant definition =====================

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
MIDPOINT_OPTIM = 10 # experiment
MUSCL = 1
# currently only for finite difference
RK4_SSP = 2

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

# self-gravity versions
SIMPLE_SOURCE_TERM = 0
DONOR_ACCOUNTING = 1
RIEMANN_SPLIT = 2
RIEMANN_SPLIT_UNSTABLE = 3
HALF_SPLIT = 4
FD_FLUX_GRAVITY = 5
WENO_FLUX_GRAVITY = 6

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

# ============================================================

# ===================== type definitions =====================

class StaticIntVector(NamedTuple):
    x: int = -1
    y: int = -1
    z: int = -1

class StaticFloatVector(NamedTuple):
    x: float = -1.0
    y: float = -1.0
    z: float = -1.0

    def __truediv__(self, other: StaticIntVector) -> "StaticFloatVector":
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

class SnapshotSettings(NamedTuple):
    """Settings for the snapshot output of the simulation."""

    #: Whether to return states during the simulation.
    return_states: bool = True

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
    left_boundary: int = OPEN_BOUNDARY
    right_boundary: int = OPEN_BOUNDARY

class BoundarySettings(NamedTuple):
    x: BoundarySettings1D = BoundarySettings1D()
    y: BoundarySettings1D = BoundarySettings1D()
    z: BoundarySettings1D = BoundarySettings1D()

class SimulationConfig(NamedTuple):
    """
    Configuration object for the simulation.
    The simulation configuration are parameters defining
    the simulation where changes necessitate recompilation.
    """

    # Static simulation parameters

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

    #: The equation of state for the simulation.
    #: NOTE: CURRENTLY ONLY IMPLEMENTED FOR 
    #: FINITE DIFFERENCE MODE.
    equation_of_state: int = IDEAL_GAS

    #: Magnetohydrodynamics switch.
    mhd: bool = False

    #: Integrator used for the magnetic part in the FV MHD scheme.
    fv_magnetic_integrator: int = IMPLICIT_MIDPOINT

    #: Enforce positivity of density and pressure.
    #: NOTE: CURRENTLY ONLY IMPLEMENTED FOR 
    #: FINITE DIFFERENCE MODE.
    enforce_positivity: bool = True

    #: Self gravity switch, currently only
    #: for periodic boundaries.
    self_gravity: bool = False

    #: Coupling of the self-gravity to the
    #: hydrodynamics.
    self_gravity_version: int = DONOR_ACCOUNTING

    #: Manual open boundary conditions in the
    #: Poisson solver.
    poisson_manual_open_boundaries: bool = False

    #: Explicit diffusion term 
    #: (currently only for finite difference mode)
    diffusion: bool = False

    #: Viscosity type - either kinematic or dynamic viscosity.
    viscosity_type: int = DYNAMIC_VISCOSITY

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

    #: Memory-efficient finite-difference WENO path. When True the WENO loop
    #: uses the in-place ``_eigen_{L,R}_*_from_blocks`` helpers and
    #: ``lax.fori_loop`` over modes — lower temp memory but slower. When False
    #: (the default) the loop is Python-unrolled and uses component-tuple
    #: eigen helpers + cross-mode optimization barriers — faster but ~50%
    #: more compiled temp memory at typical resolutions. Only affects
    #: ``solver_mode == FINITE_DIFFERENCE``.
    weno_low_memory: bool = False

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
    """Finalizes the simulation configuration."""

    # num_cells = state_shape[-1]
    # config = config._replace(num_cells=num_cells)

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

    # for now we assume the grid spacing is the same in all dimensions
    grid_spacing_vec = config.box_size / config.num_cells

    # as soon as we accept a grid spacing vector, 
    # this will not be necessary anymore
    # config = config._replace(grid_spacing=config.box_size / config.num_cells)
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
        config = config._replace(grid_spacing=config.box_size / config.num_cells)

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

    if config.self_gravity and (config.limiter != MINMOD):
        print(
            "Curiously, in self-gravitating systems, the VAN_ALBADA limiters seem to cause crashes."
        )
        print("Setting DOUBLE_MINMOD limiter for self-gravity.")
        config = config._replace(limiter=MINMOD)

    # finite difference specific checks
    if config.solver_mode == FINITE_DIFFERENCE:
        
        # if not config.mhd:
        #     raise ValueError(
        #         "Finite difference solver mode is currently " \
        #         "only supported for MHD simulations. This will be easy to extend, " \
        #         "feel free to contribute."
        #     )

        # if config.dimensionality != 3 and config.mhd:
        #     raise ValueError(
        #         "Finite difference solver mode in MHD mode is currently " \
        #         "only supported for 3D simulations. This will be easy to extend, " \
        #         "feel free to contribute."
        #     )
        
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
            # set boundary handling to periodic roll and num_ghost_cells to 0
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
            # set boundary handling to periodic roll and num_ghost_cells to 0
            print(
                "For 2D simulations with periodic boundaries, setting boundary handling to " \
                "PERIODIC_ROLL and num_ghost_cells to 0 for better performance."
            )
            config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)
        else:
            if config.dimensionality == 2:
                config = config._replace(boundary_handling=GHOST_CELLS, num_ghost_cells=4)

        if config.time_integrator != RK4_SSP:
            print(
                "Setting time integrator to RK4_SSP for finite difference solver mode."
            )
            config = config._replace(time_integrator=RK4_SSP)

        if config.boundary_handling == PERIODIC_ROLL:
            config = config._replace(num_ghost_cells=0)

        if config.boundary_handling == GHOST_CELLS and config.diffusion:
            config = config._replace(num_ghost_cells=max(config.num_ghost_cells, 6))

    # set boundary conditions if not set
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

    if (
        config.self_gravity
        and (config.riemann_solver == HLLC or config.riemann_solver == HLLC_LM)
        and config.riemann_solver != RIEMANN_SPLIT
    ):
        print("Consider using RIEMANN_SPLIT as the self_gravity_version.")

    return config

def riemann_solver_to_string(riemann_solver: int) -> str:
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
    if solver_mode == FINITE_VOLUME:
        return "FV"
    elif solver_mode == FINITE_DIFFERENCE:
        return "FD"
    
def config_to_string(config: SimulationConfig) -> str:
    if config.solver_mode == FINITE_VOLUME:
        return f"FV, {riemann_solver_to_string(config.riemann_solver)}, {limiter_to_string(config.limiter)}, {config.num_cells.x} cells"
    elif config.solver_mode == FINITE_DIFFERENCE:
        return f"FD, {config.num_cells.x} cells"