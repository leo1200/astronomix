"""
Runtime simulation parameters.

Unlike the simulation configuration, these parameters can be changed without
triggering a recompilation, and the simulation can be differentiated with
respect to them (CFL number, gas constants, viscosity, end time, ...). The
per-module parameter containers are bundled in here too.
"""

# typing
from typing import NamedTuple

# jax
import jax.numpy as jnp

# astronomix containers
from astronomix._modules._cnn_mhd_corrector._cnn_mhd_corrector_options import CNNMHDconfig
from astronomix._modules._cooling.cooling_options import CoolingParams
from astronomix._modules._cosmic_rays.cosmic_ray_options import CosmicRayParams
from astronomix._modules._neural_net_force._neural_net_force_options import NeuralNetForceParams
from astronomix._modules._stellar_wind.stellar_wind_options import WindParams
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingParams


class FixedBoundaryState1D(NamedTuple):
    """The prescribed left/right states for a single axis with FIXED_BOUNDARY."""

    #: Left state of shape (num_variables,).
    left_state: jnp.ndarray = jnp.array([])
    #: Right state of shape (num_variables,).
    right_state: jnp.ndarray = jnp.array([])


class FixedBoundaryState(NamedTuple):
    """Per-axis fixed boundary states used when a boundary is FIXED_BOUNDARY."""

    x: FixedBoundaryState1D = FixedBoundaryState1D()
    y: FixedBoundaryState1D = FixedBoundaryState1D()
    z: FixedBoundaryState1D = FixedBoundaryState1D()

class SimulationParams(NamedTuple):
    """
    Different from the simulation configuration, the simulation parameters
    do not require recompilation when changed. The simulation can be 
    differentiated with respect to them.
    """

    #: The Courant-Friedrichs-Lewy number, a factor
    #: in the time step calculation.
    C_cfl: float = 0.4

    #: Gravitational constant.
    gravitational_constant: float = 1.0

    #: External, static gravitational potential evaluated at the cell
    #: centers of the grid (without ghost cells). Only used when
    #: config.gravity_config.external_potential is True; it is added on top of the
    #: self-gravity potential in _compute_total_potential and padded to the
    #: ghost-cell-extended shape of a state field internally.
    gravitational_potential: jnp.array = jnp.array([])

    #: Dynamic or kinematic viscosity depending
    #: on the viscosity_type in SimulationConfig.
    viscosity: float = 0.0

    #: Constant thermal conductivity kappa in the conductive energy
    #: source div(kappa grad T) (config.thermal_conduction). T is taken
    #: from the ideal-gas relation T = p / rho (code units, R = 1).
    #: NOTE: CURRENTLY ONLY IMPLEMENTED FOR FINITE DIFFERENCE MODE.
    thermal_conductivity: float = 0.0

    #: The isothermal sound speed used when
    #: config.equation_of_state is ISOTHERMAL.
    #: NOTE: CURRENTLY ONLY IMPLEMENTED FOR 
    #: FINITE DIFFERENCE MODE.
    isothermal_sound_speed: float = 1.0

    #: The adiabatic index of the gas.
    gamma: float = 5/3

    #: Minimum allowed density.
    #: NOTE: CURRENTLY ONLY USED IN 
    #: FINITE DIFFERENCE MODE IF
    #: positivity protection is active.
    minimum_density: float = 1e-14

    #: Minimum allowed pressure.
    #: NOTE: CURRENTLY ONLY USED IN 
    #: FINITE DIFFERENCE MODE IF
    #: positivity protection is active.
    minimum_pressure: float = 1e-14

    #: Velocity ceiling applied to cells fixed by the REDISTRIBUTE positivity
    #: mode (mirrors HOW-MHD ``velpmx1``). Only used when a positivity mode is
    #: ``POSITIVITY_REDISTRIBUTE``.
    positivity_max_velocity: float = 50.0

    #: The maximum time step.
    dt_max: float = jnp.inf

    #: The initial (clock) time of the simulation. The time loop starts
    #: integrating from here and the snapshot grid spans [t_start, t_end].
    #: Defaults to 0.0; set to a checkpoint's time to resume a run (see
    #: astronomix.setup_helpers.restart_from_latest_checkpoint).
    t_start: float = 0.0

    #: The final time of the simulation.
    t_end: float = 0.2

    #: Snapshot timepoints
    snapshot_timepoints: jnp.array = jnp.array([0.0])

    #: The fixed boundary state if the boundary type
    #: of the specific boundary is set to FIXED_BOUNDARY.
    fixed_boundary_state: FixedBoundaryState = FixedBoundaryState()

    # parameters of physics modules

    #: The parameters of the turbulent forcing module.
    turbulent_forcing_params: TurbulentForcingParams = TurbulentForcingParams()

    #: The parameters of the stellar wind module.
    wind_params: WindParams = WindParams()

    #: Cosmic ray parameters
    cosmic_ray_params: CosmicRayParams = CosmicRayParams()

    #: The parameters of the cooling module.
    cooling_params: CoolingParams = CoolingParams()

    #: The parameters of the neural network force module.
    neural_net_force_params: NeuralNetForceParams = NeuralNetForceParams()

    #: The parameters of the CNN MHD corrector module.
    cnn_mhd_corrector_params: CNNMHDconfig = CNNMHDconfig()