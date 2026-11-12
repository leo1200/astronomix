"""
Runtime simulation parameters.

Unlike the simulation configuration, these parameters can be changed without
triggering a recompilation, and the simulation can be differentiated with
respect to them (CFL number, gas constants, viscosity, end time, ...). The
per-module parameter containers are bundled in here too.
"""

# typing
from typing import Any, NamedTuple

# jax
import jax.numpy as jnp

# astronomix containers
from astronomix._modules._cooling.cooling_options import CoolingParams
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
    #: on the viscosity_type in DiffusionConfig.
    viscosity: float = 0.0

    #: Constant ohmic resistivity eta, the coefficient of the switch of the
    #: same name, ``config.diffusion_config.resistivity``, which must be set for it to act.
    #: Only implemented for 3D finite-difference CT MHD with the isothermal
    #: EOS (no ohmic heating term).
    resistivity: float = 0.0

    #: Constant thermal conductivity kappa in the conductive energy
    #: source div(kappa grad T) (config.diffusion_config.thermal_conduction); with
    #: ``config.diffusion_config.conduction_density_weighted`` it is the diffusivity alpha of
    #: ``kappa = rho alpha`` instead. T is taken from the ideal-gas relation
    #: T = p / rho (code units, R = 1).
    #: NOTE: CURRENTLY ONLY IMPLEMENTED FOR FINITE DIFFERENCE MODE.
    thermal_conductivity: float = 0.0

    #: The isothermal sound speed used when
    #: config.equation_of_state is ISOTHERMAL.
    isothermal_sound_speed: float = 1.0

    #: The constant K and the exponent Gamma of p = K rho^Gamma, used when
    #: config.equation_of_state is POLYTROPIC.
    polytropic_constant: float = 1.0
    polytropic_exponent: float = 1.0

    #: The adiabatic index of the gas.
    gamma: float = 5/3

    #: Minimum density, the density floor of both solvers: the target of the
    #: positivity-preserving WENO scalings (capped at half the state being
    #: scaled), the per-step HARD_FLOOR (which in the VL2 finite-volume scheme
    #: also switches on AthenaPK's floors in every stage), the finite-difference
    #: MHD primitive recovery and the read-only clamps of the flux, wave-speed
    #: and time-step estimates.
    minimum_density: float = 1e-14

    #: Minimum pressure, in the same roles as ``minimum_density``.
    minimum_pressure: float = 1e-14

    #: Alfven-speed ceiling of the per-step HARD_FLOOR (MHD only): where the
    #: Alfven speed |B| / sqrt(rho) would exceed it, the density is raised to
    #: |B|^2 / maximum_alfven_speed^2, momentum and total energy kept (see
    #: ``_apply_density_floor``). ``inf`` (the default) switches it off.
    maximum_alfven_speed: float = float("inf")

    #: Ratio of the diffusive to the advective time scale of the GLM divergence
    #: cleaning in the VL2 finite-volume MHD scheme (AthenaPK ``glmmhd_alpha``,
    #: Mignone & Tzeferacos 2010 eq. 27): psi is damped by
    #: ``exp(-glm_alpha * c_h * dt / dx)`` every stage.
    glm_alpha: float = 0.1

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

    #: The parameters of the cooling module.
    cooling_params: CoolingParams = CoolingParams()

    #: Parameters of the user-supplied ``config.per_step_update`` (any
    #: pytree, for example network weights). The simulation can be
    #: differentiated with respect to them.
    per_step_update_params: Any = None