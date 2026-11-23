"""
Hook 1 of the physics modules: the explicit sources and diffusive fluxes, inside the stages of every scheme.

``explicit_sources`` returns the increment of the conserved state over a (stage) time step that the active
modules add: the stellar wind (the 3D injection of mass and thermal energy), cooling, gravity (every scheme but
VL2, which keeps the coupling of Athena++), viscosity, thermal conduction and the energy term of resistivity. The
finite-difference Runge-Kutta stages add it to their right-hand side; the finite-volume stages add it to the
conserved state after the flux update (``_finite_volume/_integrators.py``), also after a fused kernel, which does
not know the modules. Its counterpart, once per step, is ``_per_step_updates.py``.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    STATE_TYPE,
    VL2,
)
from astronomix._modules._cooling.cooling_options import SIMPLE_MIXING_LAYER_COOLING

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._equation_of_state import is_barotropic
from astronomix._fluid_equations._equations_mhd import (
    conserved_state_from_primitive_mhd,
    primitive_state_from_conserved_isothermal,
    primitive_state_from_conserved_mhd,
)
from astronomix._fluid_equations._equations import (
    conserved_state_from_primitive,
    primitive_state_from_conserved,
)
from astronomix._modules._conduction._conduction import conduction_source
from astronomix._modules._cooling._cooling import update_pressure_by_cooling
from astronomix._modules._cooling._simple_mixing_cooling import update_pressure_by_cooling_mixing
from astronomix._modules._gravity._gravity import _fd_gravity_source
from astronomix._modules._resistivity._resistivity import (
    resistive_source,
    resistive_source_active,
)
from astronomix._modules._stellar_wind.stellar_wind import _wind_ei3D_source
from astronomix._modules._viscosity._viscosity import viscosity_source


def wind_source_active(config: SimulationConfig) -> bool:
    """Whether the stellar wind enters as a source of the stages (3D); the spherical 1D injection runs once per
    step (``_per_step_updates.py``)."""
    return config.wind_config.stellar_wind and config.dimensionality == 3


def gravity_source_active(config: SimulationConfig) -> bool:
    """Whether gravity enters through this hook: every scheme but VL2, which couples it inside its stages as
    Athena++ does (``_finite_volume/_gravity_coupling.py``)."""
    uses_vl2 = config.solver_mode == FINITE_VOLUME and config.solver_config.time_integrator == VL2
    return config.gravity_config.gravity and not uses_vl2


def has_explicit_sources(config: SimulationConfig) -> bool:
    """Whether any module adds to ``explicit_sources`` (a static property of the configuration)."""
    return (
        wind_source_active(config)
        or config.cooling_config.cooling
        or gravity_source_active(config)
        or config.diffusion_config.viscosity
        or config.diffusion_config.thermal_conduction
        or resistive_source_active(config)
    )


def _primitive_from_conserved(conserved_state, gamma, config, params, registered_variables):
    """The primitive state the modules read, from the conserved state of a finite-difference stage."""
    if config.mhd:
        if is_barotropic(config):
            # No pressure slot: the general conversion would write a pressure into the last row.
            return primitive_state_from_conserved_isothermal(
                conserved_state, params.minimum_density, config, registered_variables, clamp=False
            )
        return primitive_state_from_conserved_mhd(
            conserved_state,
            params.minimum_density,
            params.minimum_pressure,
            gamma,
            config,
            registered_variables,
        )
    return primitive_state_from_conserved(conserved_state, gamma, config, registered_variables)


def _conserved_from_primitive(primitive_state, gamma, config, registered_variables):
    """The conserved state of a primitive state (an ideal gas; used for the cooling increment)."""
    if config.mhd:
        return conserved_state_from_primitive_mhd(primitive_state, gamma, registered_variables)
    return conserved_state_from_primitive(primitive_state, gamma, config, registered_variables)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def explicit_sources(
    conserved_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    density_fluxes=None,
    density_increment=None,
    primitive_state=None,
    sink_potential=None,
) -> STATE_TYPE:
    """
    The increment of the conserved state over ``dt`` from the explicit sources and diffusive fluxes of the
    active modules, evaluated at the given state.

    Args:
        conserved_state: The conserved state the increment is evaluated at.
        dt: The (stage) time step the increment is taken over.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data (the radii of the wind source).
        registered_variables: The registered variables.
        density_fluxes: The density fluxes of the stage per axis (the conservative gravity couplings), or None.
        density_increment: The density increment of the stage (the conservative gravity couplings), or None.
        primitive_state: The primitive form of ``conserved_state`` if the caller holds it (the finite-volume
            stages); otherwise it is converted here.
        sink_potential: The gravitational potential of the sink particles, which acts on the gas together
            with its own gravity, or None.

    Returns:
        The increment, laid out like the conserved state.
    """
    source_term = jnp.zeros_like(conserved_state)

    reference_conserved_state = conserved_state
    if primitive_state is None:
        primitive_state = _primitive_from_conserved(conserved_state, gamma, config, params, registered_variables)
    elif config.cooling_config.cooling:
        # The cooling increment is a difference of two conversions: both from primitives, so that a cell the
        # cooling leaves alone receives exactly nothing.
        reference_conserved_state = _conserved_from_primitive(primitive_state, gamma, config, registered_variables)

    # Stellar wind: mass and thermal energy injected around the centre of the box.
    if wind_source_active(config):
        source_term += _wind_ei3D_source(
            params.wind_params,
            conserved_state,
            dt,
            config,
            helper_data,
            config.wind_config.num_injection_cells,
            registered_variables,
        )

    # Cooling: the change of the conserved state when the pressure is cooled over dt. The grid spacing enables
    # the cooling-resolution limiter.
    if config.cooling_config.cooling:
        if config.cooling_config.cooling_curve_config.cooling_curve_type != SIMPLE_MIXING_LAYER_COOLING:
            cooled_primitive_state = update_pressure_by_cooling(
                primitive_state,
                registered_variables,
                config.cooling_config,
                params,
                dt,
                grid_spacing=config.grid_spacing,
            )
        else:
            cooled_primitive_state = update_pressure_by_cooling_mixing(
                primitive_state,
                registered_variables,
                config.cooling_config,
                params,
                dt,
            )
        cooled_conserved_state = _conserved_from_primitive(
            cooled_primitive_state, gamma, config, registered_variables
        )
        source_term += cooled_conserved_state - reference_conserved_state
        # The modules below read the cooled state.
        primitive_state = cooled_primitive_state

    # Gravity (the potential is solved in every stage).
    if gravity_source_active(config):
        source_term += _fd_gravity_source(
            primitive_state,
            density_fluxes,
            density_increment,
            dt,
            config,
            params,
            registered_variables,
            sink_potential=sink_potential,
        )

    if config.diffusion_config.viscosity:
        source_term += viscosity_source(primitive_state, params, config, registered_variables) * dt

    # Thermal conduction: div(kappa grad T) in the energy equation.
    if config.diffusion_config.thermal_conduction:
        source_term += conduction_source(primitive_state, params, config, registered_variables) * dt

    # Resistivity: the energy the resistive electric field moves (an ideal gas), and its curl on a cell-centred
    # field (the constrained transport has it in its electromotive force).
    if resistive_source_active(config):
        source_term += resistive_source(primitive_state, params, config, registered_variables) * dt

    return source_term
