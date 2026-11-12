"""
The equation of state: how the gas pressure follows from the state.

``IDEAL_GAS`` evolves the energy and stores the pressure as a state variable.
The other two are barotropic, the pressure is a function of the density alone:
``ISOTHERMAL`` (p = c_s^2 rho) and ``POLYTROPIC`` (p = K rho^Gamma). A
barotropic gas has no energy equation and its state has no pressure slot.

The functions are elementwise on their operands and choose their branch at
Python time, so they serve whole arrays and face values alike.
"""

# astronomix constants
from astronomix.option_classes.simulation_config import (
    IDEAL_GAS,
    ISOTHERMAL,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables


def is_barotropic(config: SimulationConfig) -> bool:
    """Return whether the pressure is a function of the density alone (no energy equation, no pressure slot)."""
    return config.equation_of_state != IDEAL_GAS


def barotropic_pressure(density, params: SimulationParams, config: SimulationConfig):
    """
    The pressure of a barotropic gas.

    Args:
        density: The density.
        params: The simulation parameters.
        config: The simulation configuration.

    Returns:
        ``c_s^2 rho`` (isothermal) or ``K rho^Gamma`` (polytropic).
    """
    if config.equation_of_state == ISOTHERMAL:
        return params.isothermal_sound_speed**2 * density
    return params.polytropic_constant * density**params.polytropic_exponent


def sound_speed_squared(density, pressure, params: SimulationParams, config: SimulationConfig):
    """
    The squared sound speed, dp / drho at constant entropy.

    Args:
        density: The density.
        pressure: The pressure (read for the ideal gas only; ``None`` is fine for a barotropic gas).
        params: The simulation parameters.
        config: The simulation configuration.

    Returns:
        ``gamma p / rho`` (ideal gas), the constant ``c_s^2`` (isothermal, a
        scalar) or ``Gamma K rho^(Gamma - 1)`` (polytropic).
    """
    if config.equation_of_state == IDEAL_GAS:
        return params.gamma * pressure / density
    if config.equation_of_state == ISOTHERMAL:
        return params.isothermal_sound_speed**2
    return params.polytropic_exponent * params.polytropic_constant * density ** (params.polytropic_exponent - 1.0)


def pressure_of(
    primitive_state: STATE_TYPE,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The gas pressure of a primitive state: its pressure slot, or the pressure
    the barotropic law gives for its density.

    Args:
        primitive_state: The primitive state.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The gas pressure.
    """
    if is_barotropic(config):
        return barotropic_pressure(primitive_state[registered_variables.density_index], params, config)
    return primitive_state[registered_variables.pressure_index]
