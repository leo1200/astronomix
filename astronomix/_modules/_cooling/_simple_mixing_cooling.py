"""
Simple mixing-layer cooling model based on Lancaster et al. (2026).

Provides a phenomenological net cooling/heating rate as a function of
temperature for a turbulent mixing layer, together with explicit and implicit
temperature-update steps and a driver that applies the update to the pressure
of the primitive state.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix._modules._cooling.cooling_options import (
    COOLING_CURVE_TYPE,
    EXPLICIT_COOLING,
    IMPLICIT_COOLING,
)
from astronomix.option_classes.simulation_config import FIELD_TYPE, STATE_TYPE

# astronomix containers
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
    MixingCoolingParams,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

def _cooling_rate(
    temperature: jnp.ndarray,
    density: jnp.ndarray,
    mixing_cooling_params: MixingCoolingParams,
    gamma: float = 5 / 3,
    beta_low: float = -2,
    beta_high: float = 3,
) -> jnp.ndarray:
    r"""
    Returns dT/dt in units where we have simplified
    T = P / rho, so T is in units of velocity^2.

    xi: float, # xi = t_sh / t_coolmin, in Fig. 3 of Lancaster et al. 2026: \in {10, 100, 1000}
    mach_number: float, # in Lancaster et al. 2026: \in {1/2, 1/8}
    chi: float = 1e2,
    """

    xi = mixing_cooling_params.xi
    mach_number = mixing_cooling_params.mach_number
    chi = mixing_cooling_params.density_contrast

    # The model is non-dimensionalised so the box size and the hot-medium
    # background density and pressure are all unity.
    L_box = 1.0
    rho0 = 1.0
    P0 = 1.0

    # With the simplification T = P / rho (so T carries units of velocity^2,
    # and T = (gamma - 1) * e), the local pressure follows the temperature.
    pressure = temperature * density

    # Adiabatic sound speed in the hot medium and the shear velocity set by the
    # Mach number M = v_rel / c_s_hot.
    c_s_hot = jnp.sqrt(gamma * P0 / rho0)
    v_rel = mach_number * c_s_hot

    # Shear time and minimum cooling time; xi = t_sh / t_coolmin controls how
    # strongly the layer cools relative to the shear timescale.
    t_sh = L_box / v_rel
    t_coolmin = t_sh / xi

    T_hot = P0 / rho0
    T_cold = T_hot / chi

    # Temperature at which the cooling rate peaks.
    T_pk = (T_cold ** 2 * T_hot) ** (1/3)

    # Peak cooling rate.
    edot_max = P0 / (gamma - 1) / t_coolmin * (pressure / P0) ** 2

    # Broken power-law slope of the cooling rate either side of the peak.
    beta = jnp.where(
        temperature < T_pk,
        beta_low,
        beta_high
    )

    # Cooling rate as a function of temperature.
    edot_cooling = edot_max * (temperature / T_pk) ** (-beta)

    # Heating rate as a function of temperature, with a smooth roll-off above
    # the hot-medium temperature.
    T_lim = 1.05 * T_hot
    c_heat = (T_cold / T_pk) ** ((beta_high - beta_low) * (1 + jnp.log(T_cold / T_pk) / jnp.log(chi)))
    alpha_heat = (beta_low - beta_high) * (jnp.log(T_cold / T_pk) / jnp.log(chi)) - beta_high
    heating_shape = jnp.where(
        temperature < T_lim,
        (temperature / T_pk) ** alpha_heat,
        (T_lim / T_pk) ** alpha_heat * (temperature / T_lim) ** (-beta_high - 0.5)
    )
    edot_heating = c_heat * edot_max * heating_shape

    # Net (heating minus cooling) energy rate, converted back to dT/dt.
    edot_net = edot_heating - edot_cooling

    return (gamma - 1) * edot_net / density

@partial(jax.jit, static_argnames=("cooling_curve_config",))
def update_temperature_explicit(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    time_step: float,
    gamma: float,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    """Advance the temperature one explicit (forward-Euler) cooling step.

    Args:
        density: The density field.
        temperature: The current temperature field.
        time_step: The time step.
        gamma: The adiabatic index.
        cooling_curve_config: The static cooling-curve configuration.
        cooling_curve_params: The cooling-curve parameters.

    Returns:
        The temperature field after one explicit cooling step.
    """
    return (
        temperature
        + _cooling_rate(
            temperature,
            density,
            cooling_curve_params,
            gamma,
        )
        * time_step
    )

@partial(jax.jit, static_argnames=("cooling_curve_config",))
def update_temperature_implicit(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    time_step: float,
    gamma: float,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    """Advance the temperature one implicit (backward-Euler) cooling step.

    The implicit relation is solved with a simple fixed-point iteration, which
    is robust enough for the smooth mixing-layer cooling rate.

    Args:
        density: The density field.
        temperature: The current temperature field.
        time_step: The time step.
        gamma: The adiabatic index.
        cooling_curve_config: The static cooling-curve configuration.
        cooling_curve_params: The cooling-curve parameters.

    Returns:
        The temperature field after one implicit cooling step.
    """

    def implicit_eq(T_new):
        return (temperature
        + _cooling_rate(
            T_new,
            density,
            cooling_curve_params,
            gamma,
        ) * time_step)

    # A simple fixed-point iteration suffices here; a Newton or bisection
    # solver could be substituted later if convergence ever becomes an issue.
    max_iter = 50
    tol = 1e-6

    def cond_fun(state):
        i, T_old = state
        T_candidate = implicit_eq(T_old)
        diff = jnp.max(jnp.abs(T_candidate - T_old))
        return (i < max_iter) & (diff > tol)

    def body_fun(state):
        i, T_old = state
        T_new = implicit_eq(T_old)
        return (i + 1, T_new)

    state = (0, temperature)
    _, T_final = jax.lax.while_loop(cond_fun, body_fun, state)
    return T_final


@partial(jax.jit, static_argnames=("cooling_config", "registered_variables"))
def update_pressure_by_cooling_mixing(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
    cooling_config: CoolingConfig,
    simulation_params: SimulationParams,
    time_step: float,
) -> STATE_TYPE:
    """Apply mixing-layer cooling to the pressure of the primitive state.

    Args:
        primitive_state: The primitive state array.
        registered_variables: The registered variables.
        cooling_config: The cooling configuration (selects explicit/implicit).
        simulation_params: The simulation parameters.
        time_step: The time step.

    Returns:
        The primitive state with the pressure updated by cooling.
    """

    # This model uses the simplification T = P / rho throughout.
    cooling_curve_config = cooling_config.cooling_curve_config

    # get the parameters
    cooling_params = simulation_params.cooling_params
    gamma = simulation_params.gamma

    # get the density and pressure
    density = primitive_state[registered_variables.density_index]
    pressure = primitive_state[registered_variables.pressure_index]

    # get the temperature
    temperature = pressure / density

    if cooling_config.cooling_method == IMPLICIT_COOLING:
        new_temperature = update_temperature_implicit(
            density,
            temperature,
            time_step,
            gamma,
            cooling_curve_config,
            cooling_params.cooling_curve_params,
        )
    elif cooling_config.cooling_method == EXPLICIT_COOLING:
        new_temperature = update_temperature_explicit(
            density,
            temperature,
            time_step,
            gamma,
            cooling_curve_config,
            cooling_params.cooling_curve_params,
        )

    new_temperature = jnp.where(
        (new_temperature > cooling_params.floor_temperature),
        new_temperature,
        temperature,
    )

    # update the pressure
    new_pressure = new_temperature * density

    # set the new pressure
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(
        new_pressure
    )

    # return the updated primitive state
    return primitive_state