# TOWNSEND SCHEME DOES NOT WORK CURRENTLY,
# ONLY THE SIMPLE EXPLICIT COOLING IS USED

# For proper cooling something like Grackle
# might be used. For now we are interested in
# the most simple cooling model.

# dE/dt + ... = Phi(T, rho)
# Phi(T, rho) = n_H * Gamma(T) - n_H ^ 2 * Lambda(T)

# for a simple cooling term (Lambda) see 5.3 in
# https://arxiv.org/pdf/2111.03399

# also see
# https://academic.oup.com/mnras/article/502/3/3179/6081066

# and
# https://iopscience.iop.org/article/10.1088/0067-0049/181/2/391

from typing import Tuple

from functools import partial

import jax

import equinox as eqx

from astronomix._physics_modules._cooling.cooling_options import (
    COOLING_CURVE_TYPE,
    EXPLICIT_COOLING,
    IMPLICIT_COOLING,
    NEURAL_NET_COOLING,
    NEURAL_NET_COOLING_WITH_DENSITY,
    PIECEWISE_POWER_LAW,
    SIMPLE_POWER_LAW,
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
)
from astronomix._finite_volume._state_evolution.limited_gradients import _calculate_limited_gradients
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import FIELD_TYPE, STATE_TYPE, SimulationConfig

import jax.numpy as jnp

from astronomix.option_classes.simulation_params import SimulationParams

# NOTE the rescaled units
# \tilde{T} = T * k_B / u
# \tilde{\Lambda} = \lambda / u^2


def get_effective_molecular_weights(
    hydrogen_mass_fraction: float,  # X
    metal_mass_fraction: float,  # Z
) -> Tuple[float, float, float]:
    """
    Calculate the mean molecular weight (mu)
    and the effective molecular weights for
    electrons (mu_e), hydrogen (mu_H)
    """

    # mean molecular weight
    mu = 1.0 / (
        2 * hydrogen_mass_fraction
        + 3 * (1 - hydrogen_mass_fraction - metal_mass_fraction) / 4
        + metal_mass_fraction / 2
    )

    # effective molecular weight for electrons
    mu_e = 2 * 1.0 / (1 + hydrogen_mass_fraction)

    # effective molecular weight for hydrogen
    mu_H = 1.0 / hydrogen_mass_fraction

    return mu, mu_e, mu_H


def get_particle_number_density(
    density: FIELD_TYPE, mean_molecular_weight: float
) -> FIELD_TYPE:
    return density / mean_molecular_weight


def get_pressure_from_temperature(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
) -> FIELD_TYPE:
    """
    P = n * \tilde{T}
    """

    # calculate the effective molecular weights
    mu, _, _ = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # calculate the particle number density
    n = get_particle_number_density(density, mu)

    # calculate the pressure
    return n * temperature


def get_temperature_from_pressure(
    density: FIELD_TYPE,
    pressure: FIELD_TYPE,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
) -> FIELD_TYPE:
    """
    \tilde{T} = P / \tilde{n}
    """

    # calculate the effective molecular weights
    mu, _, _ = get_effective_molecular_weights(
        hydrogen_mass_fraction, metal_mass_fraction
    )

    # calculate the particle number density
    n = get_particle_number_density(density, mu)

    # calculate the temperature
    return pressure / n  # so the density must never be zero


# \Lambda(T)
def cooling_rate_power_law(
    temperature: FIELD_TYPE,
    reference_temperature: float,
    factor: float,
    exponent: float,
):
    return factor * (temperature / reference_temperature) ** exponent


# t_cool
@partial(jax.jit, static_argnames=("cooling_curve_config",))
def cooling_time(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    gamma: float,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    """
    t_cool = (k * mu_e * mu_H * T) / ((gamma - 1) * rho * mu * Lambda(T))
    """

    # calculate the effective molecular weights
    mu, mu_e, mu_H = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # calculate the cooling rate
    cooling_rate = _cooling_rate(
        temperature,
        density,
        cooling_curve_config,
        cooling_curve_params,
    )

    # calculate the cooling time
    return (mu_e * mu_H * temperature) / ((gamma - 1) * density * mu * cooling_rate)


# Y(T)
def power_law_temporal_evolution_function(
    temperature: FIELD_TYPE,  # T
    reference_temperature: float,  # T_ref
    exponent: float,  # alpha
) -> FIELD_TYPE:
    """
    1/(1 - alpha) * (1 - (T/T_ref)^(1-alpha)) for alpha != 1
    -log(T/T_ref) for alpha = 1
    """
    return jax.lax.cond(
        exponent != 1,
        lambda: (1 / (1 - exponent))
        * (1 - (temperature / reference_temperature) ** (1 - exponent)),
        lambda: -jnp.log(temperature / reference_temperature),
    )


# Y^-1(Y)
def power_law_temporal_evolution_function_inverse(
    temporal_evolution_function: FIELD_TYPE,  # Y
    reference_temperature: float,  # T_ref
    exponent: float,  # alpha
) -> FIELD_TYPE:
    """
    T_ref * (1 - (1 - alpha) * Y)^(1/(1-alpha)) for alpha != 1
    T_ref * exp(-Y) for alpha = 1
    """
    return jax.lax.cond(
        exponent != 1,
        lambda: reference_temperature
        * (1 - (1 - exponent) * temporal_evolution_function) ** (1 / (1 - exponent)),
        lambda: reference_temperature * jnp.exp(-temporal_evolution_function),
    )


# piecewise power law
@partial(
    jnp.vectorize,
    excluded=(1, 2, 3),  # don’t vectorize over tables
    signature="()->()",  # scalar in, scalar out
)
def _evaluate_piecewise_power_law(
    T_in,
    T_table,
    Lambda_table,
    alpha_table,
):
    def eval_in_range(T_in):
        k = jnp.searchsorted(T_table, T_in) - 1

        # clip k to be in the valid range
        k = jnp.clip(k, 0, len(T_table) - 2)

        alpha_k = alpha_table[k]
        T_k = T_table[k]
        Lambda_k = Lambda_table[k]
        return Lambda_k * (T_in / T_k) ** alpha_k

    return jax.lax.cond(
        # check if T_in is in the table range
        (T_in >= T_table[0]) & (T_in <= T_table[-1]),
        eval_in_range,
        lambda _: 0.0,  # return 0 if out of range
        T_in,
    )


@partial(
    jnp.vectorize,
    excluded=(1, 2, 3, 4),  # don’t vectorize over tables
    signature="()->()",  # scalar in, scalar out
)
def _piecewise_power_law_temporal_evolution_function(
    T_in, T_table, Lambda_table, alpha_table, Y_table
):
    def eval_in_range(T_in):
        k = jnp.searchsorted(T_table, T_in) - 1

        # clip k to be in the valid range
        k = jnp.clip(k, 0, len(T_table) - 2)

        alpha_k = alpha_table[k]
        T_k = T_table[k]
        Lambda_k = Lambda_table[k]
        Y_k = Y_table[k]
        return Y_k + jax.lax.cond(
            alpha_k != 1.0,
            lambda: 1
            / (1 - alpha_k)
            * Lambda_table[-1]
            / Lambda_k
            * T_k
            / T_table[-1]
            * (1 - (T_k / T_in) ** (alpha_k - 1)),
            lambda: Lambda_table[-1]
            / Lambda_k
            * T_k
            / T_table[-1]
            * jnp.log(T_k / T_in),
        )

    return jax.lax.cond(
        # check if T_in is in the table range
        (T_in >= T_table[0]) & (T_in <= T_table[-1]),
        eval_in_range,
        lambda _: 0.0,  # return 0 if out of range
        T_in,
    )


@partial(
    jnp.vectorize,
    excluded=(1, 2, 3, 4),  # don’t vectorize over tables
    signature="()->()",  # scalar in, scalar out
)
def _piecewise_power_law_temporal_evolution_function_inverse(
    Y_in, T_table, Lambda_table, alpha_table, Y_table
):
    def eval_in_range(Y_in):
        # k such that Y_k >= Y >= Y_{k+1}
        k = jnp.searchsorted(-Y_table, -Y_in) - 1

        # clip k to be in the valid range
        k = jnp.clip(k, 0, len(Y_table) - 2)

        alpha_k = alpha_table[k]
        T_k = T_table[k]
        Lambda_k = Lambda_table[k]
        Y_k = Y_table[k]
        return jax.lax.cond(
            alpha_k != 1.0,
            lambda: T_k
            * (
                1
                - (1 - alpha_k)
                * (Y_in - Y_k)
                * Lambda_k
                / Lambda_table[-1]
                * T_table[-1]
                / T_k
            )
            ** (1 / (1 - alpha_k)),
            lambda: T_k
            * jnp.exp(-(Y_in - Y_k) * Lambda_k / Lambda_table[-1] * T_table[-1] / T_k),
        )

    return jax.lax.cond(
        # check if Y_in is in the table range,
        # Y_table is monotonically decreasing
        (Y_in >= Y_table[-1]) & (Y_in <= Y_table[0]),
        eval_in_range,
        lambda _: jnp.where(Y_in < Y_table[-1], T_table[-1], T_table[0]),
        Y_in,
    )


@partial(jax.jit, static_argnames=("cooling_curve_config",))
def _cooling_rate(
    temperature: FIELD_TYPE,
    density: FIELD_TYPE,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    if cooling_curve_config.cooling_curve_type == SIMPLE_POWER_LAW:
        return cooling_rate_power_law(
            temperature,
            cooling_curve_params.reference_temperature,
            cooling_curve_params.factor,
            cooling_curve_params.exponent,
        )
    elif cooling_curve_config.cooling_curve_type == PIECEWISE_POWER_LAW:
        return _evaluate_piecewise_power_law(
            temperature,
            10**cooling_curve_params.log10_T_table,
            10**cooling_curve_params.log10_Lambda_table,
            cooling_curve_params.alpha_table,
        )
    elif cooling_curve_config.cooling_curve_type == NEURAL_NET_COOLING:
        neural_net_params = cooling_curve_params.network_params
        neural_net_static = cooling_curve_config.cooling_net_config.network_static
        model = jax.vmap(eqx.combine(neural_net_params, neural_net_static))

        # for now we train the network in the specific code units,
        # so no appropriate rescaling here, to be changed later
        return 10 ** model(jnp.log10(temperature).reshape(-1, 1)).flatten()
    elif cooling_curve_config.cooling_curve_type == NEURAL_NET_COOLING_WITH_DENSITY:
        neural_net_params = cooling_curve_params.network_params
        neural_net_static = cooling_curve_config.cooling_net_config.network_static
        model = jax.vmap(eqx.combine(neural_net_params, neural_net_static))

        # for now we train the network in the specific code units,
        # so no appropriate rescaling here, to be changed later
        input_data = jnp.stack([jnp.log10(temperature), jnp.log10(density)], axis=-1)
        return 10 ** model(input_data).flatten()

    else:
        raise ValueError(
            f"Unknown cooling curve type: {cooling_curve_config.cooling_curve_type}"
        )


@partial(jax.jit, static_argnames=("cooling_curve_config",))
def _temporal_evolution_function(
    temperature: FIELD_TYPE,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    if cooling_curve_config.cooling_curve_type == SIMPLE_POWER_LAW:
        return power_law_temporal_evolution_function(
            temperature,
            cooling_curve_params.reference_temperature,
            cooling_curve_params.exponent,
        )
    elif cooling_curve_config.cooling_curve_type == PIECEWISE_POWER_LAW:
        return _piecewise_power_law_temporal_evolution_function(
            temperature,
            10**cooling_curve_params.log10_T_table,
            10**cooling_curve_params.log10_Lambda_table,
            cooling_curve_params.alpha_table,
            cooling_curve_params.Y_table,
        )
    else:
        raise ValueError(
            f"Unknown cooling curve type: {cooling_curve_config.cooling_curve_type}"
        )


@partial(jax.jit, static_argnames=("cooling_curve_config",))
def _temporal_evolution_function_inverse(
    temporal_evolution_function: FIELD_TYPE,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    if cooling_curve_config.cooling_curve_type == SIMPLE_POWER_LAW:
        return power_law_temporal_evolution_function_inverse(
            temporal_evolution_function,
            cooling_curve_params.reference_temperature,
            cooling_curve_params.exponent,
        )
    elif cooling_curve_config.cooling_curve_type == PIECEWISE_POWER_LAW:
        return _piecewise_power_law_temporal_evolution_function_inverse(
            temporal_evolution_function,
            10**cooling_curve_params.log10_T_table,
            10**cooling_curve_params.log10_Lambda_table,
            cooling_curve_params.alpha_table,
            cooling_curve_params.Y_table,
        )
    else:
        raise ValueError(
            f"Unknown cooling curve type: {cooling_curve_config.cooling_curve_type}"
        )


@partial(jax.jit, static_argnames=("cooling_curve_config",))
def update_temperature(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    time_step: float,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    gamma: float,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    """
    T_new = Y^-1[Y(T) + T / T_ref * \Lambda(T_ref) / \Lambda(T) * delta_t / t_cool]
    """

    reference_temperature = cooling_curve_params.reference_temperature

    # calculate the cooling time
    # not numerically stable, divisiion
    # by the cooling
    # t_cool = cooling_time(
    #     density,
    #     temperature,
    #     hydrogen_mass_fraction,
    #     metal_mass_fraction,
    #     gamma,
    #     cooling_curve_config,
    #     cooling_curve_params
    # )

    # calculate the cooling rate
    cooling_rate = _cooling_rate(
        temperature, density, cooling_curve_config, cooling_curve_params
    )

    cooling_rate_reference = _cooling_rate(
        jnp.array([reference_temperature]),
        jnp.array([density]),
        cooling_curve_config,
        cooling_curve_params,
    )

    # calculate the temporal evolution function
    temporal_evolution_function = _temporal_evolution_function(
        temperature, cooling_curve_config, cooling_curve_params
    )

    # calculate the new temperature
    # new_temperature = _temporal_evolution_function_inverse(
    #     temporal_evolution_function + (temperature / reference_temperature) * (cooling_rate_reference / cooling_rate) * time_step / t_cool,
    #     cooling_curve_config,
    #     cooling_curve_params
    # )

    mu, mu_e, mu_H = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    new_temperature = _temporal_evolution_function_inverse(
        temporal_evolution_function
        + cooling_rate_reference
        / reference_temperature
        * ((gamma - 1) * density * mu)
        / (mu_e * mu_H)
        * time_step,
        cooling_curve_config,
        cooling_curve_params,
    )

    return new_temperature


@partial(jax.jit, static_argnames=("cooling_curve_config",))
def dtemperature_dt(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    gamma: float,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    """
    T_new = T - (gamma - 1) * rho * \mu / (mu_e * mu_H * k) * Lambda(T) * delta_t
    (units absorbed in Lambda)
    """

    # calculate the cooling rate
    cooling_rate = _cooling_rate(
        temperature, density, cooling_curve_config, cooling_curve_params
    )

    mu, mu_e, mu_H = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    return -(cooling_rate * (gamma - 1) * density * mu) / (mu_e * mu_H)


@partial(jax.jit, static_argnames=("cooling_curve_config",))
def update_temperature_explicit(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    time_step: float,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    gamma: float,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:
    """
    T_new = T - (gamma - 1) * rho * \mu / (mu_e * mu_H * k) * Lambda(T) * delta_t
    (units absorbed in Lambda)
    """

    # t_cool = cooling_time(
    #     density,
    #     temperature,
    #     hydrogen_mass_fraction,
    #     metal_mass_fraction,
    #     gamma,
    #     cooling_curve_config,
    #     cooling_curve_params
    # )

    # dt_cool_min = jnp.minimum(time_step, 0.1 * t_cool)
    # num_steps_min = jnp.ceil(time_step / dt_cool_min)
    # dt_cool = time_step / num_steps_min

    # def T_update(T):
    #     # RK2 step
    #     k1 = dtemperature_dt(
    #         density,
    #         T,
    #         hydrogen_mass_fraction,
    #         metal_mass_fraction,
    #         gamma,
    #         cooling_curve_config,
    #         cooling_curve_params
    #     )
    #     k2 = dtemperature_dt(
    #         density,
    #         T + 0.5 * dt_cool * k1,
    #         hydrogen_mass_fraction,
    #         metal_mass_fraction,
    #         gamma,
    #         cooling_curve_config,
    #         cooling_curve_params
    #     )
    #     return T + dt_cool * k2

    # new_temperature = jax.lax.fori_loop(
    #     0, jnp.int32(num_steps_min[0]),
    #     lambda i, T: T_update(T),
    #     temperature
    # )
    # return new_temperature

    return (
        temperature
        + dtemperature_dt(
            density,
            temperature,
            hydrogen_mass_fraction,
            metal_mass_fraction,
            gamma,
            cooling_curve_config,
            cooling_curve_params,
        )
        * time_step
    )

@partial(jax.jit, static_argnames=("cooling_curve_config",))
def update_temperature_implicit(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    time_step: float,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    gamma: float,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
) -> FIELD_TYPE:

    def implicit_eq(T_new):
        return (temperature
        + dtemperature_dt(
            density,
            T_new,
            hydrogen_mass_fraction,
            metal_mass_fraction,
            gamma,
            cooling_curve_config,
            cooling_curve_params,
        ) * time_step)

    # use a simple fixed point iteration
    # - maybe do newton or bisection method later
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
def update_pressure_by_cooling(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
    cooling_config: CoolingConfig,
    simulation_params: SimulationParams,
    time_step: float,
) -> STATE_TYPE:
    
    cooling_curve_config = cooling_config.cooling_curve_config

    # get the parameters
    cooling_params = simulation_params.cooling_params
    hydrogen_mass_fraction = cooling_params.hydrogen_mass_fraction
    metal_mass_fraction = cooling_params.metal_mass_fraction
    gamma = simulation_params.gamma

    # get the density and pressure
    density = primitive_state[registered_variables.density_index]
    pressure = primitive_state[registered_variables.pressure_index]

    # get the temperature
    temperature = get_temperature_from_pressure(
        density,
        pressure,
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    if cooling_config.cooling_method == IMPLICIT_COOLING:
        new_temperature = update_temperature_implicit(
            density,
            temperature,
            time_step,
            hydrogen_mass_fraction,
            metal_mass_fraction,
            gamma,
            cooling_curve_config,
            cooling_params.cooling_curve_params,
        )
    elif cooling_config.cooling_method == EXPLICIT_COOLING:
        new_temperature = update_temperature_explicit(
            density,
            temperature,
            time_step,
            hydrogen_mass_fraction,
            metal_mass_fraction,
            gamma,
            cooling_curve_config,
            cooling_params.cooling_curve_params,
        )

    new_temperature = jnp.where(
        (new_temperature > cooling_params.floor_temperature),
        new_temperature,
        temperature,
    )

    # new_temperature = temperature

    # update the pressure
    new_pressure = get_pressure_from_temperature(
        density,
        new_temperature,
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # set the new pressure
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(
        new_pressure
    )

    # return the updated primitive state
    return primitive_state

@partial(jax.jit, static_argnames = ("config", "registered_variables"))
def first_order_pressure_update(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
    config: SimulationConfig,
    helper_data: HelperData,
    simulation_params: SimulationParams,
    time_step: float,
) -> STATE_TYPE:
    # dP/dt_cooling = -(gamma - 1) n_e n_H Lambda(T)
    # with n_e = rho / mu_e, n_H = rho / mu_H

    cooling_curve_config = config.cooling_config.cooling_curve_config

    # get the parameters
    cooling_params = simulation_params.cooling_params
    hydrogen_mass_fraction = cooling_params.hydrogen_mass_fraction
    metal_mass_fraction = cooling_params.metal_mass_fraction
    gamma = simulation_params.gamma

    # get the density and pressure
    density = primitive_state[registered_variables.density_index]
    pressure = primitive_state[registered_variables.pressure_index]

    # get the temperature
    temperature = get_temperature_from_pressure(
        density,
        pressure,
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # get the molecular weights
    mu, mu_e, mu_H = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # get limited gradients, support only 1d for now
    # make cartesion approximation for now
    # only along x axis for now
    limited_gradients = _calculate_limited_gradients(primitive_state, config, helper_data, axis = 1)

    # density gradient
    density_gradient = limited_gradients[registered_variables.density_index]

    # pressure gradient
    pressure_gradient = limited_gradients[registered_variables.pressure_index]

    # temperature gradient
    # n = density / mu, T = P / n = P * mu / density
    temperature_gradient = (pressure_gradient * density - pressure * density_gradient) * mu / density**2

    # finite difference approximation to the cooling rate gradient
    cooling_rate_gradient = 1/config.grid_spacing * (
        _cooling_rate(
            temperature + 0.5 * config.grid_spacing * temperature_gradient,
            density + 0.5 * config.grid_spacing * density_gradient,
            cooling_curve_config,
            cooling_params.cooling_curve_params
        ) - _cooling_rate(
            temperature - 0.5 * config.grid_spacing * temperature_gradient,
            density - 0.5 * config.grid_spacing * density_gradient,
            cooling_curve_config,
            cooling_params.cooling_curve_params
        )
    )

    # get the cooling rate
    cooling_rate = _cooling_rate(
        temperature,
        density,
        cooling_curve_config,
        cooling_params.cooling_curve_params
    )

    # update the pressure, 0th order currently
    new_pressure = (
        pressure - (gamma - 1) / (mu_e * mu_H) * time_step * (
            density**2 * cooling_rate # 0th order
            + 1/12 * config.grid_spacing ** 2 * cooling_rate * density_gradient ** 2
            + 1/6 * config.grid_spacing ** 2 * density * cooling_rate_gradient * density_gradient
        )
    )

    # jax.debug.print("mean correction = {corr}, 0th-order = {zeroth}", corr = jnp.mean(
    #     (+ 1/12 * config.grid_spacing ** 2 * cooling_rate * density_gradient ** 2
    #     + 1/6 * config.grid_spacing ** 2 * density * cooling_rate_gradient * density_gradient)
    # ), zeroth = jnp.mean(
    #     density**2 * cooling_rate
    # ))

    # calculate the new temperature
    new_temperature = get_temperature_from_pressure(
        density,
        new_pressure,
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # apply the temperature floor
    new_temperature = jnp.where(
        (new_temperature > cooling_params.floor_temperature),
        new_temperature,
        temperature
    )

    # recalculate the pressure
    new_pressure = get_pressure_from_temperature(
        density,
        new_temperature,
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # set the new pressure
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(new_pressure)

    # return the updated primitive state
    return primitive_state
