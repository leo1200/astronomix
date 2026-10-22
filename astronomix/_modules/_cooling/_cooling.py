"""
Radiative cooling of the gas.

Implements a small family of cooling-curve models (simple and piecewise power
laws, and neural-network curves) together with the temperature-update steps and
the driver that applies cooling to the pressure of the primitive state. The
governing source term is

    dE/dt + ... = Phi(T, rho),  Phi = n_H * Gamma(T) - n_H^2 * Lambda(T).

For a simple cooling term Lambda see Section 5.3 of
https://arxiv.org/pdf/2111.03399; see also
https://academic.oup.com/mnras/article/502/3/3179/6081066 and
https://iopscience.iop.org/article/10.1088/0067-0049/181/2/391.

NOTE: All temperatures and cooling rates use the rescaled units
``\\tilde{T} = T * k_B / u`` and ``\\tilde{\\Lambda} = \\lambda / u^2``.

WARNING: The temperature is advanced either explicitly (forward Euler) or
implicitly (backward Euler, solved by a safeguarded Newton iteration). The
helpers of the Townsend (2009) exact-integration scheme (the temporal evolution
function Y and its inverse) are provided but not wired into the update. Grackle
could be used for proper cooling, but here we are interested in the simplest
cooling model.
"""

# general
from functools import partial

# typing
from typing import Tuple

# jax
import jax
import jax.numpy as jnp

# neural networks
import equinox as eqx

# astronomix constants
from astronomix._modules._cooling.cooling_options import (
    COOLING_CURVE_TYPE,
    EXPLICIT_COOLING,
    IMPLICIT_COOLING,
    NEURAL_NET_COOLING,
    NEURAL_NET_COOLING_WITH_DENSITY,
    PIECEWISE_POWER_LAW,
    SIMPLE_POWER_LAW,
)
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
)

# astronomix containers
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables


# -------------------------------------------------------------
# ============== ↓ Composition and equation of state ↓ ========
# -------------------------------------------------------------


def get_effective_molecular_weights(
    hydrogen_mass_fraction: float,  # X
    metal_mass_fraction: float,  # Z
) -> Tuple[float, float, float]:
    """
    Calculate the mean molecular weight and the effective molecular weights of
    electrons and hydrogen for a fully ionised gas.

    Args:
        hydrogen_mass_fraction: The hydrogen mass fraction X.
        metal_mass_fraction: The metal mass fraction Z.

    Returns:
        The mean molecular weight mu, the electron molecular weight mu_e and
        the hydrogen molecular weight mu_H.
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
    density: FIELD_TYPE,
    mean_molecular_weight: float,
) -> FIELD_TYPE:
    """Return the particle number density n = rho / mu."""
    return density / mean_molecular_weight


def get_pressure_from_temperature(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
) -> FIELD_TYPE:
    r"""
    Pressure from the rescaled temperature, ``P = n * \tilde{T}``.

    Args:
        density: The density field.
        temperature: The rescaled temperature field.
        hydrogen_mass_fraction: The hydrogen mass fraction X.
        metal_mass_fraction: The metal mass fraction Z.

    Returns:
        The pressure field.
    """

    # calculate the effective molecular weights
    mu, _, _ = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # calculate the particle number density
    particle_number_density = get_particle_number_density(density, mu)

    # calculate the pressure
    return particle_number_density * temperature


def get_temperature_from_pressure(
    density: FIELD_TYPE,
    pressure: FIELD_TYPE,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
) -> FIELD_TYPE:
    r"""
    Rescaled temperature from the pressure, ``\tilde{T} = P / n``.

    Args:
        density: The density field (must be positive everywhere).
        pressure: The pressure field.
        hydrogen_mass_fraction: The hydrogen mass fraction X.
        metal_mass_fraction: The metal mass fraction Z.

    Returns:
        The rescaled temperature field.
    """

    # calculate the effective molecular weights
    mu, _, _ = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # calculate the particle number density
    particle_number_density = get_particle_number_density(density, mu)

    # The density enters the denominator, so it must never be zero.
    return pressure / particle_number_density


# -------------------------------------------------------------
# ============== ↑ Composition and equation of state ↑ ========
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============== ↓ Cooling curves ↓ ===========================
# -------------------------------------------------------------


def cooling_rate_power_law(
    temperature: FIELD_TYPE,
    reference_temperature: float,
    factor: float,
    exponent: float,
) -> FIELD_TYPE:
    """Single power-law cooling curve Lambda(T) = factor * (T / T_ref)^exponent."""
    return factor * (temperature / reference_temperature) ** exponent


def power_law_temporal_evolution_function(
    temperature: FIELD_TYPE,  # T
    reference_temperature: float,  # T_ref
    exponent: float,  # alpha
) -> FIELD_TYPE:
    """
    Townsend (2009) temporal evolution function Y(T) of a single power law,

        Y(T) = 1 / (1 - alpha) * (1 - (T / T_ref)^(1 - alpha))   for alpha != 1,
        Y(T) = -log(T / T_ref)                                    for alpha = 1.

    Args:
        temperature: The temperature field T.
        reference_temperature: The reference temperature T_ref.
        exponent: The power-law exponent alpha.

    Returns:
        The temporal evolution function Y(T).
    """
    return jax.lax.cond(
        exponent != 1,
        lambda: (1 / (1 - exponent))
        * (1 - (temperature / reference_temperature) ** (1 - exponent)),
        lambda: -jnp.log(temperature / reference_temperature),
    )


def power_law_temporal_evolution_function_inverse(
    temporal_evolution_function: FIELD_TYPE,  # Y
    reference_temperature: float,  # T_ref
    exponent: float,  # alpha
) -> FIELD_TYPE:
    """
    Inverse Y^-1(Y) of the single power-law temporal evolution function,

        T = T_ref * (1 - (1 - alpha) * Y)^(1 / (1 - alpha))   for alpha != 1,
        T = T_ref * exp(-Y)                                   for alpha = 1.

    Args:
        temporal_evolution_function: The temporal evolution function Y.
        reference_temperature: The reference temperature T_ref.
        exponent: The power-law exponent alpha.

    Returns:
        The temperature T.
    """
    return jax.lax.cond(
        exponent != 1,
        lambda: reference_temperature
        * (1 - (1 - exponent) * temporal_evolution_function) ** (1 / (1 - exponent)),
        lambda: reference_temperature * jnp.exp(-temporal_evolution_function),
    )


def _evaluate_piecewise_power_law(
    T_in,
    T_table,
    Lambda_table,
    alpha_table,
):
    """
    Lambda(T) on a piecewise power-law table, ``Lambda_k (T / T_k)^alpha_k``.

    Branch-free whole-array evaluation: the bin lookup and the range mask are
    ordinary batched array operations, since a per-cell ``lax.cond`` under
    ``jnp.vectorize`` would dominate the cooling solve on GPU. ``T_in`` is
    clamped inside the power so out-of-range cells cannot produce inf / NaN in
    the values or the gradients; the mask then sets them to zero.

    Args:
        T_in: The temperature field.
        T_table: The tabulated bin-edge temperatures T_k (ascending).
        Lambda_table: The tabulated cooling rates Lambda_k at the bin edges.
        alpha_table: The power-law slope alpha_k of each bin.

    Returns:
        The cooling rate, zero outside the tabulated temperature range.
    """
    table_minimum_temperature = T_table[0]
    table_maximum_temperature = T_table[-1]
    bin_index = jnp.searchsorted(T_table, T_in) - 1
    bin_index = jnp.clip(bin_index, 0, T_table.shape[0] - 2)
    alpha_k = jnp.take(alpha_table, bin_index)
    T_k = jnp.take(T_table, bin_index)
    Lambda_k = jnp.take(Lambda_table, bin_index)
    T_safe = jnp.clip(T_in, table_minimum_temperature, table_maximum_temperature)
    value = Lambda_k * (T_safe / T_k) ** alpha_k
    return jnp.where(
        (T_in >= table_minimum_temperature) & (T_in <= table_maximum_temperature),
        value,
        0.0,
    )


@partial(
    jnp.vectorize,
    excluded=(1, 2, 3, 4),  # the tables are not vectorized over
    signature="()->()",  # scalar in, scalar out
)
def _piecewise_power_law_temporal_evolution_function(
    T_in,
    T_table,
    Lambda_table,
    alpha_table,
    Y_table,
):
    """
    Townsend (2009) temporal evolution function Y(T) of a piecewise power law
    zero outside the tabulated temperature range.

    Args:
        T_in: The temperature (scalar; vectorized over fields).
        T_table: The tabulated bin-edge temperatures T_k.
        Lambda_table: The tabulated cooling rates Lambda_k.
        alpha_table: The power-law slope alpha_k of each bin.
        Y_table: The temporal evolution function Y_k at the bin edges.

    Returns:
        The temporal evolution function Y(T).
    """

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
    excluded=(1, 2, 3, 4),  # the tables are not vectorized over
    signature="()->()",  # scalar in, scalar out
)
def _piecewise_power_law_temporal_evolution_function_inverse(
    Y_in,
    T_table,
    Lambda_table,
    alpha_table,
    Y_table,
):
    """
    Inverse Y^-1(Y) of the piecewise power-law temporal evolution function
    (Townsend 2009). Values of Y beyond the table map to the table's
    end temperatures.

    Args:
        Y_in: The temporal evolution function (scalar; vectorized over fields).
        T_table: The tabulated bin-edge temperatures T_k.
        Lambda_table: The tabulated cooling rates Lambda_k.
        alpha_table: The power-law slope alpha_k of each bin.
        Y_table: The temporal evolution function Y_k at the bin edges.

    Returns:
        The temperature T.
    """

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
        # Check if Y_in is in the table range; Y_table is monotonically
        # decreasing.
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
    """
    Evaluate the cooling rate Lambda(T) for the configured cooling curve.

    Dispatches on ``cooling_curve_config.cooling_curve_type`` to the simple or
    piecewise power law, or to a (density-aware) neural-network curve.

    Args:
        temperature: The temperature field.
        density: The density field (used by the density-aware network curve).
        cooling_curve_config: The static cooling-curve configuration.
        cooling_curve_params: The cooling-curve parameters.

    Returns:
        The cooling rate evaluated at each cell.
    """
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

        # The network is trained directly in code units, so its input and
        # output are used without any rescaling.
        return 10 ** model(jnp.log10(temperature).reshape(-1, 1)).flatten()
    elif cooling_curve_config.cooling_curve_type == NEURAL_NET_COOLING_WITH_DENSITY:
        neural_net_params = cooling_curve_params.network_params
        neural_net_static = cooling_curve_config.cooling_net_config.network_static
        model = jax.vmap(eqx.combine(neural_net_params, neural_net_static))

        # The network is trained directly in code units, so its input and
        # output are used without any rescaling.
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
    """
    Evaluate the Townsend (2009) temporal evolution function Y(T) for the
    configured (simple or piecewise power-law) cooling curve.

    Args:
        temperature: The temperature field.
        cooling_curve_config: The static cooling-curve configuration.
        cooling_curve_params: The cooling-curve parameters.

    Returns:
        The temporal evolution function Y(T).
    """
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
    """
    Evaluate the inverse Y^-1 of the Townsend (2009) temporal evolution
    function for the configured (simple or piecewise power-law) cooling curve.

    Args:
        temporal_evolution_function: The temporal evolution function Y.
        cooling_curve_config: The static cooling-curve configuration.
        cooling_curve_params: The cooling-curve parameters.

    Returns:
        The temperature field.
    """
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


# -------------------------------------------------------------
# ============== ↑ Cooling curves ↑ ===========================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============== ↓ Temperature updates ↓ ======================
# -------------------------------------------------------------


@partial(jax.jit, static_argnames=("cooling_curve_config",))
def dtemperature_dt(
    density: FIELD_TYPE,
    temperature: FIELD_TYPE,
    hydrogen_mass_fraction: float,
    metal_mass_fraction: float,
    gamma: float,
    cooling_curve_config: CoolingCurveConfig,
    cooling_curve_params: COOLING_CURVE_TYPE,
    heating_rate: float = 0.0,
    heating_log10_T_table: jnp.ndarray = None,
    heating_factor_table: jnp.ndarray = None,
) -> FIELD_TYPE:
    r"""
    Net rate of change of the rescaled temperature,

        dT/dt = -(gamma - 1) * rho * \mu / (mu_e * mu_H) * Lambda(T)
                + (gamma - 1) * heating_rate

    (the physical units are absorbed in Lambda). The heating term is the
    density-independent ISM heating of ``CoolingParams.heating_rate``, making
    this the net rate Gamma - Lambda when enabled.

    Args:
        density: The density field.
        temperature: The rescaled temperature field.
        hydrogen_mass_fraction: The hydrogen mass fraction X.
        metal_mass_fraction: The metal mass fraction Z.
        gamma: The adiabatic index.
        cooling_curve_config: The static cooling-curve configuration.
        cooling_curve_params: The cooling-curve parameters.
        heating_rate: The constant heating rate (0 disables heating).

    Returns:
        The net temperature rate dT/dt at each cell.
    """

    # calculate the cooling rate
    cooling_rate = _cooling_rate(
        temperature,
        density,
        cooling_curve_config,
        cooling_curve_params,
    )

    mu, mu_e, mu_H = get_effective_molecular_weights(
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # Optional temperature-dependent heating factor f(T~) (see
    # CoolingParams.heating_factor_table); table shapes are static, so this is
    # a trace-time branch.
    heating = heating_rate
    if heating_factor_table is not None and heating_factor_table.shape[0] > 0:
        heating = heating_rate * jnp.interp(
            jnp.log10(temperature),
            heating_log10_T_table,
            heating_factor_table,
        )

    return (
        -(cooling_rate * (gamma - 1) * density * mu) / (mu_e * mu_H)
        + (gamma - 1) * heating
    )


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
    heating_rate: float = 0.0,
    heating_log10_T_table: jnp.ndarray = None,
    heating_factor_table: jnp.ndarray = None,
) -> FIELD_TYPE:
    """
    Advance the temperature by one explicit (forward-Euler) step,
    ``T_new = T + dt * dT/dt(T)`` (see ``dtemperature_dt``).

    Args:
        density: The density field.
        temperature: The current rescaled temperature field.
        time_step: The time step.
        hydrogen_mass_fraction: The hydrogen mass fraction X.
        metal_mass_fraction: The metal mass fraction Z.
        gamma: The adiabatic index.
        cooling_curve_config: The static cooling-curve configuration.
        cooling_curve_params: The cooling-curve parameters.
        heating_rate: The constant heating rate (0 disables heating).

    Returns:
        The temperature field after the step.
    """

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
            heating_rate=heating_rate,
            heating_log10_T_table=heating_log10_T_table,
            heating_factor_table=heating_factor_table,
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
    heating_rate: float = 0.0,
    heating_log10_T_table: jnp.ndarray = None,
    heating_factor_table: jnp.ndarray = None,
) -> FIELD_TYPE:
    """
    Advance the temperature by one implicit (backward-Euler) step.

    Solves ``T = T_old + dt * dT/dt(T)`` cell by cell with a safeguarded
    Newton iteration: Newton where the step stays inside a maintained bracket
    of the root, bisection otherwise.

    Args:
        density: The density field.
        temperature: The current rescaled temperature field T_old.
        time_step: The time step.
        hydrogen_mass_fraction: The hydrogen mass fraction X.
        metal_mass_fraction: The metal mass fraction Z.
        gamma: The adiabatic index.
        cooling_curve_config: The static cooling-curve configuration.
        cooling_curve_params: The cooling-curve parameters.
        heating_rate: The constant heating rate (0 disables heating).

    Returns:
        The temperature field after the step.
    """

    def net_temperature_rate(trial_temperature):
        return dtemperature_dt(
            density,
            trial_temperature,
            hydrogen_mass_fraction,
            metal_mass_fraction,
            gamma,
            cooling_curve_config,
            cooling_curve_params,
            heating_rate=heating_rate,
            heating_log10_T_table=heating_log10_T_table,
            heating_factor_table=heating_factor_table,
        )

    # Why Newton: a fixed-point sweep converges only linearly, and each sweep
    # costs a full cooling-curve evaluation over the grid. Newton converges
    # quadratically (typically 4-6 iterations); the rate is elementwise, so a
    # single JVP with a unit tangent yields the per-cell derivative
    # d(rate)/dT and no Jacobian is ever formed.
    #
    # Why the safeguard: Lambda(T) is non-monotone, so the derivative of the
    # residual, F'(T) = 1 - dt * d(rate)/dT, passes through zero on the
    # falling branch of the curve, and an unguarded Newton step then jumps the
    # wrong way. It can return a temperature orders of magnitude above T_old,
    # which is impossible for a pure sink and collapses the CFL time step.
    #
    # A bracket always exists: F(T) = T - T_old - dt * rate(T) is continuous
    # with
    #   F(0+)   = -T_old - dt * (gamma - 1) * heating  <  0  (Lambda -> 0 off-table),
    #   F(T_hi) = dt * Lambda(T_hi) * (...)            >= 0  at the no-cooling
    #             bound T_hi = T_old + dt * (gamma - 1) * heating,
    # so bisection is always available as the fallback and the iteration
    # cannot leave the physical interval whatever the curve does.
    #
    # NOTE: With heating enabled the equation can have several roots near the
    # two-phase equilibrium. At time steps far above the operating point a
    # small fraction of cells may still carry a residual above the tolerance
    # after the iteration cap; they remain inside the bracket, so they fail
    # safe.
    machine_epsilon = jnp.finfo(temperature.dtype).eps
    tolerance = jnp.maximum(1e-6, 8.0 * machine_epsilon)
    smallest_normal = jnp.finfo(temperature.dtype).tiny
    max_iterations = 30

    initial_bracket_high = temperature + time_step * (gamma - 1.0) * heating_rate
    initial_bracket_low = jnp.full_like(temperature, smallest_normal)

    def newton_body(iteration_state):
        iteration, trial_temperature, bracket_low, bracket_high, _ = iteration_state
        rate_value, rate_derivative = jax.jvp(
            net_temperature_rate,
            (trial_temperature,),
            (jnp.ones_like(trial_temperature),),
        )
        residual = trial_temperature - temperature - time_step * rate_value

        # Shrink the bracket so that F(bracket_low) < 0 <= F(bracket_high) is
        # maintained; this holds even where F is not monotone.
        bracket_low = jnp.where(residual < 0.0, trial_temperature, bracket_low)
        bracket_high = jnp.where(residual < 0.0, bracket_high, trial_temperature)

        residual_derivative = 1.0 - time_step * rate_derivative
        residual_derivative = jnp.where(
            jnp.abs(residual_derivative) > smallest_normal,
            residual_derivative,
            1.0,
        )
        newton_temperature = trial_temperature - residual / residual_derivative
        inside_bracket = (newton_temperature > bracket_low) & (
            newton_temperature < bracket_high
        )

        # Plain arithmetic bisection as the fallback. A geometric mean looks
        # more natural for a quantity spanning decades, but the fallback is
        # taken near the root, where the bracket is already narrow, not across
        # the initial decades; sqrt(low * high) would also underflow to zero
        # in single precision.
        new_temperature = jnp.where(
            inside_bracket,
            newton_temperature,
            0.5 * (bracket_low + bracket_high),
        )
        relative_change = jnp.max(
            jnp.abs(new_temperature - trial_temperature)
            / jnp.maximum(jnp.abs(new_temperature), smallest_normal)
        )
        return (iteration + 1, new_temperature, bracket_low, bracket_high, relative_change)

    def newton_cond(iteration_state):
        iteration, _, _, _, relative_change = iteration_state
        return (iteration < max_iterations) & (relative_change > tolerance)

    _, final_temperature, _, _, _ = jax.lax.while_loop(
        newton_cond,
        newton_body,
        (0, temperature, initial_bracket_low, initial_bracket_high, jnp.inf),
    )
    return final_temperature


# -------------------------------------------------------------
# ============== ↑ Temperature updates ↑ ======================
# -------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=(
        "cooling_config",
        "registered_variables",
        "grid_spacing",
    ),
)
def update_pressure_by_cooling(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
    cooling_config: CoolingConfig,
    simulation_params: SimulationParams,
    time_step: float,
    grid_spacing: float = 0.0,
) -> STATE_TYPE:
    """
    Apply cooling to the pressure of the primitive state for one time step.

    Converts pressure to temperature, applies the cooling-resolution limiter,
    advances the temperature with the chosen cooling method (explicit /
    implicit), applies the per-step cap and the temperature floor, and
    converts the result back to pressure.

    Args:
        primitive_state: The primitive state array.
        registered_variables: The registered variables.
        cooling_config: The cooling configuration (method and curve).
        simulation_params: The simulation parameters.
        time_step: The time step.
        grid_spacing: The grid spacing (static). It enables the
            cooling-resolution limiter (``CoolingParams.resolution_limiter_alpha``)
            when > 0; the default 0 disables the limiter.

    Returns:
        The primitive state with the pressure updated by cooling.
    """

    cooling_curve_config = cooling_config.cooling_curve_config

    # get the parameters
    cooling_params = simulation_params.cooling_params
    hydrogen_mass_fraction = cooling_params.hydrogen_mass_fraction
    metal_mass_fraction = cooling_params.metal_mass_fraction
    gamma = simulation_params.gamma

    # -------------------------------------------------------------
    # ============== ↓ Temperature from pressure ↓ ================
    # -------------------------------------------------------------

    density = primitive_state[registered_variables.density_index]
    pressure = primitive_state[registered_variables.pressure_index]

    temperature = get_temperature_from_pressure(
        density,
        pressure,
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    # -------------------------------------------------------------
    # ============== ↑ Temperature from pressure ↑ ================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============== ↓ Cooling-resolution limiter ↓ ===============
    # -------------------------------------------------------------

    # Suppress the cooling rate where the cooling length
    # l_cool = c_s * t_cool is unresolved (below ``resolution_limiter_alpha``
    # cells). An unresolved radiative shock layer otherwise collapses to a
    # cell-scale cold dense sheet with no pressure support and runs away under
    # ram pressure; keeping it adiabatic is the resolvable solution. This is
    # implemented by scaling the per-cell effective time step (equivalent to
    # scaling Lambda), which both the implicit (safeguarded Newton) and the
    # explicit update broadcast elementwise.
    #
    # The limiter strength rides in the (traced) SimulationParams, so its
    # on/off gate must be trace-safe: the suppression is computed
    # unconditionally and blended in with jnp.where. ``grid_spacing`` is a
    # static Python float, so that gate can stay a Python conditional.
    resolution_limiter_cells = cooling_params.resolution_limiter_alpha
    if grid_spacing > 0.0:
        # Use the net rate (cooling minus heating): at the two-phase
        # equilibrium the net rate vanishes, so the limiter leaves equilibrium
        # gas untouched.
        net_temperature_rate = dtemperature_dt(
            density,
            temperature,
            hydrogen_mass_fraction,
            metal_mass_fraction,
            gamma,
            cooling_curve_config,
            cooling_params.cooling_curve_params,
            heating_rate=cooling_params.heating_rate,
            heating_log10_T_table=cooling_params.heating_log10_T_table,
            heating_factor_table=cooling_params.heating_factor_table,
        )
        cooling_time = temperature / jnp.maximum(jnp.abs(net_temperature_rate), 1e-30)
        sound_speed = jnp.sqrt(gamma * pressure / density)
        cooling_length = sound_speed * cooling_time
        safe_resolution_limiter_cells = jnp.maximum(resolution_limiter_cells, 1e-30)
        suppression = jnp.clip(
            (cooling_length / (safe_resolution_limiter_cells * grid_spacing)) ** 2,
            0.0,
            1.0,
        )
        time_step = time_step * jnp.where(resolution_limiter_cells > 0.0, suppression, 1.0)

    # -------------------------------------------------------------
    # ============== ↑ Cooling-resolution limiter ↑ ===============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============== ↓ Temperature update ↓ =======================
    # -------------------------------------------------------------

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
            heating_rate=cooling_params.heating_rate,
            heating_log10_T_table=cooling_params.heating_log10_T_table,
            heating_factor_table=cooling_params.heating_factor_table,
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
            heating_rate=cooling_params.heating_rate,
            heating_log10_T_table=cooling_params.heating_log10_T_table,
            heating_factor_table=cooling_params.heating_factor_table,
        )
    else:
        raise ValueError(f"Unknown cooling method: {cooling_config.cooling_method}")

    # -------------------------------------------------------------
    # ============== ↑ Temperature update ↑ =======================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============== ↓ Per-step cap and temperature floor ↓ =======
    # -------------------------------------------------------------

    # Operator-splitting limiter: cap the fractional drop applied in one step
    # (see CoolingParams.max_cooling_fraction). It acts downward only, so
    # heating and the two-phase equilibrium are untouched.
    max_fraction = cooling_params.max_cooling_fraction
    new_temperature = jnp.where(
        max_fraction > 0.0,
        jnp.maximum(new_temperature, (1.0 - max_fraction) * temperature),
        new_temperature,
    )

    # Temperature floor, with two behaviours (see CoolingParams.clamp_to_floor).
    # The revert (default) discards the whole update for a cell that would
    # cross the floor; this suppresses cooling exactly in the unresolved,
    # crush-prone cells. The clamp puts such a cell on the floor instead, which
    # the explicit update needs in order to cool stiff cells at all. Cells that
    # start at or below the floor are left alone either way, since clamping
    # them would heat them.
    floor_temperature = cooling_params.floor_temperature
    clamped = jnp.where(
        temperature <= floor_temperature,
        temperature,
        jnp.maximum(new_temperature, floor_temperature),
    )
    reverted = jnp.where(
        new_temperature > floor_temperature,
        new_temperature,
        temperature,
    )
    new_temperature = jnp.where(cooling_params.clamp_to_floor, clamped, reverted)

    # -------------------------------------------------------------
    # ============== ↑ Per-step cap and temperature floor ↑ =======
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============== ↓ Back to pressure ↓ =========================
    # -------------------------------------------------------------

    new_pressure = get_pressure_from_temperature(
        density,
        new_temperature,
        hydrogen_mass_fraction,
        metal_mass_fraction,
    )

    primitive_state = primitive_state.at[registered_variables.pressure_index].set(
        new_pressure
    )

    # -------------------------------------------------------------
    # ============== ↑ Back to pressure ↑ =========================
    # -------------------------------------------------------------

    return primitive_state
