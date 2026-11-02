"""
CFL time-step estimators for the finite-difference solver.

Provides the advective CFL time-step limit for hydrodynamics and MHD, tightened
when active by the parabolic (viscous, resistive, conductive) limits and the
explicit-cooling limit. Each equation set has a full characteristic-eigenvalue
estimator plus a lower-storage "fast" estimator used by the Pallas backend that
reaches the same advective limit directly from the primitive variables, avoiding
the materialisation of the full eigenvalue stack.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import (
    Array,
    Float,
)

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    DYNAMIC_VISCOSITY,
    IDEAL_GAS,
    ISOTHERMAL,
    KINEMATIC_VISCOSITY,
    STATE_TYPE,
)
from astronomix._modules._cooling.cooling_options import EXPLICIT_COOLING

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._eigen_hydro import _eigen_all_lambdas_hydro
from astronomix._fluid_equations._eigen_hydro_iso import _eigen_all_lambdas_hydro_iso
from astronomix._fluid_equations._eigen_mhd import _eigen_all_lambdas
from astronomix._fluid_equations._eigen_mhd_iso import _eigen_all_lambdas_iso
from astronomix._fluid_equations._equations import conserved_state_from_primitive
from astronomix._fluid_equations._equations_mhd import (
    conserved_state_from_primitive_isothermal,
    conserved_state_from_primitive_mhd,
)
from astronomix._modules._cooling._cooling import (
    dtemperature_dt,
    get_temperature_from_pressure,
)
from astronomix._pallas_helpers import _backend_is_pallas


# -------------------------------------------------------------
# ========= ↓ Parabolic and source-term time-step limits ↓ ====
# -------------------------------------------------------------


def _minimum_density_for_estimates(primitive_state, params, config, registered_variables):
    """
    The smallest density of the state, floored when ``clamp_in_estimates`` is set.

    Args:
        primitive_state: The primitive state array.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The (optionally floored) minimum density.
    """
    if config.positivity_config.clamp_in_estimates:
        return jnp.maximum(
            jnp.min(primitive_state[registered_variables.density_index]),
            params.minimum_density,
        )
    return jnp.min(primitive_state[registered_variables.density_index])


def _apply_parabolic_and_source_limits(
    dt_cfl,
    primitive_state,
    grid_spacing,
    gamma,
    config,
    params,
    registered_variables,
    C_CFL,
):
    """
    Tighten a time step by the viscous, resistive, explicit-cooling and
    conductive limits (each only when the corresponding physics is active).

    The parabolic limits are the usual explicit-diffusion bound
    ``dt <= C_CFL dx^2 / (2 d D_max)`` for the largest diffusivity ``D_max`` of
    the grid. Resistivity is only reachable for 3D finite-difference MHD:
    ``finalize_config`` rejects it for every other setup.

    Args:
        dt_cfl: The time step to tighten.
        primitive_state: The primitive state array.
        grid_spacing: The grid spacing.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        C_CFL: The CFL safety factor.

    Returns:
        The tightened time step.
    """
    # --------------- ↓ Viscous limit ↓ ----------------
    if config.diffusion:
        minimum_density = _minimum_density_for_estimates(
            primitive_state,
            params,
            config,
            registered_variables,
        )

        if config.viscosity_type == DYNAMIC_VISCOSITY:
            maximum_kinematic_viscosity = params.viscosity / minimum_density
        elif config.viscosity_type == KINEMATIC_VISCOSITY:
            maximum_kinematic_viscosity = params.viscosity

        dt_viscous = C_CFL * grid_spacing**2 / (
            2.0 * config.dimensionality * maximum_kinematic_viscosity
        )
        dt_cfl = jnp.minimum(dt_cfl, dt_viscous)
    # --------------- ↑ Viscous limit ↑ ----------------

    # --------------- ↓ Resistive limit ↓ ----------------
    if config.resistivity:
        dt_resistive = C_CFL * grid_spacing ** 2 / (
            2.0 * config.dimensionality * params.resistivity
        )
        dt_cfl = jnp.minimum(dt_cfl, dt_resistive)
    # --------------- ↑ Resistive limit ↑ ----------------

    # --------------- ↓ Explicit-cooling limit ↓ ----------------
    # Mirroring AthenaK's ``srcterms_newdt``: dt <= min(e_int / |de/dt|), which
    # for an ideal gas is the temperature relaxation time min(T / |dT/dt|). Only
    # the explicit update needs it; the implicit one is unconditionally stable.
    if (config.cooling_config.cooling
            and config.cooling_config.cooling_method == EXPLICIT_COOLING
            and config.equation_of_state == IDEAL_GAS):
        cooling_params = params.cooling_params
        density = primitive_state[registered_variables.density_index]
        pressure = primitive_state[registered_variables.pressure_index]
        temperature = get_temperature_from_pressure(
            density,
            pressure,
            cooling_params.hydrogen_mass_fraction,
            cooling_params.metal_mass_fraction,
        )
        # Clamp to the cooling floor before forming the thermal time. Below the
        # floor the cooling update REVERTS the cell (it applies no cooling at
        # all), so its thermal time is INFINITE, not zero. Without this clamp a
        # single cell driven towards T -> 0 sends dt -> 0 and stalls the whole
        # run, which is not a physical time-step limit at all.
        temperature = jnp.maximum(temperature, cooling_params.floor_temperature)
        temperature_rate = dtemperature_dt(
            density,
            temperature,
            cooling_params.hydrogen_mass_fraction,
            cooling_params.metal_mass_fraction,
            gamma,
            config.cooling_config.cooling_curve_config,
            cooling_params.cooling_curve_params,
            heating_rate=cooling_params.heating_rate,
            heating_log10_T_table=cooling_params.heating_log10_T_table,
            heating_factor_table=cooling_params.heating_factor_table,
        )
        dt_cooling = C_CFL * jnp.min(
            temperature / (jnp.abs(temperature_rate) + jnp.finfo(temperature.dtype).tiny)
        )
        dt_cfl = jnp.minimum(dt_cfl, dt_cooling)
    # --------------- ↑ Explicit-cooling limit ↑ ----------------

    # --------------- ↓ Conductive limit ↓ ----------------
    if config.thermal_conduction:
        if config.conduction_density_weighted:
            # With a density-weighted conductivity (kappa = rho * alpha) the
            # temperature diffusivity is uniform, so the bound needs no minimum
            # density.
            maximum_thermal_diffusivity = (gamma - 1.0) * params.thermal_conductivity
        else:
            minimum_density = _minimum_density_for_estimates(
                primitive_state,
                params,
                config,
                registered_variables,
            )
            # The diffusivity of the internal energy is chi = (gamma - 1) kappa / rho.
            maximum_thermal_diffusivity = (
                (gamma - 1.0) * params.thermal_conductivity / minimum_density
            )
        dt_conductive = C_CFL * grid_spacing**2 / (
            2.0 * config.dimensionality * maximum_thermal_diffusivity
        )
        dt_cfl = jnp.minimum(dt_cfl, dt_conductive)
    # --------------- ↑ Conductive limit ↑ ----------------

    return dt_cfl


# -------------------------------------------------------------
# ========= ↑ Parabolic and source-term time-step limits ↑ ====
# -------------------------------------------------------------


# -------------------------------------------------------------
# =================== ↓ MHD CFL estimators ↓ ==================
# -------------------------------------------------------------


def _mhd_fast_cfl_supported(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> bool:
    """
    Whether the lower-storage MHD fast-CFL estimator can be used.

    It skips the full 7-eigenvalue stack and computes ``max(|v_d| + c_fast_d)``
    per cell directly from the primitives; the advective limit is the same as
    that of ``_cfl_time_step_fd``. Available whenever the Pallas backend is on
    and the registry exposes the velocity and magnetic indices (plus the
    pressure index for an ideal gas).
    """
    if not _backend_is_pallas(config):
        return False
    if not config.mhd:
        return False
    if not hasattr(registered_variables, "velocity_index"):
        return False
    if not hasattr(registered_variables, "magnetic_index"):
        return False
    has_pressure = hasattr(registered_variables, "pressure_index")
    if config.equation_of_state == IDEAL_GAS and not has_pressure:
        return False
    return True


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _cfl_time_step_fd_mhd_fast(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt_max: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    C_CFL: Union[float, Float[Array, ""]] = 0.8,
) -> Float[Array, ""]:
    """
    Lower-storage MHD CFL estimator used by the Pallas backend.

    Mirrors the hydro fast path: the fast magnetosonic speed along each axis,
    ``c_fast_d^2 = 0.5 (b^2/rho + c_s^2 + sqrt((b^2/rho + c_s^2)^2 - 4 (B_d^2/rho) c_s^2))``,
    is computed pointwise and the signal speed is ``max(|v_d| + c_fast_d)`` per
    axis, so no full-state characteristic-eigenvalue array is materialised.

    Args:
        primitive_state: The primitive state array.
        grid_spacing: The grid spacing.
        dt_max: The maximum allowed time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        C_CFL: The CFL safety factor.

    Returns:
        The CFL-limited time step.
    """
    density = primitive_state[registered_variables.density_index]
    velocity_x = primitive_state[registered_variables.velocity_index.x]
    if config.dimensionality >= 2:
        velocity_y = primitive_state[registered_variables.velocity_index.y]
    else:
        velocity_y = 0.0
    if config.dimensionality == 3:
        velocity_z = primitive_state[registered_variables.velocity_index.z]
    else:
        velocity_z = 0.0
    magnetic_field_x = primitive_state[registered_variables.magnetic_index.x]
    magnetic_field_y = primitive_state[registered_variables.magnetic_index.y]
    magnetic_field_z = primitive_state[registered_variables.magnetic_index.z]

    if config.equation_of_state == IDEAL_GAS:
        pressure = primitive_state[registered_variables.pressure_index]
        if config.positivity_config.clamp_in_estimates:
            density = jnp.maximum(density, params.minimum_density)
            pressure = jnp.maximum(pressure, params.minimum_pressure)
        sound_speed_squared = jnp.maximum(gamma * pressure / density, 1e-12)
    else:
        # The isothermal equation of state.
        if config.positivity_config.clamp_in_estimates:
            density = jnp.maximum(density, params.minimum_density)
        sound_speed_squared = jnp.full_like(density, params.isothermal_sound_speed ** 2)

    magnetic_field_squared = (
        magnetic_field_x * magnetic_field_x
        + magnetic_field_y * magnetic_field_y
        + magnetic_field_z * magnetic_field_z
    )
    magnetic_field_squared_over_rho = magnetic_field_squared / density

    def fast_magnetosonic_speed(normal_magnetic_field):
        normal_field_squared_over_rho = (normal_magnetic_field * normal_magnetic_field) / density
        discriminant = jnp.maximum(
            (magnetic_field_squared_over_rho + sound_speed_squared) ** 2
            - 4.0 * normal_field_squared_over_rho * sound_speed_squared,
            0.0,
        )
        return jnp.sqrt(
            jnp.maximum(
                0.5 * (
                    magnetic_field_squared_over_rho
                    + sound_speed_squared
                    + jnp.sqrt(discriminant)
                ),
                0.0,
            )
        )

    lambda_x = jnp.max(jnp.abs(velocity_x) + fast_magnetosonic_speed(magnetic_field_x))
    if config.dimensionality >= 2:
        lambda_y = jnp.max(jnp.abs(velocity_y) + fast_magnetosonic_speed(magnetic_field_y))
    else:
        lambda_y = 0.0
    if config.dimensionality == 3:
        lambda_z = jnp.max(jnp.abs(velocity_z) + fast_magnetosonic_speed(magnetic_field_z))
    else:
        lambda_z = 0.0

    dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)

    dt_cfl = _apply_parabolic_and_source_limits(
        dt_cfl,
        primitive_state,
        grid_spacing,
        gamma,
        config,
        params,
        registered_variables,
        C_CFL,
    )

    return jnp.minimum(dt_cfl, dt_max)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _cfl_time_step_fd(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt_max: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    C_CFL: Union[float, Float[Array, ""]] = 0.8,
) -> Float[Array, ""]:
    """
    Compute the MHD CFL time step from the full characteristic eigenvalues.

    For each axis the conserved state is permuted so that axis becomes the
    sweep direction, the full eigenvalue stack is evaluated, and the largest
    absolute eigenvalue is taken as the local signal speed; the advective limit
    is then combined with the optional parabolic and explicit-cooling limits.
    When the Pallas fast path is available, the equivalent lower-storage
    estimator is used instead.

    Args:
        primitive_state: The primitive state array.
        grid_spacing: The grid spacing.
        dt_max: The maximum allowed time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        C_CFL: The CFL safety factor.

    Returns:
        The CFL-limited time step.
    """
    if _mhd_fast_cfl_supported(config, registered_variables):
        return _cfl_time_step_fd_mhd_fast(
            primitive_state,
            grid_spacing,
            dt_max,
            gamma,
            config,
            params,
            registered_variables,
            C_CFL,
        )

    if config.equation_of_state == IDEAL_GAS:
        conserved_state = conserved_state_from_primitive_mhd(
            primitive_state, gamma, registered_variables
        )
    elif config.equation_of_state == ISOTHERMAL:
        conserved_state = conserved_state_from_primitive_isothermal(
            primitive_state, config, registered_variables
        )

    def maximum_signal_speed(swept_state):
        if config.equation_of_state == IDEAL_GAS:
            eigenvalues = _eigen_all_lambdas(
                swept_state,
                params.minimum_density,
                params.minimum_pressure,
                gamma,
                registered_variables,
            )
        elif config.equation_of_state == ISOTHERMAL:
            eigenvalues = _eigen_all_lambdas_iso(
                swept_state,
                params.minimum_density,
                params.isothermal_sound_speed,
                registered_variables,
            )
        return jnp.max(jnp.abs(eigenvalues))

    lambda_x = maximum_signal_speed(conserved_state)

    if config.dimensionality >= 2:
        if config.dimensionality == 2:
            state_y = jnp.transpose(conserved_state, (0, 2, 1))
        else:
            state_y = jnp.transpose(conserved_state, (0, 2, 1, 3))

        # Swap the x and y vector components so y becomes the sweep direction.
        momentum_x = state_y[registered_variables.momentum_index.x]
        momentum_y = state_y[registered_variables.momentum_index.y]
        magnetic_field_x = state_y[registered_variables.magnetic_index.x]
        magnetic_field_y = state_y[registered_variables.magnetic_index.y]
        state_y = state_y.at[registered_variables.momentum_index.x].set(momentum_y)
        state_y = state_y.at[registered_variables.momentum_index.y].set(momentum_x)
        state_y = state_y.at[registered_variables.magnetic_index.x].set(magnetic_field_y)
        state_y = state_y.at[registered_variables.magnetic_index.y].set(magnetic_field_x)

        lambda_y = maximum_signal_speed(state_y)
    else:
        lambda_y = 0.0

    if config.dimensionality == 3:
        state_z = jnp.transpose(conserved_state, (0, 3, 2, 1))

        # Swap the x and z vector components so z becomes the sweep direction.
        momentum_x = state_z[registered_variables.momentum_index.x]
        momentum_z = state_z[registered_variables.momentum_index.z]
        magnetic_field_x = state_z[registered_variables.magnetic_index.x]
        magnetic_field_z = state_z[registered_variables.magnetic_index.z]
        state_z = state_z.at[registered_variables.momentum_index.x].set(momentum_z)
        state_z = state_z.at[registered_variables.momentum_index.z].set(momentum_x)
        state_z = state_z.at[registered_variables.magnetic_index.x].set(magnetic_field_z)
        state_z = state_z.at[registered_variables.magnetic_index.z].set(magnetic_field_x)

        lambda_z = maximum_signal_speed(state_z)
    else:
        lambda_z = 0.0

    dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)

    dt_cfl = _apply_parabolic_and_source_limits(
        dt_cfl,
        primitive_state,
        grid_spacing,
        gamma,
        config,
        params,
        registered_variables,
        C_CFL,
    )

    dt_cfl = jnp.minimum(dt_cfl, dt_max)

    return dt_cfl


# -------------------------------------------------------------
# =================== ↑ MHD CFL estimators ↑ ==================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================== ↓ Hydro CFL estimators ↓ =================
# -------------------------------------------------------------


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _cfl_time_step_fd_hydro_native(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt_max: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    C_CFL: Union[float, Float[Array, ""]] = 0.8,
) -> Float[Array, ""]:
    """
    Compute the hydrodynamic CFL time step from the full characteristic
    eigenvalues.

    Mirrors the MHD estimator: for each axis the conserved state is permuted so
    that axis becomes the sweep direction, the hydro eigenvalue stack is
    evaluated, and the largest absolute eigenvalue gives the local signal speed.
    The advective limit is combined with the optional parabolic and
    explicit-cooling limits. This is the native fallback used when the Pallas
    fast path is not available.

    Args:
        primitive_state: The primitive state array.
        grid_spacing: The grid spacing.
        dt_max: The maximum allowed time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        C_CFL: The CFL safety factor.

    Returns:
        The CFL-limited time step.
    """
    if config.equation_of_state == IDEAL_GAS:
        conserved_state = conserved_state_from_primitive(
            primitive_state, gamma, config, registered_variables
        )
    elif config.equation_of_state == ISOTHERMAL:
        conserved_state = conserved_state_from_primitive_isothermal(
            primitive_state, config, registered_variables
        )

    def maximum_signal_speed(swept_state):
        if config.equation_of_state == IDEAL_GAS:
            eigenvalues = _eigen_all_lambdas_hydro(
                swept_state,
                params.minimum_density,
                params.minimum_pressure,
                gamma,
                config,
                registered_variables,
            )
        elif config.equation_of_state == ISOTHERMAL:
            eigenvalues = _eigen_all_lambdas_hydro_iso(
                swept_state,
                params.minimum_density,
                params.isothermal_sound_speed,
                config,
                registered_variables,
            )
        return jnp.max(jnp.abs(eigenvalues))

    lambda_x = maximum_signal_speed(conserved_state)

    if config.dimensionality >= 2:
        if config.dimensionality == 2:
            state_y = jnp.transpose(conserved_state, (0, 2, 1))
        else:
            state_y = jnp.transpose(conserved_state, (0, 2, 1, 3))

        # Swap the x and y momenta so y becomes the sweep direction.
        momentum_x = state_y[registered_variables.momentum_index.x]
        momentum_y = state_y[registered_variables.momentum_index.y]
        state_y = state_y.at[registered_variables.momentum_index.x].set(momentum_y)
        state_y = state_y.at[registered_variables.momentum_index.y].set(momentum_x)

        lambda_y = maximum_signal_speed(state_y)
    else:
        lambda_y = 0.0

    if config.dimensionality == 3:
        state_z = jnp.transpose(conserved_state, (0, 3, 2, 1))

        # Swap the x and z momenta so z becomes the sweep direction.
        momentum_x = state_z[registered_variables.momentum_index.x]
        momentum_z = state_z[registered_variables.momentum_index.z]
        state_z = state_z.at[registered_variables.momentum_index.x].set(momentum_z)
        state_z = state_z.at[registered_variables.momentum_index.z].set(momentum_x)

        lambda_z = maximum_signal_speed(state_z)
    else:
        lambda_z = 0.0

    dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)
    dt_cfl = jnp.minimum(dt_cfl, dt_max)

    dt_cfl = _apply_parabolic_and_source_limits(
        dt_cfl,
        primitive_state,
        grid_spacing,
        gamma,
        config,
        params,
        registered_variables,
        C_CFL,
    )

    return dt_cfl


def _hydro_fast_cfl_supported(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> bool:
    """Whether the lower-storage hydro fast-CFL estimator can be used: the
    Pallas backend must be on and the registry must expose the velocity index
    (plus the pressure index for an ideal gas)."""
    if not _backend_is_pallas(config):
        return False
    if not hasattr(registered_variables, "velocity_index"):
        return False
    has_pressure = hasattr(registered_variables, "pressure_index")
    if config.equation_of_state == IDEAL_GAS and not has_pressure:
        return False
    return True


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _cfl_time_step_fd_hydro_fast(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt_max: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    C_CFL: Union[float, Float[Array, ""]] = 0.8,
) -> Float[Array, ""]:
    """
    Lower-storage hydro CFL estimator used by the Pallas backend.

    It computes max(|v_d| + c) directly from the primitive variables rather than
    materialising all characteristic eigenvalue arrays. For the Euler equations
    this is the same advective limit.

    Args:
        primitive_state: The primitive state array.
        grid_spacing: The grid spacing.
        dt_max: The maximum allowed time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        C_CFL: The CFL safety factor.

    Returns:
        The CFL-limited time step.
    """
    density = primitive_state[registered_variables.density_index]

    if config.dimensionality == 1:
        velocity_x = primitive_state[registered_variables.velocity_index]
        velocity_y = 0.0
        velocity_z = 0.0
    else:
        velocity_x = primitive_state[registered_variables.velocity_index.x]
        velocity_y = primitive_state[registered_variables.velocity_index.y]
        if config.dimensionality == 3:
            velocity_z = primitive_state[registered_variables.velocity_index.z]
        else:
            velocity_z = 0.0

    if config.equation_of_state == IDEAL_GAS:
        pressure = primitive_state[registered_variables.pressure_index]
        if config.positivity_config.clamp_in_estimates:
            density = jnp.maximum(density, params.minimum_density)
            pressure = jnp.maximum(pressure, params.minimum_pressure)
        sound_speed = jnp.sqrt(jnp.maximum(gamma * pressure / density, 1e-12))
    elif config.equation_of_state == ISOTHERMAL:
        sound_speed = params.isothermal_sound_speed

    lambda_x = jnp.max(jnp.abs(velocity_x) + sound_speed)
    if config.dimensionality >= 2:
        lambda_y = jnp.max(jnp.abs(velocity_y) + sound_speed)
    else:
        lambda_y = 0.0
    if config.dimensionality == 3:
        lambda_z = jnp.max(jnp.abs(velocity_z) + sound_speed)
    else:
        lambda_z = 0.0

    dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)
    dt_cfl = jnp.minimum(dt_cfl, dt_max)

    dt_cfl = _apply_parabolic_and_source_limits(
        dt_cfl,
        primitive_state,
        grid_spacing,
        gamma,
        config,
        params,
        registered_variables,
        C_CFL,
    )

    return dt_cfl


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _cfl_time_step_fd_hydro(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt_max: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    C_CFL: Union[float, Float[Array, ""]] = 0.8,
) -> Float[Array, ""]:
    """
    Backend-aware hydrodynamic CFL time step.

    Dispatches to the lower-storage Pallas fast estimator when it is supported
    and otherwise to the full eigenvalue-based native estimator; both return the
    same advective limit.

    Args:
        primitive_state: The primitive state array.
        grid_spacing: The grid spacing.
        dt_max: The maximum allowed time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        C_CFL: The CFL safety factor.

    Returns:
        The CFL-limited time step.
    """
    if _hydro_fast_cfl_supported(config, registered_variables):
        return _cfl_time_step_fd_hydro_fast(
            primitive_state,
            grid_spacing,
            dt_max,
            gamma,
            config,
            params,
            registered_variables,
            C_CFL,
        )
    return _cfl_time_step_fd_hydro_native(
        primitive_state,
        grid_spacing,
        dt_max,
        gamma,
        config,
        params,
        registered_variables,
        C_CFL,
    )


# -------------------------------------------------------------
# ================== ↑ Hydro CFL estimators ↑ =================
# -------------------------------------------------------------
