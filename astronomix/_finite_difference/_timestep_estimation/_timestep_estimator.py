"""
CFL time-step estimators for the finite-difference solver.

Provides the advective (and, when active, viscous and conductive) CFL time-step
limits for hydrodynamics and MHD. Each equation set has a full
characteristic-eigenvalue estimator plus a lower-storage "fast" estimator used
by the Pallas backend that reaches the same advective limit directly from the
primitive variables, avoiding the materialisation of the full eigenvalue stack.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    DYNAMIC_VISCOSITY,
    IDEAL_GAS,
    ISOTHERMAL,
    KINEMATIC_VISCOSITY,
    PALLAS,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
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
    primitive_state_from_conserved_mhd,
)


def _mhd_fast_cfl_supported(config: SimulationConfig, registered_variables: RegisteredVariables) -> bool:
    """The MHD fast-CFL helper below skips the full 7-eigenvalue stack
    and computes ``max(|v_d| + c_fast_d)`` per cell directly from
    primitives.  Same arithmetic as ``_cfl_time_step_fd`` but no
    full-state characteristic-eigenvalue intermediate.  Available
    whenever the Pallas backend is on and the registry exposes the
    velocity/magnetic indices."""
    if config.backend != PALLAS:
        return False
    if not config.mhd:
        return False
    if not hasattr(registered_variables, "velocity_index"):
        return False
    if not hasattr(registered_variables, "magnetic_index"):
        return False
    if config.equation_of_state == IDEAL_GAS and not hasattr(registered_variables, "pressure_index"):
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
    """Per-cell fast-magnetosonic CFL for MHD, mirroring the hydro fast
    path.  ``c_fast_d² = 0.5·(b²/ρ + c_s² + √((b²/ρ + c_s²)² − 4·(B_d²/ρ)·c_s²))``
    is computed pointwise; the reduction is then ``max(|v_d| + c_fast_d)``
    per axis.  No full-state characteristic-eigenvalue array is ever
    materialised.
    """
    rho = primitive_state[registered_variables.density_index]
    vx = primitive_state[registered_variables.velocity_index.x]
    vy = primitive_state[registered_variables.velocity_index.y] if config.dimensionality >= 2 else 0.0
    vz = primitive_state[registered_variables.velocity_index.z] if config.dimensionality == 3 else 0.0
    Bx = primitive_state[registered_variables.magnetic_index.x]
    By = primitive_state[registered_variables.magnetic_index.y]
    Bz = primitive_state[registered_variables.magnetic_index.z]

    if config.equation_of_state == IDEAL_GAS:
        pressure = primitive_state[registered_variables.pressure_index]
        if config.positivity_config.clamp_in_estimates:
            rho = jnp.maximum(rho, params.minimum_density)
            pressure = jnp.maximum(pressure, params.minimum_pressure)
        cs2 = jnp.maximum(gamma * pressure / rho, 1e-12)
    else:  # ISOTHERMAL
        if config.positivity_config.clamp_in_estimates:
            rho = jnp.maximum(rho, params.minimum_density)
        cs2 = jnp.full_like(rho, params.isothermal_sound_speed ** 2)

    b2 = Bx * Bx + By * By + Bz * Bz
    b2_over_rho = b2 / rho

    def cfast(B_axis):
        bn2_over_rho = (B_axis * B_axis) / rho
        disc = jnp.maximum(
            (b2_over_rho + cs2) ** 2 - 4.0 * bn2_over_rho * cs2,
            0.0,
        )
        return jnp.sqrt(jnp.maximum(0.5 * (b2_over_rho + cs2 + jnp.sqrt(disc)), 0.0))

    lambda_x = jnp.max(jnp.abs(vx) + cfast(Bx))
    lambda_y = jnp.max(jnp.abs(vy) + cfast(By)) if config.dimensionality >= 2 else 0.0
    lambda_z = jnp.max(jnp.abs(vz) + cfast(Bz)) if config.dimensionality == 3 else 0.0

    dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)

    if config.diffusion:
        if config.positivity_config.clamp_in_estimates:
            rho_min = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min = jnp.min(primitive_state[registered_variables.density_index])
        if config.viscosity_type == DYNAMIC_VISCOSITY:
            nu_max = params.viscosity / rho_min
        elif config.viscosity_type == KINEMATIC_VISCOSITY:
            nu_max = params.viscosity
        dt_visc = C_CFL * grid_spacing ** 2 / (2.0 * config.dimensionality * nu_max)
        dt_cfl = jnp.minimum(dt_cfl, dt_visc)

    # conductive (parabolic) time step constraint
    if config.thermal_conduction:
        if config.positivity_config.clamp_in_estimates:
            rho_min_c = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min_c = jnp.min(primitive_state[registered_variables.density_index])
        # thermal diffusivity of the internal energy: chi = (gamma - 1) kappa / rho
        chi_max = (gamma - 1.0) * params.thermal_conductivity / rho_min_c
        dt_cond = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * chi_max)
        dt_cfl = jnp.minimum(dt_cfl, dt_cond)

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
    is then combined with the optional viscous and conductive limits. When the
    Pallas fast path is available, the equivalent lower-storage estimator is
    used instead.

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
            primitive_state, grid_spacing, dt_max, gamma,
            config, params, registered_variables, C_CFL,
        )

    if config.equation_of_state == IDEAL_GAS:
        conserved_state = conserved_state_from_primitive_mhd(
            primitive_state, gamma, registered_variables
        )
    elif config.equation_of_state == ISOTHERMAL:
        conserved_state = conserved_state_from_primitive_isothermal(
            primitive_state, config, registered_variables
        )

    if config.equation_of_state == IDEAL_GAS:
        lambda_x = _eigen_all_lambdas(
            conserved_state, params.minimum_density, params.minimum_pressure, gamma, registered_variables
        )
    elif config.equation_of_state == ISOTHERMAL:
        lambda_x = _eigen_all_lambdas_iso(
            conserved_state, params.minimum_density, params.isothermal_sound_speed, registered_variables
        )

    lambda_x = jnp.max(jnp.abs(lambda_x))

    if config.dimensionality >= 2:
        if config.dimensionality == 2:
            qy = jnp.transpose(conserved_state, (0, 2, 1))
        else:
            qy = jnp.transpose(conserved_state, (0, 2, 1, 3))

        momentum_x = qy[registered_variables.momentum_index.x]
        momentum_y = qy[registered_variables.momentum_index.y]
        B_x = qy[registered_variables.magnetic_index.x]
        B_y = qy[registered_variables.magnetic_index.y]
        qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
        qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)
        qy = qy.at[registered_variables.magnetic_index.x].set(B_y)
        qy = qy.at[registered_variables.magnetic_index.y].set(B_x)

        if config.equation_of_state == IDEAL_GAS:
            lambda_y = _eigen_all_lambdas(
                qy, params.minimum_density, params.minimum_pressure, gamma, registered_variables
            )
        elif config.equation_of_state == ISOTHERMAL:
            lambda_y = _eigen_all_lambdas_iso(
                qy, params.minimum_density, params.isothermal_sound_speed, registered_variables
            )
        lambda_y = jnp.max(jnp.abs(lambda_y))
    else:
        lambda_y = 0.0

    if config.dimensionality == 3:
        qz = jnp.transpose(conserved_state, (0, 3, 2, 1))

        momentum_x = qz[registered_variables.momentum_index.x]
        momentum_z = qz[registered_variables.momentum_index.z]
        B_x = qz[registered_variables.magnetic_index.x]
        B_z = qz[registered_variables.magnetic_index.z]
        qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
        qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)
        qz = qz.at[registered_variables.magnetic_index.x].set(B_z)
        qz = qz.at[registered_variables.magnetic_index.z].set(B_x)

        if config.equation_of_state == IDEAL_GAS:
            lambda_z = _eigen_all_lambdas(
                qz, params.minimum_density, params.minimum_pressure, gamma, registered_variables
            )
        elif config.equation_of_state == ISOTHERMAL:
            lambda_z = _eigen_all_lambdas_iso(
                qz, params.minimum_density, params.isothermal_sound_speed, registered_variables
            )
        lambda_z = jnp.max(jnp.abs(lambda_z))
    else:
        lambda_z = 0.0

    dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)

    # viscous time step constraint
    if config.diffusion:
        if config.positivity_config.clamp_in_estimates:
            rho_min = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min = jnp.min(primitive_state[registered_variables.density_index])

        if config.viscosity_type == DYNAMIC_VISCOSITY:
            nu_max = params.viscosity / rho_min
        elif config.viscosity_type == KINEMATIC_VISCOSITY:
            nu_max = params.viscosity

        dt_visc = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * nu_max)
        dt_cfl = jnp.minimum(dt_cfl, dt_visc)

    # conductive (parabolic) time step constraint
    if config.thermal_conduction:
        if config.positivity_config.clamp_in_estimates:
            rho_min_c = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min_c = jnp.min(primitive_state[registered_variables.density_index])
        # thermal diffusivity of the internal energy: chi = (gamma - 1) kappa / rho
        chi_max = (gamma - 1.0) * params.thermal_conductivity / rho_min_c
        dt_cond = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * chi_max)
        dt_cfl = jnp.minimum(dt_cfl, dt_cond)

    dt_cfl = jnp.minimum(dt_cfl, dt_max)

    return dt_cfl


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
    The advective limit is combined with the optional viscous and conductive
    limits. This is the native fallback used when the Pallas fast path is not
    available.

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

    if config.equation_of_state == IDEAL_GAS:
        lambda_x = _eigen_all_lambdas_hydro(
            conserved_state, params.minimum_density, params.minimum_pressure, gamma, config, registered_variables
        )
    elif config.equation_of_state == ISOTHERMAL:
        lambda_x = _eigen_all_lambdas_hydro_iso(
            conserved_state, params.minimum_density, params.isothermal_sound_speed, config, registered_variables
        )
    lambda_x = jnp.max(jnp.abs(lambda_x))

    if config.dimensionality >= 2:

        if config.dimensionality == 2:
            qy = jnp.transpose(conserved_state, (0, 2, 1))
        else:
            qy = jnp.transpose(conserved_state, (0, 2, 1, 3))
        momentum_x = qy[registered_variables.momentum_index.x]
        momentum_y = qy[registered_variables.momentum_index.y]
        qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
        qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)

        if config.equation_of_state == IDEAL_GAS:
            lambda_y = _eigen_all_lambdas_hydro(
                qy, params.minimum_density, params.minimum_pressure, gamma, config, registered_variables
            )
        elif config.equation_of_state == ISOTHERMAL:
            lambda_y = _eigen_all_lambdas_hydro_iso(
                qy, params.minimum_density, params.isothermal_sound_speed, config, registered_variables
            )

        lambda_y = jnp.max(jnp.abs(lambda_y))
    else:
        lambda_y = 0.0

    if config.dimensionality == 3:
        qz = jnp.transpose(conserved_state, (0, 3, 2, 1))
        momentum_x = qz[registered_variables.momentum_index.x]
        momentum_z = qz[registered_variables.momentum_index.z]
        qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
        qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)

        if config.equation_of_state == IDEAL_GAS:
            lambda_z = _eigen_all_lambdas_hydro(
                qz, params.minimum_density, params.minimum_pressure, gamma, config, registered_variables
            )
        elif config.equation_of_state == ISOTHERMAL:
            lambda_z = _eigen_all_lambdas_hydro_iso(
                qz, params.minimum_density, params.isothermal_sound_speed, config, registered_variables
            )

        lambda_z = jnp.max(jnp.abs(lambda_z))
    else:
        lambda_z = 0.0

    dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)
    dt_cfl = jnp.minimum(dt_cfl, dt_max)

    # viscous time step constraint
    if config.diffusion:

        if config.positivity_config.clamp_in_estimates:
            rho_min = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min = jnp.min(primitive_state[registered_variables.density_index])

        if config.viscosity_type == DYNAMIC_VISCOSITY:
            nu_max = params.viscosity / rho_min
        elif config.viscosity_type == KINEMATIC_VISCOSITY:
            nu_max = params.viscosity

        dt_visc = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * nu_max)
        dt_cfl = jnp.minimum(dt_cfl, dt_visc)

    # conductive (parabolic) time step constraint
    if config.thermal_conduction:
        if config.positivity_config.clamp_in_estimates:
            rho_min_c = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min_c = jnp.min(primitive_state[registered_variables.density_index])
        # thermal diffusivity of the internal energy: chi = (gamma - 1) kappa / rho
        chi_max = (gamma - 1.0) * params.thermal_conductivity / rho_min_c
        dt_cond = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * chi_max)
        dt_cfl = jnp.minimum(dt_cfl, dt_cond)

    return dt_cfl


# -----------------------------------------------------------------------------
# Backend-aware hydrodynamic CFL wrapper.
# -----------------------------------------------------------------------------


def _backend_name(config: SimulationConfig) -> str:
    """Return the configured backend name as an upper-case string, tolerating
    both enum-like (``.name`` / ``.value``) and plain values."""
    backend = config.backend
    name = getattr(backend, "name", None)
    if name is not None:
        return str(name).upper()
    value = getattr(backend, "value", None)
    if isinstance(value, str):
        return value.upper()
    return str(backend).upper()


def _backend_is_pallas(config: SimulationConfig) -> bool:
    """Whether the Pallas backend is selected."""
    return config.backend == PALLAS


def _hydro_fast_cfl_supported(config: SimulationConfig, registered_variables: RegisteredVariables) -> bool:
    """Whether the lower-storage hydro fast-CFL estimator can be used: the
    Pallas backend must be on and the registry must expose the velocity index
    (plus the pressure index for an ideal gas)."""
    if not _backend_is_pallas(config):
        return False
    if not hasattr(registered_variables, "velocity_index"):
        return False
    if config.equation_of_state == IDEAL_GAS and not hasattr(registered_variables, "pressure_index"):
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
    """Lower-storage hydro CFL estimator used by the Pallas backend.

    It computes max(|v_d| + c) directly from primitive variables rather than
    materialising all characteristic eigenvalue arrays.  The formula is
    equivalent for Euler hydro.
    """
    rho = primitive_state[registered_variables.density_index]

    if config.dimensionality == 1:
        vx = primitive_state[registered_variables.velocity_index]
        vy = 0.0
        vz = 0.0
    else:
        vx = primitive_state[registered_variables.velocity_index.x]
        if config.dimensionality >= 2:
            vy = primitive_state[registered_variables.velocity_index.y]
        else:
            vy = 0.0
        if config.dimensionality == 3:
            vz = primitive_state[registered_variables.velocity_index.z]
        else:
            vz = 0.0

    if config.equation_of_state == IDEAL_GAS:
        pressure = primitive_state[registered_variables.pressure_index]
        if config.positivity_config.clamp_in_estimates:
            rho = jnp.maximum(rho, params.minimum_density)
            pressure = jnp.maximum(pressure, params.minimum_pressure)
        sound_speed = jnp.sqrt(jnp.maximum(gamma * pressure / rho, 1e-12))
    elif config.equation_of_state == ISOTHERMAL:
        sound_speed = params.isothermal_sound_speed

    lambda_x = jnp.max(jnp.abs(vx) + sound_speed)
    if config.dimensionality >= 2:
        lambda_y = jnp.max(jnp.abs(vy) + sound_speed)
    else:
        lambda_y = 0.0
    if config.dimensionality == 3:
        lambda_z = jnp.max(jnp.abs(vz) + sound_speed)
    else:
        lambda_z = 0.0

    dt_cfl = C_CFL * grid_spacing / (lambda_x + lambda_y + lambda_z)
    dt_cfl = jnp.minimum(dt_cfl, dt_max)

    if config.diffusion:
        if config.positivity_config.clamp_in_estimates:
            rho_min = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min = jnp.min(primitive_state[registered_variables.density_index])

        if config.viscosity_type == DYNAMIC_VISCOSITY:
            nu_max = params.viscosity / rho_min
        elif config.viscosity_type == KINEMATIC_VISCOSITY:
            nu_max = params.viscosity

        dt_visc = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * nu_max)
        dt_cfl = jnp.minimum(dt_cfl, dt_visc)

    # conductive (parabolic) time step constraint
    if config.thermal_conduction:
        if config.positivity_config.clamp_in_estimates:
            rho_min_c = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min_c = jnp.min(primitive_state[registered_variables.density_index])
        # thermal diffusivity of the internal energy: chi = (gamma - 1) kappa / rho
        chi_max = (gamma - 1.0) * params.thermal_conductivity / rho_min_c
        dt_cond = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * chi_max)
        dt_cfl = jnp.minimum(dt_cfl, dt_cond)

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
