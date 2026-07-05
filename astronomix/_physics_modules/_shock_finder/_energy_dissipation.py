from functools import partial

import jax
import jax.numpy as jnp

from astronomix._physics_modules._shock_finder._shock_zones import (
    get_post_pre_shock_values,
)


def _thermalization_efficiency(mach, gamma):
    """
    Calculate the thermalization efficiency δ(M).

    The efficiency gives the fraction of the incoming kinetic-energy flux
    that is irreversibly converted into thermal energy at the shock.

    Based on equations (8)–(9) of Schaal & Springel:
    https://arxiv.org/pdf/1407.4117

    Args:
        mach:
            Shock Mach-number field. Mach is normally zero outside detected
            shock-surface cells.

        gamma:
            Adiabatic index of the gas, usually 5/3.

    Returns:
        Thermalization-efficiency field with the same shape as mach.
        Cells with Mach <= 1 are assigned zero efficiency.
    """

    # The efficiency equation contains M² in the denominator.
    # Mach is zero outside shock-surface cells, so temporarily replace
    # values below 1 with 1 to avoid division by zero.
    mach_safe = jnp.maximum(mach, 1.0)
    mach2 = mach_safe**2

    # Rankine-Hugoniot density compression ratio:
    #     R = rho_post / rho_pre
    density_jump = ((gamma + 1.0) * mach2 / ((gamma - 1.0) * mach2 + 2.0)
    )

    # Rankine-Hugoniot pressure ratio:
    #     P_post / P_pre
    pressure_jump = (2.0 * gamma * mach2 - (gamma - 1.0)) / (gamma + 1.0)

    # Fraction of incoming kinetic-energy flux converted into
    # irreversible thermal energy.
    efficiency = (2.0 / (gamma * (gamma - 1.0) * mach2 * density_jump) * (pressure_jump - density_jump**gamma))

    # mach_safe was only used to evaluate the formula safely.
    # Use the original Mach field to remove subsonic and non-shock cells.
    return jnp.where(
        mach > 1.0,
        jnp.maximum(efficiency, 0.0),
        0.0,
    )


@partial(
    jax.jit,
    static_argnames=[
        "config",
        "registered_variables",
        "gamma_gas",
        "sampling_steps",
    ],
)
def calculate_thermal_energy_flux(
    primitive_state,
    shock_surface,
    shock_direction,
    mach_numbers,
    config,
    registered_variables,
    gamma_gas=5.0 / 3.0,
    sampling_steps=8,
):
    """
    Calculate the dissipated thermal-energy flux at shock-surface cells.

    The calculation is:
        c_pre = sqrt(gamma * P_pre / rho_pre)
        v_pre = M * c_pre
        f_kin = 0.5 * rho_pre * v_pre^3
        f_th = delta(M) * f_kin

    where:
        c_pre is the pre-shock sound speed,
        v_pre is the upstream velocity in the shock rest frame,
        f_kin is the incoming kinetic-energy flux,
        delta(M) is the thermalization efficiency,
        f_th is the dissipated thermal-energy flux.

    Args:
        primitive_state:
            Primitive-variable array containing pressure, density,
            velocity components, and other registered variables.

        shock_surface:
            Boolean mask indicating the location of shock surfaces.

        shock_direction:
            Vector indicating the direction of the shock.

        mach_numbers:
            Array of Mach numbers at each grid point.

        config:
            Simulation configuration. Currently retained for consistency
            with the shock-finder API.

        registered_variables:
            Registry containing the pressure and density indices.

        gamma_gas:
            Adiabatic index of the gas. Default is 5/3.
        
        sampling_steps:
            Number of steps to sample pre/post-shock values along the shock direction.

    Returns:
        Thermal-energy flux with units of energy / area / time.
        Values are nonzero only on detected shock-surface cells.
    """

    # Extract pressure and density from the primitive state.
    pressure = primitive_state[
        registered_variables.pressure_index
    ]

    density = primitive_state[
        registered_variables.density_index
    ]

    # Sample pressure and density on both sides of the shock.
    # Only the pre-shock values are needed here, so the post-shock
    # outputs are ignored using "_".
    _, pressure_pre, _, density_pre = get_post_pre_shock_values(
        shock_direction,
        pressure,
        density,
        max_steps=sampling_steps,
    )

    # Avoid invalid sound-speed calculations if numerical noise produces
    # zero or slightly negative pressure or density.
    numerical_floor = 1e-30

    pressure_pre = jnp.maximum(
        pressure_pre,
        numerical_floor,
    )

    density_pre = jnp.maximum(
        density_pre,
        numerical_floor,
    )

    # Pre-shock adiabatic sound speed:
    #     c_pre = sqrt(gamma * P_pre / rho_pre)
    sound_speed_pre = jnp.sqrt(gamma_gas * pressure_pre/ density_pre)

    # Upstream velocity relative to the shock:
    #     v_pre = M * c_pre
    velocity_pre = mach_numbers * sound_speed_pre

    # Incoming kinetic-energy flux:
    #     f_kin = 1/2 * rho_pre * v_pre³
    kinetic_energy_flux = (0.5 * density_pre * velocity_pre**3)

    # Fraction of the kinetic-energy flux that is converted into heat.
    efficiency = _thermalization_efficiency(mach_numbers, gamma_gas)

    # Dissipated thermal-energy flux:
    #     f_th = delta(M) * f_kin
    thermal_energy_flux = (efficiency * kinetic_energy_flux)

    # Pre/post sampling uses max_steps=8, so values within 8 cells
    # of a boundary are not reliable because jnp.roll wraps around.
    margin = sampling_steps
    valid_interior = jnp.zeros_like(shock_surface, dtype=jnp.bool_)

    interior_slices = tuple(
        slice(margin, -margin) for _ in range(shock_surface.ndim)
    )

    valid_interior = valid_interior.at[interior_slices].set(True)

    return jnp.where(
        shock_surface & valid_interior,
        thermal_energy_flux,
        0.0,
    )