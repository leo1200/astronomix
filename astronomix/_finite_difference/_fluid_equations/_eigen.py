"""
Computations of the eigenvalues and eigenvectors for the MHD equations.

The eigenstructure was extracted from the HOW-MHD Fortran code, with altered
variable names for clarity and altered numerical safeguards.

NOTE: Problems for differentiation largely follow from square roots and divisions:
The derivative of sqrt(x) is 1/(2*sqrt(x)) and of 1/x is -1/x^2, where both expressions
are problematic for small x, especially when multiplying gradients in the backward pass,
-> exploding gradients.
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Union

from astronomix.variable_registry.registered_variables import RegisteredVariables


def diff_safe_sqrt(x):
    epsilon = 1e-12
    x_safe = jnp.maximum(x, epsilon)
    return jnp.sqrt(x_safe)


@partial(jax.jit, static_argnames=["registered_variables"])
def _eigenvalue_building_blocks(
    conserved_state,
    gamma,
    rhomin,
    pgmin,
    registered_variables: RegisteredVariables,
):
    # unpack the conserved variables
    density = conserved_state[registered_variables.density_index]
    momentum_x = conserved_state[registered_variables.momentum_index.x]
    momentum_y = conserved_state[registered_variables.momentum_index.y]
    momentum_z = conserved_state[registered_variables.momentum_index.z]
    magnetic_x = conserved_state[registered_variables.magnetic_index.x]
    magnetic_y = conserved_state[registered_variables.magnetic_index.y]
    magnetic_z = conserved_state[registered_variables.magnetic_index.z]
    energy = conserved_state[registered_variables.energy_index]

    # compute primitives
    rho = density
    velocity_x = momentum_x / rho
    velocity_y = momentum_y / rho
    velocity_z = momentum_z / rho
    velocity_squared = (
        velocity_x * velocity_x + velocity_y * velocity_y + velocity_z * velocity_z
    )
    magnetic_field_squared = (
        magnetic_x * magnetic_x + magnetic_y * magnetic_y + magnetic_z * magnetic_z
    )
    gas_pressure = (gamma - 1.0) * (
        energy - 0.5 * (rho * velocity_squared + magnetic_field_squared)
    )

    # redefine the density and pressure, and energy based on floors
    rho = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin), jnp.maximum(rho, rhomin), rho
    )
    gas_pressure = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin),
        jnp.maximum(gas_pressure, pgmin),
        gas_pressure,
    )
    energy = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin),
        gas_pressure / (gamma - 1.0)
        + 0.5 * (rho * velocity_squared + magnetic_field_squared),
        energy,
    )

    # compute derived quantities
    sound_speed_sq = jnp.maximum(0.0, gamma * jnp.abs(gas_pressure / rho))
    magnetosonic_discriminant_root = diff_safe_sqrt(
        jnp.maximum(
            0.0,
            (magnetic_field_squared / rho + sound_speed_sq) ** 2
            - 4.0 * (magnetic_x * magnetic_x) / rho * sound_speed_sq,
        )
    )
    fast_magnetosonic_velocity = diff_safe_sqrt(
        jnp.maximum(
            0.0,
            0.5
            * (
                magnetic_field_squared / rho
                + sound_speed_sq
                + magnetosonic_discriminant_root
            ),
        )
    )
    alfven_velocity = diff_safe_sqrt(jnp.maximum(0.0, (magnetic_x * magnetic_x) / rho))
    slow_magnetosonic_velocity = diff_safe_sqrt(
        jnp.maximum(
            0.0,
            0.5
            * (
                magnetic_field_squared / rho
                + sound_speed_sq
                - magnetosonic_discriminant_root
            ),
        )
    )

    return (
        velocity_x,
        fast_magnetosonic_velocity,
        alfven_velocity,
        slow_magnetosonic_velocity,
    )

@partial(jax.jit, static_argnames=["registered_variables"])
def _eigenvector_building_blocks(
    conserved_state,
    gamma,
    rhomin,
    pgmin,
    registered_variables: RegisteredVariables,
):
    # unpack conserved variables
    rho = conserved_state[registered_variables.density_index]  # density
    momentum_x = conserved_state[registered_variables.momentum_index.x]
    momentum_y = conserved_state[registered_variables.momentum_index.y]
    momentum_z = conserved_state[registered_variables.momentum_index.z]
    magnetic_x = conserved_state[registered_variables.magnetic_index.x]
    magnetic_y = conserved_state[registered_variables.magnetic_index.y]
    magnetic_z = conserved_state[registered_variables.magnetic_index.z]
    energy = conserved_state[registered_variables.energy_index]

    # compute primitives
    rho = rho
    velocity_x = momentum_x / rho
    velocity_y = momentum_y / rho
    velocity_z = momentum_z / rho
    velocity_sq = (
        velocity_x * velocity_x + velocity_y * velocity_y + velocity_z * velocity_z
    )
    magnetic_sq = (
        magnetic_x * magnetic_x + magnetic_y * magnetic_y + magnetic_z * magnetic_z
    )
    gas_pressure = (gamma - 1.0) * (energy - 0.5 * (rho * velocity_sq + magnetic_sq))

    # redefine the density and pressure, and energy based on floors
    rho = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin), jnp.maximum(rho, rhomin), rho
    )
    gas_pressure = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin),
        jnp.maximum(gas_pressure, pgmin),
        gas_pressure,
    )
    energy = jnp.where(
        (rho < rhomin) | (gas_pressure < pgmin),
        gas_pressure / (gamma - 1.0) + 0.5 * (rho * velocity_sq + magnetic_sq),
        energy,
    )

    specific_enthalpy = (energy + gas_pressure) / rho

    # periodic average to interfaces
    def avg_x(arr):
        return 0.5 * (arr + jnp.roll(arr, shift=-1, axis=0))

    rho_interface = avg_x(rho)
    velocity_x_interface = avg_x(velocity_x)
    velocity_y_interface = avg_x(velocity_y)
    velocity_z_interface = avg_x(velocity_z)
    magnetic_x_interface = avg_x(magnetic_x)
    magnetic_y_interface = avg_x(magnetic_y)
    magnetic_z_interface = avg_x(magnetic_z)
    specific_enthalpy_interface = avg_x(specific_enthalpy)

    # interface derived quantities
    velocity_sq_interface = (
        velocity_x_interface * velocity_x_interface
        + velocity_y_interface * velocity_y_interface
        + velocity_z_interface * velocity_z_interface
    )
    magnetic_sq_interface = (
        magnetic_x_interface * magnetic_x_interface
        + magnetic_y_interface * magnetic_y_interface
        + magnetic_z_interface * magnetic_z_interface
    )
    b_sq_over_rho_interface = magnetic_sq_interface / rho_interface
    bx_sq_over_rho_interface = (
        magnetic_x_interface * magnetic_x_interface
    ) / rho_interface

    # enthalpy based sound speed at interfaces
    sound_speed_sq_interface = (gamma - 1.0) * (
        specific_enthalpy_interface
        - 0.5 * (velocity_sq_interface + b_sq_over_rho_interface)
    )
    sound_speed_interface = diff_safe_sqrt(jnp.maximum(0.0, sound_speed_sq_interface))

    # calculate the characteristic velocities at the interfaces
    magnetosonic_discriminant_interface = (
        b_sq_over_rho_interface + sound_speed_sq_interface
    ) ** 2 - 4.0 * bx_sq_over_rho_interface * sound_speed_sq_interface
    magnetosonic_discriminant_root_interface = diff_safe_sqrt(
        jnp.maximum(0.0, magnetosonic_discriminant_interface)
    )

    fast_magnetosonic_velocity_interface = diff_safe_sqrt(
        jnp.maximum(
            0.0,
            0.5
            * (
                b_sq_over_rho_interface
                + sound_speed_sq_interface
                + magnetosonic_discriminant_root_interface
            ),
        )
    )
    alfven_velocity_interface = diff_safe_sqrt(
        jnp.maximum(0.0, bx_sq_over_rho_interface)
    )
    slow_magnetosonic_velocity_interface = diff_safe_sqrt(
        jnp.maximum(
            0.0,
            0.5
            * (
                b_sq_over_rho_interface
                + sound_speed_sq_interface
                - magnetosonic_discriminant_root_interface
            ),
        )
    )

    # retrieve tangential magnetic field components
    b_tangential_sq = (
        magnetic_y_interface * magnetic_y_interface
        + magnetic_z_interface * magnetic_z_interface
    )
    sgn_bx = jnp.where(magnetic_x_interface >= 0.0, 1.0, -1.0)

    b_tangential_sq = jnp.maximum(b_tangential_sq, 1.0e-12)

    # B_y / (sqrt(B_y^2 + B_z^2))
    bt_normalized_y = jnp.where(
        b_tangential_sq >= 1.0e-12,
        magnetic_y_interface / diff_safe_sqrt(b_tangential_sq),
        1.0 / diff_safe_sqrt(2.0),
    )
    # B_z / (sqrt(B_y^2 + B_z^2))
    bt_normalized_z = jnp.where(
        b_tangential_sq >= 1.0e-12,
        magnetic_z_interface / diff_safe_sqrt(b_tangential_sq),
        1.0 / diff_safe_sqrt(2.0),
    )

    # fast_mode_weighting = sqrt( c_s^2 − λ_slow^2 ) / sqrt( λ_fast^2 − λ_slow^2 )
    # slow_mode_weighting = sqrt( λ_fast^2 − c_s^2 ) / sqrt( λ_fast^2 − λ_slow^2 )
    # these are designed such that fast_mode_weighting^2 + slow_mode_weighting^2 = 1
    denom = (
        fast_magnetosonic_velocity_interface * fast_magnetosonic_velocity_interface
        - slow_magnetosonic_velocity_interface * slow_magnetosonic_velocity_interface
    )
    denom = jnp.maximum(denom, 1.0e-12)
    fast_mode_weighting = jnp.where(
        denom >= 1.0e-12,
        diff_safe_sqrt(
            jnp.maximum(
                0.0,
                sound_speed_sq_interface
                - slow_magnetosonic_velocity_interface
                * slow_magnetosonic_velocity_interface,
            )
        )
        / diff_safe_sqrt(denom),
        1.0,
    )
    slow_mode_weighting = jnp.where(
        denom >= 1e-12,
        diff_safe_sqrt(
            jnp.maximum(
                0.0,
                fast_magnetosonic_velocity_interface
                * fast_magnetosonic_velocity_interface
                - sound_speed_sq_interface,
            )
        )
        / diff_safe_sqrt(denom),
        1.0,
    )

    sqrt_rho = diff_safe_sqrt(rho_interface)

    gam0 = 1.0 - gamma
    gam1 = 0.5 * (gamma - 1.0)
    gam2 = (gamma - 2.0) / (gamma - 1.0)

    sound_speed_sq_inverse = jnp.where(
        sound_speed_sq_interface > 0.0, 1.0 / sound_speed_sq_interface, 0.0
    )

    sgn_bt = jnp.where(
        magnetic_y_interface != 0.0,
        jnp.where(magnetic_y_interface >= 0.0, 1.0, -1.0),
        jnp.where(magnetic_z_interface >= 0.0, 1.0, -1.0),
    )

    sound_speed_greater_alfven_speed = (
        sound_speed_interface >= alfven_velocity_interface
    )

    return (
        rho_interface,
        sqrt_rho,
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        velocity_sq_interface,
        magnetic_x_interface,
        magnetic_y_interface,
        magnetic_z_interface,
        bt_normalized_y,
        bt_normalized_z,
        sgn_bx,
        sgn_bt,
        sound_speed_interface,
        sound_speed_sq_interface,
        sound_speed_sq_inverse,
        sound_speed_greater_alfven_speed,
        fast_magnetosonic_velocity_interface,
        alfven_velocity_interface,
        slow_magnetosonic_velocity_interface,
        fast_mode_weighting,
        slow_mode_weighting,
        gam0,
        gam1,
        gam2,
    )


@partial(jax.jit, static_argnames=["registered_variables"])
def _eigen_R_col(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
    col: int,
):
    (
        rho_interface,
        sqrt_rho,
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        velocity_sq_interface,
        magnetic_x_interface,
        magnetic_y_interface,
        magnetic_z_interface,
        bt_normalized_y,
        bt_normalized_z,
        sgn_bx,
        sgn_bt,
        sound_speed_interface,
        sound_speed_sq_interface,
        sound_speed_sq_inverse,
        sound_speed_greater_alfven_speed,
        fast_magnetosonic_velocity_interface,
        alfven_velocity_interface,
        slow_magnetosonic_velocity_interface,
        fast_mode_weighting,
        slow_mode_weighting,
        gam0,
        gam1,
        gam2,
    ) = _eigenvector_building_blocks(
        conserved_state,
        gamma,
        rhomin,
        pgmin,
        registered_variables,
    )

    # shorter names for registry indices
    density_index = registered_variables.density_index
    momentum_index_x = registered_variables.momentum_index.x
    momentum_index_y = registered_variables.momentum_index.y
    momentum_index_z = registered_variables.momentum_index.z
    magnetic_index_y = registered_variables.magnetic_index.y
    magnetic_index_z = registered_variables.magnetic_index.z
    energy_index = registered_variables.energy_index

    def col_0():
        # Column 1 (fast -)
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(fast_mode_weighting)
        R = R.at[momentum_index_x].set(
            fast_mode_weighting
            * (velocity_x_interface - fast_magnetosonic_velocity_interface)
        )
        R = R.at[momentum_index_y].set(
            fast_mode_weighting * velocity_y_interface
            + slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        R = R.at[momentum_index_z].set(
            fast_mode_weighting * velocity_z_interface
            + slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        R = R.at[magnetic_index_y].set(
            sound_speed_interface * slow_mode_weighting * bt_normalized_y / sqrt_rho
        )
        R = R.at[magnetic_index_z].set(
            sound_speed_interface * slow_mode_weighting * bt_normalized_z / sqrt_rho
        )
        R = R.at[energy_index].set(
            fast_mode_weighting
            * (
                fast_magnetosonic_velocity_interface**2
                - fast_magnetosonic_velocity_interface * velocity_x_interface
                + 0.5 * velocity_sq_interface
                - gam2 * sound_speed_sq_interface
            )
            + slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        R = jnp.where(~sound_speed_greater_alfven_speed, R * sgn_bt, R)
        return R

    def col_1():
        # Column 2 (alfven -)
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(0.0)
        R = R.at[momentum_index_x].set(0.0)
        R = R.at[momentum_index_y].set(-bt_normalized_z)
        R = R.at[momentum_index_z].set(bt_normalized_y)
        R = R.at[magnetic_index_y].set(-bt_normalized_z * sgn_bx / sqrt_rho)
        R = R.at[magnetic_index_z].set(bt_normalized_y * sgn_bx / sqrt_rho)
        R = R.at[energy_index].set(
            bt_normalized_y * velocity_z_interface
            - bt_normalized_z * velocity_y_interface
        )
        return R

    def col_2():
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(slow_mode_weighting)
        R = R.at[momentum_index_x].set(
            slow_mode_weighting
            * (velocity_x_interface - slow_magnetosonic_velocity_interface)
        )
        R = R.at[momentum_index_y].set(
            slow_mode_weighting * velocity_y_interface
            - fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        R = R.at[momentum_index_z].set(
            slow_mode_weighting * velocity_z_interface
            - fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        R = R.at[magnetic_index_y].set(
            -sound_speed_interface * fast_mode_weighting * bt_normalized_y / sqrt_rho
        )
        R = R.at[magnetic_index_z].set(
            -sound_speed_interface * fast_mode_weighting * bt_normalized_z / sqrt_rho
        )
        R = R.at[energy_index].set(
            slow_mode_weighting
            * (
                slow_magnetosonic_velocity_interface**2
                - slow_magnetosonic_velocity_interface * velocity_x_interface
                + 0.5 * velocity_sq_interface
                - gam2 * sound_speed_sq_interface
            )
            - fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        R = jnp.where(sound_speed_greater_alfven_speed, R * sgn_bt, R)
        return R

    def col_3():
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(1.0)
        R = R.at[momentum_index_x].set(velocity_x_interface)
        R = R.at[momentum_index_y].set(velocity_y_interface)
        R = R.at[momentum_index_z].set(velocity_z_interface)
        R = R.at[magnetic_index_y].set(0.0)
        R = R.at[magnetic_index_z].set(0.0)
        R = R.at[energy_index].set(0.5 * velocity_sq_interface)
        return R

    def col_4():
        # Column 5 (slow +)
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(slow_mode_weighting)
        R = R.at[momentum_index_x].set(
            slow_mode_weighting
            * (velocity_x_interface + slow_magnetosonic_velocity_interface)
        )
        R = R.at[momentum_index_y].set(
            slow_mode_weighting * velocity_y_interface
            + fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        R = R.at[momentum_index_z].set(
            slow_mode_weighting * velocity_z_interface
            + fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        R = R.at[magnetic_index_y].set(
            -sound_speed_interface * fast_mode_weighting * bt_normalized_y / sqrt_rho
        )
        R = R.at[magnetic_index_z].set(
            -sound_speed_interface * fast_mode_weighting * bt_normalized_z / sqrt_rho
        )
        R = R.at[energy_index].set(
            slow_mode_weighting
            * (
                slow_magnetosonic_velocity_interface**2
                + slow_magnetosonic_velocity_interface * velocity_x_interface
                + 0.5 * velocity_sq_interface
                - gam2 * sound_speed_sq_interface
            )
            + fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        R = jnp.where(sound_speed_greater_alfven_speed, R * sgn_bt, R)
        return R

    def col_5():
        # Column 6 (alfven +)
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(0.0)
        R = R.at[momentum_index_x].set(0.0)
        R = R.at[momentum_index_y].set(-bt_normalized_z)
        R = R.at[momentum_index_z].set(bt_normalized_y)
        R = R.at[magnetic_index_y].set(bt_normalized_z * sgn_bx / sqrt_rho)
        R = R.at[magnetic_index_z].set(-bt_normalized_y * sgn_bx / sqrt_rho)
        R = R.at[energy_index].set(
            bt_normalized_y * velocity_z_interface
            - bt_normalized_z * velocity_y_interface
        )
        return R

    def col_6():
        # Column 7 (fast +)
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(fast_mode_weighting)
        R = R.at[momentum_index_x].set(
            fast_mode_weighting
            * (velocity_x_interface + fast_magnetosonic_velocity_interface)
        )
        R = R.at[momentum_index_y].set(
            fast_mode_weighting * velocity_y_interface
            - slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        R = R.at[momentum_index_z].set(
            fast_mode_weighting * velocity_z_interface
            - slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        R = R.at[magnetic_index_y].set(
            sound_speed_interface * slow_mode_weighting * bt_normalized_y / sqrt_rho
        )
        R = R.at[magnetic_index_z].set(
            sound_speed_interface * slow_mode_weighting * bt_normalized_z / sqrt_rho
        )
        R = R.at[energy_index].set(
            fast_mode_weighting
            * (
                fast_magnetosonic_velocity_interface**2
                + fast_magnetosonic_velocity_interface * velocity_x_interface
                + 0.5 * velocity_sq_interface
                - gam2 * sound_speed_sq_interface
            )
            - slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        R = jnp.where(~sound_speed_greater_alfven_speed, R * sgn_bt, R)
        return R

    R = jax.lax.switch(col, [col_0, col_1, col_2, col_3, col_4, col_5, col_6])

    return R


@partial(jax.jit, static_argnames=["registered_variables"])
def _eigen_L_row(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
    row: int,
):
    (
        rho_interface,
        sqrt_rho,
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        velocity_sq_interface,
        magnetic_x_interface,
        magnetic_y_interface,
        magnetic_z_interface,
        bt_normalized_y,
        bt_normalized_z,
        sgn_bx,
        sgn_bt,
        sound_speed_interface,
        sound_speed_sq_interface,
        sound_speed_sq_inverse,
        sound_speed_greater_alfven_speed,
        fast_magnetosonic_velocity_interface,
        alfven_velocity_interface,
        slow_magnetosonic_velocity_interface,
        fast_mode_weighting,
        slow_mode_weighting,
        gam0,
        gam1,
        gam2,
    ) = _eigenvector_building_blocks(
        conserved_state,
        gamma,
        rhomin,
        pgmin,
        registered_variables,
    )

    # shorter names for registry indices
    density_index = registered_variables.density_index
    momentum_index_x = registered_variables.momentum_index.x
    momentum_index_y = registered_variables.momentum_index.y
    momentum_index_z = registered_variables.momentum_index.z
    magnetic_index_y = registered_variables.magnetic_index.y
    magnetic_index_z = registered_variables.magnetic_index.z
    energy_index = registered_variables.energy_index

    def row_0():
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(
            fast_mode_weighting
            * (
                gam1 * velocity_sq_interface
                + fast_magnetosonic_velocity_interface * velocity_x_interface
            )
            - slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        L = L.at[momentum_index_x].set(
            fast_mode_weighting
            * (gam0 * velocity_x_interface - fast_magnetosonic_velocity_interface)
        )
        L = L.at[momentum_index_y].set(
            gam0 * fast_mode_weighting * velocity_y_interface
            + slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        L = L.at[momentum_index_z].set(
            gam0 * fast_mode_weighting * velocity_z_interface
            + slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        L = L.at[magnetic_index_y].set(
            gam0 * fast_mode_weighting * magnetic_y_interface
            + sound_speed_interface * slow_mode_weighting * bt_normalized_y * sqrt_rho
        )
        L = L.at[magnetic_index_z].set(
            gam0 * fast_mode_weighting * magnetic_z_interface
            + sound_speed_interface * slow_mode_weighting * bt_normalized_z * sqrt_rho
        )
        L = L.at[energy_index].set(-gam0 * fast_mode_weighting)
        L = 0.5 * L * sound_speed_sq_inverse
        L = jnp.where(~sound_speed_greater_alfven_speed, L * sgn_bt, L)
        return L

    def row_1():
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(
            bt_normalized_z * velocity_y_interface
            - bt_normalized_y * velocity_z_interface
        )
        L = L.at[momentum_index_x].set(0.0)
        L = L.at[momentum_index_y].set(-bt_normalized_z)
        L = L.at[momentum_index_z].set(bt_normalized_y)
        L = L.at[magnetic_index_y].set(-bt_normalized_z * sgn_bx * sqrt_rho)
        L = L.at[magnetic_index_z].set(bt_normalized_y * sgn_bx * sqrt_rho)
        L = L.at[energy_index].set(0.0)
        L = 0.5 * L
        return L

    def row_2():
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(
            slow_mode_weighting
            * (
                gam1 * velocity_sq_interface
                + slow_magnetosonic_velocity_interface * velocity_x_interface
            )
            + fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        L = L.at[momentum_index_x].set(
            gam0 * slow_mode_weighting * velocity_x_interface
            - slow_mode_weighting * slow_magnetosonic_velocity_interface
        )
        L = L.at[momentum_index_y].set(
            gam0 * slow_mode_weighting * velocity_y_interface
            - fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        L = L.at[momentum_index_z].set(
            gam0 * slow_mode_weighting * velocity_z_interface
            - fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        L = L.at[magnetic_index_y].set(
            gam0 * slow_mode_weighting * magnetic_y_interface
            - sound_speed_interface * fast_mode_weighting * bt_normalized_y * sqrt_rho
        )
        L = L.at[magnetic_index_z].set(
            gam0 * slow_mode_weighting * magnetic_z_interface
            - sound_speed_interface * fast_mode_weighting * bt_normalized_z * sqrt_rho
        )
        L = L.at[energy_index].set(-gam0 * slow_mode_weighting)
        L = 0.5 * L * sound_speed_sq_inverse
        L = jnp.where(sound_speed_greater_alfven_speed, L * sgn_bt, L)
        return L

    def row_3():
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(
            -sound_speed_sq_interface / gam0 - 0.5 * velocity_sq_interface
        )
        L = L.at[momentum_index_x].set(velocity_x_interface)
        L = L.at[momentum_index_y].set(velocity_y_interface)
        L = L.at[momentum_index_z].set(velocity_z_interface)
        L = L.at[magnetic_index_y].set(magnetic_y_interface)
        L = L.at[magnetic_index_z].set(magnetic_z_interface)
        L = L.at[energy_index].set(-1.0)
        L = -gam0 * L * sound_speed_sq_inverse
        return L

    def row_4():
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(
            slow_mode_weighting
            * (
                gam1 * velocity_sq_interface
                - slow_magnetosonic_velocity_interface * velocity_x_interface
            )
            - fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        L = L.at[momentum_index_x].set(
            slow_mode_weighting
            * (gam0 * velocity_x_interface + slow_magnetosonic_velocity_interface)
        )
        L = L.at[momentum_index_y].set(
            gam0 * slow_mode_weighting * velocity_y_interface
            + fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        L = L.at[momentum_index_z].set(
            gam0 * slow_mode_weighting * velocity_z_interface
            + fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        L = L.at[magnetic_index_y].set(
            gam0 * slow_mode_weighting * magnetic_y_interface
            - sound_speed_interface * fast_mode_weighting * bt_normalized_y * sqrt_rho
        )
        L = L.at[magnetic_index_z].set(
            gam0 * slow_mode_weighting * magnetic_z_interface
            - sound_speed_interface * fast_mode_weighting * bt_normalized_z * sqrt_rho
        )
        L = L.at[energy_index].set(-gam0 * slow_mode_weighting)
        L = 0.5 * L * sound_speed_sq_inverse
        L = jnp.where(sound_speed_greater_alfven_speed, L * sgn_bt, L)
        return L

    def row_5():
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(
            bt_normalized_z * velocity_y_interface
            - bt_normalized_y * velocity_z_interface
        )
        L = L.at[momentum_index_x].set(0.0)
        L = L.at[momentum_index_y].set(-bt_normalized_z)
        L = L.at[momentum_index_z].set(bt_normalized_y)
        L = L.at[magnetic_index_y].set(bt_normalized_z * sgn_bx * sqrt_rho)
        L = L.at[magnetic_index_z].set(-bt_normalized_y * sgn_bx * sqrt_rho)
        L = L.at[energy_index].set(0.0)
        L = 0.5 * L
        return L

    def row_6():
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(
            fast_mode_weighting
            * (
                gam1 * velocity_sq_interface
                - fast_magnetosonic_velocity_interface * velocity_x_interface
            )
            + slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        L = L.at[momentum_index_x].set(
            fast_mode_weighting
            * (gam0 * velocity_x_interface + fast_magnetosonic_velocity_interface)
        )
        L = L.at[momentum_index_y].set(
            gam0 * fast_mode_weighting * velocity_y_interface
            - slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        L = L.at[momentum_index_z].set(
            gam0 * fast_mode_weighting * velocity_z_interface
            - slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        L = L.at[magnetic_index_y].set(
            gam0 * fast_mode_weighting * magnetic_y_interface
            + sound_speed_interface * slow_mode_weighting * bt_normalized_y * sqrt_rho
        )
        L = L.at[magnetic_index_z].set(
            gam0 * fast_mode_weighting * magnetic_z_interface
            + sound_speed_interface * slow_mode_weighting * bt_normalized_z * sqrt_rho
        )
        L = L.at[energy_index].set(-gam0 * fast_mode_weighting)
        L = 0.5 * L * sound_speed_sq_inverse
        L = jnp.where(~sound_speed_greater_alfven_speed, L * sgn_bt, L)
        return L

    L = jax.lax.switch(row, [row_0, row_1, row_2, row_3, row_4, row_5, row_6])

    return L


@partial(jax.jit, static_argnames=["registered_variables"])
def _eigen_all_lambdas(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    (
        velocity_x,
        fast_magnetosonic_velocity,
        alfven_velocity,
        slow_magnetosonic_velocity,
    ) = _eigenvalue_building_blocks(
        conserved_state,
        gamma,
        rhomin,
        pgmin,
        registered_variables,
    )

    return jnp.stack(
        [
            velocity_x - fast_magnetosonic_velocity,
            velocity_x - alfven_velocity,
            velocity_x - slow_magnetosonic_velocity,
            velocity_x,
            velocity_x + slow_magnetosonic_velocity,
            velocity_x + alfven_velocity,
            velocity_x + fast_magnetosonic_velocity,
        ],
        axis=0,
    )


def _eigen_lambdas(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
    mode: int,
):
    (
        velocity_x,
        fast_magnetosonic_velocity,
        alfven_velocity,
        slow_magnetosonic_velocity,
    ) = _eigenvalue_building_blocks(
        conserved_state,
        gamma,
        rhomin,
        pgmin,
        registered_variables,
    )

    def mode_0():
        return velocity_x - fast_magnetosonic_velocity

    def mode_1():
        return velocity_x - alfven_velocity

    def mode_2():
        return velocity_x - slow_magnetosonic_velocity

    def mode_3():
        return velocity_x

    def mode_4():
        return velocity_x + slow_magnetosonic_velocity

    def mode_5():
        return velocity_x + alfven_velocity

    def mode_6():
        return velocity_x + fast_magnetosonic_velocity

    return jax.lax.switch(
        mode, [mode_0, mode_1, mode_2, mode_3, mode_4, mode_5, mode_6]
    )
