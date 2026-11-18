"""
Computations of the eigenvalues and eigenvectors for the MHD equations.

The eigenstructure was extracted from the HOW-MHD Fortran code, with altered
variable names for clarity and altered numerical safeguards. As in the hydro
case it is evaluated along x on a state whose axes have been permuted so that
the sweep direction comes first.

NOTE: Problems for differentiation largely follow from square roots and divisions:
the derivative of sqrt(x) is 1/(2*sqrt(x)) and that of 1/x is -1/x^2, both of which
are problematic for small x, especially when gradients are multiplied in the backward
pass (exploding gradients).
"""

# general
from functools import partial

# typing
from typing import NamedTuple, Union

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._dual_energy_switch import dual_energy_switched_pressure
from astronomix._stencil_operations._stencil_operations import _shift


def diff_safe_sqrt(x):
    """Square root with a small floor, so its derivative stays finite at x = 0.

    The derivative of sqrt(x) is 1 / (2 sqrt(x)), which blows up as x -> 0;
    clamping the argument to a tiny epsilon keeps the backward pass well-behaved.
    The floor is tighter under x64 than under x32 to match the available precision.
    """
    if jax.config.jax_enable_x64:
        epsilon = 1e-30
    else:
        epsilon = 1e-20

    x_safe = jnp.maximum(x, epsilon)
    return jnp.sqrt(x_safe)


@partial(jax.jit, static_argnames=["registered_variables"])
def _eigenvalue_building_blocks(
    conserved_state,
    gamma,
    rhomin,
    pgmin,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
    dual_eta=None,
):
    """
    Cell-centred velocity and wave speeds entering the MHD eigenvalues.

    Args:
        conserved_state: The conserved state, sweep direction first.
        gamma: The adiabatic index.
        rhomin: The density floor.
        pgmin: The pressure floor.
        registered_variables: The registered variables.
        internal_energy_density: The separately advected dual-energy ``g``, or
            None for the plain total-energy pressure recovery.
        dual_eta: The dual-energy switch threshold (``DUAL_ENERGY_ETA``);
            required when ``internal_energy_density`` is given.

    Returns:
        The normal velocity and the fast magnetosonic, Alfven and slow
        magnetosonic speeds per cell.
    """
    # Unpack the conserved variables.
    density = conserved_state[registered_variables.density_index]
    momentum_x = conserved_state[registered_variables.momentum_index.x]
    momentum_y = conserved_state[registered_variables.momentum_index.y]
    momentum_z = conserved_state[registered_variables.momentum_index.z]
    magnetic_x = conserved_state[registered_variables.magnetic_index.x]
    magnetic_y = conserved_state[registered_variables.magnetic_index.y]
    magnetic_z = conserved_state[registered_variables.magnetic_index.z]
    energy = conserved_state[registered_variables.energy_index]

    # Compute the primitive quantities.
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

    if internal_energy_density is not None:
        gas_pressure = dual_energy_switched_pressure(
            gas_pressure,
            energy,
            gamma,
            internal_energy_density,
            dual_eta,
        )

    # Apply the density and pressure floors (and make the energy consistent).
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

    # Compute the derived quantities.
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


class EigenvectorBlocks(NamedTuple):
    """
    The interface quantities the MHD eigenvectors are assembled from: the
    interface density and its square root, velocity and magnetic field
    components, normalised tangential field and sign factors, the sound, fast,
    Alfven and slow speeds, the fast/slow mode weightings and the gamma
    combinations gam0, gam1, gam2 of the HOW-MHD eigenvectors.
    """

    rho_interface: jnp.ndarray
    sqrt_rho: jnp.ndarray
    velocity_x_interface: jnp.ndarray
    velocity_y_interface: jnp.ndarray
    velocity_z_interface: jnp.ndarray
    velocity_sq_interface: jnp.ndarray
    magnetic_x_interface: jnp.ndarray
    magnetic_y_interface: jnp.ndarray
    magnetic_z_interface: jnp.ndarray
    bt_normalized_y: jnp.ndarray
    bt_normalized_z: jnp.ndarray
    sgn_bx: jnp.ndarray
    sgn_bt: jnp.ndarray
    sound_speed_interface: jnp.ndarray
    sound_speed_sq_interface: jnp.ndarray
    sound_speed_sq_inverse: jnp.ndarray
    sound_speed_greater_alfven_speed: jnp.ndarray
    fast_magnetosonic_velocity_interface: jnp.ndarray
    alfven_velocity_interface: jnp.ndarray
    slow_magnetosonic_velocity_interface: jnp.ndarray
    fast_mode_weighting: jnp.ndarray
    slow_mode_weighting: jnp.ndarray
    gam0: Union[float, jnp.ndarray]
    gam1: Union[float, jnp.ndarray]
    gam2: Union[float, jnp.ndarray]


def _interface_state(
    conserved_state,
    gamma,
    rhomin,
    pgmin,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
    dual_eta=None,
    admissible_face_state: bool = True,
):
    """
    Step (i) of the eigenvectors: the cell primitives with their floors and
    the interface state.

    The interface state between cell i and cell i + 1 is built from averages
    of the two cells.

    Args:
        conserved_state: The conserved state, sweep direction first.
        gamma: The adiabatic index.
        rhomin: The density floor.
        pgmin: The pressure floor.
        registered_variables: The registered variables.
        internal_energy_density: The separately advected dual-energy ``g``, or
            None for the plain total-energy pressure recovery.
        dual_eta: The dual-energy switch threshold (``DUAL_ENERGY_ETA``);
            required when ``internal_energy_density`` is given.
        admissible_face_state: Build the interface sound speed from the
            averaged pressure (``config.solver_config.weno_admissible_face_state``) instead of
            the averaged enthalpy.

    Returns:
        The interface density, the three velocity and the three magnetic field
        components, and the squared interface sound speed.
    """
    # Unpack the conserved variables.
    rho = conserved_state[registered_variables.density_index]
    momentum_x = conserved_state[registered_variables.momentum_index.x]
    momentum_y = conserved_state[registered_variables.momentum_index.y]
    momentum_z = conserved_state[registered_variables.momentum_index.z]
    magnetic_x = conserved_state[registered_variables.magnetic_index.x]
    magnetic_y = conserved_state[registered_variables.magnetic_index.y]
    magnetic_z = conserved_state[registered_variables.magnetic_index.z]
    energy = conserved_state[registered_variables.energy_index]

    # Compute the primitive quantities.
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

    if internal_energy_density is not None:
        gas_pressure = dual_energy_switched_pressure(
            gas_pressure,
            energy,
            gamma,
            internal_energy_density,
            dual_eta,
        )

    # Apply the density and pressure floors (and make the energy consistent).
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

    # Periodic average from cell centres to interfaces.
    def avg_x(arr):
        return 0.5 * (arr + _shift(arr, shift=-1, axis=0))

    # Average the momenta (rather than the velocities) to the interface and divide
    # by the interface density; this keeps the interface velocity consistent with
    # the averaged conserved quantities.
    rho_interface = avg_x(jnp.maximum(rho, rhomin))
    rho_interface = jnp.maximum(rho_interface, rhomin)
    velocity_x_interface = avg_x(momentum_x) / rho_interface
    velocity_y_interface = avg_x(momentum_y) / rho_interface
    velocity_z_interface = avg_x(momentum_z) / rho_interface

    magnetic_x_interface = avg_x(magnetic_x)
    magnetic_y_interface = avg_x(magnetic_y)
    magnetic_z_interface = avg_x(magnetic_z)
    specific_enthalpy_interface = avg_x(specific_enthalpy)

    # Interface quantities entering the enthalpy-based sound speed.
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

    # Enthalpy-based sound speed at the interfaces.
    sound_speed_sq_interface = (gamma - 1.0) * (
        specific_enthalpy_interface
        - 0.5 * (velocity_sq_interface + b_sq_over_rho_interface)
    )
    if admissible_face_state:
        # Build the interface state from averaged primitives so that the frozen
        # characteristic basis is that of a real, hyperbolic state; see
        # _eigen_hydro._eigenvector_building_blocks for why the enthalpy
        # average fails at strong density and velocity jumps.
        sound_speed_sq_interface = gamma * avg_x(gas_pressure) / rho_interface

    return (
        rho_interface,
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        magnetic_x_interface,
        magnetic_y_interface,
        magnetic_z_interface,
        sound_speed_sq_interface,
    )


def _wave_speeds(
    rho_interface,
    magnetic_x_interface,
    magnetic_y_interface,
    magnetic_z_interface,
    sound_speed_sq_interface,
):
    """
    Step (ii) of the eigenvectors: the fast, Alfven and slow speeds of an
    interface state,

        c_f^2, c_s^2 = 0.5 (B^2 / rho + a^2 +- sqrt((B^2 / rho + a^2)^2 - 4 a^2 B_x^2 / rho)),
        c_a^2 = B_x^2 / rho.

    The squared fast and slow speeds are returned next to the speeds. They are
    the floored arguments of the two square roots, so the speeds are exactly
    their roots, and a caller can carry c_f^2 and c_s^2 into the mode
    weightings instead of squaring the rounded speeds. The isothermal module
    uses this function with its constant squared sound speed.

    Args:
        rho_interface: The interface density.
        magnetic_x_interface: The interface field along the sweep direction.
        magnetic_y_interface: The first transverse component of the interface field.
        magnetic_z_interface: The second transverse component of the interface field.
        sound_speed_sq_interface: The squared interface sound speed.

    Returns:
        The fast, Alfven and slow speeds and the squared fast and slow speeds.
    """
    if jax.config.jax_enable_x64:
        eps = 1e-30
    else:
        eps = 1e-20

    magnetic_sq_interface = (
        magnetic_x_interface * magnetic_x_interface
        + magnetic_y_interface * magnetic_y_interface
        + magnetic_z_interface * magnetic_z_interface
    )
    b_sq_over_rho_interface = magnetic_sq_interface / rho_interface
    bx_sq_over_rho_interface = (
        magnetic_x_interface * magnetic_x_interface
    ) / rho_interface

    # Calculate the characteristic velocities at the interfaces.
    magnetosonic_discriminant_interface = (
        b_sq_over_rho_interface + sound_speed_sq_interface
    ) ** 2 - 4.0 * bx_sq_over_rho_interface * sound_speed_sq_interface
    magnetosonic_discriminant_root_interface = diff_safe_sqrt(
        jnp.maximum(0.0, magnetosonic_discriminant_interface)
    )

    # The squared fast and slow speeds, floored as diff_safe_sqrt floors its
    # argument: the speeds below are exactly their roots.
    fast_magnetosonic_velocity_sq_interface = jnp.maximum(
        jnp.maximum(
            0.0,
            0.5
            * (
                b_sq_over_rho_interface
                + sound_speed_sq_interface
                + magnetosonic_discriminant_root_interface
            ),
        ),
        eps,
    )
    slow_magnetosonic_velocity_sq_interface = jnp.maximum(
        jnp.maximum(
            0.0,
            0.5
            * (
                b_sq_over_rho_interface
                + sound_speed_sq_interface
                - magnetosonic_discriminant_root_interface
            ),
        ),
        eps,
    )

    fast_magnetosonic_velocity_interface = jnp.sqrt(
        fast_magnetosonic_velocity_sq_interface
    )
    alfven_velocity_interface = diff_safe_sqrt(
        jnp.maximum(0.0, bx_sq_over_rho_interface)
    )
    slow_magnetosonic_velocity_interface = jnp.sqrt(
        slow_magnetosonic_velocity_sq_interface
    )

    return (
        fast_magnetosonic_velocity_interface,
        alfven_velocity_interface,
        slow_magnetosonic_velocity_interface,
        fast_magnetosonic_velocity_sq_interface,
        slow_magnetosonic_velocity_sq_interface,
    )


def _mode_weightings(
    sound_speed_sq_interface,
    fast_magnetosonic_velocity_sq_interface,
    slow_magnetosonic_velocity_sq_interface,
):
    """
    The fast and slow mode weightings of the eigenvectors of an interface state,

        alpha_f = sqrt((a^2 - c_s^2) / (c_f^2 - c_s^2)),    alpha_s = sqrt((c_f^2 - a^2) / (c_f^2 - c_s^2)),

    divided by sqrt(alpha_f^2 + alpha_s^2). The rows and the columns of the eigenvectors are inverse to
    each other only if alpha_f^2 + alpha_s^2 = 1, and near c_f = c_s the rounded squared speeds do not
    enclose a^2: one numerator is cut off at zero and the sum of the squares leaves 1 without the division.
    Where the squared speeds are equal (B_t = 0 and v_A = a) both weightings are 1 / sqrt(2). The isothermal
    module uses this function with its constant squared sound speed.

    Args:
        sound_speed_sq_interface: The squared interface sound speed.
        fast_magnetosonic_velocity_sq_interface: The squared fast speed.
        slow_magnetosonic_velocity_sq_interface: The squared slow speed.

    Returns:
        The fast and the slow mode weighting.
    """
    if jax.config.jax_enable_x64:
        eps = 1e-30
    else:
        eps = 1e-20

    denom = (
        fast_magnetosonic_velocity_sq_interface
        - slow_magnetosonic_velocity_sq_interface
    )
    root_denom = diff_safe_sqrt(jnp.maximum(denom, eps))
    fast_mode_weighting = (
        diff_safe_sqrt(
            jnp.maximum(
                0.0,
                sound_speed_sq_interface - slow_magnetosonic_velocity_sq_interface,
            )
        )
        / root_denom
    )
    slow_mode_weighting = (
        diff_safe_sqrt(
            jnp.maximum(
                0.0,
                fast_magnetosonic_velocity_sq_interface - sound_speed_sq_interface,
            )
        )
        / root_denom
    )
    # The floored roots keep the norm above zero where the speeds are equal as well (there it is at
    # least sqrt(2)), so neither branch of the selection divides by zero.
    norm = jnp.sqrt(
        fast_mode_weighting * fast_mode_weighting
        + slow_mode_weighting * slow_mode_weighting
    )
    fast_mode_weighting = jnp.where(
        denom >= eps, fast_mode_weighting / norm, 1.0 / jnp.sqrt(2.0)
    )
    slow_mode_weighting = jnp.where(
        denom >= eps, slow_mode_weighting / norm, 1.0 / jnp.sqrt(2.0)
    )
    return fast_mode_weighting, slow_mode_weighting


def _eigenvector_blocks(
    rho_interface,
    velocity_x_interface,
    velocity_y_interface,
    velocity_z_interface,
    magnetic_x_interface,
    magnetic_y_interface,
    magnetic_z_interface,
    sound_speed_sq_interface,
    fast_magnetosonic_velocity_interface,
    alfven_velocity_interface,
    slow_magnetosonic_velocity_interface,
    fast_magnetosonic_velocity_sq_interface,
    slow_magnetosonic_velocity_sq_interface,
    gamma,
) -> EigenvectorBlocks:
    """
    Step (iii) of the eigenvectors: the mode weightings and the remaining
    blocks from an interface state, its wave speeds and the squared fast and
    slow speeds.

    Args:
        rho_interface: The interface density.
        velocity_x_interface: The interface velocity along the sweep direction.
        velocity_y_interface: The first transverse interface velocity.
        velocity_z_interface: The second transverse interface velocity.
        magnetic_x_interface: The interface field along the sweep direction.
        magnetic_y_interface: The first transverse component of the interface field.
        magnetic_z_interface: The second transverse component of the interface field.
        sound_speed_sq_interface: The squared interface sound speed.
        fast_magnetosonic_velocity_interface: The fast speed.
        alfven_velocity_interface: The Alfven speed.
        slow_magnetosonic_velocity_interface: The slow speed.
        fast_magnetosonic_velocity_sq_interface: The squared fast speed
            entering the mode weightings.
        slow_magnetosonic_velocity_sq_interface: The squared slow speed
            entering the mode weightings.
        gamma: The adiabatic index.

    Returns:
        The blocks the rows and columns of the eigenvectors are assembled from.
    """
    if jax.config.jax_enable_x64:
        eps = 1e-30
    else:
        eps = 1e-20

    sound_speed_interface = diff_safe_sqrt(jnp.maximum(0.0, sound_speed_sq_interface))
    velocity_sq_interface = (
        velocity_x_interface * velocity_x_interface
        + velocity_y_interface * velocity_y_interface
        + velocity_z_interface * velocity_z_interface
    )

    # Retrieve the tangential magnetic field components.
    b_tangential_sq = (
        magnetic_y_interface * magnetic_y_interface
        + magnetic_z_interface * magnetic_z_interface
    )
    sgn_bx = jnp.where(magnetic_x_interface >= 0.0, 1.0, -1.0)

    b_tangential_sq_safe = jnp.maximum(b_tangential_sq, eps)

    # B_y / (sqrt(B_y^2 + B_z^2))
    bt_normalized_y = jnp.where(
        b_tangential_sq >= eps,
        magnetic_y_interface / jnp.sqrt(b_tangential_sq_safe),
        1.0 / jnp.sqrt(2.0),
    )
    # B_z / (sqrt(B_y^2 + B_z^2))
    bt_normalized_z = jnp.where(
        b_tangential_sq >= eps,
        magnetic_z_interface / jnp.sqrt(b_tangential_sq_safe),
        1.0 / jnp.sqrt(2.0),
    )

    # mode weightings: af^2 + as^2 = 1
    fast_mode_weighting, slow_mode_weighting = _mode_weightings(
        sound_speed_sq_interface,
        fast_magnetosonic_velocity_sq_interface,
        slow_magnetosonic_velocity_sq_interface,
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

    return EigenvectorBlocks(
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


@partial(jax.jit, static_argnames=["registered_variables", "admissible_face_state"])
def _eigenvector_building_blocks(
    conserved_state,
    gamma,
    rhomin,
    pgmin,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
    dual_eta=None,
    admissible_face_state: bool = True,
):
    """
    Interface quantities entering the MHD eigenvectors.

    The interface state between cell i and cell i + 1 is built from averages
    of the two cells.

    Args:
        conserved_state: The conserved state, sweep direction first.
        gamma: The adiabatic index.
        rhomin: The density floor.
        pgmin: The pressure floor.
        registered_variables: The registered variables.
        internal_energy_density: The separately advected dual-energy ``g``, or
            None for the plain total-energy pressure recovery.
        dual_eta: The dual-energy switch threshold (``DUAL_ENERGY_ETA``);
            required when ``internal_energy_density`` is given.
        admissible_face_state: Build the interface sound speed from the
            averaged pressure (``config.solver_config.weno_admissible_face_state``) instead of
            the averaged enthalpy.

    Returns:
        The interface density and its square root, velocity and magnetic field
        components, normalised tangential field and sign factors, the sound,
        fast, Alfven and slow speeds, the fast/slow mode weightings and the
        gamma combinations gam0, gam1, gam2 of the HOW-MHD eigenvectors.
    """
    interface_state = _interface_state(
        conserved_state,
        gamma,
        rhomin,
        pgmin,
        registered_variables,
        internal_energy_density,
        dual_eta,
        admissible_face_state,
    )
    (
        rho_interface,
        _,
        _,
        _,
        magnetic_x_interface,
        magnetic_y_interface,
        magnetic_z_interface,
        sound_speed_sq_interface,
    ) = interface_state

    (
        fast_magnetosonic_velocity_interface,
        alfven_velocity_interface,
        slow_magnetosonic_velocity_interface,
        _,
        _,
    ) = _wave_speeds(
        rho_interface,
        magnetic_x_interface,
        magnetic_y_interface,
        magnetic_z_interface,
        sound_speed_sq_interface,
    )

    # The mode weightings of this function are formed from the squares of the
    # rounded speeds, as before the split into the steps (i) to (iv).
    return _eigenvector_blocks(
        *interface_state,
        fast_magnetosonic_velocity_interface,
        alfven_velocity_interface,
        slow_magnetosonic_velocity_interface,
        fast_magnetosonic_velocity_interface * fast_magnetosonic_velocity_interface,
        slow_magnetosonic_velocity_interface * slow_magnetosonic_velocity_interface,
        gamma,
    )


def _right_eigenvector_columns(
    conserved_state,
    registered_variables: RegisteredVariables,
    blocks: EigenvectorBlocks,
    fast_magnetosonic_velocity_sq_interface,
    slow_magnetosonic_velocity_sq_interface,
):
    """
    Step (iv) of the eigenvectors: the right eigenvectors (columns of R) of the
    MHD flux Jacobian at the interfaces, one function per characteristic mode.

    Args:
        conserved_state: The conserved state, sweep direction first; only its
            shape and dtype enter (every column is shaped like it).
        registered_variables: The registered variables.
        blocks: The interface quantities of ``_eigenvector_blocks``.
        fast_magnetosonic_velocity_sq_interface: The squared fast speed in the
            energy component of the two fast columns.
        slow_magnetosonic_velocity_sq_interface: The squared slow speed in the
            energy component of the two slow columns.

    Returns:
        Seven functions without arguments, ordered by wave speed (fast -,
        Alfven -, slow -, entropy, slow +, Alfven +, fast +); each returns one
        right eigenvector, shaped like ``conserved_state``.
    """
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
    ) = blocks

    # Shorter names for the registry indices.
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
                fast_magnetosonic_velocity_sq_interface
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
        # Column 3 (slow -)
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
                slow_magnetosonic_velocity_sq_interface
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
        # Column 4 (entropy)
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
                slow_magnetosonic_velocity_sq_interface
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
                fast_magnetosonic_velocity_sq_interface
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

    return [col_0, col_1, col_2, col_3, col_4, col_5, col_6]


@partial(jax.jit, static_argnames=["registered_variables", "admissible_face_state"])
def _eigen_R_col(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
    col: int,
    internal_energy_density=None,
    dual_eta=None,
    admissible_face_state: bool = True,
):
    """
    One right eigenvector (column of R) of the MHD flux Jacobian at the interfaces.

    Args:
        conserved_state: The conserved state, sweep direction first.
        rhomin: The density floor.
        pgmin: The pressure floor.
        gamma: The adiabatic index.
        registered_variables: The registered variables.
        col: The index of the characteristic mode, ordered by wave speed
            (fast -, Alfven -, slow -, entropy, slow +, Alfven +, fast +).
        internal_energy_density: The separately advected dual-energy ``g``, or
            None for the plain total-energy pressure recovery.
        dual_eta: The dual-energy switch threshold (``DUAL_ENERGY_ETA``);
            required when ``internal_energy_density`` is given.
        admissible_face_state: Build the interface sound speed from the
            averaged pressure (``config.solver_config.weno_admissible_face_state``) instead of
            the averaged enthalpy.

    Returns:
        The right eigenvector, shaped like ``conserved_state``.
    """
    blocks = _eigenvector_building_blocks(
        conserved_state,
        gamma,
        rhomin,
        pgmin,
        registered_variables,
        internal_energy_density,
        dual_eta,
        admissible_face_state,
    )

    def column(mode):
        # The squares of the rounded fast and slow speeds are taken inside the
        # selected branch, where they stood before the split of this module.
        def selected_column():
            columns = _right_eigenvector_columns(
                conserved_state,
                registered_variables,
                blocks,
                blocks.fast_magnetosonic_velocity_interface**2,
                blocks.slow_magnetosonic_velocity_interface**2,
            )
            return columns[mode]()

        return selected_column

    R = jax.lax.switch(col, [column(mode) for mode in range(7)])

    return R


def _left_eigenvector_rows(
    conserved_state,
    registered_variables: RegisteredVariables,
    blocks: EigenvectorBlocks,
):
    """
    Step (iv) of the eigenvectors: the left eigenvectors (rows of L) of the
    MHD flux Jacobian at the interfaces, one function per characteristic mode.

    Args:
        conserved_state: The conserved state, sweep direction first; only its
            shape and dtype enter (every row is shaped like it).
        registered_variables: The registered variables.
        blocks: The interface quantities of ``_eigenvector_blocks``.

    Returns:
        Seven functions without arguments, ordered by wave speed (fast -,
        Alfven -, slow -, entropy, slow +, Alfven +, fast +); each returns one
        left eigenvector, shaped like ``conserved_state``.
    """
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
    ) = blocks

    # Shorter names for the registry indices.
    density_index = registered_variables.density_index
    momentum_index_x = registered_variables.momentum_index.x
    momentum_index_y = registered_variables.momentum_index.y
    momentum_index_z = registered_variables.momentum_index.z
    magnetic_index_y = registered_variables.magnetic_index.y
    magnetic_index_z = registered_variables.magnetic_index.z
    energy_index = registered_variables.energy_index

    def row_0():
        # Row 1 (fast -)
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
        # Row 2 (alfven -)
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
        # Row 3 (slow -)
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
        # Row 4 (entropy)
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
        # Row 5 (slow +)
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
        # Row 6 (alfven +)
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
        # Row 7 (fast +)
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

    return [row_0, row_1, row_2, row_3, row_4, row_5, row_6]


@partial(jax.jit, static_argnames=["registered_variables", "admissible_face_state"])
def _eigen_L_row(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
    row: int,
    internal_energy_density=None,
    dual_eta=None,
    admissible_face_state: bool = True,
):
    """
    One left eigenvector (row of L) of the MHD flux Jacobian at the interfaces.

    Args:
        conserved_state: The conserved state, sweep direction first.
        rhomin: The density floor.
        pgmin: The pressure floor.
        gamma: The adiabatic index.
        registered_variables: The registered variables.
        row: The index of the characteristic mode, ordered by wave speed
            (fast -, Alfven -, slow -, entropy, slow +, Alfven +, fast +).
        internal_energy_density: The separately advected dual-energy ``g``, or
            None for the plain total-energy pressure recovery.
        dual_eta: The dual-energy switch threshold (``DUAL_ENERGY_ETA``);
            required when ``internal_energy_density`` is given.
        admissible_face_state: Build the interface sound speed from the
            averaged pressure (``config.solver_config.weno_admissible_face_state``) instead of
            the averaged enthalpy.

    Returns:
        The left eigenvector, shaped like ``conserved_state``.
    """
    blocks = _eigenvector_building_blocks(
        conserved_state,
        gamma,
        rhomin,
        pgmin,
        registered_variables,
        internal_energy_density,
        dual_eta,
        admissible_face_state,
    )

    L = jax.lax.switch(
        row, _left_eigenvector_rows(conserved_state, registered_variables, blocks)
    )

    return L


@partial(jax.jit, static_argnames=["registered_variables"])
def _eigen_all_lambdas(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    """
    All MHD eigenvalues per cell, stacked along a leading mode axis.

    Used by the CFL time-step estimate, which works on the plain total-energy
    pressure recovery.

    Args:
        conserved_state: The conserved state, sweep direction first.
        rhomin: The density floor.
        pgmin: The pressure floor.
        gamma: The adiabatic index.
        registered_variables: The registered variables.

    Returns:
        The eigenvalues (fast -, Alfven -, slow -, entropy, slow +, Alfven +, fast +) per cell.
    """
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
    internal_energy_density=None,
    dual_eta=None,
):
    """
    One MHD eigenvalue per cell.

    Args:
        conserved_state: The conserved state, sweep direction first.
        rhomin: The density floor.
        pgmin: The pressure floor.
        gamma: The adiabatic index.
        registered_variables: The registered variables.
        mode: The index of the characteristic mode, ordered by wave speed
            (fast -, Alfven -, slow -, entropy, slow +, Alfven +, fast +).
        internal_energy_density: The separately advected dual-energy ``g``, or
            None for the plain total-energy pressure recovery.
        dual_eta: The dual-energy switch threshold (``DUAL_ENERGY_ETA``);
            required when ``internal_energy_density`` is given.

    Returns:
        The eigenvalue of the mode per cell.
    """
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
        internal_energy_density,
        dual_eta,
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
        mode,
        [mode_0, mode_1, mode_2, mode_3, mode_4, mode_5, mode_6],
    )
