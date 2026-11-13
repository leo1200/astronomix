"""
Computations of the eigenvalues and eigenvectors for the MHD equations of a
barotropic gas: isothermal, or polytropic.

The eigenstructure was extracted from the HOW-MHD Fortran code (isothermal variant),
with altered variable names for clarity and altered numerical safeguards.

Barotropic MHD has 6 waves (no entropy wave) and 6 conserved variables
(density, 3 momenta, 2 tangential magnetic field components; Bx is constant).
The sound speed cs is the isothermal constant, or for a polytropic gas the
local a with a^2 = dp / drho, at the cell (eigenvalues) or at the averaged
interface state (eigenvectors). The functions take the law as
``barotropic_law`` (``_equation_of_state.py``): the isothermal sound speed or
the ``PolytropicLaw``.
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
from astronomix._fluid_equations._eigen_mhd import _wave_speeds
from astronomix._fluid_equations._equation_of_state import law_sound_speed
from astronomix._stencil_operations._stencil_operations import _shift


def diff_safe_sqrt(x):
    """Square root with a small floor, so its derivative stays finite at x = 0.

    The derivative of sqrt(x) is 1 / (2 sqrt(x)), which blows up as x -> 0;
    clamping the argument to a tiny epsilon keeps the backward pass well-behaved.
    The floor is tighter under x64 than under x32 to match the available precision.
    """
    if jax.config.jax_enable_x64:
        eps = 1e-30
    else:
        eps = 1e-20

    epsilon = eps
    x_safe = jnp.maximum(x, epsilon)
    return jnp.sqrt(x_safe)


def zero_safe_sqrt(x):
    """sqrt(max(x, 0)) whose derivative at x <= 0 is zero instead of infinite.

    The values are those of ``jnp.sqrt(jnp.maximum(x, 0.0))`` for every input (a NaN stays a NaN).
    Only the derivative differs: jnp.sqrt gives 1 / (2 sqrt(x)) = inf at x = 0, which the chain
    rule turns into NaN wherever the argument's own derivative vanishes there (a tangential field
    strength b_t = sqrt(b_t^2) at b_t = 0, an Alfven speed |B| / sqrt(rho) without a field). The
    derivative taken here at x <= 0 is zero: exact where the result enters only through its
    square (c_f depends on b_t^2), and the symmetric choice at the kink of |B| at B = 0.
    """
    nonpositive = x <= 0.0
    return jnp.where(nonpositive, 0.0, jnp.sqrt(jnp.where(nonpositive, 1.0, x)))


@partial(jax.jit, static_argnames=["registered_variables"])
def _eigenvalue_building_blocks(
    conserved_state,
    barotropic_law,
    rhomin,
    registered_variables: RegisteredVariables,
):
    # unpack the conserved variables
    density = conserved_state[registered_variables.density_index]
    momentum_x = conserved_state[registered_variables.momentum_index.x]
    magnetic_x = conserved_state[registered_variables.magnetic_index.x]
    magnetic_y = conserved_state[registered_variables.magnetic_index.y]
    magnetic_z = conserved_state[registered_variables.magnetic_index.z]

    # compute primitives
    rho = jnp.maximum(density, rhomin)
    velocity_x = momentum_x / rho

    magnetic_field_squared = (
        magnetic_x * magnetic_x + magnetic_y * magnetic_y + magnetic_z * magnetic_z
    )
    _, cs2 = law_sound_speed(barotropic_law, rho)

    # compute derived quantities
    b_sq_over_rho = magnetic_field_squared / rho
    bx_sq_over_rho = (magnetic_x * magnetic_x) / rho

    magnetosonic_discriminant_root = diff_safe_sqrt(
        jnp.maximum(
            0.0,
            (b_sq_over_rho + cs2) ** 2 - 4.0 * bx_sq_over_rho * cs2,
        )
    )
    fast_magnetosonic_velocity = diff_safe_sqrt(
        jnp.maximum(0.0, 0.5 * (b_sq_over_rho + cs2 + magnetosonic_discriminant_root))
    )
    alfven_velocity = diff_safe_sqrt(jnp.maximum(0.0, bx_sq_over_rho))
    slow_magnetosonic_velocity = diff_safe_sqrt(
        jnp.maximum(0.0, 0.5 * (b_sq_over_rho + cs2 - magnetosonic_discriminant_root))
    )

    return (
        velocity_x,
        fast_magnetosonic_velocity,
        alfven_velocity,
        slow_magnetosonic_velocity,
    )


class EigenvectorBlocksIso(NamedTuple):
    """
    The interface quantities the barotropic MHD eigenvectors are assembled
    from: the interface density and its square root, velocity and magnetic
    field components, normalised tangential field and sign factors, the sound
    speed with its square and inverse square, the fast, Alfven and slow speeds
    and the fast/slow mode weightings.
    """

    rho_interface: jnp.ndarray
    sqrt_rho: jnp.ndarray
    velocity_x_interface: jnp.ndarray
    velocity_y_interface: jnp.ndarray
    velocity_z_interface: jnp.ndarray
    magnetic_x_interface: jnp.ndarray
    magnetic_y_interface: jnp.ndarray
    magnetic_z_interface: jnp.ndarray
    bt_normalized_y: jnp.ndarray
    bt_normalized_z: jnp.ndarray
    sgn_bx: jnp.ndarray
    sgn_bt: jnp.ndarray
    sound_speed: Union[float, jnp.ndarray]
    cs2: Union[float, jnp.ndarray]
    cs2_inverse: Union[float, jnp.ndarray]
    sound_speed_greater_alfven_speed: jnp.ndarray
    fast_magnetosonic_velocity_interface: jnp.ndarray
    alfven_velocity_interface: jnp.ndarray
    slow_magnetosonic_velocity_interface: jnp.ndarray
    fast_mode_weighting: jnp.ndarray
    slow_mode_weighting: jnp.ndarray


def _interface_state_iso(
    conserved_state,
    rhomin,
    registered_variables: RegisteredVariables,
):
    """
    Step (i) of the eigenvectors: the cell primitives with the density floor
    and the interface state between cell i and cell i + 1, built from averages
    of the two cells.

    Args:
        conserved_state: The conserved state, sweep direction first.
        rhomin: The density floor.
        registered_variables: The registered variables.

    Returns:
        The interface density and the three velocity and the three magnetic
        field components.
    """
    # unpack conserved variables
    rho = conserved_state[registered_variables.density_index]
    momentum_x = conserved_state[registered_variables.momentum_index.x]
    momentum_y = conserved_state[registered_variables.momentum_index.y]
    momentum_z = conserved_state[registered_variables.momentum_index.z]
    magnetic_x = conserved_state[registered_variables.magnetic_index.x]
    magnetic_y = conserved_state[registered_variables.magnetic_index.y]
    magnetic_z = conserved_state[registered_variables.magnetic_index.z]

    # compute primitives
    rho = jnp.maximum(rho, rhomin)
    velocity_x = momentum_x / rho
    velocity_y = momentum_y / rho
    velocity_z = momentum_z / rho

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

    return (
        rho_interface,
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        magnetic_x_interface,
        magnetic_y_interface,
        magnetic_z_interface,
    )


def _eigenvector_blocks_iso(
    rho_interface,
    velocity_x_interface,
    velocity_y_interface,
    velocity_z_interface,
    magnetic_x_interface,
    magnetic_y_interface,
    magnetic_z_interface,
    sound_speed,
    cs2,
    fast_magnetosonic_velocity_interface,
    alfven_velocity_interface,
    slow_magnetosonic_velocity_interface,
    fast_magnetosonic_velocity_sq_interface,
    slow_magnetosonic_velocity_sq_interface,
) -> EigenvectorBlocksIso:
    """
    Step (iii) of the eigenvectors: the mode weightings and the remaining
    blocks from an interface state, its wave speeds (step (ii) is
    ``_wave_speeds`` of ``_eigen_mhd.py``) and the squared fast and slow speeds.

    Args:
        rho_interface: The interface density.
        velocity_x_interface: The interface velocity along the sweep direction.
        velocity_y_interface: The first transverse interface velocity.
        velocity_z_interface: The second transverse interface velocity.
        magnetic_x_interface: The interface field along the sweep direction.
        magnetic_y_interface: The first transverse component of the interface field.
        magnetic_z_interface: The second transverse component of the interface field.
        sound_speed: The sound speed at the interface.
        cs2: The squared sound speed at the interface.
        fast_magnetosonic_velocity_interface: The fast speed.
        alfven_velocity_interface: The Alfven speed.
        slow_magnetosonic_velocity_interface: The slow speed.
        fast_magnetosonic_velocity_sq_interface: The squared fast speed
            entering the mode weightings.
        slow_magnetosonic_velocity_sq_interface: The squared slow speed
            entering the mode weightings.

    Returns:
        The blocks the rows and columns of the eigenvectors are assembled from.
    """
    if jax.config.jax_enable_x64:
        eps = 1e-30
    else:
        eps = 1e-20

    # tangential magnetic field normalization
    b_tangential_sq = (
        magnetic_y_interface * magnetic_y_interface
        + magnetic_z_interface * magnetic_z_interface
    )
    sgn_bx = jnp.where(magnetic_x_interface >= 0.0, 1.0, -1.0)

    b_tangential_sq_safe = jnp.maximum(b_tangential_sq, eps)

    bt_normalized_y = jnp.where(
        b_tangential_sq >= eps,
        magnetic_y_interface / jnp.sqrt(b_tangential_sq_safe),
        1.0 / jnp.sqrt(2.0),
    )
    bt_normalized_z = jnp.where(
        b_tangential_sq >= eps,
        magnetic_z_interface / jnp.sqrt(b_tangential_sq_safe),
        1.0 / jnp.sqrt(2.0),
    )

    # mode weightings: af^2 + as^2 = 1
    denom = (
        fast_magnetosonic_velocity_sq_interface
        - slow_magnetosonic_velocity_sq_interface
    )
    denom_safe = jnp.maximum(denom, eps)
    fast_mode_weighting = jnp.where(
        denom >= eps,
        diff_safe_sqrt(
            jnp.maximum(0.0, cs2 - slow_magnetosonic_velocity_sq_interface)
        )
        / diff_safe_sqrt(denom_safe),
        1.0,
    )
    slow_mode_weighting = jnp.where(
        denom >= eps,
        diff_safe_sqrt(
            jnp.maximum(0.0, fast_magnetosonic_velocity_sq_interface - cs2)
        )
        / diff_safe_sqrt(denom_safe),
        1.0,
    )

    sqrt_rho = diff_safe_sqrt(rho_interface)

    cs2_inverse = jnp.where(cs2 > 0.0, 1.0 / cs2, 0.0)

    sgn_bt = jnp.where(
        magnetic_y_interface != 0.0,
        jnp.where(magnetic_y_interface >= 0.0, 1.0, -1.0),
        jnp.where(magnetic_z_interface >= 0.0, 1.0, -1.0),
    )

    sound_speed_greater_alfven_speed = sound_speed >= alfven_velocity_interface

    return EigenvectorBlocksIso(
        rho_interface,
        sqrt_rho,
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        magnetic_x_interface,
        magnetic_y_interface,
        magnetic_z_interface,
        bt_normalized_y,
        bt_normalized_z,
        sgn_bx,
        sgn_bt,
        sound_speed,
        cs2,
        cs2_inverse,
        sound_speed_greater_alfven_speed,
        fast_magnetosonic_velocity_interface,
        alfven_velocity_interface,
        slow_magnetosonic_velocity_interface,
        fast_mode_weighting,
        slow_mode_weighting,
    )


@partial(jax.jit, static_argnames=["registered_variables"])
def _eigenvector_building_blocks(
    conserved_state,
    barotropic_law,
    rhomin,
    registered_variables: RegisteredVariables,
):
    """
    Interface quantities entering the barotropic MHD eigenvectors, see the
    steps (i) to (iii) above; the sound speed is that of the interface density.
    """
    interface_state = _interface_state_iso(
        conserved_state, rhomin, registered_variables
    )
    (
        rho_interface,
        _,
        _,
        _,
        magnetic_x_interface,
        magnetic_y_interface,
        magnetic_z_interface,
    ) = interface_state

    sound_speed, cs2 = law_sound_speed(barotropic_law, rho_interface)

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
        cs2,
    )

    # The mode weightings of this function are formed from the squares of the
    # rounded speeds, as before the split into the steps (i) to (iv).
    return _eigenvector_blocks_iso(
        *interface_state,
        sound_speed,
        cs2,
        fast_magnetosonic_velocity_interface,
        alfven_velocity_interface,
        slow_magnetosonic_velocity_interface,
        fast_magnetosonic_velocity_interface * fast_magnetosonic_velocity_interface,
        slow_magnetosonic_velocity_interface * slow_magnetosonic_velocity_interface,
    )


def _right_eigenvector_columns_iso(
    conserved_state,
    registered_variables: RegisteredVariables,
    blocks: EigenvectorBlocksIso,
):
    """
    Step (iv) of the eigenvectors: the right eigenvectors (columns of R) of the
    barotropic MHD flux Jacobian at the interfaces, one function per
    characteristic mode.

    Args:
        conserved_state: The conserved state, sweep direction first; only its
            shape and dtype enter (every column is shaped like it).
        registered_variables: The registered variables.
        blocks: The interface quantities of ``_eigenvector_blocks_iso``.

    Returns:
        Six functions without arguments, ordered by wave speed (fast -,
        Alfven -, slow -, slow +, Alfven +, fast +); each returns one
        eigenvector, shaped like ``conserved_state``.
    """
    (
        rho_interface,
        sqrt_rho,
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        magnetic_x_interface,
        magnetic_y_interface,
        magnetic_z_interface,
        bt_normalized_y,
        bt_normalized_z,
        sgn_bx,
        sgn_bt,
        sound_speed_val,
        cs2,
        cs2_inverse,
        sound_speed_greater_alfven_speed,
        fast_magnetosonic_velocity_interface,
        alfven_velocity_interface,
        slow_magnetosonic_velocity_interface,
        fast_mode_weighting,
        slow_mode_weighting,
    ) = blocks

    # shorter names for registry indices
    density_index = registered_variables.density_index
    momentum_index_x = registered_variables.momentum_index.x
    momentum_index_y = registered_variables.momentum_index.y
    momentum_index_z = registered_variables.momentum_index.z
    magnetic_index_y = registered_variables.magnetic_index.y
    magnetic_index_z = registered_variables.magnetic_index.z

    def col_0():
        # fast -
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
            sound_speed_val * slow_mode_weighting * bt_normalized_y / sqrt_rho
        )
        R = R.at[magnetic_index_z].set(
            sound_speed_val * slow_mode_weighting * bt_normalized_z / sqrt_rho
        )
        R = jnp.where(~sound_speed_greater_alfven_speed, R * sgn_bt, R)
        return R

    def col_1():
        # alfven -
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(0.0)
        R = R.at[momentum_index_x].set(0.0)
        R = R.at[momentum_index_y].set(-bt_normalized_z)
        R = R.at[momentum_index_z].set(bt_normalized_y)
        R = R.at[magnetic_index_y].set(-bt_normalized_z * sgn_bx / sqrt_rho)
        R = R.at[magnetic_index_z].set(bt_normalized_y * sgn_bx / sqrt_rho)
        return R

    def col_2():
        # slow -
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
            -sound_speed_val * fast_mode_weighting * bt_normalized_y / sqrt_rho
        )
        R = R.at[magnetic_index_z].set(
            -sound_speed_val * fast_mode_weighting * bt_normalized_z / sqrt_rho
        )
        R = jnp.where(sound_speed_greater_alfven_speed, R * sgn_bt, R)
        return R

    def col_3():
        # slow +
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
            -sound_speed_val * fast_mode_weighting * bt_normalized_y / sqrt_rho
        )
        R = R.at[magnetic_index_z].set(
            -sound_speed_val * fast_mode_weighting * bt_normalized_z / sqrt_rho
        )
        R = jnp.where(sound_speed_greater_alfven_speed, R * sgn_bt, R)
        return R

    def col_4():
        # alfven +
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(0.0)
        R = R.at[momentum_index_x].set(0.0)
        R = R.at[momentum_index_y].set(-bt_normalized_z)
        R = R.at[momentum_index_z].set(bt_normalized_y)
        R = R.at[magnetic_index_y].set(bt_normalized_z * sgn_bx / sqrt_rho)
        R = R.at[magnetic_index_z].set(-bt_normalized_y * sgn_bx / sqrt_rho)
        return R

    def col_5():
        # fast +
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
            sound_speed_val * slow_mode_weighting * bt_normalized_y / sqrt_rho
        )
        R = R.at[magnetic_index_z].set(
            sound_speed_val * slow_mode_weighting * bt_normalized_z / sqrt_rho
        )
        R = jnp.where(~sound_speed_greater_alfven_speed, R * sgn_bt, R)
        return R

    return [col_0, col_1, col_2, col_3, col_4, col_5]


@partial(jax.jit, static_argnames=["registered_variables"])
def _eigen_R_col_iso(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    barotropic_law,
    registered_variables: RegisteredVariables,
    col: int,
):
    """
    One right eigenvector (column ``col`` of R) of the barotropic MHD flux
    Jacobian at the interfaces; the modes are ordered by wave speed (fast -,
    Alfven -, slow -, slow +, Alfven +, fast +).
    """
    blocks = _eigenvector_building_blocks(
        conserved_state,
        barotropic_law,
        rhomin,
        registered_variables,
    )

    R = jax.lax.switch(
        col,
        _right_eigenvector_columns_iso(conserved_state, registered_variables, blocks),
    )

    return R


def _left_eigenvector_rows_iso(
    conserved_state,
    registered_variables: RegisteredVariables,
    blocks: EigenvectorBlocksIso,
):
    """
    Step (iv) of the eigenvectors: the left eigenvectors (rows of L) of the
    barotropic MHD flux Jacobian at the interfaces, one function per
    characteristic mode.

    Args:
        conserved_state: The conserved state, sweep direction first; only its
            shape and dtype enter (every row is shaped like it).
        registered_variables: The registered variables.
        blocks: The interface quantities of ``_eigenvector_blocks_iso``.

    Returns:
        Six functions without arguments, ordered by wave speed (fast -,
        Alfven -, slow -, slow +, Alfven +, fast +); each returns one
        eigenvector, shaped like ``conserved_state``.
    """
    (
        rho_interface,
        sqrt_rho,
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        magnetic_x_interface,
        magnetic_y_interface,
        magnetic_z_interface,
        bt_normalized_y,
        bt_normalized_z,
        sgn_bx,
        sgn_bt,
        sound_speed_val,
        cs2,
        cs2_inverse,
        sound_speed_greater_alfven_speed,
        fast_magnetosonic_velocity_interface,
        alfven_velocity_interface,
        slow_magnetosonic_velocity_interface,
        fast_mode_weighting,
        slow_mode_weighting,
    ) = blocks

    # shorter names for registry indices
    density_index = registered_variables.density_index
    momentum_index_x = registered_variables.momentum_index.x
    momentum_index_y = registered_variables.momentum_index.y
    momentum_index_z = registered_variables.momentum_index.z
    magnetic_index_y = registered_variables.magnetic_index.y
    magnetic_index_z = registered_variables.magnetic_index.z

    def row_0():
        # fast -
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(
            fast_mode_weighting
            * (cs2 + fast_magnetosonic_velocity_interface * velocity_x_interface)
            - slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        L = L.at[momentum_index_x].set(
            -fast_mode_weighting * fast_magnetosonic_velocity_interface
        )
        L = L.at[momentum_index_y].set(
            slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        L = L.at[momentum_index_z].set(
            slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        L = L.at[magnetic_index_y].set(
            sound_speed_val * slow_mode_weighting * bt_normalized_y * sqrt_rho
        )
        L = L.at[magnetic_index_z].set(
            sound_speed_val * slow_mode_weighting * bt_normalized_z * sqrt_rho
        )
        L = 0.5 * L * cs2_inverse
        L = jnp.where(~sound_speed_greater_alfven_speed, L * sgn_bt, L)
        return L

    def row_1():
        # alfven -
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
        L = 0.5 * L
        return L

    def row_2():
        # slow -
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(
            slow_mode_weighting
            * (cs2 + slow_magnetosonic_velocity_interface * velocity_x_interface)
            + fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        L = L.at[momentum_index_x].set(
            -slow_mode_weighting * slow_magnetosonic_velocity_interface
        )
        L = L.at[momentum_index_y].set(
            -fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        L = L.at[momentum_index_z].set(
            -fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        L = L.at[magnetic_index_y].set(
            -sound_speed_val * fast_mode_weighting * bt_normalized_y * sqrt_rho
        )
        L = L.at[magnetic_index_z].set(
            -sound_speed_val * fast_mode_weighting * bt_normalized_z * sqrt_rho
        )
        L = 0.5 * L * cs2_inverse
        L = jnp.where(sound_speed_greater_alfven_speed, L * sgn_bt, L)
        return L

    def row_3():
        # slow +
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(
            slow_mode_weighting
            * (cs2 - slow_magnetosonic_velocity_interface * velocity_x_interface)
            - fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        L = L.at[momentum_index_x].set(
            slow_mode_weighting * slow_magnetosonic_velocity_interface
        )
        L = L.at[momentum_index_y].set(
            fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        L = L.at[momentum_index_z].set(
            fast_mode_weighting
            * fast_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        L = L.at[magnetic_index_y].set(
            -sound_speed_val * fast_mode_weighting * bt_normalized_y * sqrt_rho
        )
        L = L.at[magnetic_index_z].set(
            -sound_speed_val * fast_mode_weighting * bt_normalized_z * sqrt_rho
        )
        L = 0.5 * L * cs2_inverse
        L = jnp.where(sound_speed_greater_alfven_speed, L * sgn_bt, L)
        return L

    def row_4():
        # alfven +
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
        L = 0.5 * L
        return L

    def row_5():
        # fast +
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(
            fast_mode_weighting
            * (cs2 - fast_magnetosonic_velocity_interface * velocity_x_interface)
            + slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * (
                bt_normalized_y * velocity_y_interface
                + bt_normalized_z * velocity_z_interface
            )
            * sgn_bx
        )
        L = L.at[momentum_index_x].set(
            fast_mode_weighting * fast_magnetosonic_velocity_interface
        )
        L = L.at[momentum_index_y].set(
            -slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_y
            * sgn_bx
        )
        L = L.at[momentum_index_z].set(
            -slow_mode_weighting
            * slow_magnetosonic_velocity_interface
            * bt_normalized_z
            * sgn_bx
        )
        L = L.at[magnetic_index_y].set(
            sound_speed_val * slow_mode_weighting * bt_normalized_y * sqrt_rho
        )
        L = L.at[magnetic_index_z].set(
            sound_speed_val * slow_mode_weighting * bt_normalized_z * sqrt_rho
        )
        L = 0.5 * L * cs2_inverse
        L = jnp.where(~sound_speed_greater_alfven_speed, L * sgn_bt, L)
        return L

    return [row_0, row_1, row_2, row_3, row_4, row_5]


@partial(jax.jit, static_argnames=["registered_variables"])
def _eigen_L_row_iso(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    barotropic_law,
    registered_variables: RegisteredVariables,
    row: int,
):
    """
    One left eigenvector (row ``row`` of L) of the barotropic MHD flux
    Jacobian at the interfaces; the modes are ordered by wave speed (fast -,
    Alfven -, slow -, slow +, Alfven +, fast +).
    """
    blocks = _eigenvector_building_blocks(
        conserved_state,
        barotropic_law,
        rhomin,
        registered_variables,
    )

    L = jax.lax.switch(
        row, _left_eigenvector_rows_iso(conserved_state, registered_variables, blocks)
    )

    return L


@partial(jax.jit, static_argnames=["registered_variables"])
def _eigen_all_lambdas_iso(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    barotropic_law,
    registered_variables: RegisteredVariables,
):
    (
        velocity_x,
        fast_magnetosonic_velocity,
        alfven_velocity,
        slow_magnetosonic_velocity,
    ) = _eigenvalue_building_blocks(
        conserved_state,
        barotropic_law,
        rhomin,
        registered_variables,
    )

    return jnp.stack(
        [
            velocity_x - fast_magnetosonic_velocity,
            velocity_x - alfven_velocity,
            velocity_x - slow_magnetosonic_velocity,
            velocity_x + slow_magnetosonic_velocity,
            velocity_x + alfven_velocity,
            velocity_x + fast_magnetosonic_velocity,
        ],
        axis=0,
    )


def _eigen_lambdas_iso(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    barotropic_law,
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
        barotropic_law,
        rhomin,
        registered_variables,
    )

    def mode_0():
        return velocity_x - fast_magnetosonic_velocity

    def mode_1():
        return velocity_x - alfven_velocity

    def mode_2():
        return velocity_x - slow_magnetosonic_velocity

    def mode_3():
        return velocity_x + slow_magnetosonic_velocity

    def mode_4():
        return velocity_x + alfven_velocity

    def mode_5():
        return velocity_x + fast_magnetosonic_velocity

    return jax.lax.switch(
        mode, [mode_0, mode_1, mode_2, mode_3, mode_4, mode_5]
    )