"""
Pallas kernels of the finite-difference MHD flux of ``_riemann_flux.py``.

This module is the compiled output of the array form: it holds the same
statements, written for one tile (the values of one block of faces) instead
of the whole grid. ``_riemann_flux.py`` is the description of the scheme and
the path every derivative takes; nothing is decided here.

The functions carry the names and the order of the array form, so that the
two files can be read side by side. Where a function differs, its docstring
says how. The differences are of three kinds:

  * A tile function takes the values of the stencil cells as separate
    arguments where the array form shifts whole arrays.
  * Steps 2 and 3 use the products of the eigenvector matrices with the
    transformation between conserved and primitive variables written out
    (``_characteristic_variables``, ``_primitives_from_characteristic``): the
    interface velocity drops out of them, and the 7 by 7 (6 by 6) products
    reduce to the sums and differences of the wave pairs. The WENO-Z values
    of the two sides of a face share their stencil differences
    (``_weno5z_point_pair``), and so do the two monotonicity-preserving
    bounds (``_mp_bounded_pair``).
  * Constants in the branches of a select are typed scalars (``setup.zero``,
    ``setup.one``): a kernel has no weak types there.

Two kernels per axis, both with the stencil halo of 3 cells:

  flux kernel   steps 1 to 6 of ``_interface_flux_x``: the H-corrected flux
                F_H at the face i + 1/2 from the cells i - 2 ... i + 3
  guard kernel  step 7: the guarded flux from the cells i - 2 ... i + 3 and
                F_H at the faces i - 1/2, i + 1/2, i + 3/2

The guard at a face needs F_H of three faces, so a single kernel would compute
steps 1 to 6 three times per face.

The kernels are never differentiated: ``_riemann_flux.interface_flux`` calls
them through ``diffable_pallas_call_n``, which takes the tangent and the
cotangent from the array form. The quotients written out below therefore
carry the forward rule only; the derivative rules stay in ``_ratio_rules.py``.
"""

# general
from typing import NamedTuple, Tuple

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import HLLD_BOUNDED, HLLEM_T, IDEAL_GAS, ISOTHERMAL, RIEMANN_BY_EOS

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _backend_is_pallas,
    _pallas_call_sharded,
    _pallas_compiler_params,
    _shard_cells,
    pl,
)
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _mhd_indices_for_axis,
    _mhd_iso_indices_for_axis,
)

# The constants of ``_riemann_flux.py``. They are repeated here because that
# module imports this one at its bottom (the pattern of the package).
WENO_EPSILON_RELATIVE = 1e-6
EPSILON_FLOOR = 1e-30
DIVISION_FLOOR = 1e-30
HOLE_FRACTION = 0.5

# halo of the state in both kernels, and of F_H in the guard kernel
_HALO = 3
_GUARD_FLUX_HALO = 1


class KernelSetup(NamedTuple):
    """
    The kernel's counterpart of ``SchemeSetup``: the static choices and the
    scalars of a kernel, the latter typed like the state.
    """

    #: ``HLL``, ``HLLEM_T`` or ``HLLD_BOUNDED`` (``RIEMANN_BY_EOS`` resolved).
    riemann_solver: int
    ideal_gas: bool
    #: Adiabatic index (ideal gas, else None).
    gamma: object
    #: Isothermal sound speed and its square (isothermal gas, else None).
    sound_speed: object
    sound_speed_squared: object
    minimum_density: object
    #: Pressure floor (ideal gas, else None).
    minimum_pressure: object
    #: The floor of ``_eigen_mhd._wave_speeds`` and of the eigenvector blocks.
    eigen_floor: object
    zero: object
    one: object
    minus_one: object
    inv_sqrt_two: object


def _kernel_setup(scalars, config: SimulationConfig) -> KernelSetup:
    """
    The setup from the leading scalar inputs of a kernel: (gamma, minimum
    density, minimum pressure, eigen floor) for the ideal gas, (sound speed,
    squared sound speed, minimum density, eigen floor) for the isothermal
    gas. The squared sound speed is an input because the array form squares
    the parameter before it takes the precision of the state: the square of
    the rounded sound speed differs from it by one unit in the last place.
    """
    ideal_gas = config.equation_of_state == IDEAL_GAS
    if ideal_gas:
        gamma, minimum_density, minimum_pressure, eigen_floor = scalars[:4]
        sound_speed = sound_speed_squared = None
        zero = gamma - gamma
    else:
        sound_speed, sound_speed_squared, minimum_density, eigen_floor = scalars[:4]
        gamma = minimum_pressure = None
        zero = sound_speed - sound_speed
    riemann_solver = config.solver_config.riemann_solver
    if riemann_solver == RIEMANN_BY_EOS:
        riemann_solver = HLLEM_T if config.equation_of_state == ISOTHERMAL else HLLD_BOUNDED
    return KernelSetup(
        riemann_solver=riemann_solver,
        ideal_gas=ideal_gas,
        gamma=gamma,
        sound_speed=sound_speed,
        sound_speed_squared=sound_speed_squared,
        minimum_density=minimum_density,
        minimum_pressure=minimum_pressure,
        eigen_floor=eigen_floor,
        zero=zero,
        one=zero + 1.0,
        minus_one=zero - 1.0,
        inv_sqrt_two=zero + (1.0 / 2.0**0.5),
    )


# ============================== containers ===================================


class Primitives(NamedTuple):
    """The primitive variables V = (rho, v, B_y, B_z, P) of one cell or face side."""

    rho: object
    vx: object
    vy: object
    vz: object
    By: object
    Bz: object
    P: object


class InterfaceState(NamedTuple):
    """The interface state of the basis. Its velocity is not needed (module docstring)."""

    rho: object
    Bx: object
    By: object
    Bz: object
    a2: object


class Basis(NamedTuple):
    """
    The coefficients of the products L T (rows) and T^-1 R (columns), T the
    transformation from primitive to linearised conserved variables, in the
    frame of the transverse field direction (bt_y, bt_z).
    """

    bt_y: object
    bt_z: object
    fast_from_thermal: object
    fast_from_field: object
    fast_from_normal: object
    fast_from_parallel: object
    slow_from_thermal: object
    slow_from_field: object
    slow_from_normal: object
    slow_from_parallel: object
    alfven_from_velocity: object
    alfven_from_field: object
    entropy_from_density: object
    a2_inv: object
    a_f: object
    a_s: object
    sgn_bx: object
    inv_rho: object
    normal_from_fast: object
    normal_from_slow: object
    field_from_fast: object
    field_from_slow: object
    field_from_alfven: object
    energy_from_fast: object
    energy_from_slow: object
    By: object
    Bz: object


class FluxState(NamedTuple):
    """
    A state with what the fluxes need. ``U`` and ``F`` are tuples of the
    components that carry a flux, in the order rho, m_x, m_y, m_z, B_y, B_z
    (, E); the speeds are None where they were left out.
    """

    V: Primitives
    U: Tuple
    F: Tuple
    P_T: object
    a2: object
    c_f: object
    c_f2: object


class FanState(NamedTuple):
    """A state of the HLLD fan: density, transverse velocity and field, energy, v_t . B_t."""

    rho: object
    vy: object
    vz: object
    By: object
    Bz: object
    E: object
    vB: object


# ===================== speeds, physical flux (shared) ========================


def _sound_speed_squared(rho, P, setup: KernelSetup):
    """a^2 = gamma P / rho (ideal gas) or c_s^2 (isothermal)."""
    if setup.ideal_gas:
        return setup.gamma * P / rho
    return setup.sound_speed_squared


def _wave_speeds(rho, B_x, B_y, B_z, a2, setup: KernelSetup, slow: bool = True):
    """
    The statements of ``_eigen_mhd._wave_speeds`` (the one fast-speed formula
    of the scheme), with the floors of that function: (c_f, c_s, c_f^2,
    c_s^2), or (c_f, c_f^2) with ``slow`` off.
    """
    zero, eps = setup.zero, setup.eigen_floor
    b_sq_over_rho = (B_x * B_x + B_y * B_y + B_z * B_z) / rho
    bx_sq_over_rho = (B_x * B_x) / rho
    discriminant = (b_sq_over_rho + a2) ** 2 - 4.0 * bx_sq_over_rho * a2
    root = jnp.sqrt(jnp.maximum(jnp.maximum(zero, discriminant), eps))
    c_f2 = jnp.maximum(jnp.maximum(zero, 0.5 * (b_sq_over_rho + a2 + root)), eps)
    c_f = jnp.sqrt(c_f2)
    if not slow:
        return c_f, c_f2
    c_s2 = jnp.maximum(jnp.maximum(zero, 0.5 * (b_sq_over_rho + a2 - root)), eps)
    return c_f, jnp.sqrt(c_s2), c_f2, c_s2


def _flux_state(V: Primitives, B_x, setup: KernelSetup, speeds: bool = True) -> FluxState:
    """
    Conserved variables, physical flux, total pressure and fast speed of the
    primitive state V with the normal field B_x. ``speeds`` off leaves out
    a^2 and the fast speed (the cells of the cell-flux correction).
    """
    rho, v_x, v_y, v_z, B_y, B_z, P = V
    B2 = B_x * B_x + B_y * B_y + B_z * B_z
    P_T = P + 0.5 * B2
    U = (rho, rho * v_x, rho * v_y, rho * v_z, B_y, B_z)
    F = (
        rho * v_x,
        rho * v_x * v_x + P_T - B_x * B_x,
        rho * v_x * v_y - B_x * B_y,
        rho * v_x * v_z - B_x * B_z,
        B_y * v_x - B_x * v_y,
        B_z * v_x - B_x * v_z,
    )
    if setup.ideal_gas:
        v2 = v_x * v_x + v_y * v_y + v_z * v_z
        E = P / (setup.gamma - 1.0) + 0.5 * rho * v2 + 0.5 * B2
        F_E = (E + P_T) * v_x - (v_x * B_x + v_y * B_y + v_z * B_z) * B_x
        U = U + (E,)
        F = F + (F_E,)
    if not speeds:
        return FluxState(V, U, F, P_T, None, None, None)
    a2 = _sound_speed_squared(rho, P, setup)
    c_f, c_f2 = _wave_speeds(rho, B_x, B_y, B_z, a2, setup, slow=False)
    return FluxState(V, U, F, P_T, a2, c_f, c_f2)


# ========================= 1. cell primitives ================================


def _cell_primitives(q, setup: KernelSetup) -> Tuple[Primitives, object]:
    """
    The primitive variables and the normal field of one cell from its
    conserved variables q = (rho, m_x, m_y, m_z, B_x, B_y, B_z(, E)): the
    density raised to its floor, the momentum and the energy kept, the
    pressure raised to its floor.
    """
    rho = jnp.maximum(q[0], setup.minimum_density)
    m_x, m_y, m_z, B_x, B_y, B_z = q[1:7]
    v_x, v_y, v_z = m_x / rho, m_y / rho, m_z / rho
    if setup.ideal_gas:
        P = (setup.gamma - 1.0) * (
            q[7]
            - 0.5 * (m_x * v_x + m_y * v_y + m_z * v_z)
            - 0.5 * (B_x * B_x + B_y * B_y + B_z * B_z)
        )
        P = jnp.maximum(P, setup.minimum_pressure)
    else:
        P = setup.sound_speed_squared * rho
    return Primitives(rho, v_x, v_y, v_z, B_y, B_z, P), B_x


# ================= 2. interface state, characteristic basis ==================


def _interface_state(
    V_i: Primitives, V_ip1: Primitives, B_x_i, B_x_ip1, setup: KernelSetup
) -> InterfaceState:
    """The arithmetic mean of the cells i and i + 1, a^2 from the mean pressure."""
    rho = jnp.maximum(0.5 * (V_i.rho + V_ip1.rho), setup.minimum_density)
    a2 = _sound_speed_squared(rho, 0.5 * (V_i.P + V_ip1.P), setup)
    return InterfaceState(
        rho,
        0.5 * (B_x_i + B_x_ip1),
        0.5 * (V_i.By + V_ip1.By),
        0.5 * (V_i.Bz + V_ip1.Bz),
        a2,
    )


def _safe_sqrt(x, setup: KernelSetup):
    """``diff_safe_sqrt`` of the eigenvector modules."""
    return jnp.sqrt(jnp.maximum(x, setup.eigen_floor))


def _characteristic_basis(bar: InterfaceState, setup: KernelSetup) -> Basis:
    """
    The coefficients of the characteristic projection at the interface state:
    the mode weights a_f, a_s of ``_eigenvector_blocks`` (from the squared
    speeds of ``_wave_speeds``; no rounded root is squared) and the entries
    of L T and T^-1 R that do not vanish.
    """
    zero, one, eps = setup.zero, setup.one, setup.eigen_floor
    rho, B_x, B_y, B_z, a2 = bar
    c_f, c_s, c_f2, c_s2 = _wave_speeds(rho, B_x, B_y, B_z, a2, setup)
    if setup.ideal_gas:
        a = _safe_sqrt(jnp.maximum(zero, a2), setup)
    else:
        a = setup.sound_speed

    bt2 = B_y * B_y + B_z * B_z
    sgn_bx = jnp.where(B_x >= 0.0, one, setup.minus_one)
    bt2_safe = jnp.maximum(bt2, eps)
    bt_y = jnp.where(bt2 >= eps, B_y / jnp.sqrt(bt2_safe), setup.inv_sqrt_two)
    bt_z = jnp.where(bt2 >= eps, B_z / jnp.sqrt(bt2_safe), setup.inv_sqrt_two)

    denom = c_f2 - c_s2
    root_denom = _safe_sqrt(jnp.maximum(denom, eps), setup)
    a_f = jnp.where(denom >= eps, _safe_sqrt(jnp.maximum(zero, a2 - c_s2), setup) / root_denom, one)
    a_s = jnp.where(denom >= eps, _safe_sqrt(jnp.maximum(zero, c_f2 - a2), setup) / root_denom, one)

    sqrt_rho = _safe_sqrt(rho, setup)
    a2_inv = jnp.where(a2 > 0.0, 1.0 / a2, zero)
    half_a2_inv = 0.5 * a2_inv
    momentum_fast = rho * (a_f * c_f)
    momentum_slow = rho * (a_s * c_s)
    if setup.ideal_gas:
        # thermal variable P
        fast_from_thermal = half_a2_inv * a_f
        slow_from_thermal = half_a2_inv * a_s
        entropy_from_density = jnp.where(a2 > 0.0, one, zero)
        gam2 = (setup.gamma - 2.0) / (setup.gamma - 1.0)
        energy_from_fast = a_f * (c_f2 - gam2 * a2)
        energy_from_slow = a_s * (c_s2 - gam2 * a2)
    else:
        # thermal variable rho
        fast_from_thermal = half_a2_inv * (a_f * a2)
        slow_from_thermal = half_a2_inv * (a_s * a2)
        entropy_from_density = energy_from_fast = energy_from_slow = None
    inv_rho = 1.0 / rho
    inv_sqrt_rho = 1.0 / sqrt_rho
    return Basis(
        bt_y=bt_y,
        bt_z=bt_z,
        fast_from_thermal=fast_from_thermal,
        fast_from_field=half_a2_inv * (a * a_s * sqrt_rho),
        fast_from_normal=half_a2_inv * momentum_fast,
        fast_from_parallel=half_a2_inv * (sgn_bx * momentum_slow),
        slow_from_thermal=slow_from_thermal,
        slow_from_field=half_a2_inv * (a * a_f * sqrt_rho),
        slow_from_normal=half_a2_inv * momentum_slow,
        slow_from_parallel=half_a2_inv * (sgn_bx * momentum_fast),
        alfven_from_velocity=-0.5 * rho,
        alfven_from_field=0.5 * (sgn_bx * sqrt_rho),
        entropy_from_density=entropy_from_density,
        a2_inv=a2_inv,
        a_f=a_f,
        a_s=a_s,
        sgn_bx=sgn_bx,
        inv_rho=inv_rho,
        normal_from_fast=inv_rho * (a_f * c_f),
        normal_from_slow=inv_rho * (a_s * c_s),
        field_from_fast=inv_sqrt_rho * (a * a_s),
        field_from_slow=inv_sqrt_rho * (a * a_f),
        field_from_alfven=inv_sqrt_rho * sgn_bx,
        energy_from_fast=energy_from_fast,
        energy_from_slow=energy_from_slow,
        By=B_y,
        Bz=B_z,
    )


def _characteristic_variables(V: Primitives, basis: Basis, setup: KernelSetup) -> Tuple:
    """
    w = L T V, the characteristic variables of the primitive state of one
    cell (in place of ``_linearised_conserved`` followed by the rows): in the
    order of the eigenvector modules, fast-, Alfven-, slow-, (entropy,)
    slow+, Alfven+, fast+.
    """
    b = basis
    thermal = V.P if setup.ideal_gas else V.rho
    v_par = b.bt_y * V.vy + b.bt_z * V.vz
    v_perp = b.bt_z * V.vy - b.bt_y * V.vz
    b_par = b.bt_y * V.By + b.bt_z * V.Bz
    b_perp = b.bt_z * V.By - b.bt_y * V.Bz
    even_fast = b.fast_from_thermal * thermal + b.fast_from_field * b_par
    odd_fast = b.fast_from_normal * V.vx - b.fast_from_parallel * v_par
    even_slow = b.slow_from_thermal * thermal - b.slow_from_field * b_par
    odd_slow = b.slow_from_normal * V.vx + b.slow_from_parallel * v_par
    even_alfven = b.alfven_from_velocity * v_perp
    odd_alfven = b.alfven_from_field * b_perp
    minus = (even_fast - odd_fast, even_alfven - odd_alfven, even_slow - odd_slow)
    plus = (even_slow + odd_slow, even_alfven + odd_alfven, even_fast + odd_fast)
    if setup.ideal_gas:
        return minus + (b.entropy_from_density * V.rho - b.a2_inv * V.P,) + plus
    return minus + plus


def _primitives_from_characteristic(w, basis: Basis, setup: KernelSetup) -> Primitives:
    """
    V = T^-1 R w (in place of the columns followed by
    ``_primitives_from_linearised``), by the sums S and differences D of the
    wave pairs:

        rho = a_f S_f + a_s S_s (+ w_entropy),
        v_x = (a_f c_f D_f + a_s c_s D_s) / rho_bar,
        v_par = sgn(B_x) (a_f c_f D_s - a_s c_s D_f) / rho_bar,
        v_perp = -S_a / rho_bar,
        B_par = a (a_s S_f - a_f S_s) / sqrt(rho_bar),
        B_perp = sgn(B_x) D_a / sqrt(rho_bar),
        P = (gamma - 1) (a_f (c_f^2 - g a^2) S_f + a_s (c_s^2 - g a^2) S_s
                         - B_y,bar B_y - B_z,bar B_z),  g = (gamma - 2) / (gamma - 1).

    The isothermal pressure is set by the caller (None here).
    """
    b = basis
    if setup.ideal_gas:
        fast_minus, alfven_minus, slow_minus, entropy, slow_plus, alfven_plus, fast_plus = w
    else:
        fast_minus, alfven_minus, slow_minus, slow_plus, alfven_plus, fast_plus = w
    sum_fast, difference_fast = fast_plus + fast_minus, fast_plus - fast_minus
    sum_slow, difference_slow = slow_plus + slow_minus, slow_plus - slow_minus
    sum_alfven, difference_alfven = alfven_plus + alfven_minus, alfven_plus - alfven_minus
    v_par = b.sgn_bx * (b.normal_from_fast * difference_slow - b.normal_from_slow * difference_fast)
    v_perp = -b.inv_rho * sum_alfven
    b_par = b.field_from_fast * sum_fast - b.field_from_slow * sum_slow
    b_perp = b.field_from_alfven * difference_alfven
    rho = b.a_f * sum_fast + b.a_s * sum_slow
    v_x = b.normal_from_fast * difference_fast + b.normal_from_slow * difference_slow
    v_y = b.bt_y * v_par + b.bt_z * v_perp
    v_z = b.bt_z * v_par - b.bt_y * v_perp
    B_y = b.bt_y * b_par + b.bt_z * b_perp
    B_z = b.bt_z * b_par - b.bt_y * b_perp
    P = None
    if setup.ideal_gas:
        rho = rho + entropy
        P = (setup.gamma - 1.0) * (
            b.energy_from_fast * sum_fast + b.energy_from_slow * sum_slow - b.By * B_y - b.Bz * B_z
        )
    return Primitives(rho, v_x, v_y, v_z, B_y, B_z, P)


# ============================= 3. face states ================================


def _minmod2(a, b):
    """minmod of two values (zero unless both have one sign), by minima and maxima."""
    return jnp.maximum(jnp.minimum(a, b), 0.0) + jnp.minimum(jnp.maximum(a, b), 0.0)


def _minmod4(a, b, c, d):
    """minmod of four values."""
    smallest = jnp.minimum(jnp.minimum(a, b), jnp.minimum(c, d))
    largest = jnp.maximum(jnp.maximum(a, b), jnp.maximum(c, d))
    return jnp.maximum(smallest, 0.0) + jnp.minimum(largest, 0.0)


def _stencil_differences(w0, w1, w2, w3, w4, w5):
    """First and second differences of the six stencil values (shared by the two sides)."""
    e1, e2, e3, e4, e5 = w1 - w0, w2 - w1, w3 - w2, w4 - w3, w5 - w4
    return (e1, e2, e3, e4, e5), (e2 - e1, e3 - e2, e4 - e3, e5 - e4)


def _mp_bounded_pair(u_L, u_R, u, differences=None):
    """
    The monotonicity-preserving bound (``_mp_limit``, alpha = 4) of the two
    face values of the interface between the cells u[2] and u[3]; u holds the
    six stencil values. The two sides share the median bounds of the face and
    the stencil differences.
    """
    alpha = 4.0
    w0, w1, w2, w3, w4, w5 = u
    if differences is None:
        differences = _stencil_differences(w0, w1, w2, w3, w4, w5)
    (_, e2, e3, e4, _), (d1, d2, d3, d4) = differences
    d_m4_face = _minmod4(4.0 * d2 - d3, 4.0 * d3 - d2, d2, d3)
    u_md = 0.5 * ((w2 + w3) - d_m4_face)
    lower_md = jnp.minimum(jnp.minimum(w2, w3), u_md)
    upper_md = jnp.maximum(jnp.maximum(w2, w3), u_md)
    magnitude = jnp.maximum(jnp.abs(w2), jnp.abs(w3))

    def one_side(value, um1, u0, backward, forward, d_m, d_0):
        scale = jnp.maximum(jnp.abs(um1), magnitude) ** 2 * 1e-20 + EPSILON_FLOOR
        upwind_slope = alpha * backward
        u_mp = u0 + _minmod2(forward, upwind_slope)
        keep = (value - u0) * (value - u_mp) <= scale
        d_m4_m = _minmod4(4.0 * d_m - d_0, 4.0 * d_0 - d_m, d_m, d_0)
        u_ul = u0 + upwind_slope
        u_lc = u0 + 0.5 * backward + (4.0 / 3.0) * d_m4_m
        u_min = jnp.maximum(lower_md, jnp.minimum(jnp.minimum(u0, u_ul), u_lc))
        u_max = jnp.minimum(upper_md, jnp.maximum(jnp.maximum(u0, u_ul), u_lc))
        clipped = jnp.minimum(jnp.maximum(value, u_min), u_max)
        return jnp.where(keep, value, clipped)

    return (
        one_side(u_L, w1, w2, e2, e3, d1, d2),
        one_side(u_R, w4, w3, -e4, -e3, d4, d3),
    )


def _weno5z_point_pair(w2, w3, differences, epsilon):
    """
    The WENO5-Z point values of the two sides of the face between w2 and w3
    (``_weno5z_point_interpolation`` for both sides), written with the
    stencil differences: both values are the mean of w2 and w3 less a
    weighted second difference.
    """
    (e1, e2, e3, e4, e5), (d1, d2, d3, d4) = differences
    half_sum = 0.5 * (w2 + w3)
    curvature1, curvature2, curvature3, curvature4 = (
        (13.0 / 3.0) * (d * d) for d in (d1, d2, d3, d4)
    )
    three_e3 = 3.0 * e3
    epsilon4 = 4.0 * epsilon

    def one_side(beta0, beta1, beta2, d_far, d_near, d_other):
        tau = jnp.abs(beta0 - beta2)
        alpha0 = 1.0 + tau / (beta0 + epsilon4)
        alpha1 = 10.0 * (1.0 + tau / (beta1 + epsilon4))
        alpha2 = 5.0 * (1.0 + tau / (beta2 + epsilon4))
        weighted = alpha0 * (4.0 * d_near - 3.0 * d_far) + alpha1 * d_near + alpha2 * d_other
        return half_sum - 0.125 * (weighted / (alpha0 + alpha1 + alpha2))

    left = one_side(
        curvature1 + (3.0 * e2 - e1) ** 2,
        curvature2 + (e2 + e3) ** 2,
        curvature3 + (e4 - three_e3) ** 2,
        d1,
        d2,
        d3,
    )
    right = one_side(
        curvature4 + (e5 - 3.0 * e4) ** 2,
        curvature3 + (e3 + e4) ** 2,
        curvature2 + (three_e3 - e2) ** 2,
        d4,
        d3,
        d2,
    )
    return left, right


def _weno5z_face_values(w, differences):
    """
    The WENO5-Z values of one characteristic variable at the two sides of the
    face; w holds its six stencil values. The epsilon of the weights is
    relative to the largest square in the stencil.
    """
    scale2 = w[0] * w[0]
    for value in w[1:]:
        scale2 = jnp.maximum(scale2, value * value)
    epsilon = WENO_EPSILON_RELATIVE * scale2 + EPSILON_FLOOR
    return _weno5z_point_pair(w[2], w[3], differences, epsilon)


def _admissible(
    V_face: Primitives, V_own: Primitives, V_other: Primitives, setup: KernelSetup
) -> Primitives:
    """
    Scales the face state toward its own cell until its density (and
    pressure) is at least half the smaller of the two adjacent cell values.
    """
    one = setup.one
    rho_floor = 0.5 * jnp.minimum(V_own.rho, V_other.rho)
    vartheta = jnp.where(
        V_face.rho < rho_floor,
        (V_own.rho - rho_floor) / jnp.maximum(V_own.rho - V_face.rho, DIVISION_FLOOR),
        one,
    )
    if setup.ideal_gas:
        P_floor = 0.5 * jnp.minimum(V_own.P, V_other.P)
        vartheta_P = jnp.where(
            V_face.P < P_floor,
            (V_own.P - P_floor) / jnp.maximum(V_own.P - V_face.P, DIVISION_FLOOR),
            one,
        )
        vartheta = jnp.minimum(vartheta, vartheta_P)
    vartheta = jnp.clip(vartheta, 0.0, 1.0)
    return Primitives(*(own + vartheta * (face - own) for face, own in zip(V_face, V_own)))


def _face_states(V, basis: Basis, setup: KernelSetup) -> Tuple[Primitives, Primitives]:
    """
    The two primitive states at the face i + 1/2 from the primitives V of the
    cells i - 2 ... i + 3: WENO5-Z of the characteristic variables with the
    monotonicity-preserving bound, back to primitives, the bound once more on
    the primitives, the admissibility scaling.
    """
    w = [_characteristic_variables(V_cell, basis, setup) for V_cell in V]
    w_L, w_R = [], []
    for wave in zip(*w):  # the six stencil values of one wave
        differences = _stencil_differences(*wave)
        left, right = _weno5z_face_values(wave, differences)
        left, right = _mp_bounded_pair(left, right, wave, differences)
        w_L.append(left)
        w_R.append(right)
    V_L = _primitives_from_characteristic(w_L, basis, setup)
    V_R = _primitives_from_characteristic(w_R, basis, setup)

    bounded_L, bounded_R = [], []
    for index in range(7 if setup.ideal_gas else 6):
        left, right = _mp_bounded_pair(
            V_L[index], V_R[index], tuple(V_cell[index] for V_cell in V)
        )
        bounded_L.append(left)
        bounded_R.append(right)
    if not setup.ideal_gas:
        bounded_L.append(setup.sound_speed_squared * bounded_L[0])
        bounded_R.append(setup.sound_speed_squared * bounded_R[0])
    V_i, V_ip1 = V[2], V[3]
    return (
        _admissible(Primitives(*bounded_L), V_i, V_ip1, setup),
        _admissible(Primitives(*bounded_R), V_ip1, V_i, setup),
    )


# ============================ 4. Riemann flux ================================


def _hll_flux(L: FluxState, R: FluxState, S_L, S_R, inv) -> Tuple:
    """F_HLL = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)."""
    return tuple(
        (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) * inv
        for F_L, F_R, U_L, U_R in zip(L.F, R.F, L.U, R.U)
    )


def _anti_diffusion(mu, S_L, S_R, inv):
    """The HLLEM coefficient of a wave of speed mu (clipped into the fan)."""
    mu = jnp.clip(mu, S_L, S_R)
    return (S_R * jnp.minimum(mu, 0.0) + S_L * jnp.maximum(mu, 0.0) - S_L * S_R) * inv


def _hllem_t_flux(L: FluxState, R: FluxState, B_x, S_L, S_R, inv, setup: KernelSetup) -> Tuple:
    """
    The isothermal HLLEM flux with the two Alfven waves restored (the
    statements of the array form; components rho, m_x, m_y, m_z, B_y, B_z).
    """

    def hll_state(n):
        return (S_R * R.U[n] - S_L * L.U[n] - (R.F[n] - L.F[n])) * inv

    rho = hll_state(0)
    sqrt_rho = jnp.sqrt(rho)
    u = hll_state(1) / rho
    v_y = hll_state(2) / rho
    v_z = hll_state(3) / rho
    s = jnp.where(B_x >= 0.0, setup.one, setup.minus_one)

    d_rho = R.U[0] - L.U[0]
    d_v_y = (R.U[2] - L.U[2] - v_y * d_rho) / sqrt_rho
    d_v_z = (R.U[3] - L.U[3] - v_z * d_rho) / sqrt_rho
    d_B_y = R.U[4] - L.U[4]
    d_B_z = R.U[5] - L.U[5]

    c_a = jnp.abs(B_x) / sqrt_rho
    c_plus = _anti_diffusion(u + c_a, S_L, S_R, inv)
    c_minus = _anti_diffusion(u - c_a, S_L, S_R, inv)
    plus_y = 0.5 * c_plus * (d_v_y - s * d_B_y)
    minus_y = 0.5 * c_minus * (d_v_y + s * d_B_y)
    plus_z = 0.5 * c_plus * (d_v_z - s * d_B_z)
    minus_z = 0.5 * c_minus * (d_v_z + s * d_B_z)

    F = list(_hll_flux(L, R, S_L, S_R, inv))
    F[2] = F[2] + sqrt_rho * (plus_y + minus_y)
    F[3] = F[3] + sqrt_rho * (plus_z + minus_z)
    F[4] = F[4] + s * (minus_y - plus_y)
    F[5] = F[5] + s * (minus_z - plus_z)
    return tuple(F)


def _fast_alfven_gap(rho, B_x, B_y, B_z, a2, setup: KernelSetup):
    """G = rho (c_f^2 - B_x^2 / rho) >= B_t^2 without cancellation."""
    B_x2 = B_x * B_x
    B_t2 = B_y * B_y + B_z * B_z
    d = rho * a2 + B_t2 - B_x2
    r = jnp.sqrt(d * d + 4.0 * B_x2 * B_t2)
    falling = d < 0.0
    return jnp.where(
        falling, 2.0 * B_x2 * B_t2 / jnp.where(falling, r - d, setup.one), 0.5 * (d + r)
    )


def _hlld_star_state(
    side: FluxState, j, w, du, nu: float, B_x, S_M, P_T_star, Gamma, setup: KernelSetup
) -> FanState:
    """The state behind the outer wave of one side; nu = +1 on the left, -1 on the right."""
    V = side.V
    E = side.U[6]
    G = _fast_alfven_gap(V.rho, B_x, V.By, V.Bz, side.a2, setup)
    dP = j * du
    N = G + V.rho * (w - side.c_f) * (w + side.c_f)
    M = jnp.maximum(N - dP, G + (Gamma - 1.0) * jnp.maximum(dP, 0.0))
    positive = M > 0.0
    xi = jnp.where(positive, dP / jnp.where(positive, M, setup.one), setup.zero)
    shear = nu * (B_x / j) * xi
    v_y = V.vy + shear * V.By
    v_z = V.vz + shear * V.Bz
    B_y = (1.0 + xi) * V.By
    B_z = (1.0 + xi) * V.Bz
    rho = j / (w - du)
    vB = V.vy * V.By + V.vz * V.Bz
    vB_star = v_y * B_y + v_z * B_z
    J = nu * du * B_x + vB - vB_star
    E_star = E + (du * E + nu * (side.P_T * V.vx - P_T_star * S_M - B_x * J)) / (w - du)
    return FanState(rho, v_y, v_z, B_y, B_z, E_star, vB_star)


def _fan_flux(state: FanState, S_M, P_T_star, B_x) -> Tuple:
    """The physical flux of a state of the Riemann fan."""
    rho, v_y, v_z, B_y, B_z, E, vB = state
    return (
        rho * S_M,
        rho * S_M * S_M + P_T_star - B_x * B_x,
        rho * S_M * v_y - B_x * B_y,
        rho * S_M * v_z - B_x * B_z,
        B_y * S_M - B_x * v_y,
        B_z * S_M - B_x * v_z,
        (E + P_T_star) * S_M - B_x * (S_M * B_x + vB),
    )


def _hlld_bounded_flux(L: FluxState, R: FluxState, B_x, setup: KernelSetup) -> Tuple:
    """The HLLD flux with bounded outer speeds (ideal gas; the statements of the array form)."""
    Gamma = 0.5 * (setup.gamma + 1.0)
    s = jnp.where(B_x >= 0.0, setup.one, setup.minus_one)
    rho_L, u_L, P_T_L = L.V.rho, L.V.vx, L.P_T
    rho_R, u_R, P_T_R = R.V.rho, R.V.vx, R.P_T

    # outer speeds relative to the gas
    D = u_L - u_R
    d_L = jnp.maximum(L.c_f, R.c_f + D)
    d_R = jnp.maximum(R.c_f, L.c_f + D)
    h = 0.5 * Gamma * jnp.maximum(D, 0.0)
    D_minus = jnp.minimum(D, 0.0)
    Pi_L = Gamma * jnp.maximum(P_T_R - P_T_L + rho_R * d_R * D_minus, 0.0)
    Pi_R = Gamma * jnp.maximum(P_T_L - P_T_R + rho_L * d_L * D_minus, 0.0)
    h_L = h + 0.5 * Pi_L / (rho_R * d_R)
    h_R = h + 0.5 * Pi_R / (rho_L * d_L)
    sigma_L = jnp.minimum(
        h + jnp.sqrt(h * h + L.c_f2 + Pi_L / rho_L), h_L + jnp.sqrt(h_L * h_L + L.c_f2)
    )
    sigma_R = jnp.minimum(
        h + jnp.sqrt(h * h + R.c_f2 + Pi_R / rho_R), h_R + jnp.sqrt(h_R * h_R + R.c_f2)
    )
    w_L = jnp.maximum(d_L, jnp.maximum(u_L, sigma_L))
    w_R = jnp.maximum(d_R, jnp.maximum(-u_R, sigma_R))
    S_L = u_L - w_L
    S_R = u_R + w_R
    j_L = rho_L * w_L
    j_R = rho_R * w_R

    # middle wave, total pressure of the fan, Delta u of the two sides
    inv = 1.0 / (j_L + j_R)
    S_M = (j_L * u_L + j_R * u_R + P_T_L - P_T_R) * inv
    P_T_star = (j_R * P_T_L + j_L * P_T_R + j_L * j_R * D) * inv
    du_L = ((P_T_R - P_T_L) + j_R * D) * inv
    du_R = ((P_T_L - P_T_R) + j_L * D) * inv

    # star states
    star_L = _hlld_star_state(L, j_L, w_L, du_L, 1.0, B_x, S_M, P_T_star, Gamma, setup)
    star_R = _hlld_star_state(R, j_R, w_R, du_R, -1.0, B_x, S_M, P_T_star, Gamma, setup)

    # double-star states
    sqrt_L = jnp.sqrt(star_L.rho)
    sqrt_R = jnp.sqrt(star_R.rho)
    inv_sqrt = 1.0 / (sqrt_L + sqrt_R)
    S_star_L = S_M - jnp.abs(B_x) / sqrt_L
    S_star_R = S_M + jnp.abs(B_x) / sqrt_R

    def double_star(v_L, v_R, B_L, B_R):
        v = (sqrt_L * v_L + sqrt_R * v_R + (B_R - B_L) * s) * inv_sqrt
        B = (sqrt_L * B_R + sqrt_R * B_L + sqrt_L * sqrt_R * (v_R - v_L) * s) * inv_sqrt
        return v, B

    v_y, B_y = double_star(star_L.vy, star_R.vy, star_L.By, star_R.By)
    v_z, B_z = double_star(star_L.vz, star_R.vz, star_L.Bz, star_R.Bz)
    vB = v_y * B_y + v_z * B_z
    E_L = star_L.E - sqrt_L * (star_L.vB - vB) * s
    E_R = star_R.E + sqrt_R * (star_R.vB - vB) * s
    dstar_L = FanState(star_L.rho, v_y, v_z, B_y, B_z, E_L, vB)
    dstar_R = FanState(star_R.rho, v_y, v_z, B_y, B_z, E_R, vB)

    # the flux of the state at x / t = 0
    fan = (S_M, P_T_star, B_x)
    F_star_L, F_star_R = _fan_flux(star_L, *fan), _fan_flux(star_R, *fan)
    F_dstar_L, F_dstar_R = _fan_flux(dstar_L, *fan), _fan_flux(dstar_R, *fan)
    out = []
    for n in range(7):
        F = jnp.where(S_R > 0.0, F_star_R[n], R.F[n])
        F = jnp.where(S_star_R >= 0.0, F_dstar_R[n], F)
        F = jnp.where(S_M >= 0.0, F_dstar_L[n], F)
        F = jnp.where(S_star_L >= 0.0, F_star_L[n], F)
        out.append(jnp.where(S_L >= 0.0, L.F[n], F))
    return tuple(out)


def _riemann_flux(L: FluxState, R: FluxState, B_x, setup: KernelSetup) -> Tuple:
    """The Riemann flux of the two face states (the static choice of the solver)."""
    if setup.riemann_solver == HLLD_BOUNDED:
        return _hlld_bounded_flux(L, R, B_x, setup)
    S_L = jnp.minimum(L.V.vx - L.c_f, R.V.vx - R.c_f)
    S_R = jnp.maximum(L.V.vx + L.c_f, R.V.vx + R.c_f)
    S_L = jnp.minimum(S_L, 0.0)
    S_R = jnp.maximum(S_R, 0.0)
    inv = 1.0 / jnp.maximum(S_R - S_L, DIVISION_FLOOR)
    if setup.riemann_solver == HLLEM_T:
        return _hllem_t_flux(L, R, B_x, S_L, S_R, inv, setup)
    return _hll_flux(L, R, S_L, S_R, inv)


# ============ 5. cell-flux correction with the smoothness switch =============


def _window_not_smooth(a, b, c, d, e):
    """True where the five cell fluxes of one component are not smooth (eq. ``switch``)."""
    IS_0 = 13.0 / 12.0 * (a - 2.0 * b + c) ** 2 + 0.25 * (a - 4.0 * b + 3.0 * c) ** 2
    IS_1 = 13.0 / 12.0 * (b - 2.0 * c + d) ** 2 + 0.25 * (b - d) ** 2
    IS_2 = 13.0 / 12.0 * (c - 2.0 * d + e) ** 2 + 0.25 * (3.0 * c - 4.0 * d + e) ** 2
    IS_min = jnp.minimum(jnp.minimum(IS_0, IS_1), IS_2)
    IS_max = jnp.maximum(jnp.maximum(IS_0, IS_1), IS_2)
    return IS_max - IS_min > IS_min


def _cell_flux_correction(F_RS, F) -> Tuple:
    """
    F^c = F^RS - Delta^2 f / 24 + 7 Delta^4 f / 5760 per component, or F^RS
    where the cell fluxes of the component are not smooth; F holds the flux
    tuples of the cells i - 2 ... i + 3. The two differences are collected
    over the denominator 3840:

        -Delta^2 f / 24 + 7 Delta^4 f / 5760 = (19 outer - 137 inner) / 3840,
        outer = f_{i-2} + f_{i+3} - (f_i + f_{i+1}),
        inner = f_{i-1} + f_{i+2} - (f_i + f_{i+1}).
    """
    out = []
    for n, F_RS_n in enumerate(F_RS):
        f_m2, f_m1, f_0, f_p1, f_p2, f_p3 = (F_cell[n] for F_cell in F)
        centre = f_0 + f_p1
        outer = (f_m2 + f_p3) - centre
        inner = (f_m1 + f_p2) - centre
        F_c = F_RS_n + (19.0 * outer - 137.0 * inner) * (1.0 / 3840.0)
        not_smooth_left = _window_not_smooth(f_m2, f_m1, f_0, f_p1, f_p2)
        not_smooth_right = _window_not_smooth(f_m1, f_0, f_p1, f_p2, f_p3)
        out.append(jnp.where(not_smooth_left | not_smooth_right, F_RS_n, F_c))
    return tuple(out)


# ============================ 6. H-correction ================================


def _h_correction(F_c, U_L, U_R, eta_face) -> Tuple:
    """F^H = F^c - eta_face (U_R - U_L) / 2, eta_face = max(eta_i, eta_{i+1})."""
    return tuple(
        F_c_n - 0.5 * eta_face * (U_R_n - U_L_n) for F_c_n, U_L_n, U_R_n in zip(F_c, U_L, U_R)
    )


# ================================ 7. guard ===================================


def _lax_friedrichs_flux(cell: FluxState, cell_next: FluxState, a, a_next, count: int) -> Tuple:
    """
    The first ``count`` components of the first-order Lax-Friedrichs flux at
    the face between two cells, a = |v_x| + c_f their signal speeds. (The
    three-cell maximum of a is taken in ``_guard``.)
    """
    a_face = jnp.maximum(a, a_next)
    return tuple(
        0.5 * (cell.F[n] + cell_next.F[n]) - 0.5 * a_face * (cell_next.U[n] - cell.U[n])
        for n in range(count)
    )


def _pressure_of_conserved(q, B_x, setup: KernelSetup):
    """
    The pressure of a trial conserved vector q = (rho, m_x, m_y, m_z, B_y,
    B_z, E) of the guard with the normal field B_x of its cell, without
    floors.
    """
    rho, m_x, m_y, m_z, B_y, B_z, E = q
    m2 = m_x * m_x + m_y * m_y + m_z * m_z
    B2 = B_x * B_x + B_y * B_y + B_z * B_z
    return (setup.gamma - 1.0) * (E - 0.5 * m2 / jnp.maximum(rho, DIVISION_FLOOR) - 0.5 * B2)


def _budget_factors(t, g_low, l_plus, l_minus):
    """One linear bound g(q) >= 0 of the guard on the factors t = (t^+, t^-) of a cell."""
    t_plus, t_minus = t
    budget = jnp.maximum(g_low, 0.0)
    draw_plus = jnp.maximum(-l_plus, 0.0)
    draw_minus = jnp.maximum(-l_minus, 0.0)
    ratio = jnp.minimum(budget / jnp.maximum(draw_plus + draw_minus, DIVISION_FLOOR), 1.0)
    t_plus = jnp.where(draw_plus > 0.0, jnp.minimum(t_plus, ratio), t_plus)
    t_minus = jnp.where(draw_minus > 0.0, jnp.minimum(t_minus, ratio), t_minus)
    return t_plus, t_minus


def _guard(F_H, F_LF, q, rho, B2, B_x, v, a, reference_step, setup: KernelSetup) -> Tuple:
    """
    The guarded flux F = F^LF + theta (F^H - F^LF) at the face i + 1/2,
    theta = min(t^+_i, t^-_{i+1}): the statements of the array form for the
    two cells i and i + 1.

    Args:
        F_H, F_LF: The fluxes at the faces i - 1/2, i + 1/2, i + 3/2 (list
            positions 0, 1, 2): all components at i + 1/2, at the other two
            the components the bounds read (density and momentum; with the
            ideal gas all).
        q: The conserved variables (rho, m_x, m_y, m_z, B_y, B_z(, E)) of the
            cells i, i + 1 as stored (density not floored).
        rho, B2: Floored density and B^2 of the cells i - 2 ... i + 3.
        B_x: The cell-centred normal field of the cells i, i + 1.
        v: The velocities (v_x, v_y, v_z) of the cells i - 1 ... i + 2.
        a: The signal speeds of the cells i - 1 ... i + 2.
        reference_step: 1 / lambda.
    """
    count = len(F_H[0])  # the components of the bounds
    A = [tuple(F_H_n - F_LF_n for F_H_n, F_LF_n in zip(F_H[face], F_LF[face])) for face in range(3)]
    v_A2 = [B2_cell / rho_cell for B2_cell, rho_cell in zip(B2, rho)]

    def factors(k):
        """(t^+, t^-) of the cell i + k, k = 0 or 1."""
        updates = (
            # q_low, c_plus, c_minus
            tuple(q[k][n] - reference_step * (F_LF[k + 1][n] - F_LF[k][n]) for n in range(count)),
            tuple(-reference_step * A[k + 1][n] for n in range(count)),
            tuple(reference_step * A[k][n] for n in range(count)),
        )
        q_low, c_plus, c_minus = updates
        t = (setup.one, setup.one)

        # density with the hole bound (cell i + k is at position k + 2 of rho, B2)
        rho_ref = None
        for left, right in ((-1, 1), (-1, 2), (-2, 1)):
            v_A2_pair = jnp.maximum(v_A2[k + 2 + left], v_A2[k + 2 + right])
            rho_pair = jnp.minimum(rho[k + 2 + left], rho[k + 2 + right])
            rho_pair = jnp.minimum(rho_pair, B2[k + 2] / jnp.maximum(v_A2_pair, DIVISION_FLOOR))
            rho_ref = rho_pair if rho_ref is None else jnp.maximum(rho_ref, rho_pair)
        rho_lower = jnp.maximum(
            setup.minimum_density, jnp.minimum(q_low[0], HOLE_FRACTION * rho_ref)
        )
        t = _budget_factors(t, q_low[0] - rho_lower, c_plus[0], c_minus[0])

        # velocity components (cell i + k is at position k + 1 of v, a)
        a3 = jnp.maximum(jnp.maximum(a[k], a[k + 1]), a[k + 2])
        for index in (1, 2, 3):
            v_m, v_0, v_p = (v[k + position][index - 1] for position in range(3))
            v_max = jnp.maximum(jnp.maximum(v_m, v_0), v_p) + 2.0 * a3
            v_min = jnp.minimum(jnp.minimum(v_m, v_0), v_p) - 2.0 * a3
            t = _budget_factors(t, *(v_max * u[0] - u[index] for u in updates))
            t = _budget_factors(t, *(u[index] - v_min * u[0] for u in updates))
        t_plus, t_minus = t

        # pressure of the ideal gas
        if setup.ideal_gas:
            P_min = setup.minimum_pressure
            P_low = _pressure_of_conserved(q_low, B_x[k], setup)

            def chord(t_side, change):
                trial = tuple(q_low[n] + 2.0 * (t_side * change[n]) for n in range(count))
                P_end = _pressure_of_conserved(trial, B_x[k], setup)
                psi = jnp.where(
                    P_end < P_min,
                    (P_low - P_min) / jnp.maximum(P_low - P_end, DIVISION_FLOOR),
                    setup.one,
                )
                return jnp.clip(psi, 0.0, 1.0)

            t_plus = t_plus * chord(t_plus, c_plus)
            t_minus = t_minus * chord(t_minus, c_minus)
        return t_plus, t_minus

    t_plus, _ = factors(0)
    _, t_minus = factors(1)
    theta = jnp.minimum(t_plus, t_minus)
    return tuple(F_LF_n + theta * A_n for F_LF_n, A_n in zip(F_LF[1], A[1]))


# ============================== the kernels ==================================


class _Tiles(NamedTuple):
    """The grid of blocks of a kernel and the state slots of its components."""

    num_vars: int
    spatial_shape: Tuple[int, int, int]
    block: Tuple[int, int, int]
    grid: Tuple[int, int, int]
    #: The state rows (rho, m_x, m_y, m_z, B_x, B_y, B_z(, E)) of the sweep along the axis.
    rows: Tuple[int, ...]
    #: The state rows of the flux components (the rows without B_x).
    slots: Tuple[int, ...]


def _tiles(conserved_state, config: SimulationConfig, registered_variables, axis: int) -> _Tiles:
    _, spatial_shape = _shard_cells(conserved_state.shape[1:])
    block = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape, 3, spatial_shape=spatial_shape
    )
    grid = tuple(n // b for n, b in zip(spatial_shape, block))
    if config.equation_of_state == IDEAL_GAS:
        rows = tuple(_mhd_indices_for_axis(config, registered_variables, axis))
    else:
        rows = tuple(_mhd_iso_indices_for_axis(config, registered_variables, axis))
    slots = rows[:4] + rows[5:]
    return _Tiles(int(conserved_state.shape[0]), spatial_shape, block, grid, rows, slots)


def _tile_reader(tiles: _Tiles, axis: int):
    """
    ``read(ref, row, offset)``: the values of one row of an input at the
    cells (faces) of this block shifted by ``offset`` along the sweep axis,
    periodic in the array. On the shard of a device mesh an input padded by
    a halo holds the cells of the block after it (``_shard_cells``).
    """
    nx, ny, nz = tiles.spatial_shape
    bx_, by_, bz_ = tiles.block
    ii = (pl.program_id(0) * bx_ + jnp.arange(bx_)[:, None, None]) % nx
    jj = (pl.program_id(1) * by_ + jnp.arange(by_)[None, :, None]) % ny
    kk = (pl.program_id(2) * bz_ + jnp.arange(bz_)[None, None, :]) % nz

    def read(ref, row, offset):
        extents = ref.shape[1:]
        if extents != tiles.spatial_shape:
            indices = [
                index + (extent - cells) // 2 for index, extent, cells in zip((ii, jj, kk), extents, (nx, ny, nz))
            ]
            if offset != 0:
                indices[axis] = (indices[axis] + offset) % extents[axis]
            return ref[(row, *indices)]
        if offset == 0:
            return ref[row, ii, jj, kk]
        if axis == 0:
            return ref[row, (ii + offset) % nx, jj, kk]
        if axis == 1:
            return ref[row, ii, (jj + offset) % ny, kk]
        return ref[row, ii, jj, (kk + offset) % nz]

    return read


def _whole_array_spec(array):
    """Every block sees the whole input."""
    return pl.BlockSpec(array.shape, lambda bi, bj, bk: (0,) * array.ndim)


def _interface_flux_pallas_local(
    conserved_state,
    auxiliary,
    *scalars,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int,
):
    """
    The flux kernel of one shard: steps 1 to 6 of ``_interface_flux_x``.

    Args:
        conserved_state: The conserved state (possibly halo-padded).
        auxiliary: (face field, eta), each with a leading axis of length 1.
        scalars: (gamma, minimum density, minimum pressure, eigen floor) or
            (sound speed, squared sound speed, minimum density, eigen floor).

    Returns:
        F_H at the faces i + 1/2 (index i): the components rho, m_x, m_y,
        m_z, B_y, B_z(, E) of the sweep, stacked.
    """
    tiles = _tiles(conserved_state, config, registered_variables, axis)
    num_out = len(tiles.slots)
    auxiliary = tuple(auxiliary)
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_ref, face_field_ref, eta_ref, *other_refs):
        scalar_refs, out_ref = other_refs[:-1], other_refs[-1]
        setup = _kernel_setup(tuple(ref[()] for ref in scalar_refs), config)
        read = _tile_reader(tiles, axis)

        # 1. cell primitives of the cells i - 2 ... i + 3
        V, B_x = zip(
            *(
                _cell_primitives(tuple(read(q_ref, row, offset) for row in tiles.rows), setup)
                for offset in range(-2, 4)
            )
        )

        # 2. interface state and characteristic basis
        bar = _interface_state(V[2], V[3], B_x[2], B_x[3], setup)
        basis = _characteristic_basis(bar, setup)

        # 3. face states
        V_L, V_R = _face_states(V, basis, setup)

        # 4. Riemann flux
        face_field = read(face_field_ref, 0, 0)
        L = _flux_state(V_L, face_field, setup)
        R = _flux_state(V_R, face_field, setup)
        F_RS = _riemann_flux(L, R, face_field, setup)

        # 5. cell-flux correction with the smoothness switch
        cell_F = [_flux_state(V[k], B_x[k], setup, speeds=False).F for k in range(6)]
        F_c = _cell_flux_correction(F_RS, cell_F)

        # 6. H-correction
        eta_face = jnp.maximum(read(eta_ref, 0, 0), read(eta_ref, 0, 1))
        F_H = _h_correction(F_c, L.U, R.U, eta_face)

        for n in range(num_out):
            out_ref[n, ...] = F_H[n]

    compiler_params = _pallas_compiler_params(config)
    kwargs = {} if compiler_params is None else {"compiler_params": compiler_params}
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_out,) + tiles.spatial_shape, conserved_state.dtype),
        grid=tiles.grid,
        in_specs=[_whole_array_spec(conserved_state)]
        + [_whole_array_spec(field) for field in auxiliary]
        + [scalar_spec] * len(scalars),
        out_specs=pl.BlockSpec((num_out,) + tiles.block, lambda bi, bj, bk: (0, bi, bj, bk)),
        interpret=config.backend_config.pallas_interpret,
        name=f"riemann_flux_axis_{axis}",
        **kwargs,
    )(conserved_state, *auxiliary, *scalars)


def _guard_pallas_local(
    conserved_state,
    flux_high,
    *scalars,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int,
):
    """
    The guard kernel of one shard: step 7 of ``_interface_flux_x``.

    Args:
        conserved_state: The conserved state (possibly halo-padded).
        flux_high: F_H of the flux kernel (possibly halo-padded).
        scalars: The scalars of the flux kernel and the reference step.

    Returns:
        The guarded flux at the faces i + 1/2 (index i), shaped like the
        state (zero in the row of the normal field and in any further row).
    """
    tiles = _tiles(conserved_state, config, registered_variables, axis)
    num_components = len(tiles.slots)
    # the bounds of the isothermal gas read density and momentum only
    num_bounded = num_components if config.equation_of_state == IDEAL_GAS else 4
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_ref, F_H_ref, *other_refs):
        scalar_refs, out_ref = other_refs[:-1], other_refs[-1]
        *setup_scalars, reference_step = (ref[()] for ref in scalar_refs)
        setup = _kernel_setup(tuple(setup_scalars), config)
        read = _tile_reader(tiles, axis)

        # The cells i - 2 ... i + 3: floored density and B^2 (the hole bound);
        # the cells i - 1 ... i + 2: primitives, flux state, signal speed.
        rho, B2 = [], []
        cells, a, v, q, B_x = [], [], [], [], []
        for offset in range(-2, 4):
            if offset in (-2, 3):
                rho_cell = jnp.maximum(read(q_ref, tiles.rows[0], offset), setup.minimum_density)
                B_cell = [read(q_ref, row, offset) for row in tiles.rows[4:7]]
                rho.append(rho_cell)
                B2.append(B_cell[0] * B_cell[0] + B_cell[1] * B_cell[1] + B_cell[2] * B_cell[2])
                continue
            q_cell = tuple(read(q_ref, row, offset) for row in tiles.rows)
            V, B_x_cell = _cell_primitives(q_cell, setup)
            cell = _flux_state(V, B_x_cell, setup)
            rho.append(V.rho)
            B2.append(B_x_cell * B_x_cell + V.By * V.By + V.Bz * V.Bz)
            cells.append(cell)
            a.append(jnp.abs(V.vx) + cell.c_f)
            v.append((V.vx, V.vy, V.vz))
            if offset in (0, 1):
                q.append(q_cell[:4] + q_cell[5:])
                B_x.append(B_x_cell)

        # the faces i - 1/2, i + 1/2, i + 3/2
        counts = (num_bounded, num_components, num_bounded)
        F_LF = [
            _lax_friedrichs_flux(cells[face], cells[face + 1], a[face], a[face + 1], counts[face])
            for face in range(3)
        ]
        F_H = [
            tuple(read(F_H_ref, n, face - 1) for n in range(counts[face])) for face in range(3)
        ]
        F = _guard(F_H, F_LF, q, rho, B2, B_x, v, a, reference_step, setup)

        zero_tile = F[0] * 0.0
        for row in range(tiles.num_vars):
            if row not in tiles.slots:
                out_ref[row, ...] = zero_tile
        for n, row in enumerate(tiles.slots):
            out_ref[row, ...] = F[n]

    compiler_params = _pallas_compiler_params(config)
    kwargs = {} if compiler_params is None else {"compiler_params": compiler_params}
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((tiles.num_vars,) + tiles.spatial_shape, conserved_state.dtype),
        grid=tiles.grid,
        in_specs=[_whole_array_spec(conserved_state), _whole_array_spec(flux_high)]
        + [scalar_spec] * len(scalars),
        out_specs=pl.BlockSpec((tiles.num_vars,) + tiles.block, lambda bi, bj, bk: (0, bi, bj, bk)),
        interpret=config.backend_config.pallas_interpret,
        name=f"riemann_flux_guard_axis_{axis}",
        **kwargs,
    )(conserved_state, flux_high, *scalars)


# ============================ dispatch targets ===============================


def _riemann_flux_pallas_supported(conserved_state, config: SimulationConfig) -> bool:
    """
    Whether the kernels can run in place of ``_interface_flux_x``: the Pallas
    backend, the MHD form with the ideal or the isothermal gas, single
    precision on a GPU (or the interpreter, which takes any precision and
    device), three dimensions and a block shape that divides the grid. What is
    refused runs in the array form. The kernels hold every branch of the MHD
    form with these two gases that ``_riemann_flux._scheme_setup`` accepts.
    The stage (``_ssprk._riemann_flux_increment``) and the
    constrained transport (``_flux_form_ct_pallas_supported``) ask the same
    question, so one stage never mixes kernels with a refused condition.
    """
    if pl is None or not _backend_is_pallas(config):
        return False
    if not config.mhd or config.equation_of_state not in (IDEAL_GAS, ISOTHERMAL):
        return False
    compiled = not config.backend_config.pallas_interpret
    if compiled and (jax.config.jax_enable_x64 or jax.default_backend() == "cpu"):
        return False
    if int(config.dimensionality) != 3 or conserved_state.ndim != 4:
        return False
    block_shape = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape, 3, spatial_shape=conserved_state.shape[1:]
    )
    return all(
        int(extent) % int(block) == 0
        for extent, block in zip(conserved_state.shape[1:], block_shape, strict=True)
    )


def _interface_flux_pallas(
    conserved_state,
    face_field,
    reference_step,
    eta,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """
    ``_riemann_flux.interface_flux`` through the two kernels (the arguments
    and the result of that function, in the layout of the state: the kernels
    read along ``axis`` instead of moving it to the front).
    """
    dtype = conserved_state.dtype
    auxiliary = tuple(jnp.asarray(field, dtype=dtype)[None] for field in (face_field, eta))
    block_shape = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape, 3, spatial_shape=conserved_state.shape[1:]
    )
    halo = tuple(_HALO if d == int(axis) else 0 for d in range(3))
    flux_halo = tuple(_GUARD_FLUX_HALO if d == int(axis) else 0 for d in range(3))

    # the floor of ``_eigen_mhd._wave_speeds`` and of the eigenvector blocks
    eigen_floor = 1e-30 if jax.config.jax_enable_x64 else 1e-20
    if config.equation_of_state == IDEAL_GAS:
        scalars = (params.gamma, params.minimum_density, params.minimum_pressure, eigen_floor)
    else:
        scalars = (
            params.isothermal_sound_speed,
            params.isothermal_sound_speed**2,
            params.minimum_density,
            eigen_floor,
        )
    scalars = tuple(jnp.asarray(value, dtype=dtype) for value in scalars)

    def build_flux(state_local, *others):
        return _interface_flux_pallas_local(
            state_local,
            others[: len(auxiliary)],
            *others[len(auxiliary) :],
            config=config,
            registered_variables=registered_variables,
            axis=axis,
        )

    flux_high = _pallas_call_sharded(
        build_flux,
        state_inputs=(conserved_state,) + auxiliary,
        other_args=scalars,
        halo=halo,
        block_shape=block_shape,
        local_output=True,
    )

    def build_guard(state_local, flux_local, *guard_scalars):
        return _guard_pallas_local(
            state_local,
            flux_local,
            *guard_scalars,
            config=config,
            registered_variables=registered_variables,
            axis=axis,
        )

    return _pallas_call_sharded(
        build_guard,
        state_inputs=(conserved_state, flux_high),
        other_args=scalars + (jnp.asarray(reference_step, dtype=dtype),),
        halo=halo,
        block_shape=block_shape,
        input_halos=(halo, flux_halo),
        local_output=True,
    )
