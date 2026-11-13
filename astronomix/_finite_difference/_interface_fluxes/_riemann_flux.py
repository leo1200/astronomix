"""
Interface fluxes of the finite-difference scheme with a Riemann flux at the
faces (``FDSolverConfig.scheme = RIEMANN_FLUX``), for the ideal, the isothermal
and the polytropic gas: MHD in three dimensions, hydrodynamics in one to three.

The module follows the scheme description ``new_scheme.tex`` step by step and
uses its names. ``_interface_flux_x`` returns the flux at the faces i + 1/2
(stored at index i) along the first spatial axis in seven steps:

    1. cell primitives (rho, v, B, P) with the floors,
    2. interface state and its characteristic basis (eq. ``ifstate``, appendix
       "Eigenvectors"),
    3. face states: WENO5-Z interpolation of the characteristic variables with
       the MP bound, back-transformation, MP bound on the primitive values,
       admissibility (section "Interpolation to the faces"),
    4. Riemann flux F^RS (section "The flux at the face"): HLLEM-T for a
       barotropic gas, HLLD with bounded outer speeds for the ideal gas, or
       HLL for both,
    5. cell-flux correction F^c with the smoothness switch,
    6. H-correction F^H,
    7. guard: the flux limited towards the first-order Lax-Friedrichs flux
       (section "The guard").

The three services of a stage are ``maximum_signal_speed`` (the reference step
of the guard), ``transverse_compression`` (the eta of the H-correction) and
``interface_flux`` (the seven steps along one axis).

Layout: arrays with the variable axis first follow the variable registry and
have the rows of the state passed in (the row of the normal field B_x and any
row behind the cell-centred variables are zero in every flux and in every
vector of interpolated variables); fields have the sweep axis first.
``_at(f, k)`` is the stencil read f_{i+k} along the sweep axis.
All statements of the seven steps are stencil reads and pointwise arithmetic.

The hydrodynamic form (``mhd=False``) runs the same steps on the layout of the
hydrodynamic state: one momentum row per dimension and no field. A component
the layout does not carry (the field, the velocity across a 1D or 2D grid) is
the constant 0.0 in the containers below, so the formulas of the MHD form
serve it; the steps that differ are the characteristic basis (the Euler
waves), the Riemann fluxes (HLLC as the zero-field form of the bounded HLLD,
the zero-field form of HLLEM-T) and the guard (no hole bound).
"""

# general
from functools import partial

# typing
from typing import List, NamedTuple, Optional, Tuple, Union
from jaxtyping import Array, Bool, Float

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    HLL,
    HLLC,
    HLLD_BOUNDED,
    HLLEM_T,
    IDEAL_GAS,
    ISOTHERMAL,
    POLYTROPIC,
    RIEMANN_BY_EOS,
    STATE_TYPE,
    STATE_TYPE_ALTERED,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    SimulationConfig,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_difference._magnetic_update._flux_form_ct import (
    _weno5z_point_interpolation,
)
from astronomix._finite_difference._ratio_rules import _floored_ratio
from astronomix._fluid_equations._eigen_mhd import (
    _eigenvector_blocks,
    _left_eigenvector_rows,
    _right_eigenvector_columns,
    _wave_speeds,
)
from astronomix._fluid_equations._eigen_mhd_iso import (
    _eigenvector_blocks_iso,
    _left_eigenvector_rows_iso,
    _right_eigenvector_columns_iso,
    zero_safe_sqrt,
)
from astronomix._stencil_operations._stencil_operations import _shift

# relative epsilon of the WENO5-Z weights and its floor
WENO_EPSILON_RELATIVE = 1e-6
EPSILON_FLOOR = 1e-30

# fraction chi of the reference density in the hole bound of the guard
HOLE_FRACTION = 0.5

SCALAR = Union[float, Float[Array, ""]]


# ======================= configuration and containers ========================


class SchemeSetup(NamedTuple):
    """Static options, variable indices and parameters read by the scheme."""

    #: Riemann flux at the faces: ``HLL``, ``HLLEM_T``, ``HLLD_BOUNDED`` or
    #: ``HLLC`` (hydrodynamic form).
    riemann_solver: int
    #: The WENO weights are constants of the differentiation.
    frozen_weights: bool
    #: ``IDEAL_GAS``, ``ISOTHERMAL`` or ``POLYTROPIC``.
    equation_of_state: int
    #: The MHD form (or the hydrodynamic form).
    mhd: bool
    #: Number of spatial dimensions.
    dimensionality: int
    #: Number of rows of the state array the caller passes.
    num_vars: int
    #: Rows of density, momentum, cell-centred field and energy (ideal gas);
    #: -1 for a component the layout does not carry.
    density_index: int
    momentum_index: StaticIntVector
    magnetic_index: StaticIntVector
    energy_index: int
    #: Adiabatic index (ideal gas).
    gamma: SCALAR
    #: Sound speed and its square (isothermal gas).
    sound_speed: SCALAR
    sound_speed_squared: SCALAR
    #: K and Gamma of P = K rho^Gamma (polytropic gas).
    polytropic_constant: SCALAR
    polytropic_exponent: SCALAR
    #: Floors of density and pressure.
    minimum_density: SCALAR
    minimum_pressure: SCALAR


def _scheme_setup(
    conserved_state: Union[STATE_TYPE, STATE_TYPE_ALTERED],
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> SchemeSetup:
    """
    Reads the configuration, whose validity ``finalize_config`` has checked.
    This is the only function of the module that reads ``config`` and
    ``params``; every other function takes the returned ``SchemeSetup``.

    The state enters with its number of rows only: the stage passes the
    cell-centred rows, without the rows of the face-centred field that the
    registry counts behind them.

    ``RIEMANN_BY_EOS`` resolves to HLLD_BOUNDED (ideal-gas MHD), HLLC
    (ideal-gas hydrodynamics) or HLLEM_T (a barotropic gas, either form).
    """
    riemann_solver = config.solver_config.riemann_solver
    if riemann_solver == RIEMANN_BY_EOS:
        riemann_solver = HLLEM_T
        if config.equation_of_state == IDEAL_GAS:
            riemann_solver = HLLD_BOUNDED if config.mhd else HLLC
    momentum_index = registered_variables.momentum_index
    magnetic_index = registered_variables.magnetic_index
    if not config.mhd:
        # one momentum row per dimension (an integer in 1D), no field
        if isinstance(momentum_index, int):
            momentum_index = StaticIntVector(momentum_index)
        magnetic_index = StaticIntVector()
    return SchemeSetup(
        riemann_solver=riemann_solver,
        frozen_weights=config.solver_config.weno_ad_frozen_weights,
        equation_of_state=config.equation_of_state,
        mhd=config.mhd,
        dimensionality=config.dimensionality,
        num_vars=conserved_state.shape[0],
        density_index=registered_variables.density_index,
        momentum_index=momentum_index,
        magnetic_index=magnetic_index,
        energy_index=registered_variables.energy_index,
        gamma=params.gamma,
        sound_speed=params.isothermal_sound_speed,
        sound_speed_squared=params.isothermal_sound_speed**2,
        polytropic_constant=params.polytropic_constant,
        polytropic_exponent=params.polytropic_exponent,
        minimum_density=params.minimum_density,
        minimum_pressure=params.minimum_pressure,
    )


class Primitives(NamedTuple):
    """
    Primitive variables V = (rho, v_x, v_y, v_z, B_y, B_z, P) of the cells or
    of one side of the faces. The normal field B_x is not interpolated (at a
    face it is the face-centred field) and is carried next to V. A component
    the layout does not carry is 0.0 (``_carried_components``).
    """

    rho: FIELD_TYPE
    vx: FIELD_TYPE
    vy: FIELD_TYPE
    vz: FIELD_TYPE
    By: FIELD_TYPE
    Bz: FIELD_TYPE
    P: FIELD_TYPE


class InterfaceState(NamedTuple):
    """The interface state of eq. ``ifstate`` (``a2`` is a^2)."""

    rho: FIELD_TYPE
    vx: FIELD_TYPE
    vy: FIELD_TYPE
    vz: FIELD_TYPE
    Bx: FIELD_TYPE
    By: FIELD_TYPE
    Bz: FIELD_TYPE
    a2: Union[SCALAR, FIELD_TYPE]


class FluxState(NamedTuple):
    """
    A state as the fluxes use it: the primitive variables V, the conserved
    vector U, the flux F_x(U), the total pressure P_T = P + B^2 / 2, a^2, and
    the fast speed along x with its square.
    """

    V: Primitives
    U: STATE_TYPE_ALTERED
    F: STATE_TYPE_ALTERED
    P_T: FIELD_TYPE
    a2: Union[SCALAR, FIELD_TYPE]
    c_f: FIELD_TYPE
    c_f2: FIELD_TYPE


class FanState(NamedTuple):
    """
    A state of the fan of the bounded HLLD flux; its normal velocity is S_M,
    its total pressure P_T^* (``vB`` is v_t . B_t).
    """

    rho: FIELD_TYPE
    vy: FIELD_TYPE
    vz: FIELD_TYPE
    By: FIELD_TYPE
    Bz: FIELD_TYPE
    E: FIELD_TYPE
    vB: FIELD_TYPE


# ============================== stencil reads ================================


def _at(field: Union[Array, float], offset: int, axis: int = 0) -> Union[Array, float]:
    """
    The stencil read f_{i+offset} along ``axis``, stored at index i (periodic).
    A component the layout does not carry (the constant 0.0) reads as itself.
    """
    if offset == 0 or isinstance(field, float):
        return field
    return _shift(field, -offset, axis=axis)


def _at_state(state: STATE_TYPE_ALTERED, offset: int) -> STATE_TYPE_ALTERED:
    """``_at`` along the sweep axis for an array with the variable axis first."""
    return _at(state, offset, axis=1)


def _row(state: STATE_TYPE_ALTERED, index: int) -> Union[FIELD_TYPE, float]:
    """The row ``index`` of a state array; 0.0 for a component the layout does not carry (index -1)."""
    return state[index] if index >= 0 else 0.0


def _carried_components(setup: SchemeSetup) -> Tuple[bool, ...]:
    """Which components of ``Primitives`` the layout carries (the others are 0.0)."""
    has_y, has_z = (index >= 0 for index in setup.momentum_index[1:])
    return (True, True, has_y, has_z, setup.mhd, setup.mhd, True)


def _state_vector(
    rho: FIELD_TYPE,
    m_x: FIELD_TYPE,
    m_y: FIELD_TYPE,
    m_z: FIELD_TYPE,
    B_y: FIELD_TYPE,
    B_z: FIELD_TYPE,
    E: Optional[FIELD_TYPE],
    setup: SchemeSetup,
) -> STATE_TYPE_ALTERED:
    """
    Stacks the components of a conserved vector or of a flux in the layout of
    the state array. The row of the normal field and any row behind the
    cell-centred variables are zero; ``E`` is only read for the ideal gas, a
    component the layout does not carry is not read.
    """
    rows = [jnp.zeros_like(rho)] * setup.num_vars
    rows[setup.density_index] = rho
    components = zip(
        (*setup.momentum_index, setup.magnetic_index.y, setup.magnetic_index.z), (m_x, m_y, m_z, B_y, B_z)
    )
    for index, component in components:
        if index >= 0:
            rows[index] = component
    if setup.equation_of_state == IDEAL_GAS:
        rows[setup.energy_index] = E
    return jnp.stack(rows)


# ============================ equation of state ==============================


def _barotropic_pressure(rho: FIELD_TYPE, setup: SchemeSetup) -> FIELD_TYPE:
    """The pressure of a barotropic gas, P = c_s^2 rho (isothermal) or K rho^Gamma (polytropic)."""
    if setup.equation_of_state == POLYTROPIC:
        return setup.polytropic_constant * rho**setup.polytropic_exponent
    return setup.sound_speed_squared * rho


def _sound_speed_squared(
    rho: FIELD_TYPE, P: FIELD_TYPE, setup: SchemeSetup
) -> Union[SCALAR, FIELD_TYPE]:
    """
    a^2 = gamma P / rho (ideal gas), the constant c_s^2 (isothermal gas) or
    dP / drho = Gamma K rho^(Gamma - 1) at the density ``rho`` (polytropic gas).
    """
    if setup.equation_of_state == IDEAL_GAS:
        return setup.gamma * P / rho
    if setup.equation_of_state == POLYTROPIC:
        exponent = setup.polytropic_exponent
        return exponent * setup.polytropic_constant * rho ** (exponent - 1.0)
    return setup.sound_speed_squared


def _fast_speed(rho, B_x, B_y, B_z, a2, setup: SchemeSetup):
    """
    The fast speed along x and its square (``_wave_speeds``); without a field
    the sound speed a and a^2.
    """
    if not setup.mhd:
        return jnp.sqrt(a2), a2
    c_f, _, _, c_f2, _ = _wave_speeds(rho, B_x, B_y, B_z, a2)
    return c_f, c_f2


def _flux_state(V: Primitives, B_x: FIELD_TYPE, setup: SchemeSetup) -> FluxState:
    """
    The conserved vector, the flux and the speeds of primitive variables with
    the normal field ``B_x`` (the face-centred field at a face, the
    cell-centred field in a cell):

        F_x = (rho v_x,  rho v_x v + P_T e_x - B_x B,  B_t v_x - B_x v_t,
               (E + P_T) v_x - (v . B) B_x),    P_T = P + B^2 / 2.

    The fast speed and its square are those of the characteristic basis
    (``_fast_speed``): the module has one fast-speed formula, and the square
    is carried next to its root so that no caller squares the rounded speed.
    """
    rho, v_x, v_y, v_z, B_y, B_z, P = V
    B2 = B_x * B_x + B_y * B_y + B_z * B_z
    v2 = v_x * v_x + v_y * v_y + v_z * v_z
    P_T = P + 0.5 * B2
    E = F_E = None
    if setup.equation_of_state == IDEAL_GAS:
        E = P / (setup.gamma - 1.0) + 0.5 * rho * v2 + 0.5 * B2
        F_E = (E + P_T) * v_x - (v_x * B_x + v_y * B_y + v_z * B_z) * B_x
    U = _state_vector(rho, rho * v_x, rho * v_y, rho * v_z, B_y, B_z, E, setup)
    F = _state_vector(
        rho * v_x,
        rho * v_x * v_x + P_T - B_x * B_x,
        rho * v_x * v_y - B_x * B_y,
        rho * v_x * v_z - B_x * B_z,
        B_y * v_x - B_x * v_y,
        B_z * v_x - B_x * v_z,
        F_E,
        setup,
    )
    a2 = _sound_speed_squared(rho, P, setup)
    c_f, c_f2 = _fast_speed(rho, B_x, B_y, B_z, a2, setup)
    return FluxState(V, U, F, P_T, a2, c_f, c_f2)


# ========================== 1. cell primitives ===============================


def _cell_primitives(
    conserved_state: STATE_TYPE_ALTERED, setup: SchemeSetup
) -> Tuple[Primitives, FIELD_TYPE]:
    """
    The primitive variables of the cells and the cell-centred normal field,

        rho = max(rho, rho_min),    v = (rho v) / rho,
        P = max((gamma - 1) (E - rho v^2 / 2 - B^2 / 2), P_min)    or    P(rho).

    Every later step uses these values: there is one density, one velocity and
    one pressure per cell and flux call.
    """
    rho = jnp.maximum(conserved_state[setup.density_index], setup.minimum_density)
    m_x, m_y, m_z = (_row(conserved_state, index) for index in setup.momentum_index)
    B_x, B_y, B_z = (_row(conserved_state, index) for index in setup.magnetic_index)
    v_x, v_y, v_z = (
        m / rho if index >= 0 else 0.0 for m, index in zip((m_x, m_y, m_z), setup.momentum_index)
    )
    if setup.equation_of_state == IDEAL_GAS:
        E = conserved_state[setup.energy_index]
        P = (setup.gamma - 1.0) * (
            E
            - 0.5 * (m_x * v_x + m_y * v_y + m_z * v_z)
            - 0.5 * (B_x * B_x + B_y * B_y + B_z * B_z)
        )
        P = jnp.maximum(P, setup.minimum_pressure)
    else:
        P = _barotropic_pressure(rho, setup)
    return Primitives(rho, v_x, v_y, v_z, B_y, B_z, P), B_x


# =============== 2. interface state and characteristic basis =================


def _interface_state(
    conserved_state: STATE_TYPE_ALTERED,
    V: Primitives,
    B_x: FIELD_TYPE,
    setup: SchemeSetup,
) -> InterfaceState:
    """
    The interface state between the cells i and i + 1 (eq. ``ifstate``),

        rho = max(avg(rho), rho_min),    v = avg(rho v) / rho,    B = avg(B),
        a^2 = gamma avg(P) / rho,    c_s^2    or    dP / drho at rho,

    with avg(f) = (f_i + f_{i+1}) / 2 of the cell primitives (avg(P) >= P_min,
    as both cell pressures are).
    """

    def average(field):
        return 0.5 * (field + _at(field, 1))

    rho = jnp.maximum(average(V.rho), setup.minimum_density)
    v_x, v_y, v_z = (
        average(conserved_state[index]) / rho if index >= 0 else 0.0 for index in setup.momentum_index
    )
    a2 = _sound_speed_squared(rho, average(V.P), setup)
    return InterfaceState(
        rho, v_x, v_y, v_z, average(B_x), average(V.By), average(V.Bz), a2
    )


def _characteristic_basis(
    conserved_state: STATE_TYPE_ALTERED,
    bar: InterfaceState,
    setup: SchemeSetup,
    registered_variables: RegisteredVariables,
) -> Tuple[List[STATE_TYPE_ALTERED], List[STATE_TYPE_ALTERED]]:
    """
    The left eigenvectors L^s (rows) and right eigenvectors R^s (columns) of
    the flux Jacobian in conserved variables at the interface state ``bar``,
    in the scaling of Roe & Balsara (1996): seven waves (f-, a-, s-, e, s+,
    a+, f+) for the ideal gas, six (no entropy wave) for the isothermal gas.

    The squared fast and slow speeds go from the wave speeds into the mode
    weights; no rounded speed is squared. A polytropic gas takes the basis of
    the isothermal gas with the local a of the interface state.

    The hydrodynamic form takes ``_hydrodynamic_characteristic_basis``; the
    steps after this one do not depend on the number of waves.

    Args:
        conserved_state: Only its shape and dtype enter.

    Returns:
        The rows and the columns, one array per wave, shaped like the state.
    """
    if not setup.mhd:
        return _hydrodynamic_characteristic_basis(bar, setup)
    rho, v_x, v_y, v_z, B_x, B_y, B_z, a2 = bar
    c_f, c_a, c_s, c_f2, c_s2 = _wave_speeds(rho, B_x, B_y, B_z, a2)
    layout = (conserved_state, registered_variables)
    if setup.equation_of_state == IDEAL_GAS:
        gamma = setup.gamma
        blocks = _eigenvector_blocks(
            rho, v_x, v_y, v_z, B_x, B_y, B_z, a2, c_f, c_a, c_s, c_f2, c_s2, gamma
        )
        rows = _left_eigenvector_rows(*layout, blocks)
        columns = _right_eigenvector_columns(*layout, blocks, c_f2, c_s2)
    else:
        c_iso = setup.sound_speed if setup.equation_of_state == ISOTHERMAL else jnp.sqrt(a2)
        blocks = _eigenvector_blocks_iso(
            rho, v_x, v_y, v_z, B_x, B_y, B_z, c_iso, a2, c_f, c_a, c_s, c_f2, c_s2
        )
        rows = _left_eigenvector_rows_iso(*layout, blocks)
        columns = _right_eigenvector_columns_iso(*layout, blocks)
    return [row() for row in rows], [column() for column in columns]


def _hydrodynamic_characteristic_basis(
    bar: InterfaceState, setup: SchemeSetup
) -> Tuple[List[STATE_TYPE_ALTERED], List[STATE_TYPE_ALTERED]]:
    """
    The left eigenvectors L^s (rows) and right eigenvectors R^s (columns) of
    the Euler flux Jacobian in conserved variables (rho, rho u, rho v_t, E) at
    the interface state ``bar``: the sound waves u -+ a, the entropy wave
    (ideal gas) and one shear wave per transverse velocity component v_t of
    the layout (unit vector e_t),

        R^-+ = (1, u -+ a, v_t, H -+ u a),    R^e = (1, u, v_t, v^2 / 2),
        R^t  = (0, 0, e_t, v_t),
        L^-+ = (b_2 +- u / a, -b_1 u -+ 1 / a, -b_1 v_t, b_1) / 2,
        L^e  = (1 - b_2, b_1 u, b_1 v_t, -b_1),    L^t = (-v_t, 0, e_t, 0),

    b_1 = (gamma - 1) / a^2, b_2 = b_1 v^2 / 2, H = a^2 / (gamma - 1) + v^2 / 2.
    A barotropic gas has no energy row and no entropy wave:
    L^-+ = (a +- u, -+1, 0) / (2 a).

    Returns:
        The rows and the columns, one array per wave, shaped like the state.
    """
    one = jnp.ones_like(bar.rho)
    zero = jnp.zeros_like(bar.rho)
    ideal_gas = setup.equation_of_state == IDEAL_GAS

    def vector(density, normal, transverse, energy):
        def full(entry):
            return jnp.broadcast_to(entry, one.shape)

        return _state_vector(
            full(density), full(normal), *(full(t) for t in transverse), 0.0, 0.0,
            full(energy) if ideal_gas else None, setup,
        )

    u, v_t = bar.vx, (bar.vy, bar.vz)
    a = setup.sound_speed if setup.equation_of_state == ISOTHERMAL else jnp.sqrt(bar.a2)
    inverse_a = 1.0 / a
    if ideal_gas:
        v2 = u * u + bar.vy * bar.vy + bar.vz * bar.vz
        b_1 = (setup.gamma - 1.0) / bar.a2
        b_2 = 0.5 * b_1 * v2
        H = bar.a2 / (setup.gamma - 1.0) + 0.5 * v2
        b_1_v_t = tuple(b_1 * v for v in v_t)
        acoustic_v_t = tuple(-0.5 * b for b in b_1_v_t)
        columns = [vector(one, u - a, v_t, H - u * a), vector(one, u, v_t, 0.5 * v2)]
        rows = [
            vector(0.5 * (b_2 + u * inverse_a), -0.5 * (b_1 * u + inverse_a), acoustic_v_t, 0.5 * b_1),
            vector(1.0 - b_2, b_1 * u, b_1_v_t, -b_1),
        ]
        last_column = vector(one, u + a, v_t, H + u * a)
        last_row = vector(0.5 * (b_2 - u * inverse_a), -0.5 * (b_1 * u - inverse_a), acoustic_v_t, 0.5 * b_1)
    else:
        columns = [vector(one, u - a, v_t, None)]
        rows = [vector(0.5 * (a + u) * inverse_a, -0.5 * inverse_a, (zero, zero), None)]
        last_column = vector(one, u + a, v_t, None)
        last_row = vector(0.5 * (a - u) * inverse_a, 0.5 * inverse_a, (zero, zero), None)
    for k, (index, v) in enumerate(zip(setup.momentum_index[1:], v_t)):
        if index >= 0:
            unit = (one, zero) if k == 0 else (zero, one)
            columns.append(vector(zero, zero, unit, v))
            rows.append(vector(-v, zero, unit, zero))
    return rows + [last_row], columns + [last_column]


# ============================= 3. face states ================================


def _linearised_conserved(
    V_m: Primitives, bar: InterfaceState, setup: SchemeSetup
) -> STATE_TYPE_ALTERED:
    """
    The primitive variables of a stencil cell as a conserved vector linearised
    about the interface state ``bar``, T V_m of eq. ``charvar``:

        (rho_m,  rho v_m + v rho_m,  B_y,m,  B_z,m,
         P_m / (gamma - 1) + v^2 rho_m / 2 + rho v . v_m + B_y B_y,m + B_z B_z,m)

    with rho, v, B of the interface state.
    """
    E = None
    if setup.equation_of_state == IDEAL_GAS:
        v2 = bar.vx * bar.vx + bar.vy * bar.vy + bar.vz * bar.vz
        E = (
            V_m.P / (setup.gamma - 1.0)
            + 0.5 * v2 * V_m.rho
            + bar.rho * (bar.vx * V_m.vx + bar.vy * V_m.vy + bar.vz * V_m.vz)
            + bar.By * V_m.By
            + bar.Bz * V_m.Bz
        )
    return _state_vector(
        V_m.rho,
        bar.rho * V_m.vx + bar.vx * V_m.rho,
        bar.rho * V_m.vy + bar.vy * V_m.rho,
        bar.rho * V_m.vz + bar.vz * V_m.rho,
        V_m.By,
        V_m.Bz,
        E,
        setup,
    )


def _primitives_from_linearised(
    U: STATE_TYPE_ALTERED, bar: InterfaceState, setup: SchemeSetup
) -> Primitives:
    """
    The inverse of ``_linearised_conserved`` (eq. ``backtransform``); the
    pressure of a barotropic gas is P(rho).
    """
    rho = U[setup.density_index]
    v_x, v_y, v_z = (
        (U[index] - v * rho) / bar.rho if index >= 0 else 0.0
        for index, v in zip(setup.momentum_index, (bar.vx, bar.vy, bar.vz))
    )
    B_y, B_z = (_row(U, index) for index in setup.magnetic_index[1:])
    if setup.equation_of_state == IDEAL_GAS:
        v2 = bar.vx * bar.vx + bar.vy * bar.vy + bar.vz * bar.vz
        P = (setup.gamma - 1.0) * (
            U[setup.energy_index]
            - 0.5 * v2 * rho
            - bar.rho * (bar.vx * v_x + bar.vy * v_y + bar.vz * v_z)
            - bar.By * B_y
            - bar.Bz * B_z
        )
    else:
        P = _barotropic_pressure(rho, setup)
    return Primitives(rho, v_x, v_y, v_z, B_y, B_z, P)


def _minmod2(a, b):
    """minmod(a, b)."""
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))


def _minmod4(a, b, c, d):
    """minmod(a, b, c, d)."""
    s = (
        0.125
        * (jnp.sign(a) + jnp.sign(b))
        * jnp.abs((jnp.sign(a) + jnp.sign(c)) * (jnp.sign(a) + jnp.sign(d)))
    )
    return s * jnp.minimum(
        jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.minimum(jnp.abs(c), jnp.abs(d))
    )


def _mp_limit(u, um2, um1, u0, up1, up2, alpha=4.0):
    """
    Monotonicity-preserving bound of Suresh & Huynh (1997) on the value ``u``
    interpolated to i + 1/2 from the side of cell i (eq. ``mp``), with the cell
    values u_{i-2} .. u_{i+2}. ``u`` is kept if it lies between u_i and

        u^MP = u_i + minmod(u_{i+1} - u_i, alpha (u_i - u_{i-1})),

    up to a relative tolerance; otherwise it is clipped to the interval
    [u^min, u^max] built from u_i, u_{i+1}, u^UL, u^MD and u^LC, which allows
    smooth extrema. alpha = 4; no free parameter.
    """
    scale = (
        jnp.maximum(jnp.maximum(jnp.abs(um1), jnp.abs(u0)), jnp.abs(up1)) ** 2 * 1e-20
        + EPSILON_FLOOR
    )
    u_mp = u0 + _minmod2(up1 - u0, alpha * (u0 - um1))
    keep = (u - u0) * (u - u_mp) <= scale
    d_m = um2 - 2.0 * um1 + u0
    d_0 = um1 - 2.0 * u0 + up1
    d_p = u0 - 2.0 * up1 + up2
    d_m4_p = _minmod4(4.0 * d_0 - d_p, 4.0 * d_p - d_0, d_0, d_p)
    d_m4_m = _minmod4(4.0 * d_m - d_0, 4.0 * d_0 - d_m, d_m, d_0)
    u_ul = u0 + alpha * (u0 - um1)
    u_md = 0.5 * (u0 + up1) - 0.5 * d_m4_p
    u_lc = u0 + 0.5 * (u0 - um1) + (4.0 / 3.0) * d_m4_m
    u_min = jnp.maximum(
        jnp.minimum(jnp.minimum(u0, up1), u_md),
        jnp.minimum(jnp.minimum(u0, u_ul), u_lc),
    )
    u_max = jnp.minimum(
        jnp.maximum(jnp.maximum(u0, up1), u_md),
        jnp.maximum(jnp.maximum(u0, u_ul), u_lc),
    )
    return jnp.where(keep, u, jnp.minimum(jnp.maximum(u, u_min), u_max))


def _mp_bounded_pair(u_L, u_R, u):
    """
    The MP bound on the left and the right value at i + 1/2; ``u`` holds the
    cell values u_{i-2} .. u_{i+3} (the right value sees them mirrored).
    """
    u_L = _mp_limit(u_L, u[0], u[1], u[2], u[3], u[4])
    u_R = _mp_limit(u_R, u[5], u[4], u[3], u[2], u[1])
    return u_L, u_R


def _weno5z_face_values(w, frozen_weights: bool):
    """
    Left and right WENO5-Z point values at i + 1/2 of one characteristic
    variable, with the MP bound. ``w`` holds w_{i-2} .. w_{i+3}; the left value
    uses w_{i-2} .. w_{i+2}, the right value w_{i+3} .. w_{i-1}. Both share

        epsilon = 1e-6 max_m w_m^2 + 1e-30        (eq. ``weno_eps``)

    in the weights alpha_k = d_k (1 + tau_5 / (IS_k + epsilon)) of Borges et
    al. (2008). With ``frozen_weights`` the weights are constants of the
    differentiation.
    """
    scale2 = w[0] * w[0]
    for w_m in w[1:]:
        scale2 = jnp.maximum(scale2, w_m * w_m)
    epsilon = WENO_EPSILON_RELATIVE * scale2 + EPSILON_FLOOR
    w_L = _weno5z_point_interpolation(w[0], w[1], w[2], w[3], w[4], epsilon, frozen_weights)
    w_R = _weno5z_point_interpolation(w[5], w[4], w[3], w[2], w[1], epsilon, frozen_weights)
    return _mp_bounded_pair(w_L, w_R, w)


def _admissible(
    V_face: Primitives, V_own: Primitives, V_other: Primitives, setup: SchemeSetup
) -> Primitives:
    """
    Pulls a face state towards the state of its own cell where its density or
    pressure is below half the smaller value of the two cells of the face
    (eq. ``admissible``):

        V <- V_own + vartheta (V - V_own),
        vartheta_rho = (rho_own - rho_fl) / max(rho_own - rho, 1e-30)  if rho < rho_fl,
        rho_fl = min(rho_i, rho_{i+1}) / 2,

    likewise for P (ideal gas), vartheta = clip(min(vartheta_rho, vartheta_P), 0, 1).
    """
    rho_floor = 0.5 * jnp.minimum(V_own.rho, V_other.rho)
    vartheta = jnp.where(
        V_face.rho < rho_floor,
        _floored_ratio(V_own.rho - rho_floor, V_own.rho - V_face.rho),
        1.0,
    )
    if setup.equation_of_state == IDEAL_GAS:
        P_floor = 0.5 * jnp.minimum(V_own.P, V_other.P)
        vartheta_P = jnp.where(
            V_face.P < P_floor,
            _floored_ratio(V_own.P - P_floor, V_own.P - V_face.P),
            1.0,
        )
        vartheta = jnp.minimum(vartheta, vartheta_P)
    vartheta = jnp.clip(vartheta, 0.0, 1.0)
    carried = _carried_components(setup)
    return Primitives(
        *(own + vartheta * (face - own) if is_carried else own for own, face, is_carried in zip(V_own, V_face, carried))
    )


def _face_states(
    V: Primitives,
    bar: InterfaceState,
    rows: List[STATE_TYPE_ALTERED],
    columns: List[STATE_TYPE_ALTERED],
    setup: SchemeSetup,
) -> Tuple[Primitives, Primitives]:
    """
    The left and right primitive states at the faces i + 1/2 (section
    "Interpolation to the faces"):

        w^s_m = L^s . T V_m,  m = i - 2 .. i + 3      characteristic variables,
        w^s_L, w^s_R                                  WENO5-Z values, MP bound,
        V_L,R = T^-1 sum_s R^s w^s_L,R                back-transformation,

    then the MP bound on every primitive component with the cell values as
    neighbours, the pressure P(rho) of a barotropic gas, and the
    admissibility limit. The pressure of the polytropic gas is P(rho) of the
    admissible density: before the limit the density can be negative.
    """
    cells = [Primitives(*(_at(f, offset) for f in V)) for offset in range(-2, 4)]
    stencil = [_linearised_conserved(V_m, bar, setup) for V_m in cells]

    U_L = jnp.zeros_like(stencil[0])
    U_R = jnp.zeros_like(stencil[0])
    for row, column in zip(rows, columns):
        w = [jnp.sum(row * U_m, axis=0) for U_m in stencil]
        w_L, w_R = _weno5z_face_values(w, setup.frozen_weights)
        U_L = U_L + column * w_L[None]
        U_R = U_R + column * w_R[None]
    V_L = _primitives_from_linearised(U_L, bar, setup)
    V_R = _primitives_from_linearised(U_R, bar, setup)

    # MP bound on the interpolated primitive components (a barotropic pressure
    # is not interpolated: it follows from the bounded density)
    ideal_gas = setup.equation_of_state == IDEAL_GAS
    carried = _carried_components(setup)
    bounded = [
        _mp_bounded_pair(V_L[k], V_R[k], [V_m[k] for V_m in cells]) if carried[k] else (V_L[k], V_R[k])
        for k in range(7 if ideal_gas else 6)
    ]
    V_i, V_ip1 = cells[2], cells[3]
    polytropic = setup.equation_of_state == POLYTROPIC
    if polytropic:
        bounded.append([V_i.P, V_ip1.P])  # replaced below
    elif not ideal_gas:
        bounded.append([_barotropic_pressure(rho, setup) for rho in bounded[0]])
    V_L = Primitives(*(pair[0] for pair in bounded))
    V_R = Primitives(*(pair[1] for pair in bounded))

    # admissibility: the left state belongs to cell i, the right one to i + 1
    V_L, V_R = _admissible(V_L, V_i, V_ip1, setup), _admissible(V_R, V_ip1, V_i, setup)
    if polytropic:
        V_L, V_R = (V._replace(P=_barotropic_pressure(V.rho, setup)) for V in (V_L, V_R))
    return V_L, V_R


# ============================ 4. Riemann flux ================================


def _hll_flux(L: FluxState, R: FluxState, S_L, S_R, inv) -> STATE_TYPE_ALTERED:
    """
    The HLL flux (Harten, Lax & van Leer 1983), ``inv`` = 1 / (S_R - S_L):

        F^HLL = [S_R F_L - S_L F_R + S_L S_R (U_R - U_L)] / (S_R - S_L).
    """
    return (S_R * L.F - S_L * R.F + S_L * S_R * (R.U - L.U)) * inv


def _anti_diffusion(mu, S_L, S_R, inv):
    """
    The HLLEM anti-diffusion coefficient of a wave of speed mu (eq. ``hllem_c``),

        c(mu) = [S_R min(mu~, 0) + S_L max(mu~, 0) - S_L S_R] / (S_R - S_L) >= 0,

    with mu~ = clip(mu, S_L, S_R).
    """
    mu = jnp.clip(mu, S_L, S_R)
    return (S_R * jnp.minimum(mu, 0.0) + S_L * jnp.maximum(mu, 0.0) - S_L * S_R) * inv


def _hllem_t_flux(
    L: FluxState, R: FluxState, B_x: FIELD_TYPE, S_L, S_R, inv, setup: SchemeSetup
) -> STATE_TYPE_ALTERED:
    """
    The HLL flux with anti-diffusion in the transverse (Alfven) waves, HLLEM-T
    (section "Isothermal gas"; HLLEM of Einfeldt et al. 1991 and Dumbser &
    Balsara 2016 restricted to the four transverse vectors). With the HLL
    state U^* = [S_R U_R - S_L U_L - (F_R - F_L)] / (S_R - S_L), its velocity
    (u^*, v_t^*), c_a^* = |B_x| / sqrt(rho^*), s = sgn(B_x), Delta = R - L, and
    for each transverse direction t (eqs. ``hllem_omega``, ``hllem_flux``)

        omega_t,+- = [(Delta(rho v_t) - v_t^* Delta rho) / sqrt(rho^*)
                      -+ s Delta B_t] / 2,
        F_rho v_t = F^HLL + sqrt(rho^*) (c_+ omega_t,+ + c_- omega_t,-),
        F_B_t     = F^HLL - s (c_+ omega_t,+ - c_- omega_t,-),

    c_+- = c(u^* +- c_a^*) of ``_anti_diffusion``. Without a field c_a^* = 0
    and the transverse waves are the shear waves at u^*:

        F_rho v_t = F^HLL + c(u^*) (Delta(rho v_t) - v_t^* Delta rho).
    """

    def hll_state(index):
        return (S_R * R.U[index] - S_L * L.U[index] - (R.F[index] - L.F[index])) * inv

    def jump(index):
        return R.U[index] - L.U[index]

    # rho^* > 0: rho^* (S_R - S_L) = rho_R (S_R - u_R) + rho_L (u_L - S_L)
    rho = hll_state(setup.density_index)
    u = hll_state(setup.momentum_index.x) / rho
    zero = jnp.zeros_like(rho)
    if not setup.mhd:
        c = _anti_diffusion(u, S_L, S_R, inv)
        d_rho = jump(setup.density_index)
        shear = (
            c * (jump(index) - hll_state(index) / rho * d_rho) if index >= 0 else 0.0
            for index in setup.momentum_index[1:]
        )
        return _hll_flux(L, R, S_L, S_R, inv) + _state_vector(zero, zero, *shear, 0.0, 0.0, None, setup)
    sqrt_rho = jnp.sqrt(rho)
    v_y = hll_state(setup.momentum_index.y) / rho
    v_z = hll_state(setup.momentum_index.z) / rho
    s = jnp.where(B_x >= 0.0, 1.0, -1.0)

    d_rho = jump(setup.density_index)
    d_v_y = (jump(setup.momentum_index.y) - v_y * d_rho) / sqrt_rho
    d_v_z = (jump(setup.momentum_index.z) - v_z * d_rho) / sqrt_rho
    d_B_y = jump(setup.magnetic_index.y)
    d_B_z = jump(setup.magnetic_index.z)

    c_a = jnp.abs(B_x) / sqrt_rho
    c_plus = _anti_diffusion(u + c_a, S_L, S_R, inv)
    c_minus = _anti_diffusion(u - c_a, S_L, S_R, inv)

    # c_+ omega_t,+ and c_- omega_t,- of the two transverse directions
    plus_y = 0.5 * c_plus * (d_v_y - s * d_B_y)
    minus_y = 0.5 * c_minus * (d_v_y + s * d_B_y)
    plus_z = 0.5 * c_plus * (d_v_z - s * d_B_z)
    minus_z = 0.5 * c_minus * (d_v_z + s * d_B_z)

    return _hll_flux(L, R, S_L, S_R, inv) + _state_vector(
        zero,
        zero,
        sqrt_rho * (plus_y + minus_y),
        sqrt_rho * (plus_z + minus_z),
        s * (minus_y - plus_y),
        s * (minus_z - plus_z),
        None,
        setup,
    )


def _fast_alfven_gap(rho, B_x, B_y, B_z, a2):
    """
    G = rho (c_f^2 - B_x^2 / rho) >= B_t^2 without cancellation (eq.
    ``hlld_gap``): with d = rho a^2 + B_t^2 - B_x^2 and
    r = sqrt(d^2 + 4 B_x^2 B_t^2),

        G = (d + r) / 2  if d >= 0,    G = 2 B_x^2 B_t^2 / (r - d)  otherwise.
    """
    B_x2 = B_x * B_x
    B_t2 = B_y * B_y + B_z * B_z
    d = rho * a2 + B_t2 - B_x2
    r = zero_safe_sqrt(d * d + 4.0 * B_x2 * B_t2)
    falling = d < 0.0
    return jnp.where(
        falling, 2.0 * B_x2 * B_t2 / jnp.where(falling, r - d, 1.0), 0.5 * (d + r)
    )


def _hlld_star_state(
    side: FluxState, j, w, du, nu: float, B_x, S_M, P_T_star, Gamma, setup: SchemeSetup
) -> FanState:
    """
    The state behind the outer wave of one side (appendix "Star states of the
    bounded HLLD flux", eq. ``hlld_star``); nu = +1 on the left, -1 on the
    right. With the mass flux j = rho w through the wave, Delta u = nu (v_x -
    S_M) and the pressure rise Delta P = j Delta u,

        N = G + rho (w - c_f) (w + c_f),
        M = max(N - Delta P, G + (Gamma - 1) max(Delta P, 0)),
        xi = Delta P / M  if M > 0,  else 0,
        rho^* = j / (w - Delta u),    B_t^* = (1 + xi) B_t,
        v_t^* = v_t + nu (B_x / j) xi B_t,
        E^* = E + [Delta u E + nu (P_T v_x - P_T^* S_M - B_x J)] / (w - Delta u),
        J = nu Delta u B_x + v_t . B_t - v_t^* . B_t^*.

    The lower bound of M (which N - Delta P obeys in exact arithmetic for the
    bounded speeds) keeps -1 <= xi <= 1 / (Gamma - 1). Without a field it is
    the star state of HLLC: v_t^* = v_t, B^* = 0, J = 0.
    """
    V = side.V
    E = side.U[setup.energy_index]
    if not setup.mhd:
        E_star = E + (du * E + nu * (side.P_T * V.vx - P_T_star * S_M)) / (w - du)
        return FanState(j / (w - du), V.vy, V.vz, 0.0, 0.0, E_star, 0.0)
    G = _fast_alfven_gap(V.rho, B_x, V.By, V.Bz, side.a2)
    dP = j * du
    N = G + V.rho * (w - side.c_f) * (w + side.c_f)
    M = jnp.maximum(N - dP, G + (Gamma - 1.0) * jnp.maximum(dP, 0.0))
    positive = M > 0.0  # M = 0 only for B_t = 0: no jump of the transverse state
    xi = jnp.where(positive, dP / jnp.where(positive, M, 1.0), 0.0)
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


def _fan_flux(
    state: FanState, S_M, P_T_star, B_x, setup: SchemeSetup
) -> STATE_TYPE_ALTERED:
    """
    The physical flux of a state of the Riemann fan (normal velocity S_M,
    total pressure P_T^*). By the jump conditions it equals F + S (U^* - U).
    """
    rho, v_y, v_z, B_y, B_z, E, vB = state
    return _state_vector(
        rho * S_M,
        rho * S_M * S_M + P_T_star - B_x * B_x,
        rho * S_M * v_y - B_x * B_y,
        rho * S_M * v_z - B_x * B_z,
        B_y * S_M - B_x * v_y,
        B_z * S_M - B_x * v_z,
        (E + P_T_star) * S_M - B_x * (S_M * B_x + vB),
        setup,
    )


def _hlld_bounded_flux(
    L: FluxState, R: FluxState, B_x: FIELD_TYPE, setup: SchemeSetup
) -> STATE_TYPE_ALTERED:
    """
    The HLLD flux of Miyoshi & Kusano (2005, MK05) with bounded outer speeds
    (section "Ideal gas: HLLD with bounded outer speeds").

    Outer speeds relative to the gas (eq. ``bh_speeds``), with Gamma =
    (gamma + 1) / 2, the closing speed D = v_x,L - v_x,R, D_- = min(D, 0) and
    h = Gamma max(D, 0) / 2:

        d_L = max(c_f,L, c_f,R + D),
        Pi_L = Gamma max(P_T,R - P_T,L + rho_R d_R D_-, 0),
        h_L = h + Pi_L / (2 rho_R d_R),
        sigma_L = min(h + sqrt(h^2 + c_f,L^2 + Pi_L / rho_L),
                      h_L + sqrt(h_L^2 + c_f,L^2)),
        w_L = max(d_L, v_x,L, sigma_L),    S_L = v_x,L - w_L,    j_L = rho_L w_L,

    mirrored on the right (w_R = max(d_R, -v_x,R, sigma_R), S_R = v_x,R + w_R).
    d is the speed of Davis (1988) with S_L <= 0 <= S_R; sigma obeys the
    relation of a gas shock between its speed and its pressure rise,
    rho w^2 >= rho c_f^2 + Gamma max(Delta P, 0) (Toro 2009), which keeps the
    Alfven waves inside the fan and every state of the fan bounded.

    Middle wave and total pressure of the fan (eq. ``bh_middle``, MK05 eqs. 38
    and 41):

        S_M = (j_L v_x,L + j_R v_x,R + P_T,L - P_T,R) / (j_L + j_R),
        P_T^* = (j_R P_T,L + j_L P_T,R + j_L j_R D) / (j_L + j_R).

    Star states: ``_hlld_star_state``; double-star states between the Alfven
    waves S^*_L,R = S_M -+ |B_x| / sqrt(rho^*_L,R) (eq. ``hlld_dstar``, MK05
    eqs. 59 to 63). The flux is the physical flux of the state that holds
    x / t = 0 (eq. ``hlld_select``). Without a field it is the HLLC flux with
    the same speeds (the hydrodynamic form, ``HLLC``): the Alfven waves merge
    with the middle wave and the double-star states are not formed.
    """
    Gamma = 0.5 * (setup.gamma + 1.0)
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
    du_L = ((P_T_R - P_T_L) + j_R * D) * inv  # v_x,L - S_M
    du_R = ((P_T_L - P_T_R) + j_L * D) * inv  # S_M - v_x,R
    fan = (B_x, S_M, P_T_star)

    # star states
    star_L = _hlld_star_state(L, j_L, w_L, du_L, 1.0, *fan, Gamma, setup)
    star_R = _hlld_star_state(R, j_R, w_R, du_R, -1.0, *fan, Gamma, setup)
    fan = (S_M, P_T_star, B_x, setup)
    if not setup.mhd:
        F = jnp.where(S_R > 0.0, _fan_flux(star_R, *fan), R.F)
        F = jnp.where(S_M >= 0.0, _fan_flux(star_L, *fan), F)
        return jnp.where(S_L >= 0.0, L.F, F)

    # double-star states
    s = jnp.where(B_x >= 0.0, 1.0, -1.0)
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
    F = jnp.where(S_R > 0.0, _fan_flux(star_R, *fan), R.F)
    F = jnp.where(S_star_R >= 0.0, _fan_flux(dstar_R, *fan), F)
    F = jnp.where(S_M >= 0.0, _fan_flux(dstar_L, *fan), F)
    F = jnp.where(S_star_L >= 0.0, _fan_flux(star_L, *fan), F)
    return jnp.where(S_L >= 0.0, L.F, F)


def _riemann_flux(
    L: FluxState, R: FluxState, B_x: FIELD_TYPE, setup: SchemeSetup
) -> STATE_TYPE_ALTERED:
    """
    The Riemann flux F^RS of the left and right face states with the face
    field ``B_x`` (section "The flux at the face"). HLL and HLLEM-T use the
    speeds of Davis (1988) extended to include zero (eq. ``hllspeeds``),

        S_L = min(v_x,L - c_f,L, v_x,R - c_f,R, 0),
        S_R = max(v_x,L + c_f,L, v_x,R + c_f,R, 0).
    """
    if setup.riemann_solver in (HLLD_BOUNDED, HLLC):
        return _hlld_bounded_flux(L, R, B_x, setup)

    S_L = jnp.minimum(L.V.vx - L.c_f, R.V.vx - R.c_f)
    S_R = jnp.maximum(L.V.vx + L.c_f, R.V.vx + R.c_f)
    S_L = jnp.minimum(S_L, 0.0)
    S_R = jnp.maximum(S_R, 0.0)
    inv = _floored_ratio(1.0, S_R - S_L)
    if setup.riemann_solver == HLLEM_T:
        return _hllem_t_flux(L, R, B_x, S_L, S_R, inv, setup)
    return _hll_flux(L, R, S_L, S_R, inv)


# ================ 5. cell-flux correction, smoothness switch =================


def _window_not_smooth(a, b, c, d, e) -> Bool[Array, "..."]:
    """
    The five-point window (a, b, c, d, e) is not smooth if max_k IS_k >
    2 min_k IS_k for the three smoothness indicators IS_k of Jiang & Shu
    (eq. ``switch``).
    """
    IS_0 = 13.0 / 12.0 * (a - 2.0 * b + c) ** 2 + 0.25 * (a - 4.0 * b + 3.0 * c) ** 2
    IS_1 = 13.0 / 12.0 * (b - 2.0 * c + d) ** 2 + 0.25 * (b - d) ** 2
    IS_2 = 13.0 / 12.0 * (c - 2.0 * d + e) ** 2 + 0.25 * (3.0 * c - 4.0 * d + e) ** 2
    IS_min = jnp.minimum(jnp.minimum(IS_0, IS_1), IS_2)
    IS_max = jnp.maximum(jnp.maximum(IS_0, IS_1), IS_2)
    return IS_max - IS_min > IS_min


def _cell_flux_correction(
    F_RS: STATE_TYPE_ALTERED, F: STATE_TYPE_ALTERED
) -> STATE_TYPE_ALTERED:
    """
    The cell-flux correction of Jiang, Shu & Zhang (2013) with the smoothness
    switch (eqs. ``jsz``, ``switch``),

        F^c = F^RS - Delta^2 f / 24 + 7 Delta^4 f / 5760,

    Delta^2 f and Delta^4 f the central differences of the cell fluxes
    f_m = F_x(q_m), m = i - 2 .. i + 3, for dx^2 f_xx and dx^4 f_xxxx at
    i + 1/2. A flux component whose cell fluxes are not smooth on the window
    i - 2 .. i + 2 or i - 1 .. i + 3 keeps F^c = F^RS.
    """
    f_m2, f_m1, f_0, f_p1, f_p2, f_p3 = (_at_state(F, k) for k in range(-2, 4))
    d2f = (-5.0 * (f_m2 + f_p3) + 39.0 * (f_m1 + f_p2) - 34.0 * (f_0 + f_p1)) / 48.0
    d4f = (f_m2 - 3.0 * f_m1 + 2.0 * f_0 + 2.0 * f_p1 - 3.0 * f_p2 + f_p3) / 2.0
    F_c = F_RS - d2f / 24.0 + 7.0 * d4f / 5760.0
    not_smooth_left = _window_not_smooth(f_m2, f_m1, f_0, f_p1, f_p2)
    not_smooth_right = _window_not_smooth(f_m1, f_0, f_p1, f_p2, f_p3)
    return jnp.where(not_smooth_left | not_smooth_right, F_RS, F_c)


# ============================ 6. H-correction ================================


def _h_correction(
    F_c: STATE_TYPE_ALTERED,
    U_L: STATE_TYPE_ALTERED,
    U_R: STATE_TYPE_ALTERED,
    eta: FIELD_TYPE,
) -> STATE_TYPE_ALTERED:
    """
    The H-correction of Sanders et al. (1998) (eq. ``hcorr``),

        F^H = F^c - max(eta_i, eta_{i+1}) (U_R - U_L) / 2,

    eta the transverse compression of the cells (``transverse_compression``).
    """
    eta_face = jnp.maximum(eta, _at(eta, 1))
    return F_c - 0.5 * eta_face * (U_R - U_L)


# ================================ 7. guard ===================================


def _lax_friedrichs_flux(cell: FluxState) -> Tuple[STATE_TYPE_ALTERED, FIELD_TYPE]:
    """
    The first-order Lax-Friedrichs flux of the cell values (eq. ``lf``),

        F^LF = [F_x(q_i) + F_x(q_{i+1})] / 2 - max(a_i, a_{i+1}) (U_{i+1} - U_i) / 2,

    with the signal speed a_i = |v_x,i| + c_f,i, and the three-cell maximum
    a^(3)_i = max(a_{i-1}, a_i, a_{i+1}).
    """
    U, F = cell.U, cell.F
    a = jnp.abs(cell.V.vx) + cell.c_f
    a_face = jnp.maximum(a, _at(a, 1))
    F_LF = 0.5 * (F + _at_state(F, 1)) - 0.5 * a_face * (_at_state(U, 1) - U)
    a3 = jnp.maximum(jnp.maximum(_at(a, -1), a), _at(a, 1))
    return F_LF, a3


def _pressure_of_conserved(q: STATE_TYPE_ALTERED, setup: SchemeSetup) -> FIELD_TYPE:
    """
    The pressure (gamma - 1) (E - |rho v|^2 / (2 rho) - B^2 / 2) of a trial
    conserved vector of the guard, without floors (its sign is what the guard
    tests).
    """
    rho = q[setup.density_index]
    m_x, m_y, m_z = (_row(q, index) for index in setup.momentum_index)
    B_x, B_y, B_z = (_row(q, index) for index in setup.magnetic_index)
    m2 = m_x * m_x + m_y * m_y + m_z * m_z
    B2 = B_x * B_x + B_y * B_y + B_z * B_z
    return (setup.gamma - 1.0) * (
        q[setup.energy_index] - _floored_ratio(0.5 * m2, rho) - 0.5 * B2
    )


def _budget_factors(t, g_low, l_plus, l_minus):
    """
    One linear bound g(q) >= 0 of the guard on the factors t = (t^+, t^-) of a
    cell (eq. ``guard_ratio``, Zalesak 1979). ``g_low`` is g of the low-order
    update, ``l_plus`` and ``l_minus`` the linear part of g applied to the
    changes c^+ and c^- by the two faces of the cell. With the budget
    B = max(g_low, 0) and the draws D^+- = max(-l^+-, 0),

        r = min(B / max(D^+ + D^-, 1e-30), 1),
        t^+- <- min(t^+-, r)  where D^+- > 0.
    """
    t_plus, t_minus = t
    budget = jnp.maximum(g_low, 0.0)
    draw_plus = jnp.maximum(-l_plus, 0.0)
    draw_minus = jnp.maximum(-l_minus, 0.0)
    ratio = jnp.minimum(_floored_ratio(budget, draw_plus + draw_minus), 1.0)
    t_plus = jnp.where(draw_plus > 0.0, jnp.minimum(t_plus, ratio), t_plus)
    t_minus = jnp.where(draw_minus > 0.0, jnp.minimum(t_minus, ratio), t_minus)
    return t_plus, t_minus


def _guard(
    F_H: STATE_TYPE_ALTERED,
    F_LF: STATE_TYPE_ALTERED,
    conserved_state: STATE_TYPE_ALTERED,
    V: Primitives,
    B_x: FIELD_TYPE,
    a3: FIELD_TYPE,
    reference_step: SCALAR,
    setup: SchemeSetup,
) -> STATE_TYPE_ALTERED:
    """
    The guard (section "The guard"): the flux (eq. ``guard_flux``)

        F = F^LF + theta (F^H - F^LF),    theta_{i+1/2} = min(t^+_i, t^-_{i+1}),

    with the largest factors t^+-_i of the two faces of cell i for which the
    forward-Euler update of the cell along this axis with the reference step
    1 / lambda (eq. ``guard_low``),

        q^low_i + t^+_i c^+_i + t^-_i c^-_i,
        q^low_i = q_i - (F^LF_{i+1/2} - F^LF_{i-1/2}) / lambda,
        c^+_i = -A_{i+1/2} / lambda,    c^-_i = +A_{i-1/2} / lambda,    A = F^H - F^LF,

    keeps the bounds of eq. ``guard_bounds``, one budget per cell and bound:

        rho >= max[rho_min, min(rho^low_i, chi rho^ref_i)]     (hole bound, chi = 1/2),
        v^min_c rho <= rho v_c <= v^max_c rho                  (c = x, y, z),
        P >= P_min                                             (ideal gas, chord),

    with v^max_c, v^min_c the maximum, minimum of v_c over the cells i - 1,
    i, i + 1 widened by 2 a^(3)_i, and the reference density (eq. ``hole``)

        rho^ref_i = max over (l, r) of min[rho_l, rho_r, B_i^2 / max(v_A,l^2, v_A,r^2)],

    (l, r) = (i-1, i+1), (i-1, i+2), (i-2, i+1), v_A^2 = B^2 / rho. Without a
    field rho^ref = 0 and the density bound is rho >= rho_min. The
    pressure is concave in q: the scaled changes are limited once more with
    the chord between q^low and q^low + 2 t^+- c^+- (eq. ``guard_pressure``).

    Args:
        reference_step: 1 / lambda, lambda the maximum signal speed of the grid
            along this axis (``maximum_signal_speed``).
    """
    r = setup.density_index
    A = F_H - F_LF
    q_low = conserved_state - reference_step * (F_LF - _at_state(F_LF, -1))
    c_plus = -reference_step * A
    c_minus = reference_step * _at_state(A, -1)
    updates = (q_low, c_plus, c_minus)
    t = (jnp.ones_like(V.rho), jnp.ones_like(V.rho))

    # density with the hole bound
    rho_lower = setup.minimum_density
    if setup.mhd:
        B2 = B_x * B_x + V.By * V.By + V.Bz * V.Bz
        v_A2 = B2 / V.rho
        rho_ref = None
        for left, right in ((-1, 1), (-1, 2), (-2, 1)):
            v_A2_pair = jnp.maximum(_at(v_A2, left), _at(v_A2, right))
            rho_pair = jnp.minimum(_at(V.rho, left), _at(V.rho, right))
            rho_pair = jnp.minimum(rho_pair, _floored_ratio(B2, v_A2_pair))
            rho_ref = rho_pair if rho_ref is None else jnp.maximum(rho_ref, rho_pair)
        rho_lower = jnp.maximum(
            setup.minimum_density, jnp.minimum(q_low[r], HOLE_FRACTION * rho_ref)
        )
    t = _budget_factors(t, q_low[r] - rho_lower, c_plus[r], c_minus[r])

    # velocity components: the two bounds are linear in q
    for index, v in zip(setup.momentum_index, (V.vx, V.vy, V.vz)):
        if index < 0:
            continue
        v_max = jnp.maximum(jnp.maximum(_at(v, -1), v), _at(v, 1)) + 2.0 * a3
        v_min = jnp.minimum(jnp.minimum(_at(v, -1), v), _at(v, 1)) - 2.0 * a3
        t = _budget_factors(t, *(v_max * q[r] - q[index] for q in updates))
        t = _budget_factors(t, *(q[index] - v_min * q[r] for q in updates))
    t_plus, t_minus = t

    # pressure of the ideal gas
    if setup.equation_of_state == IDEAL_GAS:
        P_min = setup.minimum_pressure
        P_low = _pressure_of_conserved(q_low, setup)

        def chord(change):
            P_end = _pressure_of_conserved(q_low + 2.0 * change, setup)
            psi = jnp.where(
                P_end < P_min,
                _floored_ratio(P_low - P_min, P_low - P_end),
                1.0,
            )
            return jnp.clip(psi, 0.0, 1.0)

        t_plus = t_plus * chord(t_plus * c_plus)
        t_minus = t_minus * chord(t_minus * c_minus)

    theta = jnp.minimum(t_plus, _at(t_minus, 1))
    return F_LF + theta * A


# ====================== the flux along the first axis ========================


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _interface_flux_x(
    conserved_state: STATE_TYPE_ALTERED,
    face_field: FIELD_TYPE,
    reference_step: SCALAR,
    eta: FIELD_TYPE,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE_ALTERED:
    """
    The guarded flux at the faces i + 1/2 (index i) along the first spatial
    axis; its two-point difference is the stage increment of the cells.

    Args:
        conserved_state: The conserved state, sweep axis first.
        face_field: The face-centred field normal to the faces (B_x at
            i + 1/2), the normal field of the Riemann problem; None in the
            hydrodynamic form.
        reference_step: The reference step 1 / lambda of the guard.
        eta: The transverse compression of the cells for the H-correction.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The flux, shaped like the state.
    """
    setup = _scheme_setup(conserved_state, params, config, registered_variables)
    if face_field is None:
        face_field = 0.0

    # 1. cell primitives
    V, B_x = _cell_primitives(conserved_state, setup)

    # 2. interface state and characteristic basis
    bar = _interface_state(conserved_state, V, B_x, setup)
    rows, columns = _characteristic_basis(conserved_state, bar, setup, registered_variables)

    # 3. face states
    V_L, V_R = _face_states(V, bar, rows, columns, setup)

    # 4. Riemann flux
    L = _flux_state(V_L, face_field, setup)
    R = _flux_state(V_R, face_field, setup)
    F_RS = _riemann_flux(L, R, face_field, setup)

    # 5. cell-flux correction with the smoothness switch
    cell = _flux_state(V, B_x, setup)
    F_c = _cell_flux_correction(F_RS, cell.F)

    # 6. H-correction
    F_H = _h_correction(F_c, L.U, R.U, eta)

    # 7. guard
    F_LF, a3 = _lax_friedrichs_flux(cell)
    return _guard(F_H, F_LF, conserved_state, V, B_x, a3, reference_step, setup)


# ======================== the services of a stage ============================


def _sweep_layout(
    state: Union[STATE_TYPE, STATE_TYPE_ALTERED], axis: int, setup: SchemeSetup
) -> Union[STATE_TYPE, STATE_TYPE_ALTERED]:
    """
    Exchanges the first spatial axis with ``axis`` and the x components of
    momentum and field with the components along ``axis``: the layout in which
    a sweep along ``axis`` is a sweep along x. The exchange is its own inverse.
    """
    if axis == 0:
        return state
    swapped = jnp.swapaxes(state, 1, 1 + axis)
    rows = list(range(setup.num_vars))
    for index in (setup.momentum_index, setup.magnetic_index):
        if index.x >= 0:
            rows[index.x], rows[index[axis]] = index[axis], index.x
    return jnp.stack([swapped[row] for row in rows])


def maximum_signal_speed(
    conserved_state: STATE_TYPE,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int,
) -> Float[Array, ""]:
    """
    The maximum over the grid of the signal speed along ``axis``,

        lambda = max_i (|v_axis,i| + c_f,i),

    c_f the fast speed along ``axis``. 1 / lambda is the reference step of the
    guard (and the step of the CFL condition along this axis). The velocity,
    the pressure and the fast speed are those of the cells in the flux call
    along ``axis`` (``_cell_primitives`` in the sweep layout, ``_fast_speed``
    with the normal field first).
    """
    setup = _scheme_setup(conserved_state, params, config, registered_variables)
    V, B_x = _cell_primitives(_sweep_layout(conserved_state, axis, setup), setup)
    a2 = _sound_speed_squared(V.rho, V.P, setup)
    c_f, _ = _fast_speed(V.rho, B_x, V.By, V.Bz, a2, setup)
    return jnp.max(jnp.abs(V.vx) + c_f)


def transverse_compression(
    conserved_state: STATE_TYPE,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> List[FIELD_TYPE]:
    """
    The eta of the H-correction for every axis (eq. ``hcorr_eta``): for the
    sweep along x

        eta^(x)_i = max over the other directions d of
                    max(0, -(v_d|_{+1} - v_d|_{-1}) / 2),

    the differences taken along d (zero in one dimension).
    """
    setup = _scheme_setup(conserved_state, params, config, registered_variables)
    rho = jnp.maximum(conserved_state[setup.density_index], setup.minimum_density)
    compression = []
    for d in range(setup.dimensionality):
        v_d = conserved_state[setup.momentum_index[d]] / rho
        compression.append(
            jnp.maximum(0.0, -0.5 * (_at(v_d, 1, axis=d) - _at(v_d, -1, axis=d)))
        )
    eta = []
    for axis in range(setup.dimensionality):
        eta_axis = jnp.zeros_like(rho)
        for d in range(setup.dimensionality):
            if d != axis:
                eta_axis = jnp.maximum(eta_axis, compression[d])
        eta.append(eta_axis)
    return eta


def interface_flux(
    conserved_state: STATE_TYPE,
    face_field: FIELD_TYPE,
    reference_step: SCALAR,
    eta: FIELD_TYPE,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int,
) -> STATE_TYPE:
    """
    The guarded flux at the faces along ``axis`` (index i holds the face
    i + 1/2 along ``axis``), in the layout of the state.

    Args:
        conserved_state: The conserved state.
        face_field: The face-centred field component along ``axis`` (None in
            the hydrodynamic form).
        reference_step: 1 / ``maximum_signal_speed`` of this axis.
        eta: The entry of ``transverse_compression`` for this axis.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        axis: The sweep axis.

    Returns:
        The flux, shaped like the state.
    """
    setup = _scheme_setup(conserved_state, params, config, registered_variables)

    def native_branch(state, field, step, compression, parameters):
        flux = _interface_flux_x(
            _sweep_layout(state, axis, setup),
            None if field is None else jnp.swapaxes(field, 0, axis),
            step,
            jnp.swapaxes(compression, 0, axis),
            parameters,
            config,
            registered_variables,
        )
        return _sweep_layout(flux, axis, setup)

    primals = (conserved_state, face_field, reference_step, eta, params)
    if _riemann_flux_pallas_supported(conserved_state, config):

        def pallas_branch(state, field, step, compression, parameters):
            if setup.dimensionality < 3:
                # the kernels tile three dimensions: a 1D or 2D grid is a layer of one cell across
                layer = (1,) * (3 - setup.dimensionality)
                flux = _interface_flux_pallas(
                    state.reshape(state.shape + layer), field, step, compression.reshape(compression.shape + layer),
                    parameters, config, registered_variables, axis=axis,
                )
                return flux.reshape(state.shape)
            return _interface_flux_pallas(
                state, field, step, compression, parameters, config, registered_variables, axis=axis
            )

        # the kernels compute the values, a derivative takes the array form
        return diffable_pallas_call_n(primals, pallas_branch=pallas_branch, native_branch=native_branch)
    return native_branch(*primals)


# The kernel module is the compiled form of this module and is imported at the
# bottom, as everywhere in the package.
from astronomix._pallas_helpers import diffable_pallas_call_n  # noqa: E402
from astronomix._finite_difference._interface_fluxes._riemann_flux_pallas import (  # noqa: E402
    _interface_flux_pallas,
    _riemann_flux_pallas_supported,
)
