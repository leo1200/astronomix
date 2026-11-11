"""
The VL2 finite-volume scheme with constrained transport (VL2 + CT).

The second-order MHD integrator of Athena (Stone & Gardiner 2009, New
Astronomy 14, 139, section 4, "SG09" below; the corner electromotive forces are
those of Gardiner & Stone 2005, J. Comput. Phys. 205, 509, "GS05", and 2008,
J. Comput. Phys. 227, 4123, "GS08"). It is the constrained-transport
counterpart of the GLM scheme in :mod:`._van_leer_integrator`, selected by
``config.vl2_constrained_transport``; the reconstruction, the Riemann solvers
and the conversions between conserved and primitive variables are the ones of
that module.

Variables
---------
The primary magnetic variable is the face-centred field: ``b_x`` on the
x-faces ``(i ± 1/2, j, k)``, ``b_y`` on the y-faces and ``b_z`` on the z-faces.
The cell-centred field, which the Riemann problems, the energy and the
time-step estimate use, is the average of the two faces of a cell (SG09 eq.
13)::

    B_x(i, j, k) = [b_x(i - 1/2, j, k) + b_x(i + 1/2, j, k)] / 2    (and cyclic)

The state is ``(rho, v_x, v_y, v_z, p, B_x, B_y, B_z, b_x, b_y, b_z)`` in any
dimensionality; there is no cleaning scalar psi. In the state the face rows
follow the convention of the finite-difference layout: index ``i`` holds the
value at ``i + 1/2``. Inside this module all face and edge arrays use the
convention of the VL2 flux arrays instead: index ``i`` holds the value at
``i - 1/2`` (face arrays along their normal, edge arrays along both axes normal
to the edge direction); ``_evolve_state_vl2_ct`` converts on entry and exit.

One step
--------
From ``W^n`` (conserved ``U^n``, face field ``b^n``) with time step ``dt``
(SG09 section 4.2, steps 1 to 9)::

    predictor  F = donor-cell Riemann fluxes of W^n, normal field = b^n
               E = edge electromotive forces of (F, W^n)
               U^{n+1/2} = U^n - dt/2 div F,    b^{n+1/2} = b^n - dt/2 curl E
    corrector  F = PLM (van Leer slope) Riemann fluxes of W^{n+1/2},
                   normal field = b^{n+1/2}
               E = edge electromotive forces of (F, W^{n+1/2})
               U^{n+1} = U^n - dt div F,        b^{n+1} = b^n - dt curl E

After each stage the cell-centred field is the face average, the total energy
(updated in flux form) contains the magnetic energy of that cell-centred field,
and the primitive variables follow from it.

Riemann problems. At a face normal to axis ``a`` the left and right states
carry the reconstructed transverse cell-centred field components and the
single-valued face field as normal component on both sides (SG09 section 4.2
step 1). The interface solvers of the GLM scheme are called unchanged with
that normal field on both sides and psi = 0: their interface normal field
``(B_L + B_R)/2 - (psi_R - psi_L)/(2 c_h)`` is then the face field, and the
flux of the normal field vanishes.

Face electromotive forces. The induction equation ``dB/dt = -curl E`` with
``E = -v x B`` makes the flux of a transverse field component an electromotive
force: for cyclic ``(a, b, c)``, ``F_a(B_b) = -E_c`` and ``F_b(B_a) = +E_c``.

Edge electromotive force (GS05 eq. 41, 45 and 50; in three dimensions GS08
section 4.3, each component by the same formula in the plane normal to it).
For ``E_c`` at the edge ``(i - 1/2, j - 1/2)`` of the ``(a, b)`` plane, with
``Ea`` the value on the a-faces, ``Eb`` on the b-faces and ``Ecc`` the
cell-centred reference ``-(v x B)_c`` of the stage's primitive state::

    E_c(i-1/2, j-1/2) = 1/4 [Ea(i-1/2, j) + Ea(i-1/2, j-1) + Eb(i, j-1/2) + Eb(i-1, j-1/2)]
                      + 1/4 [D_b(i-1/2, j-3/4) - D_b(i-1/2, j-1/4)]
                      + 1/4 [D_a(i-3/4, j-1/2) - D_a(i-1/4, j-1/2)]

    D_b(i*, j-1/4) = Ecc(i*, j) - Eb(i*, j-1/2),   D_b(i*, j-3/4) = Eb(i*, j-1/2) - Ecc(i*, j-1)
    D_a(i-1/4, j*) = Ecc(i, j*) - Ea(i-1/2, j*),   D_a(i-3/4, j*) = Ea(i-1/2, j*) - Ecc(i-1, j*)

The ``D`` are differences of ``E_c`` over half a cell (``dx/2`` times the
derivative estimates of GS05 eq. 45). ``D_b(i-1/2, .)`` is taken from the
upwind side of the a-face it lies on (``i* = i - 1`` if the mass flux through
that face is positive, ``i* = i`` if negative, the mean of both if it is
zero), and ``D_a(., j-1/2)`` likewise from the upwind side of its b-face (GS05
eq. 50, upwinding by the contact mode). For a uniform state the formula
returns ``Ecc``; for a flow that varies along one axis only it returns the
face value of the one-dimensional Riemann problem, which is the design
condition of GS05.

Constrained transport (SG09 eq. 15 to 17; ``dx`` the cubic grid spacing)::

    b_a(face) -= dt/dx { [E_c(b + 1/2) - E_c(b - 1/2)] - [E_b(c + 1/2) - E_b(c - 1/2)] }

for cyclic ``(a, b, c)``, the differences taken between the two edges of the
face in direction ``b`` and ``c``. Each edge value enters the four faces
around the edge with signs that cancel in the face divergence
``sum_a [b_a(a + 1/2) - b_a(a - 1/2)] / dx``, which therefore keeps its
initial value (zero) to rounding.

One and two dimensions run the same three-dimensional code on a state with
singleton trailing axes: along such an axis left and right states coincide,
the Riemann flux is the physical flux of the cell, and the edge formula
reduces to the face value of the remaining axes (the two-dimensional and
one-dimensional forms of the algorithm).

Optional first-order flux correction (``config.first_order_flux_correction``).
The face fluxes of a stage, including their field rows, which are the face
electromotive forces, are replaced by donor-cell local Lax-Friedrichs fluxes
on all faces of every cell whose trial update has a non-positive density or
pressure; the edge electromotive forces are then built from the corrected
fluxes, so the field stays divergence-free. The trial update is the
finite-volume one of the GLM scheme (the cell-centred field updated with the
divergence of the face fluxes), which serves as the predictor of the
constrained-transport update; the flagging, the four attempts and the role of
the floors are those of ``_first_order_flux_correction``. Athena++'s
first-order flux correction with CT is built the same way (corrected face
fluxes, then corner electromotive forces); this implementation was written
from that description, not checked line by line against the Athena++ source.

Not supported: non-periodic boundaries (the staggered update is written for
the periodic-roll layout only), gravity, the Pallas backend (array-form JAX
only), non-cubic cells, the extended GLM source (there is no GLM field).
``finalize_config`` rejects the first two and the last.
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
    LAX_FRIEDRICHS,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._state_evolution._van_leer_integrator import (
    NUM_FLUX_CORRECTION_ATTEMPTS,
    _conserved_from_primitive_vl2,
    _corrected_fluxes,
    _flux_divergence,
    _interface_flux_from_states,
    _newly_flagged_cells,
    _positivity_failure_code,
    _primitive_from_conserved_vl2,
    _reconstructed_face_states,
    _state_components_from_frame,
)

# -------------------------------------------------------------
# ===================== ↓ Module constants ↓ ===================
# -------------------------------------------------------------

#: The gas variables and the cell-centred field, ``(rho, v_x, v_y, v_z, p, B_x,
#: B_y, B_z)``: the leading rows of the state, on which the fluxes act.
NUM_CELL_CENTRED_VARIABLES = 8

#: Position of the normal field in the interface frame ``(rho, v_n, v_t1,
#: v_t2, p, B_n, B_t1, B_t2)``.
FRAME_NORMAL_FIELD = 5

# -------------------------------------------------------------
# ===================== ↑ Module constants ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================== ↓ Interface fluxes ↓ ======================
# -------------------------------------------------------------


def _ct_frame_indices(axis: int, registered_variables: RegisteredVariables):
    """
    The state indices of the cell-centred variables in the frame of an
    interface normal to ``axis`` (1-based), in cyclic order: normal, then
    transverse 1 and transverse 2 (x: y, z; y: z, x; z: x, y).

    Args:
        axis: The spatial axis (1, 2 or 3) normal to the interface.
        registered_variables: The registered variables.

    Returns:
        The state indices of ``(rho, v_n, v_t1, v_t2, p, B_n, B_t1, B_t2)``.
    """
    velocity = tuple(registered_variables.velocity_index)
    field = tuple(registered_variables.magnetic_index)
    normal = axis - 1
    transverse_1 = axis % 3
    transverse_2 = (axis + 1) % 3
    return (
        registered_variables.density_index,
        velocity[normal],
        velocity[transverse_1],
        velocity[transverse_2],
        registered_variables.pressure_index,
        field[normal],
        field[transverse_1],
        field[transverse_2],
    )


def _ct_interface_fluxes(
    cell_state: STATE_TYPE,
    normal_face_field,
    axis: int,
    piecewise_linear: bool,
    riemann_solver: int,
    gamma,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    The interface fluxes along one axis with the face field as the normal
    field of both Riemann states: ``flux[:, i]`` is the flux through the
    interface ``i - 1/2``. The field rows hold the fluxes of the cell-centred
    field, i.e. the face electromotive forces (``F_a(B_b) = -E_c``,
    ``F_b(B_a) = +E_c`` for cyclic ``(a, b, c)``); the row of the normal field
    is zero.

    Args:
        cell_state: The cell-centred primitive variables (three spatial axes).
        normal_face_field: The face field of this axis (value at ``i - 1/2``).
        axis: The spatial axis (1-based).
        piecewise_linear: Use PLM (else donor-cell) reconstruction.
        riemann_solver: The Riemann solver.
        gamma: The adiabatic index.
        registered_variables: The registered variables.

    Returns:
        The flux array, laid out like ``cell_state``.
    """
    left_states, right_states = _reconstructed_face_states(cell_state, axis, piecewise_linear)
    frame_indices = _ct_frame_indices(axis, registered_variables)
    zero = jnp.zeros_like(normal_face_field)

    def frame_values(states):
        values = [states[index] for index in frame_indices]
        values[FRAME_NORMAL_FIELD] = normal_face_field
        # The solvers of the GLM scheme take psi as a ninth value.
        return tuple(values) + (zero,)

    # With equal normal fields and psi = 0 on both sides the GLM interface
    # solve returns the face field and a vanishing normal-field flux for any
    # cleaning speed; the value 1 only keeps its division finite.
    flux = _interface_flux_from_states(
        frame_values(left_states),
        frame_values(right_states),
        gamma,
        1.0,
        riemann_solver,
        True,
    )
    flux_values = list(flux[:NUM_CELL_CENTRED_VARIABLES])
    flux_values[FRAME_NORMAL_FIELD] = zero
    return jnp.stack(
        _state_components_from_frame(flux_values, frame_indices, NUM_CELL_CENTRED_VARIABLES),
        axis=0,
    )


def _ct_fluxes_per_axis(
    cell_state: STATE_TYPE,
    face_field,
    piecewise_linear: bool,
    riemann_solver: int,
    gamma,
    registered_variables: RegisteredVariables,
):
    """
    The interface fluxes of all three axes (see ``_ct_interface_fluxes``).

    Args:
        cell_state: The cell-centred primitive variables (three spatial axes).
        face_field: The three face-field components (value at ``i - 1/2``).
        piecewise_linear: Use PLM (else donor-cell) reconstruction.
        riemann_solver: The Riemann solver.
        gamma: The adiabatic index.
        registered_variables: The registered variables.

    Returns:
        The interface fluxes per axis.
    """
    fluxes_per_axis = tuple(
        _ct_interface_fluxes(
            cell_state,
            face_field[axis - 1],
            axis,
            piecewise_linear,
            riemann_solver,
            gamma,
            registered_variables,
        )
        for axis in (1, 2, 3)
    )
    # Each flux array has many consumers (the flux divergence and, shifted in
    # several ways, the edge electromotive forces). Without the barrier XLA
    # fuses the Riemann solver into every consumer and evaluates it once per
    # use; the barrier makes it evaluate the fluxes once. The values are
    # unchanged (measured at 32^3 on four CPU cores in double precision: 388 ms
    # per step without the barrier, 43 ms with it, identical results).
    return list(jax.lax.optimization_barrier(fluxes_per_axis))


def _ct_first_order_flux_correction(
    fluxes_per_axis,
    cell_state: STATE_TYPE,
    face_field,
    base_conserved_state: STATE_TYPE,
    stage_time_step,
    gamma,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The first-order flux correction of the constrained-transport stage: the
    procedure of ``_first_order_flux_correction`` on the face fluxes, with
    donor-cell local Lax-Friedrichs fluxes that use the face field as normal
    field. The trial update is the finite-volume update of all cell-centred
    variables (see the module docstring); the edge electromotive forces are
    built from the returned fluxes.

    Args:
        fluxes_per_axis: The stage's interface fluxes per axis.
        cell_state: The cell-centred primitive variables the stage started from.
        face_field: The three face-field components of that state.
        base_conserved_state: The conserved cell-centred variables the stage
            update starts from (``U^n`` in both stages).
        stage_time_step: The stage's time-step weight ``beta * dt``.
        gamma: The adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The corrected interface fluxes per axis.
    """
    first_order_fluxes = _ct_fluxes_per_axis(
        cell_state,
        face_field,
        False,
        LAX_FRIEDRICHS,
        gamma,
        registered_variables,
    )

    correction_mask = jnp.zeros(cell_state.shape[1:], dtype=bool)
    for attempt in range(NUM_FLUX_CORRECTION_ATTEMPTS):
        fluxes = _corrected_fluxes(fluxes_per_axis, first_order_fluxes, correction_mask)
        trial_state = base_conserved_state - stage_time_step * _flux_divergence(fluxes, config)
        failure_codes = _positivity_failure_code(list(trial_state), config, registered_variables)
        correction_mask = correction_mask | _newly_flagged_cells(failure_codes, attempt)

    return _corrected_fluxes(fluxes_per_axis, first_order_fluxes, correction_mask)


# -------------------------------------------------------------
# ================== ↑ Interface fluxes ↑ ======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ============ ↓ Edge electromotive forces and CT ↓ ============
# -------------------------------------------------------------


def _upwind_value(mass_flux, upstream_for_positive, upstream_for_negative):
    """
    The value on the upwind side of a face: the first candidate where the mass
    flux through the face is positive, the second where it is negative, their
    mean where it vanishes (GS05 eq. 50).

    Args:
        mass_flux: The mass flux through the face.
        upstream_for_positive: The value in the cell on the lower side.
        upstream_for_negative: The value in the cell on the upper side.

    Returns:
        The upwinded value.
    """
    return jnp.where(
        mass_flux > 0.0,
        upstream_for_positive,
        jnp.where(
            mass_flux < 0.0,
            upstream_for_negative,
            0.5 * (upstream_for_positive + upstream_for_negative),
        ),
    )


def _edge_electromotive_force(
    fluxes_per_axis,
    cell_state: STATE_TYPE,
    component: int,
    registered_variables: RegisteredVariables,
):
    """
    The electromotive force component ``c`` on the cell edges along ``c``, by
    the upwind formula of Gardiner & Stone (2005, eq. 41, 45, 50; see the
    module docstring for the formula in the index convention used here). With
    ``(a, b, c)`` cyclic, the result at array index ``(i, j)`` along ``(a, b)``
    is the value at the edge ``(i - 1/2, j - 1/2)``.

    Args:
        fluxes_per_axis: The interface fluxes per axis (at ``i - 1/2``).
        cell_state: The cell-centred primitive variables the fluxes were
            computed from (reference electromotive force).
        component: The component ``c`` (0, 1 or 2).
        registered_variables: The registered variables.

    Returns:
        The edge electromotive force.
    """
    axis_a = (component + 1) % 3
    axis_b = (component + 2) % 3
    velocity = tuple(registered_variables.velocity_index)
    field = tuple(registered_variables.magnetic_index)
    density_index = registered_variables.density_index

    def lower(values, axis):
        """The values one cell lower along a spatial axis (index i - 1)."""
        return jnp.roll(values, 1, axis=axis)

    # Face values from the fluxes of the cell-centred field: F_a(B_b) = -E_c
    # on the a-faces (i - 1/2, j), F_b(B_a) = +E_c on the b-faces (i, j - 1/2).
    emf_a_face = -fluxes_per_axis[axis_a][field[axis_b]]
    emf_b_face = fluxes_per_axis[axis_b][field[axis_a]]
    # Cell-centred reference E_c = -(v x B)_c = v_b B_a - v_a B_b.
    emf_cell = (
        cell_state[velocity[axis_b]] * cell_state[field[axis_a]]
        - cell_state[velocity[axis_a]] * cell_state[field[axis_b]]
    )
    mass_flux_a = fluxes_per_axis[axis_a][density_index]
    mass_flux_b = fluxes_per_axis[axis_b][density_index]

    # Half-cell differences along b in cell column i:
    # D_b(i, j - 1/4) = Ecc(i, j) - Eb(i, j - 1/2),
    # D_b(i, j - 3/4) = Eb(i, j - 1/2) - Ecc(i, j - 1);
    # each upwinded across the a-face it borders: (i - 1/2, j) for the first,
    # (i - 1/2, j - 1) for the second.
    difference_b_upper = emf_cell - emf_b_face
    difference_b_lower = emf_b_face - lower(emf_cell, axis_b)
    upwind_b_upper = _upwind_value(
        mass_flux_a,
        lower(difference_b_upper, axis_a),
        difference_b_upper,
    )
    upwind_b_lower = _upwind_value(
        lower(mass_flux_a, axis_b),
        lower(difference_b_lower, axis_a),
        difference_b_lower,
    )

    # Half-cell differences along a in cell row j:
    # D_a(i - 1/4, j) = Ecc(i, j) - Ea(i - 1/2, j),
    # D_a(i - 3/4, j) = Ea(i - 1/2, j) - Ecc(i - 1, j);
    # each upwinded across the b-face it borders: (i, j - 1/2) for the first,
    # (i - 1, j - 1/2) for the second.
    difference_a_upper = emf_cell - emf_a_face
    difference_a_lower = emf_a_face - lower(emf_cell, axis_a)
    upwind_a_upper = _upwind_value(
        mass_flux_b,
        lower(difference_a_upper, axis_b),
        difference_a_upper,
    )
    upwind_a_lower = _upwind_value(
        lower(mass_flux_b, axis_a),
        lower(difference_a_lower, axis_b),
        difference_a_lower,
    )

    face_mean = 0.25 * (
        emf_a_face + lower(emf_a_face, axis_b) + emf_b_face + lower(emf_b_face, axis_a)
    )
    return (
        face_mean
        + 0.25 * (upwind_b_lower - upwind_b_upper)
        + 0.25 * (upwind_a_lower - upwind_a_upper)
    )


def _constrained_transport_update(base_face_field, edge_emf, time_step_over_spacing):
    """
    The constrained-transport update of the face field (SG09 eq. 15 to 17)::

        b_a -= dt/dx { [E_c(b + 1/2) - E_c(b - 1/2)] - [E_b(c + 1/2) - E_b(c - 1/2)] }

    for cyclic ``(a, b, c)``, with faces and edges stored at their lower
    index.

    Args:
        base_face_field: The three face-field components the update starts from.
        edge_emf: The three edge electromotive force components.
        time_step_over_spacing: ``beta dt / dx``.

    Returns:
        The updated face-field components.
    """
    new_face_field = []
    for axis_a in range(3):
        axis_b = (axis_a + 1) % 3
        axis_c = (axis_a + 2) % 3
        circulation = (jnp.roll(edge_emf[axis_c], -1, axis=axis_b) - edge_emf[axis_c]) - (
            jnp.roll(edge_emf[axis_b], -1, axis=axis_c) - edge_emf[axis_b]
        )
        new_face_field.append(base_face_field[axis_a] - time_step_over_spacing * circulation)
    return new_face_field


def _cell_centred_field(face_field):
    """
    The cell-centred field as the average of the two faces of each cell
    (SG09 eq. 13), from face components stored at their lower index.

    Args:
        face_field: The three face-field components.

    Returns:
        The three cell-centred components.
    """
    return [
        0.5 * (component + jnp.roll(component, -1, axis=axis))
        for axis, component in enumerate(face_field)
    ]


# -------------------------------------------------------------
# ============ ↑ Edge electromotive forces and CT ↑ ============
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ VL2 + CT time step ↓ =================
# -------------------------------------------------------------


def _vl2_ct_stage(
    stage_cell_state: STATE_TYPE,
    stage_face_field,
    base_conserved_state: STATE_TYPE,
    base_face_field,
    stage_time_step,
    piecewise_linear: bool,
    gamma,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):
    """
    One stage of VL2 + CT: ``U = U_base - beta dt div F(W_stage)`` for the gas
    variables, ``b = b_base - beta dt curl E(W_stage)`` for the face field,
    the cell-centred field as the face average, then the primitive variables.

    Args:
        stage_cell_state: The cell-centred primitive variables the fluxes are
            computed from.
        stage_face_field: The face field of that state (normal field of the
            Riemann problems).
        base_conserved_state: The conserved cell-centred variables the update
            starts from.
        base_face_field: The face field the update starts from.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        piecewise_linear: PLM (corrector) or donor-cell (predictor) fluxes.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The cell-centred primitive variables and the face field after the
        stage.
    """
    fluxes_per_axis = _ct_fluxes_per_axis(
        stage_cell_state,
        stage_face_field,
        piecewise_linear,
        config.riemann_solver,
        gamma,
        registered_variables,
    )

    if config.first_order_flux_correction:
        fluxes_per_axis = _ct_first_order_flux_correction(
            fluxes_per_axis,
            stage_cell_state,
            stage_face_field,
            base_conserved_state,
            stage_time_step,
            gamma,
            config,
            registered_variables,
        )

    edge_emf = [
        _edge_electromotive_force(fluxes_per_axis, stage_cell_state, component, registered_variables)
        for component in range(3)
    ]
    face_field = _constrained_transport_update(
        base_face_field,
        edge_emf,
        stage_time_step / config.grid_spacing,
    )

    # The flux divergence updates density, momentum and energy; the field rows
    # are replaced by the average of the updated faces.
    conserved_state = base_conserved_state - stage_time_step * _flux_divergence(
        fluxes_per_axis,
        config,
    )
    for index, component in zip(
        tuple(registered_variables.magnetic_index),
        _cell_centred_field(face_field),
    ):
        conserved_state = conserved_state.at[index].set(component)

    cell_state = _primitive_from_conserved_vl2(
        conserved_state,
        gamma,
        config,
        params,
        registered_variables,
    )
    return cell_state, face_field


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _evolve_state_vl2_ct(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Advance the primitive state by one step of VL2 with constrained transport
    (see the module docstring): a donor-cell predictor to the half step, then
    the corrector with the van Leer limited PLM, or with donor cell when
    ``config.first_order_fallback`` is set; both stages with their own
    constrained-transport update of the face field.

    Args:
        primitive_state: The primitive state ``W^n`` in the layout ``(rho,
            v_x, v_y, v_z, p, B_x, B_y, B_z, b_x, b_y, b_z)``, the face rows
            holding the value at ``i + 1/2``.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data (unused, kept for the common signature).
        registered_variables: The registered variables.

    Returns:
        The primitive state ``W^{n+1}``.
    """
    face_indices = tuple(registered_variables.interface_magnetic_field_index)
    if min(face_indices) < NUM_CELL_CENTRED_VARIABLES:
        raise ValueError("The VL2 + CT layout keeps the face field behind the cell-centred rows.")

    # One code path for every dimensionality: singleton trailing axes.
    state_shape = primitive_state.shape
    primitive_state = primitive_state.reshape(
        state_shape + (1,) * (3 - config.dimensionality)
    )

    cell_state = primitive_state[:NUM_CELL_CENTRED_VARIABLES]
    # State convention (value at i + 1/2) to the module's (value at i - 1/2).
    face_field = [
        jnp.roll(primitive_state[index], 1, axis=axis) for axis, index in enumerate(face_indices)
    ]

    # The cell-centred field is a function of the face field; rebuilding it
    # here keeps the two consistent whatever happened to the state between
    # steps.
    for index, component in zip(
        tuple(registered_variables.magnetic_index),
        _cell_centred_field(face_field),
    ):
        cell_state = cell_state.at[index].set(component)

    initial_conserved_state = _conserved_from_primitive_vl2(
        cell_state,
        gamma,
        config,
        registered_variables,
    )

    half_step_cell_state, half_step_face_field = _vl2_ct_stage(
        cell_state,
        face_field,
        initial_conserved_state,
        face_field,
        0.5 * dt,
        False,
        gamma,
        config,
        params,
        registered_variables,
    )
    new_cell_state, new_face_field = _vl2_ct_stage(
        half_step_cell_state,
        half_step_face_field,
        initial_conserved_state,
        face_field,
        dt,
        not config.first_order_fallback,
        gamma,
        config,
        params,
        registered_variables,
    )

    new_primitive_state = primitive_state.at[:NUM_CELL_CENTRED_VARIABLES].set(new_cell_state)
    for axis, (index, component) in enumerate(zip(face_indices, new_face_field)):
        new_primitive_state = new_primitive_state.at[index].set(jnp.roll(component, -1, axis=axis))
    return new_primitive_state.reshape(state_shape)


# -------------------------------------------------------------
# ===================== ↑ VL2 + CT time step ↑ =================
# -------------------------------------------------------------
