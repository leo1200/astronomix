"""
The VL2 finite-volume scheme with constrained transport (VL2 + CT).

The second-order MHD integrator of Athena (Stone & Gardiner 2009, New
Astronomy 14, 139, section 4, "SG09" below; the corner electromotive forces are
those of Gardiner & Stone 2005, J. Comput. Phys. 205, 509, "GS05", and 2008,
J. Comput. Phys. 227, 4123, "GS08"). It is the constrained-transport
counterpart of the GLM scheme, selected by
``divergence_control=CONSTRAINED_TRANSPORT``: the step (``vl2_step`` in
``_integrators.py``), the stage (``fv_stage`` in ``_stage.py``), the
reconstruction, the Riemann solvers, the first-order flux correction and the
conversions between conserved and primitive variables are the ones of that
scheme. This file holds what the divergence control adds to them: the interface
fluxes with the face field as normal field, the update of the magnetic field
by the edge electromotive forces, and the two conversions between the layout
of the state and the layout used here.

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
to the edge direction); ``vl2_step`` converts on entry and exit
(``_ct_layout_from_state``, ``_state_from_ct_layout``). In between, the state
array keeps all its rows: the conversions between conserved and primitive
variables pass the face rows through, and the flux arrays carry zeros there.

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

Optional first-order flux correction (``config.solver_config.first_order_flux_correction``).
The face fluxes of a stage, including their field rows, which are the face
electromotive forces, are replaced by donor-cell local Lax-Friedrichs fluxes
on all faces of every cell whose trial update has a non-positive density or
pressure; the edge electromotive forces are then built from the corrected
fluxes, so the field stays divergence-free. The trial update is the
finite-volume one of the GLM scheme (the cell-centred field updated with the
divergence of the face fluxes), which serves as the predictor of the
constrained-transport update: the correction is the stage's
``_first_order_flux_correction``, given first-order fluxes that use the face
field as normal field. Athena++'s
first-order flux correction with CT is built the same way (corrected face
fluxes, then corner electromotive forces); this implementation was written
from that description, not checked line by line against the Athena++ source.

Not supported: non-periodic boundaries (the staggered update is written for
the periodic-roll layout only), gravity, the Pallas backend (array-form JAX
only), non-cubic cells, the extended GLM source (there is no GLM field).
``finalize_config`` rejects the first two and the last.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._interface_frame import (
    BarotropicLaw,
    _interface_flux_from_states,
    _state_components_from_frame,
)
from astronomix._finite_volume._reconstruction import _reconstructed_face_states

# -------------------------------------------------------------
# ===================== ↓ Module constants ↓ ===================
# -------------------------------------------------------------

#: The interface frame of the fluxes, ``(rho, v_n, v_t1, v_t2, p, B_n, B_t1,
#: B_t2)``: its size and the position of the normal field.
NUM_FRAME_VARIABLES = 8
FRAME_NORMAL_FIELD = 5

# -------------------------------------------------------------
# ===================== ↑ Module constants ↑ ===================
# -------------------------------------------------------------


def _num_cell_centred_variables(registered_variables: RegisteredVariables) -> int:
    """
    The number of leading state rows on which the fluxes act: the gas
    variables and the cell-centred field, ``(rho, v_x, v_y, v_z, p, B_x, B_y,
    B_z)``, without the pressure for a barotropic gas. The face field follows.
    """
    return max(registered_variables.magnetic_index) + 1


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
        The state indices of ``(rho, v_n, v_t1, v_t2, p, B_n, B_t1, B_t2)``
        (``None`` for the pressure of a layout without pressure slot).
    """
    velocity = tuple(registered_variables.velocity_index)
    field = tuple(registered_variables.magnetic_index)
    pressure_index = registered_variables.pressure_index
    normal = axis - 1
    transverse_1 = axis % 3
    transverse_2 = (axis + 1) % 3
    return (
        registered_variables.density_index,
        velocity[normal],
        velocity[transverse_1],
        velocity[transverse_2],
        pressure_index if pressure_index >= 0 else None,
        field[normal],
        field[transverse_1],
        field[transverse_2],
    )


def _ct_interface_fluxes(
    cell_state: STATE_TYPE,
    normal_face_field,
    axis: int,
    reconstruction: int,
    limiter: int,
    riemann_solver: int,
    gamma,
    registered_variables: RegisteredVariables,
    barotropic_law: BarotropicLaw = None,
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
        reconstruction: ``DONOR_CELL`` or ``PIECEWISE_LINEAR``.
        limiter: The slope limiter of the piecewise-linear reconstruction.
        riemann_solver: The Riemann solver.
        gamma: The adiabatic index.
        registered_variables: The registered variables.
        barotropic_law: ``None`` for the ideal gas, or the law of a barotropic gas.

    Returns:
        The flux array, laid out like ``cell_state``.
    """
    left_states, right_states = _reconstructed_face_states(cell_state, axis, reconstruction, limiter)
    frame_indices = _ct_frame_indices(axis, registered_variables)
    zero = jnp.zeros_like(normal_face_field)

    def frame_values(states):
        # A layout without pressure slot: the solver call fills the pressure of the barotropic law in.
        values = [states[index] if index is not None else zero for index in frame_indices]
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
        barotropic_law,
    )
    flux_values = list(flux[:NUM_FRAME_VARIABLES])
    flux_values[FRAME_NORMAL_FIELD] = zero
    return jnp.stack(
        _state_components_from_frame(flux_values, frame_indices, cell_state.shape[0]),
        axis=0,
    )


def _ct_fluxes_per_axis(
    primitive_state: STATE_TYPE,
    reconstruction: int,
    limiter: int,
    riemann_solver: int,
    gamma,
    registered_variables: RegisteredVariables,
    barotropic_law: BarotropicLaw = None,
):
    """
    The interface fluxes of all three axes (see ``_ct_interface_fluxes``), laid
    out like the state: the face field has no flux, so its rows are zero and
    the flux divergence leaves the face rows of the state alone.

    Args:
        primitive_state: The primitive state in the layout of this module
            (three spatial axes, the face rows holding the value at ``i - 1/2``).
        reconstruction: ``DONOR_CELL`` or ``PIECEWISE_LINEAR``.
        limiter: The slope limiter of the piecewise-linear reconstruction.
        riemann_solver: The Riemann solver.
        gamma: The adiabatic index.
        registered_variables: The registered variables.
        barotropic_law: ``None`` for the ideal gas, or the law of a barotropic gas.

    Returns:
        The interface fluxes per axis.
    """
    cell_state = primitive_state[: _num_cell_centred_variables(registered_variables)]
    face_indices = tuple(registered_variables.interface_magnetic_field_index)
    fluxes_per_axis = tuple(
        _ct_interface_fluxes(
            cell_state,
            primitive_state[face_indices[axis - 1]],
            axis,
            reconstruction,
            limiter,
            riemann_solver,
            gamma,
            registered_variables,
            barotropic_law,
        )
        for axis in (1, 2, 3)
    )
    # Each flux array has many consumers (the flux divergence and, shifted in
    # several ways, the edge electromotive forces). Without the barrier XLA
    # fuses the Riemann solver into every consumer and evaluates it once per
    # use; the barrier makes it evaluate the fluxes once. The values are
    # unchanged (measured at 32^3 on four CPU cores in double precision: 388 ms
    # per step without the barrier, 43 ms with it, identical results).
    fluxes_per_axis = jax.lax.optimization_barrier(fluxes_per_axis)

    # The zero rows of the face field are added behind the barrier, where they
    # stay part of the expressions that use the fluxes; in front of it every
    # flux array would hold them in memory (measured at 128^3 with the flux
    # correction on one GPU in double precision: a peak of 3.5 GiB and 26.2 ms
    # per step in front of the barrier, 2.9 GiB and 22.4 ms behind it).
    num_face_rows = registered_variables.num_vars - cell_state.shape[0]
    return [jnp.pad(fluxes, ((0, num_face_rows), (0, 0), (0, 0), (0, 0))) for fluxes in fluxes_per_axis]


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


def _ct_field_update(
    conserved_state: STATE_TYPE,
    fluxes_per_axis,
    stage_primitive_state: STATE_TYPE,
    base_conserved_state: STATE_TYPE,
    stage_time_step,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    The magnetic field of a stage with constrained transport: the face field
    ``b = b_base - beta dt curl E(W_stage)`` from the edge electromotive
    forces of the stage's fluxes, and the cell-centred field as its face
    average.

    Args:
        conserved_state: The conserved state after the flux-divergence update.
        fluxes_per_axis: The stage's interface fluxes per axis.
        stage_primitive_state: The primitive state the fluxes were computed
            from.
        base_conserved_state: The register the stage update starts from; its
            face rows are the face field the update starts from.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The conserved state with the updated face field and its cell-centred
        field.
    """
    face_indices = tuple(registered_variables.interface_magnetic_field_index)
    edge_emf = [
        _edge_electromotive_force(fluxes_per_axis, stage_primitive_state, component, registered_variables)
        for component in range(3)
    ]
    face_field = _constrained_transport_update(
        [base_conserved_state[index] for index in face_indices],
        edge_emf,
        stage_time_step / config.grid_spacing,
    )

    # The new rows go into a list of the components, stacked once: faster than
    # a scatter per row (measured at 128^3 with the flux correction on one GPU
    # in double precision: 25.9 ms per step against 28.6 ms).
    components = list(conserved_state)
    for index, component in zip(face_indices, face_field):
        components[index] = component

    # The flux divergence updates density, momentum and energy; the field rows
    # are replaced by the average of the updated faces.
    for index, component in zip(
        tuple(registered_variables.magnetic_index),
        _cell_centred_field(face_field),
    ):
        components[index] = component
    return jnp.stack(components, axis=0)


# -------------------------------------------------------------
# ============ ↑ Edge electromotive forces and CT ↑ ============
# -------------------------------------------------------------

# -------------------------------------------------------------
# ====================== ↓ State layout ↓ ======================
# -------------------------------------------------------------


def _ct_layout_from_state(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    The primitive state in the layout of this module: three spatial axes
    (singleton trailing axes in one and two dimensions), the face rows holding
    the value at ``i - 1/2``, the cell-centred field rebuilt from the faces.

    Args:
        primitive_state: The primitive state in the layout ``(rho, v_x, v_y,
            v_z, p, B_x, B_y, B_z, b_x, b_y, b_z)``, the face rows holding the
            value at ``i + 1/2``.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The primitive state in the layout of this module.
    """
    face_indices = tuple(registered_variables.interface_magnetic_field_index)
    if min(face_indices) < _num_cell_centred_variables(registered_variables):
        raise ValueError("The VL2 + CT layout keeps the face field behind the cell-centred rows.")

    # One code path for every dimensionality: singleton trailing axes.
    primitive_state = primitive_state.reshape(
        primitive_state.shape + (1,) * (3 - config.dimensionality)
    )

    # State convention (value at i + 1/2) to the module's (value at i - 1/2).
    face_field = [
        jnp.roll(primitive_state[index], 1, axis=axis) for axis, index in enumerate(face_indices)
    ]

    # One stack of the components instead of a scatter per row, as in
    # ``_ct_field_update`` (same measurement: 22.7 ms per step against 25.9 ms).
    components = list(primitive_state)
    for index, component in zip(face_indices, face_field):
        components[index] = component

    # The cell-centred field is a function of the face field; rebuilding it
    # here keeps the two consistent whatever happened to the state between
    # steps.
    for index, component in zip(
        tuple(registered_variables.magnetic_index),
        _cell_centred_field(face_field),
    ):
        components[index] = component
    return jnp.stack(components, axis=0)


def _state_from_ct_layout(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    The primitive state back in the layout of the state: the face rows holding
    the value at ``i + 1/2``, the spatial axes of the configuration.

    Args:
        primitive_state: The primitive state in the layout of this module.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The primitive state in the layout of the state.
    """
    face_indices = tuple(registered_variables.interface_magnetic_field_index)

    # A scatter per row here: with one stack as in ``_ct_layout_from_state``
    # the step time stays within 2 %, but on a GPU the components that vanish
    # up to rounding (v_z, B_z and b_z of a two-dimensional flow) come out
    # with other values (seen in the Orszag-Tang test; all else bitwise equal).
    for axis, index in enumerate(face_indices):
        primitive_state = primitive_state.at[index].set(jnp.roll(primitive_state[index], -1, axis=axis))
    return primitive_state.reshape(primitive_state.shape[: config.dimensionality + 1])


# -------------------------------------------------------------
# ====================== ↑ State layout ↑ ======================
# -------------------------------------------------------------
