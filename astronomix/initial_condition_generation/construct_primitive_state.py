"""
Assemble the primitive state array from individual primitive fields.

Given the per-variable fields (density, velocities, magnetic field components,
pressures, passive scalars, ...) this stacks them into the single state array
used throughout the solver, placing each field at the index dictated by
``registered_variables`` for the active configuration (dimensionality, MHD,
solver mode, equation of state, dual energy and passive scalars).
"""

# general
from functools import partial

# typing
from typing import Union
from types import NoneType

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CONSTRAINED_TRANSPORT,
    FIELD_TYPE,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    IDEAL_GAS,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    SimulationConfig,
    StaticIntVector,
)
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)
from astronomix.option_classes.simulation_config import get_solver_config


def _uses_vl2_constrained_transport(config: SimulationConfig) -> bool:
    """Whether the state has the layout of the VL2 scheme with constrained transport.

    The initial state is built before ``finalize_config``, so the solver
    settings are read through ``get_solver_config``.
    """
    return (
        config.solver_mode == FINITE_VOLUME
        and get_solver_config(config).divergence_control == CONSTRAINED_TRANSPORT
    )


@partial(jax.jit, static_argnames=["registered_variables", "config", "sharding"])
def _assemble_primitive_state(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    density: FIELD_TYPE,
    velocity_x: Union[FIELD_TYPE, NoneType] = None,
    velocity_y: Union[FIELD_TYPE, NoneType] = None,
    velocity_z: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_x: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_y: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_z: Union[FIELD_TYPE, NoneType] = None,
    interface_magnetic_field_x: Union[FIELD_TYPE, NoneType] = None,
    interface_magnetic_field_y: Union[FIELD_TYPE, NoneType] = None,
    interface_magnetic_field_z: Union[FIELD_TYPE, NoneType] = None,
    gas_pressure: Union[FIELD_TYPE, NoneType] = None,
    passive_scalars: Union[STATE_TYPE, NoneType] = None,
    sharding=None,
) -> STATE_TYPE:
    """Stack the primitive variables into the state array.

    In 1D set only the x-components, in 2D set the x- and y-components, and in
    3D set the x-, y- and z-components.

    Args:
        config: The simulation configuration.
        registered_variables: The indices of the variables in the state array.
        density: The density of the fluid.
        velocity_x: The x-component of the velocity of the fluid.
        velocity_y: The y-component of the velocity of the fluid.
        velocity_z: The z-component of the velocity of the fluid.
        magnetic_field_x: The x-component of the magnetic field in B / sqrt(mu_0).
        magnetic_field_y: The y-component of the magnetic field in B / sqrt(mu_0).
        magnetic_field_z: The z-component of the magnetic field in B / sqrt(mu_0).
        interface_magnetic_field_x: The x-component of the face-centered
            (interface) magnetic field, used by the finite-difference solver.
        interface_magnetic_field_y: The y-component of the face-centered
            (interface) magnetic field, used by the finite-difference solver.
        interface_magnetic_field_z: The z-component of the face-centered
            (interface) magnetic field, used by the finite-difference solver.
        gas_pressure: The thermal pressure of the fluid.
        passive_scalars: The user-defined passive scalars, stacked along a
            leading axis with shape ``(config.passive_scalar_config.num_scalars,) + grid``.
        sharding: An optional sharding to apply to the allocated state array.

    Returns:
        The state array.
    """
    # Allocate the (optionally sharded) empty state array; the per-variable
    # fields are written into their registered slots below.
    if sharding is not None:
        state = jax.lax.with_sharding_constraint(
            jnp.zeros((registered_variables.num_vars, *density.shape)), sharding
        )
    else:
        state = jnp.zeros((registered_variables.num_vars, *density.shape))

    state = state.at[registered_variables.density_index].set(density)

    # The velocity index is a scalar in 1D and a per-axis vector otherwise —
    # except for the layouts that always carry all three components (the FD
    # and the VL2 MHD schemes), where it is a vector in 1D too.
    if config.dimensionality == 1:
        if isinstance(registered_variables.velocity_index, StaticIntVector):
            state = state.at[registered_variables.velocity_index.x].set(velocity_x)
        else:
            state = state.at[registered_variables.velocity_index].set(velocity_x)
    elif config.dimensionality == 2:
        state = state.at[registered_variables.velocity_index.x].set(velocity_x)
        state = state.at[registered_variables.velocity_index.y].set(velocity_y)
    elif config.dimensionality == 3:
        state = state.at[registered_variables.velocity_index.x].set(velocity_x)
        state = state.at[registered_variables.velocity_index.y].set(velocity_y)
        state = state.at[registered_variables.velocity_index.z].set(velocity_z)

    # The VL2 GLM-MHD layout carries the divergence-cleaning scalar psi, which
    # simply starts at zero, and stores all three velocity and field components
    # in any dimensionality.
    # The VL2 layout with constrained transport stores all three components
    # too, and the face-centred field next to the cell-centred one.
    is_vl2_ct = _uses_vl2_constrained_transport(config)
    # Every finite-volume MHD layout stores all three velocity and field components.
    is_finite_volume_mhd = config.solver_mode == FINITE_VOLUME and config.mhd
    if config.mhd:
        if config.dimensionality >= 2 or is_finite_volume_mhd:
            if magnetic_field_x is not None:
                state = state.at[registered_variables.magnetic_index.x].set(
                    magnetic_field_x
                )
            if magnetic_field_y is not None:
                state = state.at[registered_variables.magnetic_index.y].set(
                    magnetic_field_y
                )
            if magnetic_field_z is not None:
                state = state.at[registered_variables.magnetic_index.z].set(
                    magnetic_field_z
                )

        if config.solver_mode == FINITE_DIFFERENCE or is_finite_volume_mhd:
            # The finite-difference and VL2 MHD states always carry all three
            # velocity components; any not supplied stay zero by default.
            if velocity_y is not None:
                state = state.at[registered_variables.velocity_index.y].set(velocity_y)
            if velocity_z is not None:
                state = state.at[registered_variables.velocity_index.z].set(velocity_z)

        # Only the finite-difference layout and the VL2 layout with
        # constrained transport store face-centred fields (index i holds the
        # value at i + 1/2); the cell-centred VL2 GLM layout ignores any that a
        # shared setup passes.
        if config.solver_mode == FINITE_DIFFERENCE or is_vl2_ct:
            if interface_magnetic_field_x is not None:
                state = state.at[
                    registered_variables.interface_magnetic_field_index.x
                ].set(interface_magnetic_field_x)
            if interface_magnetic_field_y is not None:
                state = state.at[
                    registered_variables.interface_magnetic_field_index.y
                ].set(interface_magnetic_field_y)
            if interface_magnetic_field_z is not None:
                state = state.at[
                    registered_variables.interface_magnetic_field_index.z
                ].set(interface_magnetic_field_z)

    # VL2 with constrained transport: the cell-centred field is by definition
    # the average of the two faces of the cell (Stone & Gardiner 2009, eq. 13),
    # B_d(i) = [b_d(i - 1/2) + b_d(i + 1/2)] / 2, so it is derived here from the
    # face field and replaces any cell-centred values given above. Along an
    # axis the grid does not have (1D, 2D) the two faces coincide.
    if config.mhd and is_vl2_ct:
        for component, (cell_index, face_index) in enumerate(
            zip(
                tuple(registered_variables.magnetic_index),
                tuple(registered_variables.interface_magnetic_field_index),
            )
        ):
            face_field = state[face_index]
            if component < config.dimensionality:
                face_field = 0.5 * (jnp.roll(face_field, 1, axis=component) + face_field)
            state = state.at[cell_index].set(face_field)

    # For an ideal gas the pressure is an independent variable; in the
    # isothermal case it instead follows from p = c_s^2 * rho and is not stored.
    if config.equation_of_state == IDEAL_GAS:
        state = state.at[registered_variables.pressure_index].set(gas_pressure)

    if registered_variables.passive_scalars_active:
        num_scalars = registered_variables.num_passive_scalars
        if passive_scalars is None:
            raise ValueError(
                f"config.passive_scalar_config.num_scalars = {num_scalars} but no "
                "`passive_scalars` were supplied to construct_primitive_state"
            )
        passive_scalars = jnp.asarray(passive_scalars)
        if passive_scalars.shape[0] != num_scalars:
            raise ValueError(
                f"expected {num_scalars} passive scalars (config."
                f"passive_scalar_config.num_scalars), got {passive_scalars.shape[0]}"
            )
        scalars_start = registered_variables.passive_scalar_index
        state = state.at[scalars_start:scalars_start + num_scalars].set(passive_scalars)

    return state


def construct_primitive_state(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    density: FIELD_TYPE,
    velocity_x: Union[FIELD_TYPE, NoneType] = None,
    velocity_y: Union[FIELD_TYPE, NoneType] = None,
    velocity_z: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_x: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_y: Union[FIELD_TYPE, NoneType] = None,
    magnetic_field_z: Union[FIELD_TYPE, NoneType] = None,
    interface_magnetic_field_x: Union[FIELD_TYPE, NoneType] = None,
    interface_magnetic_field_y: Union[FIELD_TYPE, NoneType] = None,
    interface_magnetic_field_z: Union[FIELD_TYPE, NoneType] = None,
    gas_pressure: Union[FIELD_TYPE, NoneType] = None,
    passive_scalars: Union[STATE_TYPE, NoneType] = None,
    sharding=None,
) -> STATE_TYPE:
    """Stack the primitive variables into the state array, checking for NaNs.

    Thin wrapper around the jitted :func:`_assemble_primitive_state` that adds a
    one-time sanity check: a NaN in the freshly built primitive state almost
    always means a bad initial condition (a divide-by-zero, an out-of-range log,
    a mismatched field shape, ...) that would otherwise only surface much later
    as an opaque solver blow-up. Catching it here points the user straight at
    their setup. The check reads back a single scalar, so it is cheap; it is
    skipped when the state is a tracer so differentiating or jitting through the
    setup still works.

    See :func:`_assemble_primitive_state` for the argument semantics.

    Returns:
        The state array.

    Raises:
        ValueError: If the assembled primitive state contains NaNs.
    """
    # The finite-difference MHD scheme evolves face-centered (interface)
    # magnetic fields for constrained transport, but a user setting up an
    # initial condition typically only has the cell-centered field to hand. If
    # none of the interface components were supplied, derive them here from the
    # cell-centered field (4th-order center-to-face interpolation, exactly what
    # every FD-MHD setup does by hand) so the resulting state is consistent
    # rather than silently carrying zero interface fields. Any component the
    # user did supply explicitly is left untouched.
    interface_fields_missing = (
        interface_magnetic_field_x is None
        and interface_magnetic_field_y is None
        and interface_magnetic_field_z is None
    )
    cell_centered_field_given = (
        magnetic_field_x is not None
        or magnetic_field_y is not None
        or magnetic_field_z is not None
    )
    if (
        config.mhd
        and config.solver_mode == FINITE_DIFFERENCE
        and interface_fields_missing
        and cell_centered_field_given
    ):
        # The interpolation needs a concrete array per axis; substitute zeros
        # for any cell-centered component the user left out.
        zero_field = jnp.zeros_like(density)
        (
            interface_magnetic_field_x,
            interface_magnetic_field_y,
            interface_magnetic_field_z,
        ) = initialize_interface_fields(
            magnetic_field_x if magnetic_field_x is not None else zero_field,
            magnetic_field_y if magnetic_field_y is not None else zero_field,
            magnetic_field_z if magnetic_field_z is not None else zero_field,
            config.dimensionality,
        )

    # VL2 with constrained transport evolves the face-centred field. If no
    # face component is given, each is set to the average of the two adjacent
    # cell-centred values (periodic), b_x(i + 1/2) = [B_x(i) + B_x(i + 1)] / 2.
    # The face divergence of this field equals the central-difference
    # divergence of the cell-centred field: zero for uniform fields and for
    # fields that vary along one axis only, but not in general. Any other field
    # must be given on the faces (from a vector potential on the cell edges).
    if (
        config.mhd
        and _uses_vl2_constrained_transport(config)
        and interface_fields_missing
        and cell_centered_field_given
    ):
        zero_field = jnp.zeros_like(density)
        face_fields = []
        for component, cell_field in enumerate(
            (magnetic_field_x, magnetic_field_y, magnetic_field_z)
        ):
            cell_field = zero_field if cell_field is None else jnp.asarray(cell_field)
            if component < config.dimensionality:
                cell_field = 0.5 * (cell_field + jnp.roll(cell_field, -1, axis=component))
            face_fields.append(cell_field)
        (
            interface_magnetic_field_x,
            interface_magnetic_field_y,
            interface_magnetic_field_z,
        ) = face_fields

    state = _assemble_primitive_state(
        config,
        registered_variables,
        density,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_z=velocity_z,
        magnetic_field_x=magnetic_field_x,
        magnetic_field_y=magnetic_field_y,
        magnetic_field_z=magnetic_field_z,
        interface_magnetic_field_x=interface_magnetic_field_x,
        interface_magnetic_field_y=interface_magnetic_field_y,
        interface_magnetic_field_z=interface_magnetic_field_z,
        gas_pressure=gas_pressure,
        passive_scalars=passive_scalars,
        sharding=sharding,
    )

    if not isinstance(state, jax.core.Tracer) and bool(jnp.isnan(state).any()):
        raise ValueError(
            "construct_primitive_state produced NaNs in the primitive state. "
            "This almost always points to a problem in the initial-condition "
            "fields you passed (a divide-by-zero, an out-of-range log or sqrt, "
            "a wrongly shaped field, ...). Check the density, velocity, "
            "pressure and magnetic-field arrays before starting the simulation."
        )

    return state
