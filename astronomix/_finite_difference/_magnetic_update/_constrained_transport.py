"""
Constrained Transport (CT) implementation for MHD.
Based on the HOW-MHD paper (Seo & Ryu 2023,
see https://arxiv.org/abs/2304.04360).

Algorithm summary
-----------------

We carry interface magnetic fields

b_x at x-interfaces,
b_y at y-interfaces,
b_z at z-interfaces

through the simulation, updating them using the CT
algorithm such that (ignoring floating point errors) the
divergence of B remains zero. This is achieved by updating
the interfaces based on the discrete curl of an electric
field defined at cell edges.

NOTE: While the scheme theoretically keeps div B = 0,
floating point errors seem to accumulate over time,
especially in single precision. Projecting this divergence
out seemed to help with the divergence of the magnetic field
but comes at additional cost.

The Pallas backend of these helpers lives in
``_constrained_transport_pallas.py``; it is opt-in
(``backend_config.pallas_ct``) and the predicates below fall
through to the native implementation otherwise.

The edge EMFs above are built by central high-order interpolation and
carry upwind dissipation only through the WENO fluxes of the cell-centred
field, which is itself a central interpolation of the interface fields.
Near the Nyquist wavenumber that leaves the interface fields too little
damping: a uniform field oblique to the grid lets grid-scale perturbations
grow. ``ct_upwind_dissipation_rhs`` adds the missing upwind dissipation
directly on the interface fields (``FDSolverConfig.ct_upwind_dissipation``).
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_difference._magnetic_update._constrained_transport_pallas import (
    _ct_rhs_pallas,
    _ct_rhs_pallas_supported,
    _ct_update_cell_center_fields_pallas,
    _ct_update_cell_center_fields_pallas_supported,
    _ct_upwind_dissipation_pallas,
    _ct_upwind_pallas_supported,
)
from astronomix._spatial_operators._differencing import finite_difference_int6
from astronomix._spatial_operators._interpolate import (
    interp_center_to_face,
    interp_face_to_center,
    point_values_to_averages,
    point_values_to_averages_single_axis,
)
from astronomix._stencil_operations._stencil_operations import _shift

XAXIS = 0
YAXIS = 1
ZAXIS = 2

# The transverse velocity components v_y and v_z are kept and used for the CT
# electromotive force even in 1D and 2D MHD runs, because the out-of-plane field
# components still evolve through the edge EMFs.


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def constrained_transport_rhs(
    conserved_state,
    weno_flux_x,
    weno_flux_y,
    weno_flux_z,
    dtdx,
    dtdy,
    dtdz,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Compute the CT magnetic-field RHS from the full WENO interface fluxes.

    Extracts the six magnetic-flux slices CT needs and delegates to
    ``_constrained_transport_rhs_from_slices``. Memory-aware callers extract
    the slices themselves and call that helper directly, so the full
    ``dF_x/y/z`` arrays can be freed before CT runs.

    Args:
        conserved_state: The conserved MHD state.
        weno_flux_x: The WENO interface flux along x.
        weno_flux_y: The WENO interface flux along y (unused in 1D).
        weno_flux_z: The WENO interface flux along z (unused in 1D and 2D).
        dtdx: The time step over the grid spacing along x.
        dtdy: The time step over the grid spacing along y.
        dtdz: The time step over the grid spacing along z.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The increments of the interface fields (bx, by, bz).
    """
    By_flux_x = weno_flux_x[registered_variables.magnetic_index.y]
    Bz_flux_x = weno_flux_x[registered_variables.magnetic_index.z]
    if config.dimensionality >= 2:
        Bx_flux_y = weno_flux_y[registered_variables.magnetic_index.x]
        Bz_flux_y = weno_flux_y[registered_variables.magnetic_index.z]
    else:
        Bx_flux_y = 0.0
        Bz_flux_y = 0.0
    if config.dimensionality == 3:
        Bx_flux_z = weno_flux_z[registered_variables.magnetic_index.x]
        By_flux_z = weno_flux_z[registered_variables.magnetic_index.y]
    else:
        Bx_flux_z = 0.0
        By_flux_z = 0.0
    return _constrained_transport_rhs_from_slices(
        conserved_state,
        By_flux_x,
        Bz_flux_x,
        Bx_flux_y,
        Bz_flux_y,
        Bx_flux_z,
        By_flux_z,
        dtdx,
        dtdy,
        dtdz,
        config,
        registered_variables,
    )


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _constrained_transport_rhs_from_slices(
    conserved_state,
    By_flux_x_interface,
    Bz_flux_x_interface,
    Bx_flux_y_interface,
    Bz_flux_y_interface,
    Bx_flux_z_interface,
    By_flux_z_interface,
    dtdx,
    dtdy,
    dtdz,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Compute the CT magnetic-field RHS from the six magnetic-flux slices.

    Taking only the six single-channel slices CT needs (instead of the three
    full ``dF_x/y/z`` arrays) lets callers free the full flux buffers as soon
    as the fluid-flux divergence is done, keeping only about 6/8 of a state
    worth of magnetic flux around for the EMF computation.

    The Pallas implementation in ``_constrained_transport_pallas`` runs
    a three-stage split pipeline (modified flux → edge EMF → smoothed
    curl) so each Pallas kernel has bounded halo and compiles fast.

    Args:
        conserved_state: The conserved MHD state.
        By_flux_x_interface: The WENO flux of B_y at the x-interfaces.
        Bz_flux_x_interface: The WENO flux of B_z at the x-interfaces.
        Bx_flux_y_interface: The WENO flux of B_x at the y-interfaces (0 in 1D).
        Bz_flux_y_interface: The WENO flux of B_z at the y-interfaces (0 in 1D).
        Bx_flux_z_interface: The WENO flux of B_x at the z-interfaces (0 below 3D).
        By_flux_z_interface: The WENO flux of B_y at the z-interfaces (0 below 3D).
        dtdx: The time step over the grid spacing along x.
        dtdy: The time step over the grid spacing along y.
        dtdz: The time step over the grid spacing along z.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The increments of the interface fields (bx, by, bz).
    """
    if _ct_rhs_pallas_supported(conserved_state, config):
        return _ct_rhs_pallas(
            conserved_state,
            By_flux_x_interface,
            Bz_flux_x_interface,
            Bx_flux_y_interface,
            Bz_flux_y_interface,
            Bx_flux_z_interface,
            By_flux_z_interface,
            dtdx,
            dtdy,
            dtdz,
            config,
            registered_variables,
        )

    # Cell-centered variables.
    rho = conserved_state[registered_variables.density_index]
    vx = conserved_state[registered_variables.momentum_index.x] / rho
    vy = conserved_state[registered_variables.momentum_index.y] / rho
    vz = conserved_state[registered_variables.momentum_index.z] / rho
    Bx = conserved_state[registered_variables.magnetic_index.x]
    By = conserved_state[registered_variables.magnetic_index.y]
    Bz = conserved_state[registered_variables.magnetic_index.z]

    # Step 1: Compute the modified magnetic field fluxes (Eqs. 12-17).
    # Products are computed at cell centers, then interpolated together
    # to the interface (NOT interpolating factors separately and multiplying).

    # At x-interfaces.
    Bx_vy = Bx * vy
    Bx_vz = Bx * vz
    By_flux_x_interface_mod = By_flux_x_interface + interp_center_to_face(Bx_vy, XAXIS)
    Bz_flux_x_interface_mod = Bz_flux_x_interface + interp_center_to_face(Bx_vz, XAXIS)

    # At y-interfaces.
    if config.dimensionality == 1:
        # In a collapsed y direction no WENO y-flux exists (Bx_flux_y = 0), so
        # the modified flux G* = G + By vx must be supplied whole: G* = Bx vy
        # (and likewise Bz vy for the B_z component).
        Bx_flux_y_interface_mod = Bx_flux_y_interface + Bx * vy
        Bz_flux_y_interface_mod = Bz_flux_y_interface + Bz * vy
    else:
        By_vx = By * vx
        By_vz = By * vz
        Bx_flux_y_interface_mod = Bx_flux_y_interface + interp_center_to_face(By_vx, YAXIS)
        Bz_flux_y_interface_mod = Bz_flux_y_interface + interp_center_to_face(By_vz, YAXIS)

    # At z-interfaces.
    if config.dimensionality <= 2:
        # In a collapsed z direction no WENO z-flux exists (Bx_flux_z = 0), so
        # the modified flux H* = H + Bz vx must be supplied whole: H* = Bx vz
        # (and likewise By vz for the B_y component).
        Bx_flux_z_interface_mod = Bx_flux_z_interface + Bx * vz
        By_flux_z_interface_mod = By_flux_z_interface + By * vz
    else:
        Bz_vx = Bz * vx
        Bz_vy = Bz * vy
        Bx_flux_z_interface_mod = Bx_flux_z_interface + interp_center_to_face(Bz_vx, ZAXIS)
        By_flux_z_interface_mod = By_flux_z_interface + interp_center_to_face(Bz_vy, ZAXIS)

    # Step 2: Compute the electric field components at the cell edges (Eqs. 19-21).

    # Interpolate from the y interfaces to the (x,y) edges.
    g_star_x_edge = interp_center_to_face(Bx_flux_y_interface_mod, XAXIS)

    # Interpolate from the x interfaces to the (x,y) edges.
    if config.dimensionality == 1:
        f_star_y_edge = By_flux_x_interface_mod
    else:
        f_star_y_edge = interp_center_to_face(By_flux_x_interface_mod, YAXIS)

    # Electric field component at the (x,y) edges.
    Omega_z_edge = g_star_x_edge - f_star_y_edge

    # Interpolate from the z interfaces to the (y,z) edges.
    if config.dimensionality == 1:
        h_star_y_edge = By_flux_z_interface_mod
    else:
        h_star_y_edge = interp_center_to_face(By_flux_z_interface_mod, YAXIS)

    # Interpolate from the y interfaces to the (y,z) edges.
    if config.dimensionality <= 2:
        g_star_z_edge = Bz_flux_y_interface_mod
    else:
        g_star_z_edge = interp_center_to_face(Bz_flux_y_interface_mod, ZAXIS)

    # Electric field component at the (y,z) edges.
    Omega_x_edge = h_star_y_edge - g_star_z_edge

    # Interpolate from the x interfaces to the (z,x) edges.
    if config.dimensionality <= 2:
        f_star_z_edge = Bz_flux_x_interface_mod
    else:
        f_star_z_edge = interp_center_to_face(Bz_flux_x_interface_mod, ZAXIS)

    # Interpolate from the z interfaces to the (z,x) edges.
    h_star_x_edge = interp_center_to_face(Bx_flux_z_interface_mod, XAXIS)

    # Electric field component at the (z,x) edges.
    Omega_y_edge = f_star_z_edge - h_star_x_edge

    # Step 3: Convert the edge point values to edge averages.
    if config.dimensionality == 1:
        Omega_z_bar = point_values_to_averages_single_axis(Omega_z_edge, XAXIS)
        Omega_x_bar = Omega_x_edge
        Omega_y_bar = point_values_to_averages_single_axis(Omega_y_edge, XAXIS)
    if config.dimensionality == 2:
        Omega_z_bar = point_values_to_averages(Omega_z_edge, XAXIS, YAXIS)
        Omega_x_bar = point_values_to_averages_single_axis(Omega_x_edge, YAXIS)
        Omega_y_bar = point_values_to_averages_single_axis(Omega_y_edge, XAXIS)
    if config.dimensionality == 3:
        Omega_z_bar = point_values_to_averages(Omega_z_edge, XAXIS, YAXIS)
        Omega_x_bar = point_values_to_averages(Omega_x_edge, YAXIS, ZAXIS)
        Omega_y_bar = point_values_to_averages(Omega_y_edge, XAXIS, ZAXIS)

    # Update the interface magnetic fields via the discrete curl.
    if config.dimensionality == 1:
        rhs_bx = 0.0
        rhs_by = dtdx * finite_difference_int6(Omega_z_bar, XAXIS)
        rhs_bz = - dtdx * finite_difference_int6(Omega_y_bar, XAXIS)
    if config.dimensionality == 2:
        rhs_bx = - dtdy * finite_difference_int6(Omega_z_bar, YAXIS)
        rhs_by = dtdx * finite_difference_int6(Omega_z_bar, XAXIS)
        rhs_bz = - dtdx * finite_difference_int6(Omega_y_bar, XAXIS) \
                 + dtdy * finite_difference_int6(Omega_x_bar, YAXIS)
    if config.dimensionality == 3:
        rhs_bx = - dtdy * finite_difference_int6(Omega_z_bar, YAXIS) \
                + dtdz * finite_difference_int6(Omega_y_bar, ZAXIS)

        rhs_by = - dtdz * finite_difference_int6(Omega_x_bar, ZAXIS) \
                + dtdx * finite_difference_int6(Omega_z_bar, XAXIS)

        rhs_bz = - dtdx * finite_difference_int6(Omega_y_bar, XAXIS) \
                + dtdy * finite_difference_int6(Omega_x_bar, YAXIS)

    return rhs_bx, rhs_by, rhs_bz


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def update_cell_center_fields(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Update the cell-centered B field from the interface values using
    6th-order interpolation, and update the total energy accordingly so
    that it is conserved.

    The Pallas implementation in ``_constrained_transport_pallas`` is
    used transparently when the Pallas backend is active and the
    predicate applies (3D ideal-gas MHD).

    Args:
        conserved_state: The conserved MHD state.
        bx_interface: The B_x field at the x-interfaces.
        by_interface: The B_y field at the y-interfaces.
        bz_interface: The B_z field at the z-interfaces.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The conserved state with updated cell-centered fields (and energy).
    """
    if _ct_update_cell_center_fields_pallas_supported(conserved_state, config):
        return _ct_update_cell_center_fields_pallas(
            conserved_state,
            bx_interface,
            by_interface,
            bz_interface,
            config,
            registered_variables,
        )

    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z

    if config.equation_of_state == IDEAL_GAS:
        b2_old = (
            conserved_state[BX] ** 2 + conserved_state[BY] ** 2 + conserved_state[BZ] ** 2
        )

    # Interpolate from the interfaces back to the cell centers.
    Bx_center = interp_face_to_center(bx_interface, XAXIS)
    if config.dimensionality == 1:
        By_center = by_interface
        Bz_center = bz_interface
    if config.dimensionality == 2:
        By_center = interp_face_to_center(by_interface, YAXIS)
        Bz_center = bz_interface
    if config.dimensionality == 3:
        By_center = interp_face_to_center(by_interface, YAXIS)
        Bz_center = interp_face_to_center(bz_interface, ZAXIS)

    conserved_new = conserved_state.at[BX].set(Bx_center)
    conserved_new = conserved_new.at[BY].set(By_center)
    conserved_new = conserved_new.at[BZ].set(Bz_center)

    if config.equation_of_state == IDEAL_GAS:
        b2_new = conserved_new[BX] ** 2 + conserved_new[BY] ** 2 + conserved_new[BZ] ** 2

        # Update the total energy: E_new = E_old + 0.5 * (b2_new - b2_old).
        conserved_new = conserved_new.at[registered_variables.pressure_index].add(
            0.5 * (b2_new - b2_old)
        )
    # There is no energy to update for the isothermal equation of state.

    return conserved_new


def _upwind_edge_jump(face_field, axis):
    """
    The jump at the half index i + 1/2 along ``axis`` of a field stored at the
    integer positions i: its fifth-order upwind-biased interpolation from the
    right (positions i - 1 ... i + 3) minus the one from the left (positions
    i - 2 ... i + 2). It is O(dx^5) on smooth fields and about half the
    first-order jump at the Nyquist wavenumber.

    Args:
        face_field: The field, stored at the integer positions along ``axis``.
        axis: The axis of the jump.

    Returns:
        The jump, stored at index i for the half index i + 1/2.
    """

    def value_at(offset):
        return _shift(face_field, -offset, axis=axis)

    from_left = (
        2.0 * value_at(-2)
        - 13.0 * value_at(-1)
        + 47.0 * value_at(0)
        + 27.0 * value_at(1)
        - 3.0 * value_at(2)
    ) / 60.0
    from_right = (
        -3.0 * value_at(-1)
        + 27.0 * value_at(0)
        + 47.0 * value_at(1)
        - 13.0 * value_at(2)
        + 2.0 * value_at(3)
    ) / 60.0
    return from_right - from_left


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def ct_upwind_dissipation_rhs(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    dt_tilde,
    grid_spacing,
    gamma,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Upwind dissipation of the edge EMFs on the interface fields, in the form
    of the upwind constrained transport of Londrillo & Del Zanna (2004): every
    edge EMF gets the jumps of the two interface-field components that meet at
    the edge, reconstructed to the edge from both sides, times half the local
    fast signal speed across each jump and the strength kappa
    (``config.solver_config.ct_upwind_strength``; 1 is the full upwind form),

        Omega_z += -kappa (a_y / 2) (B_x^+y - B_x^-y) + kappa (a_x / 2) (B_y^+x - B_y^-x)

    and cyclically for Omega_x and Omega_y, with a_d = |v_d| + c_fast,d the
    largest over the (up to) four cells around the edge. The increments are the
    discrete curl of these EMFs with the CT's own staggered derivative, so
    div B = 0 is kept exactly. With fifth-order reconstructions the jumps are
    O(dx^5) on smooth fields: the scheme keeps its order, while the
    grid-scale modes of the interface fields, which the central CT
    interpolations leave undamped, are damped at the local signal speed.
    In 1D and 2D the fields are invariant along the collapsed axes. In 3D on
    the Pallas backend the same arithmetic runs in two kernels
    (``_constrained_transport_pallas._ct_upwind_dissipation_pallas``).

    Args:
        conserved_state: The conserved state (its cell-centred fields give
            the signal speeds).
        bx_interface: The B_x field at the x-interfaces.
        by_interface: The B_y field at the y-interfaces.
        bz_interface: The B_z field at the z-interfaces.
        dt_tilde: The (stage) time step.
        grid_spacing: The cell size.
        gamma: The adiabatic index (ideal gas).
        params: The simulation parameters (minimum density and pressure,
            isothermal sound speed).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The increments of the interface fields (bx, by, bz).
    """

    if _ct_upwind_pallas_supported(conserved_state, config):
        increments = _ct_upwind_dissipation_pallas(
            conserved_state,
            jnp.stack([bx_interface, by_interface, bz_interface]),
            dt_tilde / grid_spacing,
            gamma,
            params,
            config,
            registered_variables,
        )
        return increments[0], increments[1], increments[2]

    # -------------------------------------------------------------
    # ========== ↓ Fast signal speed along every axis ↓ ===========
    # -------------------------------------------------------------

    density = jnp.maximum(
        conserved_state[registered_variables.density_index],
        params.minimum_density,
    )
    momentum_indices = (
        registered_variables.momentum_index.x,
        registered_variables.momentum_index.y,
        registered_variables.momentum_index.z,
    )
    magnetic_indices = (
        registered_variables.magnetic_index.x,
        registered_variables.magnetic_index.y,
        registered_variables.magnetic_index.z,
    )
    velocity = [conserved_state[index] / density for index in momentum_indices]
    magnetic_field = [conserved_state[index] for index in magnetic_indices]
    magnetic_field_squared = sum(component * component for component in magnetic_field)

    if config.equation_of_state == IDEAL_GAS:
        kinetic_energy = 0.5 * density * sum(component * component for component in velocity)
        gas_pressure = (gamma - 1.0) * (
            conserved_state[registered_variables.energy_index]
            - kinetic_energy
            - 0.5 * magnetic_field_squared
        )
        sound_speed_squared = gamma * jnp.maximum(gas_pressure, params.minimum_pressure) / density
    else:
        sound_speed_squared = params.isothermal_sound_speed ** 2

    alfven_speed_squared = magnetic_field_squared / density
    total_speed_squared = sound_speed_squared + alfven_speed_squared
    signal_speed = []
    for axis in range(3):
        normal_alfven_speed_squared = magnetic_field[axis] * magnetic_field[axis] / density
        discriminant = jnp.maximum(
            total_speed_squared * total_speed_squared
            - 4.0 * sound_speed_squared * normal_alfven_speed_squared,
            0.0,
        )
        fast_speed = jnp.sqrt(0.5 * (total_speed_squared + jnp.sqrt(discriminant)))
        signal_speed.append(jnp.abs(velocity[axis]) + fast_speed)

    # -------------------------------------------------------------
    # ========== ↑ Fast signal speed along every axis ↑ ===========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================ ↓ Upwind edge EMFs and curl ↓ ==============
    # -------------------------------------------------------------

    # Omega_z lives on the (x, y) edges, Omega_x on the (y, z) edges and
    # Omega_y on the (z, x) edges, with the CT's sign convention (the
    # increments below are its curl, so each jump term diffuses its field
    # component across the jump). Below 3D the fields are invariant along the
    # collapsed axes, so jumps and differences along them vanish and the edge
    # maxima run over the active axes only.
    dimensionality = int(config.dimensionality)

    def edge_maximum(cell_field, first_axis, second_axis):
        maximum = cell_field
        for axis in (first_axis, second_axis):
            if axis < dimensionality:
                maximum = jnp.maximum(maximum, _shift(maximum, -1, axis=axis))
        return maximum

    def upwind_term(speed_axis, first_axis, second_axis, face_field, jump_axis):
        if jump_axis >= dimensionality:
            return 0.0
        return (
            edge_maximum(signal_speed[speed_axis], first_axis, second_axis)
            * _upwind_edge_jump(face_field, jump_axis)
        )

    def difference(edge_field, axis):
        # An EMF built only from collapsed jumps is the plain number 0.
        if axis >= dimensionality or isinstance(edge_field, float):
            return 0.0
        return finite_difference_int6(edge_field, axis)

    half_strength = 0.5 * config.solver_config.ct_upwind_strength
    emf_z = half_strength * (
        -upwind_term(1, XAXIS, YAXIS, bx_interface, YAXIS)
        + upwind_term(0, XAXIS, YAXIS, by_interface, XAXIS)
    )
    emf_x = half_strength * (
        -upwind_term(2, YAXIS, ZAXIS, by_interface, ZAXIS)
        + upwind_term(1, YAXIS, ZAXIS, bz_interface, YAXIS)
    )
    emf_y = half_strength * (
        -upwind_term(0, ZAXIS, XAXIS, bz_interface, XAXIS)
        + upwind_term(2, ZAXIS, XAXIS, bx_interface, ZAXIS)
    )

    time_step_over_spacing = dt_tilde / grid_spacing
    rhs_bx = time_step_over_spacing * (-difference(emf_z, YAXIS) + difference(emf_y, ZAXIS))
    rhs_by = time_step_over_spacing * (-difference(emf_x, ZAXIS) + difference(emf_z, XAXIS))
    rhs_bz = time_step_over_spacing * (-difference(emf_y, XAXIS) + difference(emf_x, YAXIS))

    # -------------------------------------------------------------
    # ================ ↑ Upwind edge EMFs and curl ↑ ==============
    # -------------------------------------------------------------

    return rhs_bx, rhs_by, rhs_bz


@partial(jax.jit, static_argnames=["dimensionality"])
def initialize_interface_fields(
    magnetic_field_x,
    magnetic_field_y,
    magnetic_field_z,
    dimensionality: int = 3,
):
    """
    Initialize the magnetic field at the interfaces from the cell centers.

    Args:
        magnetic_field_x: The cell-centered B_x.
        magnetic_field_y: The cell-centered B_y.
        magnetic_field_z: The cell-centered B_z.
        dimensionality: The number of spatial dimensions; components along
            collapsed directions are taken over unchanged.

    Returns:
        The interface fields (bx, by, bz).
    """
    # Use fourth-order interpolation.
    if dimensionality == 1:
        bx_interface = interp_center_to_face(magnetic_field_x, XAXIS)
        by_interface = magnetic_field_y
        bz_interface = magnetic_field_z
    if dimensionality == 2:
        bx_interface = interp_center_to_face(magnetic_field_x, XAXIS)
        by_interface = interp_center_to_face(magnetic_field_y, YAXIS)
        bz_interface = magnetic_field_z
    if dimensionality == 3:
        bx_interface = interp_center_to_face(magnetic_field_x, XAXIS)
        by_interface = interp_center_to_face(magnetic_field_y, YAXIS)
        bz_interface = interp_center_to_face(magnetic_field_z, ZAXIS)

    return bx_interface, by_interface, bz_interface
