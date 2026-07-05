"""
Constrained Transport (CT) implementation for MHD.
Based on HOW-MHD paper (Seo & Ryu 2023,
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
"""

# general
from functools import partial

# typing
from jaxtyping import Array, Float

# jax
import jax

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._spatial_operators._differencing import finite_difference_int6
from astronomix._spatial_operators._interpolate import interp_center_to_face, point_values_to_averages_single_axis
from astronomix._spatial_operators._interpolate import interp_face_to_center
from astronomix._spatial_operators._interpolate import point_values_to_averages

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
    """Public entry point — accepts the full 8-var WENO flux arrays and
    delegates to ``_constrained_transport_rhs_from_slices``.  Kept
    backwards-compatible with the original signature; new memory-aware
    callers should pre-extract the six magnetic-flux slices and call the
    private helper directly so the full ``dF_x/y/z`` arrays can be freed
    before CT runs.
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
        By_flux_x, Bz_flux_x,
        Bx_flux_y, Bz_flux_y,
        Bx_flux_z, By_flux_z,
        dtdx, dtdy, dtdz,
        config, registered_variables,
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
    """Compute the CT magnetic-field RHS from the six single-channel
    magnetic-flux slices CT actually needs (instead of the three 8-var
    ``dF_x/y/z`` arrays).  This is the body of the original
    ``constrained_transport_rhs``; the only change is the input signature
    — letting callers free the full ``dF_x/y/z`` buffers as soon as the
    fluid-flux divergence step is done and keep only ~6× state/8 worth of
    magnetic flux around for the EMF computation.

    The Pallas implementation in ``_constrained_transport_pallas`` runs
    a three-stage split pipeline (modified flux → edge EMF → smoothed
    curl) so each Pallas kernel has bounded halo and compiles fast.
    """
    if _ct_rhs_pallas_supported(conserved_state, config):
        return _ct_rhs_pallas(
            conserved_state,
            By_flux_x_interface, Bz_flux_x_interface,
            Bx_flux_y_interface, Bz_flux_y_interface,
            Bx_flux_z_interface, By_flux_z_interface,
            dtdx, dtdy, dtdz,
            config, registered_variables,
        )

    # cell-centered variables
    rho = conserved_state[registered_variables.density_index]
    vx = conserved_state[registered_variables.momentum_index.x] / rho
    vy = conserved_state[registered_variables.momentum_index.y] / rho
    vz = conserved_state[registered_variables.momentum_index.z] / rho
    Bx = conserved_state[registered_variables.magnetic_index.x]
    By = conserved_state[registered_variables.magnetic_index.y]
    Bz = conserved_state[registered_variables.magnetic_index.z]

    # Step 1: Compute modified magnetic field fluxes (Eq. 12 - 17)
    # Products are computed at cell centers, then interpolated together
    # to the interface (NOT interpolating factors separately and multiplying).

    # At x-interfaces
    Bx_vy = Bx * vy
    Bx_vz = Bx * vz
    By_flux_x_interface_mod = By_flux_x_interface + interp_center_to_face(Bx_vy, XAXIS)
    Bz_flux_x_interface_mod = Bz_flux_x_interface + interp_center_to_face(Bx_vz, XAXIS)

    # At y-interfaces
    if config.dimensionality == 1:
        By_vx = By * vx
        By_vz = By * vz
        Bx_flux_y_interface_mod = Bx_flux_y_interface + By_vx
        Bz_flux_y_interface_mod = Bz_flux_y_interface + By_vz
    else:
        By_vx = By * vx
        By_vz = By * vz
        Bx_flux_y_interface_mod = Bx_flux_y_interface + interp_center_to_face(By_vx, YAXIS)
        Bz_flux_y_interface_mod = Bz_flux_y_interface + interp_center_to_face(By_vz, YAXIS)

    # At z-interfaces
    if config.dimensionality <= 2:
        Bz_vx = Bz * vx
        Bz_vy = Bz * vy
        Bx_flux_z_interface_mod = Bx_flux_z_interface + Bz_vx
        By_flux_z_interface_mod = By_flux_z_interface + Bz_vy
    else:
        Bz_vx = Bz * vx
        Bz_vy = Bz * vy
        Bx_flux_z_interface_mod = Bx_flux_z_interface + interp_center_to_face(Bz_vx, ZAXIS)
        By_flux_z_interface_mod = By_flux_z_interface + interp_center_to_face(Bz_vy, ZAXIS)

    # Step 2: Compute electric field components at cell edges (Equations 19-21)

    # interpolate from the y interfaces to the (x,y) edges
    g_star_x_edge = interp_center_to_face(Bx_flux_y_interface_mod, XAXIS)

    # interpolate from the x interfaces to the (x,y) edges
    if config.dimensionality == 1:
        f_star_y_edge = By_flux_x_interface_mod
    else:
        f_star_y_edge = interp_center_to_face(By_flux_x_interface_mod, YAXIS)

    # electric field component at (x,y) edges
    Omega_z_edge = g_star_x_edge - f_star_y_edge

    # interpolate from the z interfaces to the (y,z) edges
    if config.dimensionality == 1:
        h_star_y_edge = By_flux_z_interface_mod
    else:
        h_star_y_edge = interp_center_to_face(By_flux_z_interface_mod, YAXIS)

    # interpolate from the y interfaces to the (y,z) edges
    if config.dimensionality <= 2:
        g_star_z_edge = Bz_flux_y_interface_mod
    else:
        g_star_z_edge = interp_center_to_face(Bz_flux_y_interface_mod, ZAXIS)

    # electric field component at (y,z) edges
    Omega_x_edge = h_star_y_edge - g_star_z_edge

    # interpolate from the x interfaces to the (z,x) edges
    if config.dimensionality <= 2:
        f_star_z_edge = Bz_flux_x_interface_mod
    else:
        f_star_z_edge = interp_center_to_face(Bz_flux_x_interface_mod, ZAXIS)

    # interpolate from the z interfaces to the (z,x) edges
    h_star_x_edge = interp_center_to_face(Bx_flux_z_interface_mod, XAXIS)

    # electric field component at (z,x) edges
    Omega_y_edge = f_star_z_edge - h_star_x_edge

    # Step 3: point values to averages
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

    # Update interface magnetic fields via discrete curl
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


# Finite difference derivatives
# Interpolation from interfaces back to centers
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
    Update cell-centered B field from interface values
    using 6th order interpolation. Update the total energy
    accordingly to conserve total energy.

    The Pallas implementation in ``_constrained_transport_pallas`` is
    used transparently when the Pallas backend is active and the
    predicate applies (3D ideal-gas MHD).
    """
    if _ct_update_cell_center_fields_pallas_supported(conserved_state, config):
        return _ct_update_cell_center_fields_pallas(
            conserved_state,
            bx_interface, by_interface, bz_interface,
            config, registered_variables,
        )

    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z

    if config.equation_of_state == IDEAL_GAS:
        b2_old = (
            conserved_state[BX] ** 2 + conserved_state[BY] ** 2 + conserved_state[BZ] ** 2
        )

    # interpolate from interfaces back to cell centers
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

        # update total energy: E_new = E_old + 0.5 * (b2_new - b2_old)
        conserved_new = conserved_new.at[registered_variables.pressure_index].add(
            0.5 * (b2_new - b2_old)
        )
    # no pressure update for the isothermal equation of state

    return conserved_new


@partial(jax.jit, static_argnames=["dimensionality"])
def initialize_interface_fields(
    magnetic_field_x,
    magnetic_field_y,
    magnetic_field_z,
    dimensionality: int = 3,
):
    """Initialize magnetic field at interfaces from cell centers."""

    # Use fourth-order interpolation
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

# -----------------------------------------------------------------------------
# Pallas backend symbols (full implementation in ``_constrained_transport_pallas.py``).
# Bottom-of-file import to avoid the circular-import trap; callers should hit
# these predicates first and fall through to the native versions above.
# -----------------------------------------------------------------------------
from astronomix._finite_difference._magnetic_update._constrained_transport_pallas import (  # noqa: E402
    _ct_rhs_pallas,
    _ct_rhs_pallas_supported,
    _ct_update_cell_center_fields_pallas,
    _ct_update_cell_center_fields_pallas_supported,
)


# -----------------------------------------------------------------------------
# Note on the Pallas CT port
# -----------------------------------------------------------------------------
# An initial single-kernel Pallas port of these helpers
# (``_constrained_transport_pallas.py``) was written but trips a Triton
# compile-time issue: the fused EMF kernel chains four stencil stages —
# modified-flux interpolation (halo 2) → edge EMF interpolation (halo 2)
# → point-values-to-averages smoothing (halo 1) → 6th-order curl (halo 3).
# At small block shapes the worst-case combined halo is ~8 cells per
# axis, and Triton's lowering of the deeply nested closure-call graph
# produces enough IR that compilation exceeds reasonable budgets.  Both
# Pallas helpers are therefore gated off by default (see the early
# ``return False`` in their ``_*_pallas_supported`` predicates).
#
# The pallasify skill recipe handles this by splitting a deeply-chained
# stencil into multiple smaller kernels, each with bounded halo:
#   1. ``modified_flux_pallas``      — 6 outputs, halo 2/axis
#   2. ``edge_emf_pallas``           — 3 outputs, halo 2/axis
#   3. ``edge_average_pallas``       — 3 outputs, halo 1/axis (or fused with 4)
#   4. ``curl_pallas`` (rhs_b{x,y,z}) — halo 3/axis
# Each stage materialises one tile-of-output worth of data, so the
# stages chain via JAX arrays at full state shape (still smaller than
# the current native intermediates because there are fewer of them).
# This split is what the skill currently documents in §4.4 of the guide.
