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

import jax
from functools import partial
from typing import Tuple, Union
from jaxtyping import Array, Float

from astronomix._finite_difference._maths._differencing import finite_difference_int6
from astronomix._finite_difference._maths._interpolate import interp_center_to_face
from astronomix._finite_difference._maths._interpolate import interp_face_to_center
from astronomix._finite_difference._maths._interpolate import point_values_to_averages
from astronomix.variable_registry.registered_variables import RegisteredVariables

XAXIS = 0
YAXIS = 1
ZAXIS = 2

@partial(jax.jit, static_argnames=["registered_variables"])
def constrained_transport_rhs(
    conserved_state,
    weno_flux_x,
    weno_flux_y,
    weno_flux_z,
    dtdx, 
    dtdy,
    dtdz,
    registered_variables: RegisteredVariables,
):
    """
    Here we compute the RHS updates for the interface magnetic fields.
    """

    # Step 0: retrieve variables

    # at index i, the fluxes are at interfaces i+1/2
    By_flux_x_interface = weno_flux_x[registered_variables.magnetic_index.y]
    Bz_flux_x_interface = weno_flux_x[registered_variables.magnetic_index.z]
    Bx_flux_y_interface = weno_flux_y[registered_variables.magnetic_index.x]
    Bz_flux_y_interface = weno_flux_y[registered_variables.magnetic_index.z]
    Bx_flux_z_interface = weno_flux_z[registered_variables.magnetic_index.x]
    By_flux_z_interface = weno_flux_z[registered_variables.magnetic_index.y]

    # cell-centered variables
    rho = conserved_state[registered_variables.density_index]
    vx = conserved_state[registered_variables.momentum_index.x] / rho
    vy = conserved_state[registered_variables.momentum_index.y] / rho
    vz = conserved_state[registered_variables.momentum_index.z] / rho
    Bx = conserved_state[registered_variables.magnetic_index.x]
    By = conserved_state[registered_variables.magnetic_index.y]
    Bz = conserved_state[registered_variables.magnetic_index.z]

    # Step 1: Compute modified magnetic field fluxes (Eq. 12 - 17)

    # At x-interfaces, after the interpolation at index i,
    # we have values at i+1/2
    Bx_x_interface = interp_center_to_face(Bx, XAXIS)
    vy_x_interface = interp_center_to_face(vy, XAXIS)
    vz_x_interface = interp_center_to_face(vz, XAXIS)

    By_flux_x_interface_mod = By_flux_x_interface + Bx_x_interface * vy_x_interface
    Bz_flux_x_interface_mod = Bz_flux_x_interface + Bx_x_interface * vz_x_interface

    # At y-interfaces
    By_y_interface = interp_center_to_face(By, YAXIS)
    vx_y_interface = interp_center_to_face(vx, YAXIS)
    vz_y_interface = interp_center_to_face(vz, YAXIS)

    Bx_flux_y_interface_mod = Bx_flux_y_interface + By_y_interface * vx_y_interface
    Bz_flux_y_interface_mod = Bz_flux_y_interface + By_y_interface * vz_y_interface

    # At z-interfaces
    Bz_z_interface = interp_center_to_face(Bz, ZAXIS)
    vx_z_interface = interp_center_to_face(vx, ZAXIS)
    vy_z_interface = interp_center_to_face(vy, ZAXIS)

    Bx_flux_z_interface_mod = Bx_flux_z_interface + Bz_z_interface * vx_z_interface
    By_flux_z_interface_mod = By_flux_z_interface + Bz_z_interface * vy_z_interface

    # Step 2: Compute electric field components at cell edges (Equations 19-21)

    # interpolate from the y interfaces to the (x,y) edges
    g_star_x_edge = interp_center_to_face(Bx_flux_y_interface_mod, XAXIS)

    # interpolate from the x interfaces to the (x,y) edges
    f_star_y_edge = interp_center_to_face(By_flux_x_interface_mod, YAXIS)

    # electric field component at (x,y) edges
    Omega_z_edge = g_star_x_edge - f_star_y_edge

    # interpolate from the z interfaces to the (y,z) edges
    h_star_y_edge = interp_center_to_face(By_flux_z_interface_mod, YAXIS)

    # interpolate from the y interfaces to the (y,z) edges
    g_star_z_edge = interp_center_to_face(Bz_flux_y_interface_mod, ZAXIS)

    # electric field component at (y,z) edges
    Omega_x_edge = h_star_y_edge - g_star_z_edge

    # interpolate from the x interfaces to the (z,x) edges
    f_star_z_edge = interp_center_to_face(Bz_flux_x_interface_mod, ZAXIS)

    # interpolate from the z interfaces to the (z,x) edges
    h_star_x_edge = interp_center_to_face(Bx_flux_z_interface_mod, XAXIS)

    # electric field component at (z,x) edges
    Omega_y_edge = f_star_z_edge - h_star_x_edge

    # Step 3: Maintain high-order accuracy in spite of
    # dimensional splitting by converting point values to averages
    # In the paper (Seo & Ryu 2023) they say "the advective
    # fluxes are modified to approximate “point values” at grid
    # cell edges", but the formula used (also the finite differencing)
    # is that from point values to averages, compare 
    # Buchmüller & Helzel 2014.
    Omega_z_bar = point_values_to_averages(Omega_z_edge, XAXIS, YAXIS)
    Omega_x_bar = point_values_to_averages(Omega_x_edge, YAXIS, ZAXIS)
    Omega_y_bar = point_values_to_averages(Omega_y_edge, XAXIS, ZAXIS)

    # Update the interface magnetic fields based on the discrete curl of
    # the electric field at edges (Eq. 24 - 26)
    rhs_bx = - dtdy * finite_difference_int6(Omega_z_bar, YAXIS) \
             + dtdz * finite_difference_int6(Omega_y_bar, ZAXIS)

    rhs_by = - dtdz * finite_difference_int6(Omega_x_bar, ZAXIS) \
             + dtdx * finite_difference_int6(Omega_z_bar, XAXIS)

    rhs_bz = - dtdx * finite_difference_int6(Omega_y_bar, XAXIS) \
             + dtdy * finite_difference_int6(Omega_x_bar, YAXIS)

    return rhs_bx, rhs_by, rhs_bz


# Finite difference derivatives
# Interpolation from interfaces back to centers
@partial(jax.jit, static_argnames=["registered_variables"])
def update_cell_center_fields(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    registered_variables: RegisteredVariables,
):
    """
    Update cell-centered B field from interface values
    using 6th order interpolation. Update the total energy
    accordingly to conserve total energy.
    """

    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z

    b2_old = (
        conserved_state[BX] ** 2 + conserved_state[BY] ** 2 + conserved_state[BZ] ** 2
    )

    # interpolate from interfaces back to cell centers
    Bx_center = interp_face_to_center(bx_interface, XAXIS)
    By_center = interp_face_to_center(by_interface, YAXIS)
    Bz_center = interp_face_to_center(bz_interface, ZAXIS)

    conserved_new = conserved_state.at[BX].set(Bx_center)
    conserved_new = conserved_new.at[BY].set(By_center)
    conserved_new = conserved_new.at[BZ].set(Bz_center)

    b2_new = conserved_new[BX] ** 2 + conserved_new[BY] ** 2 + conserved_new[BZ] ** 2

    # update total energy: E_new = E_old + 0.5 * (b2_new - b2_old)
    conserved_new = conserved_new.at[registered_variables.pressure_index].add(
        0.5 * (b2_new - b2_old)
    )

    return conserved_new


@jax.jit
def initialize_interface_fields(
    magnetic_field_x,
    magnetic_field_y,
    magnetic_field_z,
):
    """Initialize magnetic field at interfaces from cell centers."""

    # Use fourth-order interpolation
    bx_interface = interp_center_to_face(magnetic_field_x, XAXIS)
    by_interface = interp_center_to_face(magnetic_field_y, YAXIS)
    bz_interface = interp_center_to_face(magnetic_field_z, ZAXIS)

    return bx_interface, by_interface, bz_interface