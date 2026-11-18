"""
Poynting-consistent energy flux of the finite-difference scheme with a Riemann
flux (``FDSolverConfig.scheme = RIEMANN_FLUX``, ideal gas).

The fluid's interface flux carries, in its energy row, the magnetic energy flux
of the fluid's own field rows; the constrained transport moves the cell-centred
field with different fluxes, and the rebuild of the cell-centred field
(``_rebuild_energy.py``, energy held) would hand the difference of the two
magnetic-energy changes to the thermal energy. That difference is mostly a
transport mismatch of two discretisations of the same ideal flux, which at low
plasma beta exceeds the thermal energy.

Here the magnetic part of the energy flux is made the CT's own:

1. The CT's increment of the cell-centred field B^c = I_fc(b) is exactly a
   flux difference of face fluxes built from the edge EMFs,

       dB^c_c = -(dt/h) sum_a [F^CT_{c,a}(i+1/2) - F^CT_{c,a}(i-1/2)],
       F^CT_{c,a} = eps_{c a e} I_fc^c( W^a[E_e] ),

   with W^a the "wide" edge combination c1 E + c2 (3-sum) + c3 (5-sum) whose
   difference along a is the staggered sixth-order derivative D6
   (``finite_difference_int6``), I_fc^c the face-to-centre interpolation along
   the component's own axis (absent along a collapsed axis), and F^CT_{a,a} = 0.

2. The fluid's energy flux along a is corrected by

       dF_E,a = sum_c B_f,a[c] (F^CT_{c,a} - F^fluid_{c,a}),
       B_f,a = (B_i + B_{i+1}) / 2   (two-point face value along a),

   so the energy equation transports B_f . F^CT, the magnetic energy of the
   field the CT moves. With the exact identity
   -(dt/h) delta(B_f . F) = B . dB + (dt/h) (E.J)_disc, where
   (E.J)_disc = -((B_f^+ - B_i).F^+ - (B_f^- - B_i).F^-), the thermal energy
   after the energy-held rebuild changes by the fluid's own thermal update with
   the fluid's magnetic E.J replaced by the CT's (its Lorentz work and its
   Joule heating, where its dissipation acts) plus O(dt^2) terms, and no
   transport mismatch.

The correction is a conservative flux (total energy conserved to rounding) and
O(h^5) on smooth data. Reach from a face: W two edges to either side along a,
I_fc from -3 to +2 along c, and the EMF's own reach.

It is applied in two halves that need different inputs:

* the fluid half, -B_f . F^fluid, is subtracted from the fluid's energy flux
  before the flux divergence of each axis (``magnetic_energy_corrected_flux``),
  while the fluid's flux exists;
* the CT half, B_f . F^CT, needs only the EMFs and the cell-centred field and
  is added after the CT's EMFs are built (``ct_energy_flux_rhs``), so the EMFs
  need not exist during the fluid fluxes.
"""

# astronomix functions
from astronomix._spatial_operators._differencing import finite_difference_int6
from astronomix._spatial_operators._interpolate import interp_face_to_center
from astronomix._stencil_operations._stencil_operations import _shift

#: The staggered sixth-order derivative coefficients (``finite_difference_int6``).
_C1, _C2, _C3 = 75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0

#: (field component c, flux axis a) -> (EMF component e, sign eps_{c a e}).
_LEVI_CIVITA = {
    (0, 1): (2, 1.0),
    (1, 0): (2, -1.0),
    (1, 2): (0, 1.0),
    (2, 1): (0, -1.0),
    (2, 0): (1, 1.0),
    (0, 2): (1, -1.0),
}


def wide_edge_combination(edge_field, axis):
    """
    W^axis[E] at index j (edge position j + 1/2): the combination whose
    difference W(j + 1/2) - W(j - 1/2) is ``finite_difference_int6`` at j.
    """

    def at(offset):
        return _shift(edge_field, -offset, axis=axis)

    three = at(1) + edge_field + at(-1)
    five = three + at(2) + at(-2)
    return _C1 * edge_field + _C2 * three + _C3 * five


def exact_center_fluxes_along(emfs, axis, dimensionality):
    """
    The face fluxes F^CT_{c,axis} (c != axis) whose flux differences along
    ``axis`` reproduce the CT's increment of the cell-centred field exactly.

    Args:
        emfs: The edge EMFs ``(E_x, E_y, E_z)`` as the CT's curl uses them.
        axis: The flux axis.
        dimensionality: The number of active axes.

    Returns:
        A dict ``{c: flux}`` over the two transverse components.
    """
    fluxes = {}
    for component in range(3):
        if component == axis:
            continue
        emf_component, sign = _LEVI_CIVITA[(component, axis)]
        wide = wide_edge_combination(emfs[emf_component], axis)
        if component < dimensionality:
            wide = interp_face_to_center(wide, component)
        fluxes[component] = sign * wide
    return fluxes


def _face_field(cell_field, axis):
    """The two-point face value (B_i + B_{i+1}) / 2 at i + 1/2 along ``axis``."""
    return 0.5 * (cell_field + _shift(cell_field, -1, axis=axis))


def magnetic_energy_corrected_flux(interface_flux, magnetic_state, energy_flux_channels, axis: int):
    """
    The fluid half of the correction: ``interface_flux`` with the magnetic
    energy flux of the fluid's own field rows removed from the energy row,
    F_E - sum_c B_f[c] F_{B_c}.

    Args:
        interface_flux: The flux through the right face of each cell.
        magnetic_state: The state whose cell-centred field gives B_f.
        energy_flux_channels: ``(energy channel, (B_x, B_y, B_z) channels)``.
        axis: The spatial axis of the flux (0 = x).

    Returns:
        The corrected flux.
    """
    energy_channel, magnetic_channels = energy_flux_channels
    magnetic_flux = 0.0
    for channel in magnetic_channels:
        magnetic_flux = magnetic_flux + _face_field(magnetic_state[channel], axis) * interface_flux[channel]
    return interface_flux.at[energy_channel].add(-magnetic_flux)


def ct_magnetic_energy_flux(cell_magnetic, emfs, axis, dimensionality):
    """
    sum_{c != axis} B_f[c] F^CT_{c,axis}: the magnetic energy flux of the field
    the CT moves. Its normal component has no CT flux: the fluid's normal-field
    row is diffusion of B_n, which the CT never applies, so its energy is
    not transported either.

    Args:
        cell_magnetic: ``(B_x, B_y, B_z)`` at the cell centres (the stage state).
        emfs: The edge EMFs ``(E_x, E_y, E_z)``.
        axis: The flux axis.
        dimensionality: The number of active axes.

    Returns:
        The flux at the faces along ``axis``.
    """
    ct_fluxes = exact_center_fluxes_along(emfs, axis, dimensionality)
    flux = 0.0
    for component in range(3):
        if component == axis:
            continue
        flux = flux + _face_field(cell_magnetic[component], axis) * ct_fluxes[component]
    return flux


def ct_energy_flux_rhs(cell_magnetic, emfs, dt_over_dx, dimensionality):
    """
    The CT half of the correction, -dt/h sum_a delta_a(sum_{c != a} B_f[c]
    F^CT_{c,a}) over the active axes.

    Args:
        cell_magnetic: ``(B_x, B_y, B_z)`` at the cell centres (the stage state).
        emfs: The edge EMFs ``(E_x, E_y, E_z)``.
        dt_over_dx: The (stage) time step over the cell size.
        dimensionality: The number of active axes.

    Returns:
        The energy increment.
    """
    increment = 0.0
    for axis in range(dimensionality):
        flux = ct_magnetic_energy_flux(cell_magnetic, emfs, axis, dimensionality)
        increment = increment - dt_over_dx * (flux - _shift(flux, 1, axis=axis))
    return increment


def edge_emf_curl(emfs, dt_over_dx, dimensionality):
    """
    The interface-field increments dt/h curl(E) of the edge EMFs with the
    staggered sixth-order derivative (the CT's curl, any EMF).

    Args:
        emfs: The edge EMFs ``(E_x, E_y, E_z)``.
        dt_over_dx: The (stage) time step over the cell size.
        dimensionality: The number of active axes.

    Returns:
        The increments ``(rhs_bx, rhs_by, rhs_bz)``.
    """

    def difference(edge_field, axis):
        if axis >= dimensionality:
            return 0.0
        return finite_difference_int6(edge_field, axis)

    emf_x, emf_y, emf_z = emfs
    rhs_bx = dt_over_dx * (-difference(emf_z, 1) + difference(emf_y, 2))
    rhs_by = dt_over_dx * (-difference(emf_x, 2) + difference(emf_z, 0))
    rhs_bz = dt_over_dx * (-difference(emf_y, 0) + difference(emf_x, 1))
    return rhs_bx, rhs_by, rhs_bz
