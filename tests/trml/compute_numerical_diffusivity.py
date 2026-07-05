"""Dissipative-flux estimate of the numerical diffusivity on a saved TRML state.

The WENO flux in this code is assembled as

    F_{i+1/2} = F_central  +  (characteristic-mode corrections)

where ``F_central = 1/12 (-F_{i-1} + 7 F_i + 7 F_{i+1} - F_{i+2})`` is the
4th-order *centered*, non-dissipative reconstruction of the physical flux and
the mode corrections carry the Lax-Friedrichs / WENO numerical dissipation.
The dissipative part of the numerical flux is therefore

    F_diss = F_WENO - F_central .

Modelling the numerical dissipation of the *density* field (phase mixing =
density mixing, which sets the T = P/rho PDF at fixed P) as Fickian,
``F_diss^rho = -D_num d rho/dx``, we invert per cell

    D_num = dx * sum_faces |F_diss^rho|  /  sum_faces |Delta rho| ,

summed over the (up to 6) axis faces of the cell.  This is WENO-adaptive: it is
small in smooth, resolved regions (where the mode corrections nearly vanish)
and large at the interface.  Cells with a negligible density gradient
(``sum |Delta rho| / rho < REL_GRAD_FLOOR``) are flagged invalid -- numerical
diffusion does not act where there is nothing to diffuse, and the inversion is
ill-posed there.

Saves ``data/diffusivity/trml_N{N}_numdiff.npz`` with the cell-centered
``D_num_diss`` field and a boolean ``valid`` mask (interior, unpadded).

Usage (from tests/trml, repo root on PYTHONPATH):
    PYTHONPATH=<repo-root> python compute_numerical_diffusivity.py --num-cells 64
"""

import argparse
import os

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import numpy as np
import jax.numpy as jnp

from trml_diffusivity import build, DATA_DIR

from astronomix.time_stepping._utils import _pad, _unpad
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix._fluid_equations._equations import (
    conserved_state_from_primitive,
    primitive_state_from_conserved,
)
from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux_x_native,
    _weno_flux_y_native,
    _weno_flux_z_native,
)
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix.option_classes.simulation_config import CONSERVATIVE_GAS_STATE

REL_GRAD_FLOOR = 1e-3   # |sum Delta rho| / rho below this -> cell flagged invalid

# native per-axis WENO flux + matching physical-flux direction index (1=x,2=y,3=z)
_WENO = {1: _weno_flux_x_native, 2: _weno_flux_y_native, 3: _weno_flux_z_native}


def _centered_flux(F_phys, axis):
    """4th-order centered (non-dissipative) face flux, matching the WENO
    F_interface term: 1/12 (-F_{i-1} + 7 F_i + 7 F_{i+1} - F_{i+2})."""
    return (
        -_shift(F_phys, 1, axis=axis)
        + 7.0 * F_phys
        + 7.0 * _shift(F_phys, -1, axis=axis)
        - _shift(F_phys, -2, axis=axis)
    ) / 12.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cells", type=int, required=True)
    args = ap.parse_args()
    N = args.num_cells

    config, params, rv, _init, meta = build(N)
    gamma = params.gamma
    dx = config.grid_spacing
    rho_idx = rv.density_index

    data = np.load(os.path.join(DATA_DIR, f"trml_N{N}.npz"))
    primitive = jnp.asarray(data["final_state"])  # interior, primitive

    # to conserved, pad with ghost cells, apply boundary conditions
    conserved = conserved_state_from_primitive(primitive, gamma, config, rv)
    q = _pad(conserved, config)
    q = _boundary_handler(q, config, rv, params, CONSERVATIVE_GAS_STATE)
    prim_pad = primitive_state_from_conserved(q, gamma, config, rv)

    rho = q[rho_idx]

    sum_absFdiss = jnp.zeros_like(rho)
    sum_absDrho = jnp.zeros_like(rho)

    for axis in (1, 2, 3):  # x, y, z array axes
        F_weno = _WENO[axis](q, params, config, rv)              # flux at i+1/2
        F_phys = _euler_flux(prim_pad, gamma, config, rv, axis)  # physical flux
        F_cent = _centered_flux(F_phys, axis)
        F_diss = F_weno - F_cent                                 # dissipative flux

        fd_rho = F_diss[rho_idx]                 # numerical mass-dissipation flux
        sax = axis - 1                            # spatial axis on the 3D fields
        # right face (i+1/2) at this cell, left face (i-1/2) = shift by +1
        drho_right = _shift(rho, -1, axis=sax) - rho
        drho_left = rho - _shift(rho, 1, axis=sax)
        fd_right = fd_rho
        fd_left = _shift(fd_rho, 1, axis=sax)

        sum_absFdiss = sum_absFdiss + 0.5 * (jnp.abs(fd_right) + jnp.abs(fd_left))
        sum_absDrho = sum_absDrho + 0.5 * (jnp.abs(drho_right) + jnp.abs(drho_left))

    # Fickian inversion (gradient-weighted), regularised
    D_num = dx * sum_absFdiss / (sum_absDrho + 1e-30)
    valid = (sum_absDrho / jnp.maximum(rho, 1e-30)) > REL_GRAD_FLOOR

    # back to the interior grid (strip the ghost ring on each spatial axis)
    ng = config.num_ghost_cells
    interior = (slice(ng, -ng),) * 3 if ng > 0 else (slice(None),) * 3
    D_num = np.asarray(D_num)[interior]
    valid = np.asarray(valid)[interior]

    out = os.path.join(DATA_DIR, f"trml_N{N}_numdiff.npz")
    np.savez(out, D_num_diss=D_num, valid=valid, rel_grad_floor=REL_GRAD_FLOOR)
    vd = D_num[valid]
    print(f"[N={N}] valid cells: {valid.mean()*100:.1f}%  "
          f"D_num_diss median={np.median(vd):.2e}  "
          f"[{np.percentile(vd,5):.2e}, {np.percentile(vd,95):.2e}]")
    print(f"[N={N}] saved -> {out}")


if __name__ == "__main__":
    main()
