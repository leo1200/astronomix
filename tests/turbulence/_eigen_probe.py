#!/usr/bin/env python
"""Adversarial probe of the isothermal-MHD eigensystem.

For a grid of extreme/degenerate interface states (deep-void rho, strong B,
Bt->0, Bx->0, cf~cs, large |v|), check:
  (1) L_row / R_col / lambdas are all FINITE,
  (2) the 6x6 matrix M[i,j] = L_row(i) . R_col(j) equals the identity
      (eigenvector consistency; a violation => unstable characteristic
      reconstruction, the kind that can drive a marginal blow-up).

Run on GPU via autocvd (tiny). Pinpoints whether the deep-void NaN is an
eigensystem bug (non-finite or L.R != I) or lives elsewhere (overflow).
"""
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
import itertools
import numpy as np
import jax
import jax.numpy as jnp

from astronomix import SimulationConfig, get_registered_variables
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, ISOTHERMAL, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
)
from astronomix._fluid_equations._eigen_mhd_iso import (
    _eigen_L_row_iso, _eigen_R_col_iso, _eigen_all_lambdas_iso,
)

cfg = SimulationConfig(
    solver_mode=FINITE_DIFFERENCE, equation_of_state=ISOTHERMAL, mhd=True,
    dimensionality=3, num_cells=8, box_size=1.0,
    boundary_settings=BoundarySettings(
        *[BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY,
                             right_boundary=PERIODIC_BOUNDARY) for _ in range(3)]),
)
rv = get_registered_variables(cfg)
nv = rv.num_vars
di = rv.density_index
mi = rv.momentum_index
bi = rv.magnetic_index
rhomin = 0.02
cs = 0.05  # the M20 (deep-void) sound speed

# --- build a battery of adversarial 1D interface states (along axis 0) -------
rng = np.random.default_rng(0)
rho_vals = [rhomin, 0.01, 0.05, 1.0]
v_vals = [0.0, 1.0, 50.0, -30.0]
# (Bx, By, Bz) covering: zero-field, Bx->0 (pure tangential), Bt->0 (pure normal),
# strong, near-degenerate (cf~cs needs B small or aligned), random
B_sets = [
    (0.0, 0.0, 0.0), (1e-8, 0.0, 0.0), (0.0, 1e-8, 1e-8), (3.0, 0.0, 0.0),
    (0.0, 3.0, 0.0), (3.0, 3.0, 3.0), (1e-6, 1e-6, 1e-6), (0.4472, 0.0, 0.0),
    (cs * 1.0000001, 0.0, 0.0),  # cf ~ cs degeneracy (vA_x ~ cs)
]
cases = []
for rho, vx, (Bx, By, Bz) in itertools.product(rho_vals, v_vals, B_sets):
    cases.append((rho, vx, vx * 0.3, -vx * 0.5, Bx, By, Bz))
ncell = len(cases) + 4
state = np.zeros((nv, ncell, 1, 1), dtype=np.float32)
state[di, :, 0, 0] = rhomin
for c, (rho, vx, vy, vz, Bx, By, Bz) in enumerate(cases):
    state[di, c, 0, 0] = rho
    state[mi.x, c, 0, 0] = rho * vx
    state[mi.y, c, 0, 0] = rho * vy
    state[mi.z, c, 0, 0] = rho * vz
    state[bi.x, c, 0, 0] = Bx
    state[bi.y, c, 0, 0] = By
    state[bi.z, c, 0, 0] = Bz
state = jnp.asarray(state)

# --- eigenvectors + lambdas --------------------------------------------------
L = jnp.stack([_eigen_L_row_iso(state, rhomin, cs, rv, r) for r in range(6)])  # (6,nv,ncell,1,1)
R = jnp.stack([_eigen_R_col_iso(state, rhomin, cs, rv, c) for c in range(6)])  # (6,nv,...)
lam = _eigen_all_lambdas_iso(state, rhomin, cs, rv)

Lf = np.asarray(L); Rf = np.asarray(R); lamf = np.asarray(lam)
print("L finite:", bool(np.all(np.isfinite(Lf))), " R finite:", bool(np.all(np.isfinite(Rf))),
      " lambda finite:", bool(np.all(np.isfinite(lamf))))

# M[i,j] = sum_var L_i . R_j, per cell. Identity check.
M = np.einsum("ivxyz,jvxyz->ijxyz", Lf, Rf)  # (6,6,ncell,1,1)
I6 = np.eye(6)[:, :, None, None, None]
err = np.abs(M - I6)
# per-cell max identity error (over the finite cells)
percell = err.reshape(6, 6, ncell, -1).max(axis=(0, 1)).reshape(ncell, -1).max(axis=1)
worst = np.argsort(percell)[::-1][:6]
print(f"L.R identity: max|M-I| = {np.nanmax(err):.3e}, median = {np.nanmedian(percell):.3e}")
print("worst cells (idx: identity-err, state rho,vx,Bx,By,Bz):")
for w in worst:
    if w < len(cases):
        rho, vx, vy, vz, Bx, By, Bz = cases[w]
        print(f"  cell {w}: err={percell[w]:.3e}  rho={rho:.3g} vx={vx:.3g} "
              f"B=({Bx:.3g},{By:.3g},{Bz:.3g}) finite_M={bool(np.all(np.isfinite(M[:,:,w])))}")
    else:
        print(f"  cell {w}: err={percell[w]:.3e} (padding cell)")
print("ANY non-finite in M:", bool(not np.all(np.isfinite(M))))
