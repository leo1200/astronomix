"""
Computations of the eigenvalues and eigenvectors for the MHD equations.
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Union

from jf1uids.variable_registry.registered_variables import RegisteredVariables

EPSILON = 1e-7

@partial(jax.jit, static_argnames=["registered_variables"])
def _eigen_x(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    # registry indices (conserved ordering)
    DI = registered_variables.density_index
    MX = registered_variables.momentum_index.x
    MY = registered_variables.momentum_index.y
    MZ = registered_variables.momentum_index.z
    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z
    IE = registered_variables.energy_index

    N_vars, Nx, Ny, Nz = conserved_state.shape

    # unpack conserved variables (centers)
    DD = conserved_state[DI]    # density
    Mx = conserved_state[MX]
    My = conserved_state[MY]
    Mz = conserved_state[MZ]
    Bx = conserved_state[BX]
    By = conserved_state[BY]
    Bz = conserved_state[BZ]
    EE = conserved_state[IE]

    # compute primitives (like Fortran primit)
    rho = DD
    vx = Mx / rho
    vy = My / rho
    vz = Mz / rho

    vv2 = vx * vx + vy * vy + vz * vz
    BB2 = Bx * Bx + By * By + Bz * Bz

    pg = (gamma - 1.0) * (EE - 0.5 * (rho * vv2 + BB2))

    # protection (use same small floors as before)
    rhomin = 1.0e-12
    pgmin = 1.0e-12

    mask_bad = (rho < rhomin) | (pg < pgmin)
    rho = jnp.where(mask_bad, jnp.maximum(rho, rhomin), rho)
    pg = jnp.where(mask_bad, jnp.maximum(pg, pgmin), pg)
    EE = jnp.where(mask_bad, pg / (gamma - 1.0) + 0.5 * (rho * vv2 + BB2), EE)

    HH = (EE + pg) / rho

    # center-derived quantities for eigenvalues (center block, matches Fortran)
    bbn2 = BB2 / rho
    bnx2 = (Bx * Bx) / rho

    cs2_center = jnp.maximum(0.0, gamma * jnp.abs(pg / rho))

    disc_c = (bbn2 + cs2_center) ** 2 - 4.0 * bnx2 * cs2_center
    root_c = jnp.sqrt(jnp.maximum(EPSILON, disc_c))

    lf_c = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2 + cs2_center + root_c)))
    la_c = jnp.sqrt(jnp.maximum(EPSILON, bnx2))
    ls_c = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2 + cs2_center - root_c)))

    # eigenvalues at centers: (7, Nx, Ny, Nz)
    lambdas = jnp.stack([
        vx - lf_c,
        vx - la_c,
        vx - ls_c,
        vx,
        vx + ls_c,
        vx + la_c,
        vx + lf_c
    ], axis=0)

    # helper: periodic average to interfaces using jnp.roll (keeps Nx shape)
    def avg_x(arr):
        return 0.5 * (arr + jnp.roll(arr, shift=-1, axis=0))

    rho_i = avg_x(rho)
    vx_i  = avg_x(vx)
    vy_i  = avg_x(vy)
    vz_i  = avg_x(vz)
    Bx_i  = avg_x(Bx)
    By_i  = avg_x(By)
    Bz_i  = avg_x(Bz)
    pg_i  = avg_x(pg)
    HH_i  = avg_x(HH)

    # interface derived quantities
    vv2_i = vx_i * vx_i + vy_i * vy_i + vz_i * vz_i
    BB2_i = Bx_i * Bx_i + By_i * By_i + Bz_i * Bz_i
    bbn2_i = BB2_i / rho_i
    bnx2_i = (Bx_i * Bx_i) / rho_i

    # cs2 at interfaces per Fortran (enthalpy-based)
    cs2_i = (gamma - 1.0) * (HH_i - 0.5 * (vv2_i + bbn2_i))
    cs_i = jnp.sqrt(jnp.maximum(EPSILON, cs2_i))

    disc_i = (bbn2_i + cs2_i) ** 2 - 4.0 * bnx2_i * cs2_i
    root_i = jnp.sqrt(jnp.maximum(EPSILON, disc_i))

    lf_i = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2_i + cs2_i + root_i)))
    la_i = jnp.sqrt(jnp.maximum(EPSILON, bnx2_i))   # sqrt(Bx^2/rho) -> |Bx|/sqrt(rho)
    ls_i = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2_i + cs2_i - root_i)))

    # degeneracy handling (tangential B)
    Bt2 = By_i * By_i + Bz_i * Bz_i
    sgnBx = jnp.where(Bx_i >= 0.0, 1.0, -1.0)

    Bt2 = jnp.maximum(Bt2, 1.0e-31)
    
    # fallback for tiny Bt
    bty = jnp.where(Bt2 >= 1.0e-30, By_i / jnp.sqrt(Bt2), 1.0 / jnp.sqrt(2.0))
    btz = jnp.where(Bt2 >= 1.0e-30, Bz_i / jnp.sqrt(Bt2), 1.0 / jnp.sqrt(2.0))

    denom = lf_i * lf_i - ls_i * ls_i
    tiny = 1.0e-30
    denom = jnp.maximum(denom, 1e-31)


    af = jnp.where(denom >= tiny,
                   jnp.sqrt(jnp.maximum(EPSILON, cs2_i - ls_i * ls_i)) / jnp.sqrt(denom),
                   1.0)
    as_ = jnp.where(denom >= tiny,
                    jnp.sqrt(jnp.maximum(EPSILON, lf_i * lf_i - cs2_i)) / jnp.sqrt(denom),
                    1.0)

    sqrt_rho = jnp.sqrt(rho_i)

    gam0 = 1.0 - gamma
    gam1 = 0.5 * (gamma - 1.0)
    gam2 = (gamma - 2.0) / (gamma - 1.0)

    # allocate R and L on interfaces (Nx,Ny,Nz)
    R = jnp.zeros((N_vars, 7, Nx, Ny, Nz))
    L = jnp.zeros((7, N_vars, Nx, Ny, Nz))

    # helper setters using registry indices for conserved ordering
    def rset(idx, mode, value):
        return R.at[idx, mode, ...].set(value)

    def lset(mode, idx, value):
        return L.at[mode, idx, ...].set(value)

    # map conserved ordering indices for eigenvector slots
    idx_D  = DI
    idx_Mx = MX
    idx_My = MY
    idx_Mz = MZ
    idx_By = BY
    idx_Bz = BZ
    idx_E  = IE

    # Fill left eigenvectors L (rows 1..7 -> 0..6) exactly as Fortran
    L = lset(0, idx_D,  af * (gam1 * vv2_i + lf_i * vx_i) - as_ * ls_i * (bty * vy_i + btz * vz_i) * sgnBx)
    L = lset(0, idx_Mx, af * (gam0 * vx_i - lf_i))
    L = lset(0, idx_My, gam0 * af * vy_i + as_ * ls_i * bty * sgnBx)
    L = lset(0, idx_Mz, gam0 * af * vz_i + as_ * ls_i * btz * sgnBx)
    L = lset(0, idx_By, gam0 * af * By_i + cs_i * as_ * bty * sqrt_rho)
    L = lset(0, idx_Bz, gam0 * af * Bz_i + cs_i * as_ * btz * sqrt_rho)
    L = lset(0, idx_E,  -gam0 * af)

    L = lset(1, idx_D,  btz * vy_i - bty * vz_i)
    L = lset(1, idx_Mx, 0.0)
    L = lset(1, idx_My, -btz)
    L = lset(1, idx_Mz, bty)
    L = lset(1, idx_By, -btz * sgnBx * sqrt_rho)
    L = lset(1, idx_Bz, bty * sgnBx * sqrt_rho)
    L = lset(1, idx_E,  0.0)

    L = lset(2, idx_D,  as_ * (gam1 * vv2_i + ls_i * vx_i) + af * lf_i * (bty * vy_i + btz * vz_i) * sgnBx)
    L = lset(2, idx_Mx, gam0 * as_ * vx_i - as_ * ls_i)
    L = lset(2, idx_My, gam0 * as_ * vy_i - af * lf_i * bty * sgnBx)
    L = lset(2, idx_Mz, gam0 * as_ * vz_i - af * lf_i * btz * sgnBx)
    L = lset(2, idx_By, gam0 * as_ * By_i - cs_i * af * bty * sqrt_rho)
    L = lset(2, idx_Bz, gam0 * as_ * Bz_i - cs_i * af * btz * sqrt_rho)
    L = lset(2, idx_E,  -gam0 * as_)

    L = lset(3, idx_D,  -cs2_i / gam0 - 0.5 * vv2_i)
    L = lset(3, idx_Mx, vx_i)
    L = lset(3, idx_My, vy_i)
    L = lset(3, idx_Mz, vz_i)
    L = lset(3, idx_By, By_i)
    L = lset(3, idx_Bz, Bz_i)
    L = lset(3, idx_E,  -1.0)

    L = lset(4, idx_D,  as_ * (gam1 * vv2_i - ls_i * vx_i) - af * lf_i * (bty * vy_i + btz * vz_i) * sgnBx)
    L = lset(4, idx_Mx, as_ * (gam0 * vx_i + ls_i))
    L = lset(4, idx_My, gam0 * as_ * vy_i + af * lf_i * bty * sgnBx)
    L = lset(4, idx_Mz, gam0 * as_ * vz_i + af * lf_i * btz * sgnBx)
    L = lset(4, idx_By, gam0 * as_ * By_i - cs_i * af * bty * sqrt_rho)
    L = lset(4, idx_Bz, gam0 * as_ * Bz_i - cs_i * af * btz * sqrt_rho)
    L = lset(4, idx_E,  -gam0 * as_)

    L = lset(5, idx_D,  btz * vy_i - bty * vz_i)
    L = lset(5, idx_Mx, 0.0)
    L = lset(5, idx_My, -btz)
    L = lset(5, idx_Mz, bty)
    L = lset(5, idx_By, btz * sgnBx * sqrt_rho)
    L = lset(5, idx_Bz, -bty * sgnBx * sqrt_rho)
    L = lset(5, idx_E,  0.0)

    L = lset(6, idx_D,  af * (gam1 * vv2_i - lf_i * vx_i) + as_ * ls_i * (bty * vy_i + btz * vz_i) * sgnBx)
    L = lset(6, idx_Mx, af * (gam0 * vx_i + lf_i))
    L = lset(6, idx_My, gam0 * af * vy_i - as_ * ls_i * bty * sgnBx)
    L = lset(6, idx_Mz, gam0 * af * vz_i - as_ * ls_i * btz * sgnBx)
    L = lset(6, idx_By, gam0 * af * By_i + cs_i * as_ * bty * sqrt_rho)
    L = lset(6, idx_Bz, gam0 * af * Bz_i + cs_i * as_ * btz * sqrt_rho)
    L = lset(6, idx_E,  -gam0 * af)

    # normalization scalings (Fortran)
    inv_cs2 = jnp.where(cs2_i > 0.0, 1.0 / cs2_i, 0.0)
    L = L.at[0, :, ...].set(0.5 * L[0, :, ...] * inv_cs2)
    L = L.at[1, :, ...].set(0.5 * L[1, :, ...])
    L = L.at[2, :, ...].set(0.5 * L[2, :, ...] * inv_cs2)
    L = L.at[3, :, ...].set(-gam0 * L[3, :, ...] * inv_cs2)
    L = L.at[4, :, ...].set(0.5 * L[4, :, ...] * inv_cs2)
    L = L.at[5, :, ...].set(0.5 * L[5, :, ...])
    L = L.at[6, :, ...].set(0.5 * L[6, :, ...] * inv_cs2)

    # Fill right eigenvectors R (columns) exactly as Fortran
    # Column 1 (fast -)
    R = rset(idx_D, 0, af)
    R = rset(idx_Mx, 0, af * (vx_i - lf_i))
    R = rset(idx_My, 0, af * vy_i + as_ * ls_i * bty * sgnBx)
    R = rset(idx_Mz, 0, af * vz_i + as_ * ls_i * btz * sgnBx)
    R = rset(idx_By, 0, cs_i * as_ * bty / sqrt_rho)
    R = rset(idx_Bz, 0, cs_i * as_ * btz / sqrt_rho)
    R = rset(idx_E,  0, af * (lf_i**2 - lf_i * vx_i + 0.5 * vv2_i - gam2 * cs2_i) + as_ * ls_i * (bty * vy_i + btz * vz_i) * sgnBx)

    # Column 2 (alfven -)
    R = rset(idx_D, 1, 0.0)
    R = rset(idx_Mx, 1, 0.0)
    R = rset(idx_My, 1, -btz)
    R = rset(idx_Mz, 1, bty)
    R = rset(idx_By, 1, -btz * sgnBx / sqrt_rho)
    R = rset(idx_Bz, 1, bty * sgnBx / sqrt_rho)
    R = rset(idx_E,  1, bty * vz_i - btz * vy_i)

    # Column 3 (slow -)
    R = rset(idx_D, 2, as_)
    R = rset(idx_Mx, 2, as_ * (vx_i - ls_i))
    R = rset(idx_My, 2, as_ * vy_i - af * lf_i * bty * sgnBx)
    R = rset(idx_Mz, 2, as_ * vz_i - af * lf_i * btz * sgnBx)
    R = rset(idx_By, 2, -cs_i * af * bty / sqrt_rho)
    R = rset(idx_Bz, 2, -cs_i * af * btz / sqrt_rho)
    R = rset(idx_E,  2, as_ * (ls_i**2 - ls_i * vx_i + 0.5 * vv2_i - gam2 * cs2_i) - af * lf_i * (bty * vy_i + btz * vz_i) * sgnBx)

    # Column 4 (entropy)
    R = rset(idx_D, 3, 1.0)
    R = rset(idx_Mx, 3, vx_i)
    R = rset(idx_My, 3, vy_i)
    R = rset(idx_Mz, 3, vz_i)
    R = rset(idx_By, 3, 0.0)
    R = rset(idx_Bz, 3, 0.0)
    R = rset(idx_E,  3, 0.5 * vv2_i)

    # Column 5 (slow +)
    R = rset(idx_D, 4, as_)
    R = rset(idx_Mx, 4, as_ * (vx_i + ls_i))
    R = rset(idx_My, 4, as_ * vy_i + af * lf_i * bty * sgnBx)
    R = rset(idx_Mz, 4, as_ * vz_i + af * lf_i * btz * sgnBx)
    R = rset(idx_By, 4, -cs_i * af * bty / sqrt_rho)
    R = rset(idx_Bz, 4, -cs_i * af * btz / sqrt_rho)
    R = rset(idx_E,  4, as_ * (ls_i**2 + ls_i * vx_i + 0.5 * vv2_i - gam2 * cs2_i) + af * lf_i * (bty * vy_i + btz * vz_i) * sgnBx)

    # Column 6 (alfven +)
    R = rset(idx_D, 5, 0.0)
    R = rset(idx_Mx, 5, 0.0)
    R = rset(idx_My, 5, -btz)
    R = rset(idx_Mz, 5, bty)
    R = rset(idx_By, 5, btz * sgnBx / sqrt_rho)
    R = rset(idx_Bz, 5, -bty * sgnBx / sqrt_rho)
    R = rset(idx_E,  5, bty * vz_i - btz * vy_i)

    # Column 7 (fast +)
    R = rset(idx_D, 6, af)
    R = rset(idx_Mx, 6, af * (vx_i + lf_i))
    R = rset(idx_My, 6, af * vy_i - as_ * ls_i * bty * sgnBx)
    R = rset(idx_Mz, 6, af * vz_i - as_ * ls_i * btz * sgnBx)
    R = rset(idx_By, 6, cs_i * as_ * bty / sqrt_rho)
    R = rset(idx_Bz, 6, cs_i * as_ * btz / sqrt_rho)
    R = rset(idx_E,  6, af * (lf_i**2 + lf_i * vx_i + 0.5 * vv2_i - gam2 * cs2_i) - as_ * ls_i * (bty * vy_i + btz * vz_i) * sgnBx)

    # continuity sign flips (sgnBt) and conditional multiplication as in Fortran
    sgnBt = jnp.where(By_i != 0.0, jnp.where(By_i >= 0.0, 1.0, -1.0), jnp.where(Bz_i >= 0.0, 1.0, -1.0))
    mask_cs_ge_la = cs_i >= la_i

    # apply sign flips to L
    L = L.at[2, :, ...].set(jnp.where(mask_cs_ge_la, L[2, :, ...] * sgnBt, L[2, :, ...]))
    L = L.at[4, :, ...].set(jnp.where(mask_cs_ge_la, L[4, :, ...] * sgnBt, L[4, :, ...]))
    L = L.at[0, :, ...].set(jnp.where(~mask_cs_ge_la, L[0, :, ...] * sgnBt, L[0, :, ...]))
    L = L.at[6, :, ...].set(jnp.where(~mask_cs_ge_la, L[6, :, ...] * sgnBt, L[6, :, ...]))

    # apply sign flips to R
    R = R.at[:, 2, ...].set(jnp.where(mask_cs_ge_la, R[:, 2, ...] * sgnBt, R[:, 2, ...]))
    R = R.at[:, 4, ...].set(jnp.where(mask_cs_ge_la, R[:, 4, ...] * sgnBt, R[:, 4, ...]))
    R = R.at[:, 0, ...].set(jnp.where(~mask_cs_ge_la, R[:, 0, ...] * sgnBt, R[:, 0, ...]))
    R = R.at[:, 6, ...].set(jnp.where(~mask_cs_ge_la, R[:, 6, ...] * sgnBt, R[:, 6, ...]))

    return lambdas, R, L

# I've added these functions to avoid 
# computing the full R and L downstream

@partial(jax.jit, static_argnames=["registered_variables"])
def _eigen_R_col(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
    col: int,
):
    # registry indices (conserved ordering)
    DI = registered_variables.density_index
    MX = registered_variables.momentum_index.x
    MY = registered_variables.momentum_index.y
    MZ = registered_variables.momentum_index.z
    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z
    IE = registered_variables.energy_index

    N_vars, Nx, Ny, Nz = conserved_state.shape

    # unpack conserved variables (centers)
    DD = conserved_state[DI]    # density
    Mx = conserved_state[MX]
    My = conserved_state[MY]
    Mz = conserved_state[MZ]
    Bx = conserved_state[BX]
    By = conserved_state[BY]
    Bz = conserved_state[BZ]
    EE = conserved_state[IE]

    # compute primitives (like Fortran primit)
    rho = DD
    vx = Mx / rho
    vy = My / rho
    vz = Mz / rho

    vv2 = vx * vx + vy * vy + vz * vz
    BB2 = Bx * Bx + By * By + Bz * Bz

    pg = (gamma - 1.0) * (EE - 0.5 * (rho * vv2 + BB2))

    mask_bad = (rho < rhomin) | (pg < pgmin)
    rho = jnp.where(mask_bad, jnp.maximum(rho, rhomin), rho)
    pg = jnp.where(mask_bad, jnp.maximum(pg, pgmin), pg)
    EE = jnp.where(mask_bad, pg / (gamma - 1.0) + 0.5 * (rho * vv2 + BB2), EE)

    HH = (EE + pg) / rho

    # helper: periodic average to interfaces using jnp.roll (keeps Nx shape)
    def avg_x(arr):
        return 0.5 * (arr + jnp.roll(arr, shift=-1, axis=0))

    rho_i = avg_x(rho)
    vx_i  = avg_x(vx)
    vy_i  = avg_x(vy)
    vz_i  = avg_x(vz)
    Bx_i  = avg_x(Bx)
    By_i  = avg_x(By)
    Bz_i  = avg_x(Bz)
    HH_i  = avg_x(HH)

    # interface derived quantities
    vv2_i = vx_i * vx_i + vy_i * vy_i + vz_i * vz_i
    BB2_i = Bx_i * Bx_i + By_i * By_i + Bz_i * Bz_i
    bbn2_i = BB2_i / rho_i
    bnx2_i = (Bx_i * Bx_i) / rho_i

    # cs2 at interfaces per Fortran (enthalpy-based)
    cs2_i = (gamma - 1.0) * (HH_i - 0.5 * (vv2_i + bbn2_i))
    cs_i = jnp.sqrt(jnp.maximum(EPSILON, cs2_i))

    disc_i = (bbn2_i + cs2_i) ** 2 - 4.0 * bnx2_i * cs2_i
    root_i = jnp.sqrt(jnp.maximum(EPSILON, disc_i))

    lf_i = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2_i + cs2_i + root_i)))
    la_i = jnp.sqrt(jnp.maximum(EPSILON, bnx2_i))   # sqrt(Bx^2/rho) -> |Bx|/sqrt(rho)
    ls_i = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2_i + cs2_i - root_i)))

    # degeneracy handling (tangential B)
    Bt2 = By_i * By_i + Bz_i * Bz_i
    sgnBx = jnp.where(Bx_i >= 0.0, 1.0, -1.0)

    Bt2 = jnp.maximum(Bt2, 1.0e-31)
    
    # fallback for tiny Bt
    bty = jnp.where(Bt2 >= 1.0e-30, By_i / jnp.sqrt(Bt2), 1.0 / jnp.sqrt(2.0))
    btz = jnp.where(Bt2 >= 1.0e-30, Bz_i / jnp.sqrt(Bt2), 1.0 / jnp.sqrt(2.0))

    denom = lf_i * lf_i - ls_i * ls_i
    tiny = 1.0e-30
    denom = jnp.maximum(denom, 1e-31)

    af = jnp.where(denom >= tiny,
                   jnp.sqrt(jnp.maximum(EPSILON, cs2_i - ls_i * ls_i)) / jnp.sqrt(denom),
                   1.0)
    as_ = jnp.where(denom >= tiny,
                    jnp.sqrt(jnp.maximum(EPSILON, lf_i * lf_i - cs2_i)) / jnp.sqrt(denom),
                    1.0)

    sqrt_rho = jnp.sqrt(rho_i)

    gam2 = (gamma - 2.0) / (gamma - 1.0)

    # continuity sign flips (sgnBt) and conditional multiplication as in Fortran
    sgnBt = jnp.where(By_i != 0.0, jnp.where(By_i >= 0.0, 1.0, -1.0), jnp.where(Bz_i >= 0.0, 1.0, -1.0))
    mask_cs_ge_la = cs_i >= la_i

    # map conserved ordering indices for eigenvector slots
    idx_D  = DI
    idx_Mx = MX
    idx_My = MY
    idx_Mz = MZ
    idx_By = BY
    idx_Bz = BZ
    idx_E  = IE

    def col_0():
        # Fill right eigenvectors R (columns) exactly as Fortran
        # Column 1 (fast -)
        R = jnp.zeros((N_vars, Nx, Ny, Nz))
        R = R.at[idx_D].set(af)
        R = R.at[idx_Mx].set(af * (vx_i - lf_i))
        R = R.at[idx_My].set(af * vy_i + as_ * ls_i * bty * sgnBx)
        R = R.at[idx_Mz].set(af * vz_i + as_ * ls_i * btz * sgnBx)
        R = R.at[idx_By].set(cs_i * as_ * bty / sqrt_rho)
        R = R.at[idx_Bz].set(cs_i * as_ * btz / sqrt_rho)
        R = R.at[idx_E].set(af * (lf_i**2 - lf_i * vx_i + 0.5 * vv2_i - gam2 * cs2_i) + as_ * ls_i * (bty * vy_i + btz * vz_i) * sgnBx)
        R = jnp.where(~mask_cs_ge_la, R * sgnBt, R)
        return R
    
    def col_1():
        # Column 2 (alfven -)
        R = jnp.zeros((N_vars, Nx, Ny, Nz))
        R = R.at[idx_D].set(0.0)
        R = R.at[idx_Mx].set(0.0)
        R = R.at[idx_My].set(-btz)
        R = R.at[idx_Mz].set(bty)
        R = R.at[idx_By].set(-btz * sgnBx / sqrt_rho)
        R = R.at[idx_Bz].set(bty * sgnBx / sqrt_rho)
        R = R.at[idx_E].set(bty * vz_i - btz * vy_i)
        return R
    
    def col_2():
        R = jnp.zeros((N_vars, Nx, Ny, Nz))
        R = R.at[idx_D].set(as_)
        R = R.at[idx_Mx].set(as_ * (vx_i - ls_i))
        R = R.at[idx_My].set(as_ * vy_i - af * lf_i * bty * sgnBx)
        R = R.at[idx_Mz].set(as_ * vz_i - af * lf_i * btz * sgnBx)
        R = R.at[idx_By].set(-cs_i * af * bty / sqrt_rho)
        R = R.at[idx_Bz].set(-cs_i * af * btz / sqrt_rho)
        R = R.at[idx_E].set(as_ * (ls_i**2 - ls_i * vx_i + 0.5 * vv2_i - gam2 * cs2_i) - af * lf_i * (bty * vy_i + btz * vz_i) * sgnBx)
        R = jnp.where(mask_cs_ge_la, R * sgnBt, R)
        return R
    
    def col_3():
        R = jnp.zeros((N_vars, Nx, Ny, Nz))
        R = R.at[idx_D].set(1.0)
        R = R.at[idx_Mx].set(vx_i)
        R = R.at[idx_My].set(vy_i)
        R = R.at[idx_Mz].set(vz_i)
        R = R.at[idx_By].set(0.0)
        R = R.at[idx_Bz].set(0.0)
        R = R.at[idx_E].set(0.5 * vv2_i)
        return R
    
    def col_4():
        # Column 5 (slow +)
        R = jnp.zeros((N_vars, Nx, Ny, Nz))
        R = R.at[idx_D].set(as_)
        R = R.at[idx_Mx].set(as_ * (vx_i + ls_i))
        R = R.at[idx_My].set(as_ * vy_i + af * lf_i * bty * sgnBx)
        R = R.at[idx_Mz].set(as_ * vz_i + af * lf_i * btz * sgnBx)
        R = R.at[idx_By].set(-cs_i * af * bty / sqrt_rho)
        R = R.at[idx_Bz].set(-cs_i * af * btz / sqrt_rho)
        R = R.at[idx_E].set(as_ * (ls_i**2 + ls_i * vx_i + 0.5 * vv2_i - gam2 * cs2_i) + af * lf_i * (bty * vy_i + btz * vz_i) * sgnBx)
        R = jnp.where(mask_cs_ge_la, R * sgnBt, R)
        return R
    
    def col_5():
        # Column 6 (alfven +)
        R = jnp.zeros((N_vars, Nx, Ny, Nz))
        R = R.at[idx_D].set(0.0)
        R = R.at[idx_Mx].set(0.0)
        R = R.at[idx_My].set(-btz)
        R = R.at[idx_Mz].set(bty)
        R = R.at[idx_By].set(btz * sgnBx / sqrt_rho)
        R = R.at[idx_Bz].set(-bty * sgnBx / sqrt_rho)
        R = R.at[idx_E].set(bty * vz_i - btz * vy_i)
        return R
    
    def col_6():
        # Column 7 (fast +)
        R = jnp.zeros((N_vars, Nx, Ny, Nz))
        R = R.at[idx_D].set(af)
        R = R.at[idx_Mx].set(af * (vx_i + lf_i))
        R = R.at[idx_My].set(af * vy_i - as_ * ls_i * bty * sgnBx)
        R = R.at[idx_Mz].set(af * vz_i - as_ * ls_i * btz * sgnBx)
        R = R.at[idx_By].set(cs_i * as_ * bty / sqrt_rho)
        R = R.at[idx_Bz].set(cs_i * as_ * btz / sqrt_rho)
        R = R.at[idx_E].set(af * (lf_i**2 + lf_i * vx_i + 0.5 * vv2_i - gam2 * cs2_i) - as_ * ls_i * (bty * vy_i + btz * vz_i) * sgnBx)
        R = jnp.where(~mask_cs_ge_la, R * sgnBt, R)
        return R

    R = jax.lax.switch(col, [col_0, col_1, col_2, col_3, col_4, col_5, col_6])

    return R

@partial(jax.jit, static_argnames=["registered_variables"])
def _eigen_L_row(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
    row: int,
):
    # registry indices (conserved ordering)
    DI = registered_variables.density_index
    MX = registered_variables.momentum_index.x
    MY = registered_variables.momentum_index.y
    MZ = registered_variables.momentum_index.z
    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z
    IE = registered_variables.energy_index

    N_vars, Nx, Ny, Nz = conserved_state.shape

    # unpack conserved variables (centers)
    DD = conserved_state[DI]    # density
    Mx = conserved_state[MX]
    My = conserved_state[MY]
    Mz = conserved_state[MZ]
    Bx = conserved_state[BX]
    By = conserved_state[BY]
    Bz = conserved_state[BZ]
    EE = conserved_state[IE]

    # compute primitives (like Fortran primit)
    rho = DD
    vx = Mx / rho
    vy = My / rho
    vz = Mz / rho

    vv2 = vx * vx + vy * vy + vz * vz
    BB2 = Bx * Bx + By * By + Bz * Bz

    pg = (gamma - 1.0) * (EE - 0.5 * (rho * vv2 + BB2))

    mask_bad = (rho < rhomin) | (pg < pgmin)
    rho = jnp.where(mask_bad, jnp.maximum(rho, rhomin), rho)
    pg = jnp.where(mask_bad, jnp.maximum(pg, pgmin), pg)
    EE = jnp.where(mask_bad, pg / (gamma - 1.0) + 0.5 * (rho * vv2 + BB2), EE)

    HH = (EE + pg) / rho

    # helper: periodic average to interfaces using jnp.roll (keeps Nx shape)
    def avg_x(arr):
        return 0.5 * (arr + jnp.roll(arr, shift=-1, axis=0))

    rho_i = avg_x(rho)
    vx_i  = avg_x(vx)
    vy_i  = avg_x(vy)
    vz_i  = avg_x(vz)
    Bx_i  = avg_x(Bx)
    By_i  = avg_x(By)
    Bz_i  = avg_x(Bz)
    HH_i  = avg_x(HH)

    # interface derived quantities
    vv2_i = vx_i * vx_i + vy_i * vy_i + vz_i * vz_i
    BB2_i = Bx_i * Bx_i + By_i * By_i + Bz_i * Bz_i
    bbn2_i = BB2_i / rho_i
    bnx2_i = (Bx_i * Bx_i) / rho_i

    # cs2 at interfaces per Fortran (enthalpy-based)
    cs2_i = (gamma - 1.0) * (HH_i - 0.5 * (vv2_i + bbn2_i))
    cs_i = jnp.sqrt(jnp.maximum(EPSILON, cs2_i))

    disc_i = (bbn2_i + cs2_i) ** 2 - 4.0 * bnx2_i * cs2_i
    root_i = jnp.sqrt(jnp.maximum(EPSILON, disc_i))

    lf_i = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2_i + cs2_i + root_i)))
    la_i = jnp.sqrt(jnp.maximum(EPSILON, bnx2_i))   # sqrt(Bx^2/rho) -> |Bx|/sqrt(rho)
    ls_i = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2_i + cs2_i - root_i)))

    # degeneracy handling (tangential B)
    Bt2 = By_i * By_i + Bz_i * Bz_i
    sgnBx = jnp.where(Bx_i >= 0.0, 1.0, -1.0)

    Bt2 = jnp.maximum(Bt2, 1.0e-31)
    
    # fallback for tiny Bt
    bty = jnp.where(Bt2 >= 1.0e-30, By_i / jnp.sqrt(Bt2), 1.0 / jnp.sqrt(2.0))
    btz = jnp.where(Bt2 >= 1.0e-30, Bz_i / jnp.sqrt(Bt2), 1.0 / jnp.sqrt(2.0))

    denom = lf_i * lf_i - ls_i * ls_i
    tiny = 1.0e-30
    denom = jnp.maximum(denom, 1e-31)


    af = jnp.where(denom >= tiny,
                   jnp.sqrt(jnp.maximum(EPSILON, cs2_i - ls_i * ls_i)) / jnp.sqrt(denom),
                   1.0)
    as_ = jnp.where(denom >= tiny,
                    jnp.sqrt(jnp.maximum(EPSILON, lf_i * lf_i - cs2_i)) / jnp.sqrt(denom),
                    1.0)

    sqrt_rho = jnp.sqrt(rho_i)

    gam0 = 1.0 - gamma
    gam1 = 0.5 * (gamma - 1.0)

    # normalization scalings (Fortran)
    safe_cs2 = jnp.where(cs2_i > 0.0, cs2_i, 1.0)
    inv_cs2 = jnp.where(cs2_i > 0.0, 1.0 / safe_cs2, 0.0)

    # continuity sign flips (sgnBt) and conditional multiplication as in Fortran
    sgnBt = jnp.where(By_i != 0.0, jnp.where(By_i >= 0.0, 1.0, -1.0), jnp.where(Bz_i >= 0.0, 1.0, -1.0))
    mask_cs_ge_la = cs_i >= la_i

    # map conserved ordering indices for eigenvector slots
    idx_D  = DI
    idx_Mx = MX
    idx_My = MY
    idx_Mz = MZ
    idx_By = BY
    idx_Bz = BZ
    idx_E  = IE

    def row_0():
        L = jnp.zeros((N_vars, Nx, Ny, Nz))
        L = L.at[idx_D].set(af * (gam1 * vv2_i + lf_i * vx_i) - as_ * ls_i * (bty * vy_i + btz * vz_i) * sgnBx)
        L = L.at[idx_Mx].set(af * (gam0 * vx_i - lf_i))
        L = L.at[idx_My].set(gam0 * af * vy_i + as_ * ls_i * bty * sgnBx)
        L = L.at[idx_Mz].set(gam0 * af * vz_i + as_ * ls_i * btz * sgnBx)
        L = L.at[idx_By].set(gam0 * af * By_i + cs_i * as_ * bty * sqrt_rho)
        L = L.at[idx_Bz].set(gam0 * af * Bz_i + cs_i * as_ * btz * sqrt_rho)
        L = L.at[idx_E].set(-gam0 * af)
        L = 0.5 * L * inv_cs2
        L = jnp.where(~mask_cs_ge_la, L * sgnBt, L)
        return L
    
    def row_1():
        L = jnp.zeros((N_vars, Nx, Ny, Nz))
        L = L.at[idx_D].set(btz * vy_i - bty * vz_i)
        L = L.at[idx_Mx].set(0.0)
        L = L.at[idx_My].set(-btz)
        L = L.at[idx_Mz].set(bty)
        L = L.at[idx_By].set(-btz * sgnBx * sqrt_rho)
        L = L.at[idx_Bz].set(bty * sgnBx * sqrt_rho)
        L = L.at[idx_E].set(0.0)
        L = 0.5 * L
        return L

    def row_2():
        L = jnp.zeros((N_vars, Nx, Ny, Nz))
        L = L.at[idx_D].set(as_ * (gam1 * vv2_i + ls_i * vx_i) + af * lf_i * (bty * vy_i + btz * vz_i) * sgnBx)
        L = L.at[idx_Mx].set(gam0 * as_ * vx_i - as_ * ls_i)
        L = L.at[idx_My].set(gam0 * as_ * vy_i - af * lf_i * bty * sgnBx)
        L = L.at[idx_Mz].set(gam0 * as_ * vz_i - af * lf_i * btz * sgnBx)
        L = L.at[idx_By].set(gam0 * as_ * By_i - cs_i * af * bty * sqrt_rho)
        L = L.at[idx_Bz].set(gam0 * as_ * Bz_i - cs_i * af * btz * sqrt_rho)
        L = L.at[idx_E].set(-gam0 * as_)
        L = 0.5 * L * inv_cs2
        L = jnp.where(mask_cs_ge_la, L * sgnBt, L)
        return L

    def row_3():
        L = jnp.zeros((N_vars, Nx, Ny, Nz))
        L = L.at[idx_D].set(-cs2_i / gam0 - 0.5 * vv2_i)
        L = L.at[idx_Mx].set(vx_i)
        L = L.at[idx_My].set(vy_i)
        L = L.at[idx_Mz].set(vz_i)
        L = L.at[idx_By].set(By_i)
        L = L.at[idx_Bz].set(Bz_i)
        L = L.at[idx_E].set(-1.0)
        L = -gam0 * L * inv_cs2
        return L

    def row_4():
        L = jnp.zeros((N_vars, Nx, Ny, Nz))
        L = L.at[idx_D].set(as_ * (gam1 * vv2_i - ls_i * vx_i) - af * lf_i * (bty * vy_i + btz * vz_i) * sgnBx)
        L = L.at[idx_Mx].set(as_ * (gam0 * vx_i + ls_i))
        L = L.at[idx_My].set(gam0 * as_ * vy_i + af * lf_i * bty * sgnBx)
        L = L.at[idx_Mz].set(gam0 * as_ * vz_i + af * lf_i * btz * sgnBx)
        L = L.at[idx_By].set(gam0 * as_ * By_i - cs_i * af * bty * sqrt_rho)
        L = L.at[idx_Bz].set(gam0 * as_ * Bz_i - cs_i * af * btz * sqrt_rho)
        L = L.at[idx_E].set(-gam0 * as_)
        L = 0.5 * L * inv_cs2
        L = jnp.where(mask_cs_ge_la, L * sgnBt, L)
        return L

    def row_5():
        L = jnp.zeros((N_vars, Nx, Ny, Nz))
        L = L.at[idx_D].set(btz * vy_i - bty * vz_i)
        L = L.at[idx_Mx].set(0.0)
        L = L.at[idx_My].set(-btz)
        L = L.at[idx_Mz].set(bty)
        L = L.at[idx_By].set(btz * sgnBx * sqrt_rho)
        L = L.at[idx_Bz].set(-bty * sgnBx * sqrt_rho)
        L = L.at[idx_E].set(0.0)
        L = 0.5 * L
        return L

    def row_6():
        L = jnp.zeros((N_vars, Nx, Ny, Nz))
        L = L.at[idx_D].set(af * (gam1 * vv2_i - lf_i * vx_i) + as_ * ls_i * (bty * vy_i + btz * vz_i) * sgnBx)
        L = L.at[idx_Mx].set(af * (gam0 * vx_i + lf_i))
        L = L.at[idx_My].set(gam0 * af * vy_i - as_ * ls_i * bty * sgnBx)
        L = L.at[idx_Mz].set(gam0 * af * vz_i - as_ * ls_i * btz * sgnBx)
        L = L.at[idx_By].set(gam0 * af * By_i + cs_i * as_ * bty * sqrt_rho)
        L = L.at[idx_Bz].set(gam0 * af * Bz_i + cs_i * as_ * btz * sqrt_rho)
        L = L.at[idx_E].set(-gam0 * af)
        L = 0.5 * L * inv_cs2
        L = jnp.where(~mask_cs_ge_la, L * sgnBt, L)
        return L

    L = jax.lax.switch(row, [row_0, row_1, row_2, row_3, row_4, row_5, row_6])

    return L

@partial(jax.jit, static_argnames=["registered_variables"])
def _eigen_all_lambdas(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    
    # registry indices (conserved ordering)
    DI = registered_variables.density_index
    MX = registered_variables.momentum_index.x
    MY = registered_variables.momentum_index.y
    MZ = registered_variables.momentum_index.z
    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z
    IE = registered_variables.energy_index

    # unpack conserved variables (centers)
    DD = conserved_state[DI]    # density
    Mx = conserved_state[MX]
    My = conserved_state[MY]
    Mz = conserved_state[MZ]
    Bx = conserved_state[BX]
    By = conserved_state[BY]
    Bz = conserved_state[BZ]
    EE = conserved_state[IE]

    # compute primitives (like Fortran primit)
    rho = DD
    vx = Mx / rho
    vy = My / rho
    vz = Mz / rho

    vv2 = vx * vx + vy * vy + vz * vz
    BB2 = Bx * Bx + By * By + Bz * Bz

    pg = (gamma - 1.0) * (EE - 0.5 * (rho * vv2 + BB2))

    mask_bad = (rho < rhomin) | (pg < pgmin)
    rho = jnp.where(mask_bad, jnp.maximum(rho, rhomin), rho)
    pg = jnp.where(mask_bad, jnp.maximum(pg, pgmin), pg)
    EE = jnp.where(mask_bad, pg / (gamma - 1.0) + 0.5 * (rho * vv2 + BB2), EE)

    # center-derived quantities for eigenvalues (center block, matches Fortran)
    bbn2 = BB2 / rho
    bnx2 = (Bx * Bx) / rho

    cs2_center = jnp.maximum(0.0, gamma * jnp.abs(pg / rho))

    disc_c = (bbn2 + cs2_center) ** 2 - 4.0 * bnx2 * cs2_center
    root_c = jnp.sqrt(jnp.maximum(EPSILON, disc_c))

    lf_c = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2 + cs2_center + root_c)))
    la_c = jnp.sqrt(jnp.maximum(EPSILON, bnx2))
    ls_c = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2 + cs2_center - root_c)))

    return jnp.stack([
        vx - lf_c,
        vx - la_c,
        vx - ls_c,
        vx,
        vx + ls_c,
        vx + la_c,
        vx + lf_c
    ], axis=0)

def _eigen_lambdas(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
    mode: int,
):
    
    # registry indices (conserved ordering)
    DI = registered_variables.density_index
    MX = registered_variables.momentum_index.x
    MY = registered_variables.momentum_index.y
    MZ = registered_variables.momentum_index.z
    BX = registered_variables.magnetic_index.x
    BY = registered_variables.magnetic_index.y
    BZ = registered_variables.magnetic_index.z
    IE = registered_variables.energy_index

    # unpack conserved variables (centers)
    DD = conserved_state[DI]    # density
    Mx = conserved_state[MX]
    My = conserved_state[MY]
    Mz = conserved_state[MZ]
    Bx = conserved_state[BX]
    By = conserved_state[BY]
    Bz = conserved_state[BZ]
    EE = conserved_state[IE]

    # compute primitives (like Fortran primit)
    rho = DD
    vx = Mx / rho
    vy = My / rho
    vz = Mz / rho

    vv2 = vx * vx + vy * vy + vz * vz
    BB2 = Bx * Bx + By * By + Bz * Bz

    pg = (gamma - 1.0) * (EE - 0.5 * (rho * vv2 + BB2))

    mask_bad = (rho < rhomin) | (pg < pgmin)
    rho = jnp.where(mask_bad, jnp.maximum(rho, rhomin), rho)
    pg = jnp.where(mask_bad, jnp.maximum(pg, pgmin), pg)
    EE = jnp.where(mask_bad, pg / (gamma - 1.0) + 0.5 * (rho * vv2 + BB2), EE)

    # center-derived quantities for eigenvalues (center block, matches Fortran)
    bbn2 = BB2 / rho
    bnx2 = (Bx * Bx) / rho

    cs2_center = jnp.maximum(0.0, gamma * jnp.abs(pg / rho))

    disc_c = (bbn2 + cs2_center) ** 2 - 4.0 * bnx2 * cs2_center
    root_c = jnp.sqrt(jnp.maximum(EPSILON, disc_c))

    lf_c = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2 + cs2_center + root_c)))
    la_c = jnp.sqrt(jnp.maximum(EPSILON, bnx2))
    ls_c = jnp.sqrt(jnp.maximum(EPSILON, 0.5 * (bbn2 + cs2_center - root_c)))

    def mode_0():
        return vx - lf_c
    
    def mode_1():
        return vx - la_c
    
    def mode_2():
        return vx - ls_c
    
    def mode_3():
        return vx
    
    def mode_4():
        return vx + ls_c
    
    def mode_5():
        return vx + la_c
    
    def mode_6():
        return vx + lf_c

    return jax.lax.switch(mode, [mode_0, mode_1, mode_2, mode_3, mode_4, mode_5, mode_6])