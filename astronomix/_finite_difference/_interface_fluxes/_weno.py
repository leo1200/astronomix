"""
Here we calculate weighted essentially non-oscillatory 
(WENO) fluxes for the MHD equations.

The idea of WENO is to find interface fluxes by interpolating
the cell centered fluxes using several stencils, and then
weighting the stencils based on their smoothness.

The reconstruction is done in characteristic variables to
better capture the underlying wave structure. At each interface,
we compute the eigenstructure (evaluated at the average of the
left and right states), and project all stencil
characteristic space.

Consider the interface at i + 1/2. Our vector of conserved
variables is q = (rho, rho*v_x, rho*v_y, rho*v_z, B_x, B_y, B_z, E)^T
with N_vars = 8 variables. In the eigenstructure of the MHD equations,
we have N_char = 7 characteristic waves.

We calculate the flux as follows:

1. We retrieve the eigenstructure given by the right 
   and left eigenvector matrices R_{i+1/2} \in R^{N_vars x N_char} 
   and L_{i+1/2} \in R^{N_char x N_vars}, as well 
   as the eigenvalues lambda at 
   q_{i+1/2} ~ 0.5 * (q_i + q_{i+1}).

2. In the stencil m = i - 2, ..., i + 2, we project the fluxes
   F_m and conserved variables q_m into characteristic space:
   F_s_m = L^s_{i+1/2} * F_m, q_s_m = L^s_{i+1/2} * q_m, where L^s_{i+1/2}
   is the s-th row of L so F_s_m and q_s_m are scalar fields. All
   fluxes and conserved variables in the stencil m = i - 2, ..., i + 2
   are projected using the same L^s_{i+1/2} at the interface i + 1/2.

3. We compute the differences ΔF_s_{m+1/2} = F_s_{m+1} - F_s_m and
   Δq_s_{m+1/2} = q_s_{m+1} - q_s_m for m = i - 2, ..., i + 1.

4. We use local Lax-Friedrichs flux splitting to split the fluxes
   into F_s^+ and F_s^- such that \partial_u F_s^+ only has non-negative
   eigenvalues, and \partial_u F_s^- only has non-positive eigenvalues.
   Both can then be properly upwinded with skewed stencils (for F_s^+ we
   use a left-biased stencil, for F_s^- we use a right-biased stencil, 
   see step 5).

   ΔF_s^+_{m+1/2} = 0.5 * (ΔF_s_{m+1/2} + alpha^s * Δq_s_{m+1/2}), m = i - 2, ..., i + 1
   ΔF_s^-_{m+1/2} = 0.5 * (ΔF_s_{m+1/2} - alpha^s * Δq_s_{m+1/2}), m = i - 1, ..., i + 2

   where alpha^s = max(|lambda^s_m|) over the 
   stencil m = i - 2, ..., i + 3.

5. We can compactly write the WENO flux reconstruction as:
   
   F_{i+1/2} = 1/12 * (-F_{i-1} + 7*F_i + 7*F_{i+1} - F_{i+2})
                +sum_{s = 1}^{N_char} [
                    -\phi(ΔF_s^+_{i-3/2}, ΔF_s^+_{i-1/2}, ΔF_s^+_{i+1/2}, ΔF_s^+_{i+3/2})
                    +\phi(ΔF_s^-_{i+5/2}, ΔF_s^-_{i+3/2}, ΔF_s^-_{i+1/2}, ΔF_s^-_{i-1/2})
                ] * R^s_{i+1/2}
    
    where R^s_{i+1/2} is the s-th column of R at the interface i + 1/2,
    and \phi is the WENO interpolant function given by:

    \phi(a, b, c, d) = 1/3 ω_0 (a - 2b + c) + 1/6 (ω_2 - 1/2) (b - 2c + d)

    with weight functions:
    
    ω_0 = α_0 / (α_0 + α_1 + α_2)
    ω_2 = α_2 / (α_0 + α_1 + α_2)

    α_0 = 1 / (ε + IS_0)^2
    α_1 = 6 / (ε + IS_1)^2
    α_2 = 3 / (ε + IS_2)^2

    and smoothness indicators:

    IS_0 = 13 (a - b)^2 + 3 (a - 3b)^2
    IS_1 = 13 (b - c)^2 + 3 (b + c)^2
    IS_2 = 13 (c - d)^2 + 3 (3c - d)^2

    ε is a small parameter to avoid 
    division by zero, here taken as 1e-8.

NOTE: I have seen formulations where the first part of the flux
is also calculated in characteristic space and then transformed back,
but I found that at single precision this introduces small perturbations
as RL is not exactly the identity matrix by finite precision effects.

For literature references, see:

 - High Order ENO and WENO Schemes for Computational Fluid Dynamics by Chi-Wang Shu (1997)
   (https://doi.org/10.1007/978-3-662-03882-6_5)

Concretely we implement the 5th-order WENO scheme as described in 

- HOW-MHD: A High-Order WENO-Based Magnetohydrodynamic Code with a High-Order 
  Constrained Transport Algorithm for Astrophysical Applications by Seo & Ryu 2023
  (https://arxiv.org/abs/2304.04360)
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Union

from jax import checkpoint

from astronomix._finite_difference._fluid_equations._eigen import _eigen_L_row, _eigen_R_col, _eigen_lambdas
from astronomix._finite_difference._fluid_equations._fluxes import _mhd_flux_x
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_x(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    WENO flux reconstruction.
    """

    epsilon = 1e-8

    # retrieve the center fluxes
    F = _mhd_flux_x(conserved_state, rhomin, pgmin, gamma, config, registered_variables)
    
    # with this we can already compute the first part of the flux
    F_interface = 1/12 * (
        -jnp.roll(F, 1, axis=1) + 7 * F + 7 * jnp.roll(F, -1, axis=1) - jnp.roll(F, -2, axis=1)
    )

    def mode_flux(mode, F_current):

        # get eigenstructure for this mode
        lambdas_center = _eigen_lambdas(conserved_state, rhomin, pgmin, gamma, registered_variables, mode)
        L_row = _eigen_L_row(conserved_state, rhomin, pgmin, gamma, registered_variables, mode)

        F0 = jnp.roll(F,  2, axis=1)   # shape (N_vars, Nx, Ny, Nz) — i-2 at target i
        F1 = jnp.roll(F,  1, axis=1)   # i-1
        F2 = F                         # i
        F3 = jnp.roll(F, -1, axis=1)   # i+1
        F4 = jnp.roll(F, -2, axis=1)   # i+2
        F5 = jnp.roll(F, -3, axis=1)   # i+3

        s0 = jnp.einsum('nxyz,nxyz->xyz', L_row, F0)
        s1 = jnp.einsum('nxyz,nxyz->xyz', L_row, F1)
        s2 = jnp.einsum('nxyz,nxyz->xyz', L_row, F2)
        s3 = jnp.einsum('nxyz,nxyz->xyz', L_row, F3)
        s4 = jnp.einsum('nxyz,nxyz->xyz', L_row, F4)
        s5 = jnp.einsum('nxyz,nxyz->xyz', L_row, F5)

        q0 = jnp.einsum('nxyz,nxyz->xyz', L_row, jnp.roll(conserved_state, 2, axis=1))
        q1 = jnp.einsum('nxyz,nxyz->xyz', L_row, jnp.roll(conserved_state, 1, axis=1))
        q2 = jnp.einsum('nxyz,nxyz->xyz', L_row, conserved_state)
        q3 = jnp.einsum('nxyz,nxyz->xyz', L_row, jnp.roll(conserved_state, -1, axis=1))
        q4 = jnp.einsum('nxyz,nxyz->xyz', L_row, jnp.roll(conserved_state, -2, axis=1))
        q5 = jnp.einsum('nxyz,nxyz->xyz', L_row, jnp.roll(conserved_state, -3, axis=1))

        # dFsk identical to original: d0 = s1 - s0, d1 = s2 - s1, ...
        d0 = s1 - s0
        d1 = s2 - s1
        d2 = s3 - s2
        d3 = s4 - s3
        d4 = s5 - s4

        dq0 = q1 - q0
        dq1 = q2 - q1
        dq2 = q3 - q2
        dq3 = q4 - q3
        dq4 = q5 - q4

        # compute amx over the same stencil (take abs then max over the six entries)
        lam0 = jnp.roll(lambdas_center,  2, axis=0)
        lam1 = jnp.roll(lambdas_center,  1, axis=0)
        lam2 = lambdas_center
        lam3 = jnp.roll(lambdas_center, -1, axis=0)
        lam4 = jnp.roll(lambdas_center, -2, axis=0)
        lam5 = jnp.roll(lambdas_center, -3, axis=0)
        lam_stack = jnp.stack([lam0, lam1, lam2, lam3, lam4, lam5], axis=0)
        amx = jnp.max(jnp.abs(lam_stack), axis=0)

        # Now use the exact same definitions as original for aterm/bterm/cterm/dterm
        aterm_p = 0.5 * (d0 + amx * dq0)
        bterm_p = 0.5 * (d1 + amx * dq1)
        cterm_p = 0.5 * (d2 + amx * dq2)
        dterm_p = 0.5 * (d3 + amx * dq3)

        IS0_p = 13.0 * (aterm_p - bterm_p)**2 + 3.0 * (aterm_p - 3.0*bterm_p)**2
        IS1_p = 13.0 * (bterm_p - cterm_p)**2 + 3.0 * (bterm_p + cterm_p)**2
        IS2_p = 13.0 * (cterm_p - dterm_p)**2 + 3.0 * (3.0*cterm_p - dterm_p)**2

        alpha0_p = 1.0 / (epsilon + IS0_p)**2
        alpha1_p = 6.0 / (epsilon + IS1_p)**2
        alpha2_p = 3.0 / (epsilon + IS2_p)**2

        alpha_sum_p = alpha0_p + alpha1_p + alpha2_p
        alpha_sum_p = jnp.maximum(alpha_sum_p, 1e-14)  # prevent division by zero

        omega0_p = alpha0_p / alpha_sum_p
        omega2_p = alpha2_p / alpha_sum_p

        second = (omega0_p * (aterm_p - 2.0*bterm_p + cterm_p) / 3.0
                  + (omega2_p - 0.5) * (bterm_p - 2.0*cterm_p + dterm_p) / 6.0)

        # Backward WENO similarly with the matching stencil differences:
        aterm_m = 0.5 * (d4 - amx * dq4)   # corresponds to original dFsk[4] etc.
        bterm_m = 0.5 * (d3 - amx * dq3)
        cterm_m = 0.5 * (d2 - amx * dq2)
        dterm_m = 0.5 * (d1 - amx * dq1)

        IS0_m = 13.0 * (aterm_m - bterm_m)**2 + 3.0 * (aterm_m - 3.0*bterm_m)**2
        IS1_m = 13.0 * (bterm_m - cterm_m)**2 + 3.0 * (bterm_m + cterm_m)**2
        IS2_m = 13.0 * (cterm_m - dterm_m)**2 + 3.0 * (3.0*cterm_m - dterm_m)**2

        alpha0_m = 1.0 / (epsilon + IS0_m)**2
        alpha1_m = 6.0 / (epsilon + IS1_m)**2
        alpha2_m = 3.0 / (epsilon + IS2_m)**2

        alpha_sum_m = alpha0_m + alpha1_m + alpha2_m
        alpha_sum_m = jnp.maximum(alpha_sum_m, 1e-14)  # prevent division by zero

        omega0_m = alpha0_m / alpha_sum_m
        omega2_m = alpha2_m / alpha_sum_m

        third = (omega0_m * (aterm_m - 2.0*bterm_m + cterm_m) / 3.0
                 + (omega2_m - 0.5) * (bterm_m - 2.0*cterm_m + dterm_m) / 6.0)

        Fs = -second + third

        # transform back and add to current flux
        R_col = _eigen_R_col(conserved_state, rhomin, pgmin, gamma, registered_variables, mode)
        dF = jnp.einsum('nxyz,xyz->nxyz', R_col, Fs)
        return F_current + dF
    
    return jax.lax.fori_loop(
        0, 7,
        mode_flux,
        F_interface
    )

@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_y(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    # Transpose to make y the "x" direction
    qy = jnp.transpose(conserved_state, (0, 2, 1, 3))
    
    # Swap components
    momentum_x = qy[registered_variables.momentum_index.x]
    momentum_y = qy[registered_variables.momentum_index.y]
    B_x = qy[registered_variables.magnetic_index.x]
    B_y = qy[registered_variables.magnetic_index.y]
    
    qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
    qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)
    qy = qy.at[registered_variables.magnetic_index.x].set(B_y)
    qy = qy.at[registered_variables.magnetic_index.y].set(B_x)
    
    Fy = _weno_flux_x(qy, rhomin, pgmin, gamma, config, registered_variables)
    
    # Transpose back
    Fy = jnp.transpose(Fy, (0, 2, 1, 3))
    
    # Swap components back
    Fmomentum_x = Fy[registered_variables.momentum_index.x]
    Fmomentum_y = Fy[registered_variables.momentum_index.y]
    FB_x = Fy[registered_variables.magnetic_index.x]
    FB_y = Fy[registered_variables.magnetic_index.y]
    
    Fy = Fy.at[registered_variables.momentum_index.x].set(Fmomentum_y)
    Fy = Fy.at[registered_variables.momentum_index.y].set(Fmomentum_x)
    Fy = Fy.at[registered_variables.magnetic_index.x].set(FB_y)
    Fy = Fy.at[registered_variables.magnetic_index.y].set(FB_x)
    
    return Fy


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_z(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    pgmin: Union[float, jnp.ndarray],
    gamma: Union[float, jnp.ndarray],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    # Transpose to make z the "x" direction
    qz = jnp.transpose(conserved_state, (0, 3, 2, 1))
    
    # Swap components
    momentum_x = qz[registered_variables.momentum_index.x]
    momentum_z = qz[registered_variables.momentum_index.z]
    B_x = qz[registered_variables.magnetic_index.x]
    B_z = qz[registered_variables.magnetic_index.z]
    
    qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
    qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)
    qz = qz.at[registered_variables.magnetic_index.x].set(B_z)
    qz = qz.at[registered_variables.magnetic_index.z].set(B_x)
    
    Fz = _weno_flux_x(qz, rhomin, pgmin, gamma, config, registered_variables)
    
    # Transpose back
    Fz = jnp.transpose(Fz, (0, 3, 2, 1))
    
    # Swap components back
    Fmomentum_x = Fz[registered_variables.momentum_index.x]
    Fmomentum_z = Fz[registered_variables.momentum_index.z]
    FB_x = Fz[registered_variables.magnetic_index.x]
    FB_z = Fz[registered_variables.magnetic_index.z]
    
    Fz = Fz.at[registered_variables.momentum_index.x].set(Fmomentum_z)
    Fz = Fz.at[registered_variables.momentum_index.z].set(Fmomentum_x)
    Fz = Fz.at[registered_variables.magnetic_index.x].set(FB_z)
    Fz = Fz.at[registered_variables.magnetic_index.z].set(FB_x)
    
    return Fz

# also seems to be slower
# @partial(jax.jit, static_argnames=["registered_variables"])
# def _weno_flux_x(
#     conserved_state,
#     rhomin: Union[float, jnp.ndarray],
#     pgmin: Union[float, jnp.ndarray],
#     gamma: Union[float, jnp.ndarray],
#     registered_variables: RegisteredVariables,
# ):
#     """
#     WENO flux reconstruction.
#     """
    
#     q = conserved_state
#     epsilon = 1e-8
    
#     # Get eigenstructure
#     lambdas_center, R, L = _eigen_x(q, rhomin, pgmin, gamma, registered_variables)
    
#     # Get physical fluxes at cell centers
#     F = _mhd_flux_x(q, gamma, registered_variables)

#     F_stencil = jnp.stack([
#         jnp.roll(F, 2, axis=1),   # i-2
#         jnp.roll(F, 1, axis=1),   # i-1
#         F,                        # i
#         jnp.roll(F, -1, axis=1),  # i+1
#         jnp.roll(F, -2, axis=1),  # i+2
#         jnp.roll(F, -3, axis=1),  # i+3
#     ], axis=0)
    
#     q_stencil = jnp.stack([
#         jnp.roll(q, 2, axis=1),
#         jnp.roll(q, 1, axis=1),
#         q,
#         jnp.roll(q, -1, axis=1),
#         jnp.roll(q, -2, axis=1),
#         jnp.roll(q, -3, axis=1),
#     ], axis=0)
    
#     # Get maximum eigenvalue over stencil
#     lambda_stencil = jnp.stack([
#         jnp.roll(lambdas_center, 2, axis=1),
#         jnp.roll(lambdas_center, 1, axis=1),
#         lambdas_center,
#         jnp.roll(lambdas_center, -1, axis=1),
#         jnp.roll(lambdas_center, -2, axis=1),
#         jnp.roll(lambdas_center, -3, axis=1),
#     ], axis=0)
    
#     amx = jnp.max(jnp.abs(lambda_stencil), axis=0)  # Shape: (7, Nx, Ny, Nz)
    
#     # Transform stencil to characteristic variables
#     # L has shape (7, N_vars, Nx, Ny, Nz) at interfaces
#     # F_stencil: (6, N_vars, Nx, Ny, Nz)
#     # Result: (6, 7, Nx, Ny, Nz)
#     Fsk = jnp.einsum('mnxyz,snxyz->smxyz', L, F_stencil)
#     qsk = jnp.einsum('mnxyz,snxyz->smxyz', L, q_stencil)
    
#     # Compute differences
#     dFsk = Fsk[1:, ...] - Fsk[:-1, ...]  # (5, 7, Nx, Ny, Nz)
#     dqsk = qsk[1:, ...] - qsk[:-1, ...]
    
#     # Forward WENO (positive flux)
#     aterm_p = 0.5 * (dFsk[0, ...] + amx * dqsk[0, ...])
#     bterm_p = 0.5 * (dFsk[1, ...] + amx * dqsk[1, ...])
#     cterm_p = 0.5 * (dFsk[2, ...] + amx * dqsk[2, ...])
#     dterm_p = 0.5 * (dFsk[3, ...] + amx * dqsk[3, ...])
    
#     IS0_p = 13.0 * (aterm_p - bterm_p)**2 + 3.0 * (aterm_p - 3.0*bterm_p)**2
#     IS1_p = 13.0 * (bterm_p - cterm_p)**2 + 3.0 * (bterm_p + cterm_p)**2
#     IS2_p = 13.0 * (cterm_p - dterm_p)**2 + 3.0 * (3.0*cterm_p - dterm_p)**2
    
#     alpha0_p = 1.0 / (epsilon + IS0_p)**2
#     alpha1_p = 6.0 / (epsilon + IS1_p)**2
#     alpha2_p = 3.0 / (epsilon + IS2_p)**2
    
#     alpha_sum_p = alpha0_p + alpha1_p + alpha2_p
#     alpha_sum_p = jnp.maximum(alpha_sum_p, 1e-14)  # prevent division by zero

#     omega0_p = alpha0_p / alpha_sum_p
#     omega2_p = alpha2_p / alpha_sum_p
    
#     second = (omega0_p * (aterm_p - 2.0*bterm_p + cterm_p) / 3.0 
#               + (omega2_p - 0.5) * (bterm_p - 2.0*cterm_p + dterm_p) / 6.0)
    
#     # Backward WENO (negative flux)
#     aterm_m = 0.5 * (dFsk[4, ...] - amx * dqsk[4, ...])
#     bterm_m = 0.5 * (dFsk[3, ...] - amx * dqsk[3, ...])
#     cterm_m = 0.5 * (dFsk[2, ...] - amx * dqsk[2, ...])
#     dterm_m = 0.5 * (dFsk[1, ...] - amx * dqsk[1, ...])
    
#     IS0_m = 13.0 * (aterm_m - bterm_m)**2 + 3.0 * (aterm_m - 3.0*bterm_m)**2
#     IS1_m = 13.0 * (bterm_m - cterm_m)**2 + 3.0 * (bterm_m + cterm_m)**2
#     IS2_m = 13.0 * (cterm_m - dterm_m)**2 + 3.0 * (3.0*cterm_m - dterm_m)**2
    
#     alpha0_m = 1.0 / (epsilon + IS0_m)**2
#     alpha1_m = 6.0 / (epsilon + IS1_m)**2
#     alpha2_m = 3.0 / (epsilon + IS2_m)**2
    
#     alpha_sum_m = alpha0_m + alpha1_m + alpha2_m
#     alpha_sum_m = jnp.maximum(alpha_sum_m, 1e-14)  # prevent division by zero

#     omega0_m = alpha0_m / alpha_sum_m
#     omega2_m = alpha2_m / alpha_sum_m
    
#     third = (omega0_m * (aterm_m - 2.0*bterm_m + cterm_m) / 3.0 
#              + (omega2_m - 0.5) * (bterm_m - 2.0*cterm_m + dterm_m) / 6.0)
    
#     # Combine
#     Fs = - second + third  # Shape: (7, Nx, Ny, Nz)
    
#     # Transform back to physical variables
#     # R has shape (N_vars, 7, Nx, Ny, Nz)
#     dF = jnp.einsum('nmxyz,mxyz->nxyz', R, Fs)  # Shape: (N_vars, Nx, Ny, Nz)

#     # Base 4th-order flux
#     # first = (-Fsk[1, ...] + 7.0*Fsk[2, ...] + 7.0*Fsk[3, ...] - Fsk[4, ...]) / 12.0
#     first = 1/12 * (
#         -jnp.roll(F, 1, axis=1) + 7 * F + 7 * jnp.roll(F, -1, axis=1) - jnp.roll(F, -2, axis=1)
#     )

#     dF = first + dF

#     return dF