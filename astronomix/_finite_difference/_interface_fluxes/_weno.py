r"""
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

# general
from functools import partial

# typing
from typing import Union

# jax
import jax
import jax.numpy as jnp
from jax import checkpoint

# optional Pallas backend (absent on platforms without a Pallas/Triton build)
try:
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - optional backend
    pl = None

try:
    from jax.experimental.pallas import triton as pltriton
except Exception:  # pragma: no cover - optional backend
    pltriton = None

# astronomix constants
from astronomix.option_classes.simulation_config import BACKWARDS, IDEAL_GAS, ISOTHERMAL, PALLAS

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._eigen_hydro import _eigen_L_row_hydro, _eigen_R_col_hydro, _eigen_lambdas_hydro
from astronomix._fluid_equations._eigen_hydro_iso import _eigen_L_row_hydro_iso, _eigen_R_col_hydro_iso, _eigen_lambdas_hydro_iso
from astronomix._fluid_equations._eigen_mhd import _eigen_L_row, _eigen_R_col, _eigen_lambdas
from astronomix._fluid_equations._eigen_mhd_iso import _eigen_L_row_iso, _eigen_R_col_iso, _eigen_lambdas_iso
from astronomix._fluid_equations._fluxes_mhd import _euler_flux_isothermal_x, _mhd_flux_isothermal_x, _mhd_flux_x
from astronomix._fluid_equations._equations import primitive_state_from_conserved
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix._stencil_operations._stencil_operations import _shift


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_x_native(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    WENO flux reconstruction.
    """

    epsilon = 1e-7

    # only used in the IDEAL_GAS case
    rhomin = params.minimum_density
    pgmin = params.minimum_pressure
    gamma = params.gamma

    # only used in the ISOTHERMAL case
    isothermal_sound_speed = params.isothermal_sound_speed

    if config.equation_of_state == IDEAL_GAS:
        # retrieve the center fluxes
        if config.mhd:
            F = _mhd_flux_x(
                conserved_state,
                rhomin,
                pgmin,
                gamma,
                config,
                registered_variables
            )
        else:
            F =  _euler_flux(
                primitive_state_from_conserved(
                    conserved_state, gamma, config, registered_variables
                ),
                gamma, config, registered_variables, 1
            )
    elif config.equation_of_state == ISOTHERMAL:
        if config.mhd:
            F = _mhd_flux_isothermal_x(
                conserved_state,
                rhomin,
                isothermal_sound_speed,
                config,
                registered_variables,
            )
        else:
            F = _euler_flux_isothermal_x(
                conserved_state,
                rhomin,
                isothermal_sound_speed,
                config,
                registered_variables,
            )

        
    # with this we can already compute the first part of the flux
    F_interface = 1/12 * (
        -_shift(F, 1, axis=1) + 7 * F + 7 * _shift(F, -1, axis=1) - _shift(F, -2, axis=1)
    )

    def mode_flux(mode, F_current):

        # get eigenstructure for this mode
        if config.equation_of_state == IDEAL_GAS:
            if config.mhd:
                lambdas_center = _eigen_lambdas(conserved_state, rhomin, pgmin, gamma, registered_variables, mode)
                L_row = _eigen_L_row(conserved_state, rhomin, pgmin, gamma, registered_variables, mode)
            else:
                lambdas_center = _eigen_lambdas_hydro(conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode)
                L_row = _eigen_L_row_hydro(conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode)
        elif config.equation_of_state == ISOTHERMAL:
            if config.mhd:
                lambdas_center = _eigen_lambdas_iso(conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode)
                L_row = _eigen_L_row_iso(conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode)
            else:
                lambdas_center = _eigen_lambdas_hydro_iso(conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode)
                L_row = _eigen_L_row_hydro_iso(conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode)

        F0 = _shift(F,  2, axis=1)   # shape (N_vars, Nx, Ny, Nz) — i-2 at target i
        F1 = _shift(F,  1, axis=1)   # i-1
        F2 = F                         # i
        F3 = _shift(F, -1, axis=1)   # i+1
        F4 = _shift(F, -2, axis=1)   # i+2
        F5 = _shift(F, -3, axis=1)   # i+3

        if config.dimensionality == 3:
            s0 = jnp.einsum('nxyz,nxyz->xyz', L_row, F0)
            s1 = jnp.einsum('nxyz,nxyz->xyz', L_row, F1)
            s2 = jnp.einsum('nxyz,nxyz->xyz', L_row, F2)
            s3 = jnp.einsum('nxyz,nxyz->xyz', L_row, F3)
            s4 = jnp.einsum('nxyz,nxyz->xyz', L_row, F4)
            s5 = jnp.einsum('nxyz,nxyz->xyz', L_row, F5)

            q0 = jnp.einsum('nxyz,nxyz->xyz', L_row, _shift(conserved_state, 2, axis=1))
            q1 = jnp.einsum('nxyz,nxyz->xyz', L_row, _shift(conserved_state, 1, axis=1))
            q2 = jnp.einsum('nxyz,nxyz->xyz', L_row, conserved_state)
            q3 = jnp.einsum('nxyz,nxyz->xyz', L_row, _shift(conserved_state, -1, axis=1))
            q4 = jnp.einsum('nxyz,nxyz->xyz', L_row, _shift(conserved_state, -2, axis=1))
            q5 = jnp.einsum('nxyz,nxyz->xyz', L_row, _shift(conserved_state, -3, axis=1))
        elif config.dimensionality == 2:
            s0 = jnp.einsum('nxy,nxy->xy', L_row, F0)
            s1 = jnp.einsum('nxy,nxy->xy', L_row, F1)
            s2 = jnp.einsum('nxy,nxy->xy', L_row, F2)
            s3 = jnp.einsum('nxy,nxy->xy', L_row, F3)
            s4 = jnp.einsum('nxy,nxy->xy', L_row, F4)
            s5 = jnp.einsum('nxy,nxy->xy', L_row, F5)

            q0 = jnp.einsum('nxy,nxy->xy', L_row, _shift(conserved_state, 2, axis=1))
            q1 = jnp.einsum('nxy,nxy->xy', L_row, _shift(conserved_state, 1, axis=1))
            q2 = jnp.einsum('nxy,nxy->xy', L_row, conserved_state)
            q3 = jnp.einsum('nxy,nxy->xy', L_row, _shift(conserved_state, -1, axis=1))
            q4 = jnp.einsum('nxy,nxy->xy', L_row, _shift(conserved_state, -2, axis=1))
            q5 = jnp.einsum('nxy,nxy->xy', L_row, _shift(conserved_state, -3, axis=1))
        else:
            s0 = jnp.einsum('nx,nx->x', L_row, F0)
            s1 = jnp.einsum('nx,nx->x', L_row, F1)
            s2 = jnp.einsum('nx,nx->x', L_row, F2)
            s3 = jnp.einsum('nx,nx->x', L_row, F3)
            s4 = jnp.einsum('nx,nx->x', L_row, F4)
            s5 = jnp.einsum('nx,nx->x', L_row, F5)

            q0 = jnp.einsum('nx,nx->x', L_row, _shift(conserved_state, 2, axis=1))
            q1 = jnp.einsum('nx,nx->x', L_row, _shift(conserved_state, 1, axis=1))
            q2 = jnp.einsum('nx,nx->x', L_row, conserved_state)
            q3 = jnp.einsum('nx,nx->x', L_row, _shift(conserved_state, -1, axis=1))
            q4 = jnp.einsum('nx,nx->x', L_row, _shift(conserved_state, -2, axis=1))
            q5 = jnp.einsum('nx,nx->x', L_row, _shift(conserved_state, -3, axis=1))

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
        lam0 = _shift(lambdas_center,  2, axis=0)
        lam1 = _shift(lambdas_center,  1, axis=0)
        lam2 = lambdas_center
        lam3 = _shift(lambdas_center, -1, axis=0)
        lam4 = _shift(lambdas_center, -2, axis=0)
        lam5 = _shift(lambdas_center, -3, axis=0)
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
        if config.equation_of_state == IDEAL_GAS:
            if config.mhd:
                R_col = _eigen_R_col(conserved_state, rhomin, pgmin, gamma, registered_variables, mode)
            else:
                R_col = _eigen_R_col_hydro(conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode)
        elif config.equation_of_state == ISOTHERMAL:
            if config.mhd:
                R_col = _eigen_R_col_iso(conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode)
            else:
                R_col = _eigen_R_col_hydro_iso(conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode)

        if config.dimensionality == 3:
            dF = jnp.einsum('nxyz,xyz->nxyz', R_col, Fs)
        elif config.dimensionality == 2:
            dF = jnp.einsum('nxy,xy->nxy', R_col, Fs)
        else:
            dF = jnp.einsum('nx,x->nx', R_col, Fs)
        return F_current + dF
    
    if config.mhd:
        num_modes = 7
    else:
        num_modes = config.dimensionality + 2

    if config.equation_of_state == ISOTHERMAL:
        num_modes -= 1
    
    # I went for the for loop instead of one einsum
    # because of memory considerations (the full projection
    # matrix does not need to be materialized)
    # But probably this (as I originally had it)
    # would be faster (?).
    return jax.lax.fori_loop(
        0, num_modes,
        mode_flux,
        F_interface
    )

@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_y_native(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    WENO flux reconstruction in the y-direction.

    Reuses the x-direction kernel by transposing the state so that y becomes
    the leading spatial axis (and swapping the x/y momentum and magnetic
    components), running ``_weno_flux_x_native``, then undoing both the
    transpose and the component swap on the resulting flux.

    Args:
        conserved_state: The conserved state array.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The WENO interface fluxes in the y-direction.
    """

    # Transpose to make y the "x" direction
    if config.dimensionality == 2:
        qy = jnp.transpose(conserved_state, (0, 2, 1))
    elif config.dimensionality == 3:
        qy = jnp.transpose(conserved_state, (0, 2, 1, 3))
    
    # Swap components
    momentum_x = qy[registered_variables.momentum_index.x]
    momentum_y = qy[registered_variables.momentum_index.y]

    if config.mhd:
        B_x = qy[registered_variables.magnetic_index.x]
        B_y = qy[registered_variables.magnetic_index.y]
    
    qy = qy.at[registered_variables.momentum_index.x].set(momentum_y)
    qy = qy.at[registered_variables.momentum_index.y].set(momentum_x)

    if config.mhd:
        qy = qy.at[registered_variables.magnetic_index.x].set(B_y)
        qy = qy.at[registered_variables.magnetic_index.y].set(B_x)
    
    Fy = _weno_flux_x_native(qy, params, config, registered_variables)
    
    # Transpose back
    if config.dimensionality == 2:
        Fy = jnp.transpose(Fy, (0, 2, 1))
    elif config.dimensionality == 3:
        Fy = jnp.transpose(Fy, (0, 2, 1, 3))
    
    # Swap components back
    Fmomentum_x = Fy[registered_variables.momentum_index.x]
    Fmomentum_y = Fy[registered_variables.momentum_index.y]

    if config.mhd:
        FB_x = Fy[registered_variables.magnetic_index.x]
        FB_y = Fy[registered_variables.magnetic_index.y]
    
    Fy = Fy.at[registered_variables.momentum_index.x].set(Fmomentum_y)
    Fy = Fy.at[registered_variables.momentum_index.y].set(Fmomentum_x)

    if config.mhd:
        Fy = Fy.at[registered_variables.magnetic_index.x].set(FB_y)
        Fy = Fy.at[registered_variables.magnetic_index.y].set(FB_x)
    
    return Fy


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_z_native(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    WENO flux reconstruction in the z-direction.

    Reuses the x-direction kernel by transposing the state so that z becomes
    the leading spatial axis (and swapping the x/z momentum and magnetic
    components), running ``_weno_flux_x_native``, then undoing both the
    transpose and the component swap on the resulting flux.

    Args:
        conserved_state: The conserved state array.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The WENO interface fluxes in the z-direction.
    """

    # Transpose to make z the "x" direction
    qz = jnp.transpose(conserved_state, (0, 3, 2, 1))
    
    # Swap components
    momentum_x = qz[registered_variables.momentum_index.x]
    momentum_z = qz[registered_variables.momentum_index.z]

    if config.mhd:
        B_x = qz[registered_variables.magnetic_index.x]
        B_z = qz[registered_variables.magnetic_index.z]
    
    qz = qz.at[registered_variables.momentum_index.x].set(momentum_z)
    qz = qz.at[registered_variables.momentum_index.z].set(momentum_x)

    if config.mhd:
        qz = qz.at[registered_variables.magnetic_index.x].set(B_z)
        qz = qz.at[registered_variables.magnetic_index.z].set(B_x)
    
    Fz = _weno_flux_x_native(qz, params, config, registered_variables)
    
    # Transpose back
    Fz = jnp.transpose(Fz, (0, 3, 2, 1))
    
    # Swap components back
    Fmomentum_x = Fz[registered_variables.momentum_index.x]
    Fmomentum_z = Fz[registered_variables.momentum_index.z]

    if config.mhd:
        FB_x = Fz[registered_variables.magnetic_index.x]
        FB_z = Fz[registered_variables.magnetic_index.z]
    
    Fz = Fz.at[registered_variables.momentum_index.x].set(Fmomentum_z)
    Fz = Fz.at[registered_variables.momentum_index.z].set(Fmomentum_x)

    if config.mhd:
        Fz = Fz.at[registered_variables.magnetic_index.x].set(FB_z)
        Fz = Fz.at[registered_variables.magnetic_index.z].set(FB_x)
    
    return Fz


# -----------------------------------------------------------------------------
# Pallas backend symbols (full implementation lives in ``_weno_pallas.py``).
#
# Importing them at the *bottom* of this file lets ``_weno_pallas.py`` perform
# a lazy import of the ``_weno_flux_{x,y,z}_native`` symbols above without
# tripping a circular import: those names are already bound in this module's
# global namespace by the time ``_weno_pallas`` is loaded.  A developer who
# only touches native JAX never needs to look at the Pallas module.
# -----------------------------------------------------------------------------
from astronomix._finite_difference._interface_fluxes._weno_pallas import (  # noqa: E402
    _hydro_pallas_flux_supported,
    _mhd_iso_pallas_flux_supported,
    _mhd_pallas_flux_supported,
    _weno_flux_hydro_pallas,
    _weno_flux_hydro_pallas_rhs,
    _weno_flux_hydro_pallas_vjp_local,
    _weno_flux_mhd_iso_pallas,
    _weno_flux_mhd_pallas,
    _weno_flux_mhd_pallas_vjp_local,
)


from astronomix._pallas_helpers import diffable_pallas_call, pallas_vjp_call  # noqa: E402


def _weno_flux_native_for_axis(axis: int):
    """Return the native-JAX WENO flux function for the given spatial axis."""
    if axis == 0:
        return _weno_flux_x_native
    if axis == 1:
        return _weno_flux_y_native
    return _weno_flux_z_native


def _weno_flux_axis_dispatch(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int,
):
    """Pick the Pallas flux for the supported equation set, falling back to
    the native per-axis JAX flux.

    Forward-mode AD wraps the call through ``diffable_pallas_call`` (custom_jvp:
    Pallas primal, native tangent).  In reverse-mode (``differentiation_mode ==
    BACKWARDS``) the ideal-gas hydro path instead uses ``pallas_vjp_call`` so the
    backward stays on the GPU via the hand-derived explicit Pallas adjoint
    kernel, rather than transposing the native tangent.  The Pallas reverse path
    differentiates w.r.t. the conserved STATE only (params are physical
    constants for the flux) and is single-device; correct 3D y/z gradients need
    jax >= ~0.8 (older jaxlib miscompiles the adjoint kernel on Triton).

    The adjoint kernel is the exact transpose of the forward WENO kernel (whose
    stencil shifts are periodic ``custom_roll``), so it is correct for both
    periodic and ghost-cell boundaries — validated bit-exact (~1e-15) vs the
    native VJP for smooth states under either boundary handling.  At
    discontinuities (shocks) the WENO gradient is inherently FP-ill-conditioned
    and may differ from the native VJP by a sub-gradient amount, exactly as the
    native WENO AD does."""
    if _hydro_pallas_flux_supported(conserved_state, config):
        if axis == 0 or (axis == 1 and int(config.dimensionality) >= 2) or (axis == 2 and int(config.dimensionality) == 3):
            if config.differentiation_mode == BACKWARDS:
                return pallas_vjp_call(
                    conserved_state,
                    params,
                    pallas_forward=lambda s, p: _weno_flux_hydro_pallas(
                        s, p, config, registered_variables, axis=axis
                    ),
                    pallas_backward=lambda s, p, ct: _weno_flux_hydro_pallas_vjp_local(
                        s, ct, p, config, registered_variables, axis=axis
                    ),
                )
            pallas = lambda s, p: _weno_flux_hydro_pallas(  # noqa: E731
                s, p, config, registered_variables, axis=axis
            )
            native = lambda s, p: _weno_flux_native_for_axis(axis)(  # noqa: E731
                s, p, config, registered_variables
            )
            return diffable_pallas_call(
                conserved_state, params, pallas_branch=pallas, native_branch=native,
            )
    if _mhd_pallas_flux_supported(conserved_state, config):
        if config.differentiation_mode == BACKWARDS:
            # Reverse mode: keep the backward on the GPU via the explicit Pallas
            # adjoint (in-kernel jax.vjp of the shared MHD window), exactly like
            # the hydro path, instead of transposing the native tangent.  3D,
            # all axes; state-only; single-device — the inverse-problem regime.
            return pallas_vjp_call(
                conserved_state,
                params,
                pallas_forward=lambda s, p: _weno_flux_mhd_pallas(
                    s, p, config, registered_variables, axis=axis
                ),
                pallas_backward=lambda s, p, ct: _weno_flux_mhd_pallas_vjp_local(
                    s, ct, p, config, registered_variables, axis=axis
                ),
            )
        pallas = lambda s, p: _weno_flux_mhd_pallas(  # noqa: E731
            s, p, config, registered_variables, axis=axis
        )
        native = lambda s, p: _weno_flux_native_for_axis(axis)(  # noqa: E731
            s, p, config, registered_variables
        )
        return diffable_pallas_call(
            conserved_state, params, pallas_branch=pallas, native_branch=native,
        )
    if _mhd_iso_pallas_flux_supported(conserved_state, config):
        pallas = lambda s, p: _weno_flux_mhd_iso_pallas(  # noqa: E731
            s, p, config, registered_variables, axis=axis
        )
        native = lambda s, p: _weno_flux_native_for_axis(axis)(  # noqa: E731
            s, p, config, registered_variables
        )
        return diffable_pallas_call(
            conserved_state, params, pallas_branch=pallas, native_branch=native,
        )
    return _weno_flux_native_for_axis(axis)(
        conserved_state, params, config, registered_variables
    )


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_x(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """WENO interface flux in the x-direction (Pallas backend where supported,
    native JAX otherwise)."""
    return _weno_flux_axis_dispatch(
        conserved_state, params, config, registered_variables, axis=0,
    )


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_y(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """WENO interface flux in the y-direction (Pallas backend where supported,
    native JAX otherwise)."""
    return _weno_flux_axis_dispatch(
        conserved_state, params, config, registered_variables, axis=1,
    )


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_z(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """WENO interface flux in the z-direction (Pallas backend where supported,
    native JAX otherwise)."""
    return _weno_flux_axis_dispatch(
        conserved_state, params, config, registered_variables, axis=2,
    )
