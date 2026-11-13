r"""
Here we calculate weighted essentially non-oscillatory
(WENO) fluxes for the MHD equations.

The idea of WENO is to find interface fluxes by interpolating
the cell centered fluxes using several stencils, and then
weighting the stencils based on their smoothness.

The reconstruction is done in characteristic variables to
better capture the underlying wave structure. At each interface,
we compute the eigenstructure (evaluated at the average of the
left and right states), and project all stencil values into
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

    ε is a small parameter to avoid division by zero (``WENO_EPSILON``).

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

The per-axis dispatchers at the bottom of the module run the Pallas kernels of
``_weno_pallas.py`` where they support the equation set and fall back to the
native-JAX reconstruction above otherwise.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    IDEAL_GAS,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._eigen_hydro import (
    _eigen_L_row_hydro,
    _eigen_R_col_hydro,
    _eigen_lambdas_hydro,
)
from astronomix._fluid_equations._eigen_hydro_baro import (
    _eigen_L_row_hydro_iso,
    _eigen_R_col_hydro_iso,
    _eigen_lambdas_hydro_iso,
)
from astronomix._fluid_equations._eigen_mhd import (
    _eigen_L_row,
    _eigen_R_col,
    _eigen_lambdas,
)
from astronomix._fluid_equations._eigen_mhd_baro import (
    _eigen_L_row_iso,
    _eigen_R_col_iso,
    _eigen_lambdas_iso,
)
from astronomix._fluid_equations._equation_of_state import (
    barotropic_law,
    is_barotropic,
)
from astronomix._fluid_equations._fluxes_mhd import (
    _euler_flux_barotropic_x,
    _mhd_flux_barotropic_x,
    _mhd_flux_x,
)
from astronomix._fluid_equations._equations import primitive_state_from_conserved
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix._fluid_equations._dual_energy_switch import DUAL_ENERGY_ETA
from astronomix._finite_difference._interface_fluxes._weno_weights import (
    WENO_EPSILON,
    _weno_omega_weights_ad,
    _weno_omega_weights_z,
)
from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    mass_free_modes,
    positivity_preserving_interface_flux,
    stencil_maximum,
)
from astronomix._pallas_helpers import (
    _pallas_call_sharded,
    diffable_pallas_call,
    diffable_pallas_call_n,
    pallas_vjp_call,
)

# astronomix functions (Pallas backend; ``_weno_pallas.py`` imports the native
# fluxes of this module only inside its functions, so there is no import cycle)
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _hydro_pallas_flux_supported,
    _mhd_iso_pallas_flux_supported,
    _mhd_pallas_flux_supported,
    _weno_flux_hydro_pallas,
    _weno_flux_hydro_pallas_vjp,
    _weno_flux_mhd_iso_pallas,
    _weno_flux_mhd_pallas,
    _weno_flux_mhd_pallas_vjp,
)


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_x_native(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
    inflow_reference=None,
):
    """
    WENO flux reconstruction along x (the first spatial axis), native JAX.

    Args:
        conserved_state: The conserved state array.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The dual-energy ``g``, aligned with
            ``conserved_state`` along the active axis, or None. When given, it
            is used by the cell-centred pressure recovery in both the physical
            flux and the eigenstructure, so the reconstruction never sees the
            cancellation-corrupted pressure.
        inflow_reference: Ideal MHD with ``weno_positivity_preserving``: the
            ``(B, sum S)`` pair of ``mhd_inflow_reference`` in this sweep's
            layout, for the joint limiting of each cell's inflow faces; None
            otherwise.

    Returns:
        The WENO interface fluxes at i + 1/2, aligned with cell i.
    """
    dual_eta = DUAL_ENERGY_ETA

    epsilon = WENO_EPSILON
    # ``_weno_omega_weights_ad`` gives the same weights bit for bit as the
    # plain Jiang-Shu weights, with an overflow-free derivative for the
    # exact-weight tangent (see its docstring).
    omega_weights = _weno_omega_weights_z if config.solver_config.weno_z else _weno_omega_weights_ad
    positivity_preserving = config.solver_config.weno_positivity_preserving
    admissible_face_state = config.solver_config.weno_admissible_face_state
    if config.solver_config.weno_ad_frozen_weights:
        unfrozen_omega_weights = omega_weights

        def omega_weights(*args):
            return tuple(jax.lax.stop_gradient(w) for w in unfrozen_omega_weights(*args))

    # The floors and gamma are only used for an ideal gas.
    minimum_density = params.minimum_density
    minimum_pressure = params.minimum_pressure
    gamma = params.gamma

    # The law of a barotropic gas: the isothermal sound speed, or the
    # polytropic K and Gamma (the eigen-systems take the local sound speed).
    law = barotropic_law(params, config) if is_barotropic(config) else None

    # -------------------------------------------------------------
    # ================= ↓ Cell-centred fluxes ↓ ===================
    # -------------------------------------------------------------

    if config.equation_of_state == IDEAL_GAS:
        if config.mhd:
            F = _mhd_flux_x(
                conserved_state,
                minimum_density,
                minimum_pressure,
                gamma,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
            )
        else:
            F = _euler_flux(
                primitive_state_from_conserved(
                    conserved_state,
                    gamma,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy_density,
                ),
                gamma,
                config,
                registered_variables,
                1,
            )
    else:
        if config.mhd:
            F = _mhd_flux_barotropic_x(
                conserved_state,
                minimum_density,
                law,
                config,
                registered_variables,
            )
        else:
            F = _euler_flux_barotropic_x(
                conserved_state,
                minimum_density,
                law,
                config,
                registered_variables,
            )

    # With the cell-centred fluxes we can already compute the central part of
    # the interface flux.
    F_interface = 1/12 * (
        -_shift(F, 1, axis=1) + 7 * F + 7 * _shift(F, -1, axis=1) - _shift(F, -2, axis=1)
    )

    # -------------------------------------------------------------
    # ================= ↑ Cell-centred fluxes ↑ ===================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ======== ↓ Eigensystem of one characteristic field ↓ ========
    # -------------------------------------------------------------

    if config.mhd:
        num_modes = 7
    else:
        num_modes = config.dimensionality + 2

    if is_barotropic(config):
        num_modes -= 1

    def mode_eigenvalues(mode):
        """The eigenvalue of characteristic field ``mode`` in every cell."""
        if config.equation_of_state == IDEAL_GAS:
            if config.mhd:
                return _eigen_lambdas(
                    conserved_state,
                    minimum_density,
                    minimum_pressure,
                    gamma,
                    registered_variables,
                    mode,
                    internal_energy_density=internal_energy_density,
                    dual_eta=dual_eta,
                )
            return _eigen_lambdas_hydro(
                conserved_state,
                minimum_density,
                minimum_pressure,
                gamma,
                config,
                registered_variables,
                mode,
                internal_energy_density=internal_energy_density,
                dual_eta=dual_eta,
            )
        if config.mhd:
            return _eigen_lambdas_iso(
                conserved_state,
                minimum_density,
                law,
                registered_variables,
                mode,
            )
        return _eigen_lambdas_hydro_iso(
            conserved_state,
            minimum_density,
            law,
            config,
            registered_variables,
            mode,
        )

    def mode_left_row(mode):
        """Row ``mode`` of the left eigenvector matrix at every interface."""
        if config.equation_of_state == IDEAL_GAS:
            if config.mhd:
                return _eigen_L_row(
                    conserved_state,
                    minimum_density,
                    minimum_pressure,
                    gamma,
                    registered_variables,
                    mode,
                    internal_energy_density=internal_energy_density,
                    dual_eta=dual_eta,
                    admissible_face_state=admissible_face_state,
                )
            return _eigen_L_row_hydro(
                conserved_state,
                minimum_density,
                minimum_pressure,
                gamma,
                config,
                registered_variables,
                mode,
                internal_energy_density=internal_energy_density,
                dual_eta=dual_eta,
            )
        if config.mhd:
            return _eigen_L_row_iso(
                conserved_state,
                minimum_density,
                law,
                registered_variables,
                mode,
            )
        return _eigen_L_row_hydro_iso(
            conserved_state,
            minimum_density,
            law,
            config,
            registered_variables,
            mode,
        )

    def mode_right_column(mode):
        """Column ``mode`` of the right eigenvector matrix at every interface."""
        if config.equation_of_state == IDEAL_GAS:
            if config.mhd:
                return _eigen_R_col(
                    conserved_state,
                    minimum_density,
                    minimum_pressure,
                    gamma,
                    registered_variables,
                    mode,
                    internal_energy_density=internal_energy_density,
                    dual_eta=dual_eta,
                    admissible_face_state=admissible_face_state,
                )
            return _eigen_R_col_hydro(
                conserved_state,
                minimum_density,
                minimum_pressure,
                gamma,
                config,
                registered_variables,
                mode,
                internal_energy_density=internal_energy_density,
                dual_eta=dual_eta,
            )
        if config.mhd:
            return _eigen_R_col_iso(
                conserved_state,
                minimum_density,
                law,
                registered_variables,
                mode,
            )
        return _eigen_R_col_hydro_iso(
            conserved_state,
            minimum_density,
            law,
            config,
            registered_variables,
            mode,
        )

    # -------------------------------------------------------------
    # ======== ↑ Eigensystem of one characteristic field ↑ ========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ====== ↓ Positivity-preserving splitting speed ↓ ============
    # -------------------------------------------------------------

    if positivity_preserving:
        # The common splitting speed of the fields that carry mass: the
        # largest wave speed anywhere on the stencil (see _weno_positivity.py).
        spectral_radius = jnp.max(
            jnp.stack([jnp.abs(mode_eigenvalues(mode)) for mode in range(num_modes)]),
            axis=0,
        )
        common_speed = stencil_maximum(spectral_radius)
        if config.solver_config.weno_ad_frozen_weights:
            common_speed = jax.lax.stop_gradient(common_speed)
        safe_common_speed = jnp.maximum(common_speed, 1e-30)

        # Fields that carry no mass keep their own splitting speed.
        keeps_own_speed = jnp.array(
            [mode in mass_free_modes(config) for mode in range(num_modes)]
        )
        # The split-state shifts only exist when some field keeps its own speed.
        carries_shifts = len(mass_free_modes(config)) > 0
        zero = jnp.zeros_like(F_interface)

    # -------------------------------------------------------------
    # ====== ↑ Positivity-preserving splitting speed ↑ ============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ====== ↓ Per-mode characteristic reconstruction ↓ ===========
    # -------------------------------------------------------------

    def mode_flux(mode, F_current):
        """Add the WENO correction of characteristic field ``mode`` to the
        flux accumulated over the previous fields (the split-flux tuple under
        positivity preservation)."""

        lambdas_center = mode_eigenvalues(mode)
        L_row = mode_left_row(mode)
        if config.solver_config.weno_ad_frozen_weights:
            # The eigensystem and the splitting speed are frozen with the
            # weights: L carries 1/c^2 terms, so in cold gas d L / d c ~ 1/c^3
            # and the tangent is amplified ~20x per step in cold dense knots.
            lambdas_center = jax.lax.stop_gradient(lambdas_center)
            L_row = jax.lax.stop_gradient(L_row)

        # The six stencil cells i - 2 ... i + 3 of the interface i + 1/2, as
        # fields aligned with cell i; each has shape (N_vars, Nx, Ny, Nz).
        F0 = _shift(F,  2, axis=1)
        F1 = _shift(F,  1, axis=1)
        F2 = F
        F3 = _shift(F, -1, axis=1)
        F4 = _shift(F, -2, axis=1)
        F5 = _shift(F, -3, axis=1)

        # Project the fluxes (s_k = F_s_m) and the conserved variables
        # (q_k = q_s_m) of the stencil onto the characteristic field.
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

        # The differences ΔF_s and Δq_s of neighbouring stencil cells.
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

        # The splitting speed alpha^s: the largest |lambda^s| over the six
        # stencil cells.
        lam0 = _shift(lambdas_center,  2, axis=0)
        lam1 = _shift(lambdas_center,  1, axis=0)
        lam2 = lambdas_center
        lam3 = _shift(lambdas_center, -1, axis=0)
        lam4 = _shift(lambdas_center, -2, axis=0)
        lam5 = _shift(lambdas_center, -3, axis=0)
        lam_stack = jnp.stack([lam0, lam1, lam2, lam3, lam4, lam5], axis=0)
        amx = jnp.max(jnp.abs(lam_stack), axis=0)
        if positivity_preserving:
            amx = jnp.where(keeps_own_speed[mode], amx, common_speed)

        # The left-biased reconstruction of the split flux ΔF_s^+ (the
        # arguments a, b, c, d of phi in the module docstring).
        aterm_p = 0.5 * (d0 + amx * dq0)
        bterm_p = 0.5 * (d1 + amx * dq1)
        cterm_p = 0.5 * (d2 + amx * dq2)
        dterm_p = 0.5 * (d3 + amx * dq3)

        IS0_p = 13.0 * (aterm_p - bterm_p)**2 + 3.0 * (aterm_p - 3.0*bterm_p)**2
        IS1_p = 13.0 * (bterm_p - cterm_p)**2 + 3.0 * (bterm_p + cterm_p)**2
        IS2_p = 13.0 * (cterm_p - dterm_p)**2 + 3.0 * (3.0*cterm_p - dterm_p)**2

        omega0_p, omega2_p = omega_weights(IS0_p, IS1_p, IS2_p, epsilon, 1e-14)

        second = (omega0_p * (aterm_p - 2.0*bterm_p + cterm_p) * (1.0 / 3.0)
                  + (omega2_p - 0.5) * (bterm_p - 2.0*cterm_p + dterm_p) * (1.0 / 6.0))

        # The right-biased reconstruction of ΔF_s^-, on the mirrored stencil.
        aterm_m = 0.5 * (d4 - amx * dq4)
        bterm_m = 0.5 * (d3 - amx * dq3)
        cterm_m = 0.5 * (d2 - amx * dq2)
        dterm_m = 0.5 * (d1 - amx * dq1)

        IS0_m = 13.0 * (aterm_m - bterm_m)**2 + 3.0 * (aterm_m - 3.0*bterm_m)**2
        IS1_m = 13.0 * (bterm_m - cterm_m)**2 + 3.0 * (bterm_m + cterm_m)**2
        IS2_m = 13.0 * (cterm_m - dterm_m)**2 + 3.0 * (3.0*cterm_m - dterm_m)**2

        omega0_m, omega2_m = omega_weights(IS0_m, IS1_m, IS2_m, epsilon, 1e-14)

        third = (omega0_m * (aterm_m - 2.0*bterm_m + cterm_m) * (1.0 / 3.0)
                 + (omega2_m - 0.5) * (bterm_m - 2.0*cterm_m + dterm_m) * (1.0 / 6.0))

        Fs = -second + third

        # Transform back to conserved variables and add to the current flux.
        R_col = mode_right_column(mode)
        if config.solver_config.weno_ad_frozen_weights:
            R_col = jax.lax.stop_gradient(R_col)

        if positivity_preserving:
            # Keep the two split fluxes apart. A field on its own (smaller)
            # speed also shifts the central part and the upwind cells' split
            # states along its eigenvector, by (own - common) speed.
            if not carries_shifts:
                plus_correction, minus_correction = F_current
                return (
                    plus_correction - R_col * second[None],
                    minus_correction + R_col * third[None],
                )
            plus_correction, minus_correction, plus_owner_shift, minus_owner_shift = F_current
            speed_offset = (amx - common_speed)[None]
            central_projection = (1.0 / 12.0) * (-q1 + 7.0 * q2 + 7.0 * q3 - q4)
            relative_offset = speed_offset / safe_common_speed[None]
            central_shift = 0.5 * speed_offset * R_col * central_projection[None]
            return (
                plus_correction - R_col * second[None] + central_shift,
                minus_correction + R_col * third[None] - central_shift,
                plus_owner_shift + relative_offset * R_col * q2[None],
                minus_owner_shift + relative_offset * R_col * q3[None],
            )

        if config.dimensionality == 3:
            dF = jnp.einsum('nxyz,xyz->nxyz', R_col, Fs)
        elif config.dimensionality == 2:
            dF = jnp.einsum('nxy,xy->nxy', R_col, Fs)
        else:
            dF = jnp.einsum('nx,x->nx', R_col, Fs)
        return F_current + dF

    # -------------------------------------------------------------
    # ====== ↑ Per-mode characteristic reconstruction ↑ ===========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ====== ↓ Positivity-preserving recombination ↓ ==============
    # -------------------------------------------------------------

    if positivity_preserving:
        if carries_shifts:
            plus_face_flux, minus_face_flux, plus_owner_shift, minus_owner_shift = (
                jax.lax.fori_loop(0, num_modes, mode_flux, (zero, zero, zero, zero))
            )
        else:
            plus_face_flux, minus_face_flux = jax.lax.fori_loop(
                0,
                num_modes,
                mode_flux,
                (zero, zero),
            )
            plus_owner_shift = minus_owner_shift = 0.0
        # The central parts of the two split fluxes, at the common speed (the
        # fields' own speeds entered through the shifts above).
        central_flux = F_interface
        central_state = (1.0 / 12.0) * (
            -_shift(conserved_state, 1, axis=1) + 7.0 * conserved_state
            + 7.0 * _shift(conserved_state, -1, axis=1) - _shift(conserved_state, -2, axis=1)
        )
        plus_face_flux = plus_face_flux + 0.5 * (
            central_flux + common_speed[None] * central_state
        )
        minus_face_flux = minus_face_flux + 0.5 * (
            central_flux - common_speed[None] * central_state
        )
        return positivity_preserving_interface_flux(
            conserved_state,
            F,
            common_speed,
            plus_face_flux,
            minus_face_flux,
            plus_owner_shift,
            minus_owner_shift,
            params,
            config,
            registered_variables,
            inflow_reference=inflow_reference,
        )

    # -------------------------------------------------------------
    # ====== ↑ Positivity-preserving recombination ↑ ==============
    # -------------------------------------------------------------

    # A loop over the characteristic fields instead of one einsum over all of
    # them keeps the full projection matrix from being materialised (memory);
    # a single einsum might be faster.
    return jax.lax.fori_loop(
        0,
        num_modes,
        mode_flux,
        F_interface,
    )


def _reference_in_sweep_layout(
    inflow_reference,
    axis,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Move ``mhd_inflow_reference`` into the layout of the native y / z sweep:
    the active axis first and its momentum / field components swapped with x,
    exactly as the state.

    Args:
        inflow_reference: The ``(B, sum S)`` pair of ``mhd_inflow_reference``
            in the untransposed layout, or None.
        axis: The sweep axis (1 or 2).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The pair in the sweep's layout, or None.
    """
    if inflow_reference is None:
        return None
    reference_state, speed_sum = inflow_reference
    state_order = _sweep_permutation(axis, config)
    component = "yz"[axis - 1]
    reference_state = jnp.transpose(reference_state, state_order)
    speed_sum = jnp.transpose(speed_sum, tuple(order - 1 for order in state_order[1:]))
    for index in (registered_variables.momentum_index, registered_variables.magnetic_index):
        first, other = index.x, getattr(index, component)
        swapped = jnp.stack([reference_state[other], reference_state[first]])
        reference_state = reference_state.at[jnp.array([first, other])].set(swapped)
    return reference_state, speed_sum


def _sweep_permutation(axis: int, config: SimulationConfig):
    """
    The transpose that makes ``axis`` (1 or 2) the leading spatial axis of a state with a leading variable axis;
    it is its own inverse.
    """
    if axis == 1:
        return (0, 2, 1) if config.dimensionality == 2 else (0, 2, 1, 3)
    return (0, 3, 2, 1)


def _swap_with_x_components(state, axis: int, config: SimulationConfig, registered_variables: RegisteredVariables):
    """
    The state (or flux) with the x components of momentum and, in MHD, of the magnetic field exchanged with their
    components along ``axis`` (1 or 2).
    """
    momentum_index = registered_variables.momentum_index
    momentum_other = (momentum_index.y, momentum_index.z)[axis - 1]

    momentum_x = state[momentum_index.x]
    momentum_along = state[momentum_other]
    if config.mhd:
        magnetic_index = registered_variables.magnetic_index
        magnetic_other = (magnetic_index.y, magnetic_index.z)[axis - 1]
        B_x = state[magnetic_index.x]
        B_along = state[magnetic_other]

    state = state.at[momentum_index.x].set(momentum_along)
    state = state.at[momentum_other].set(momentum_x)
    if config.mhd:
        state = state.at[magnetic_index.x].set(B_along)
        state = state.at[magnetic_other].set(B_x)
    return state


@partial(jax.jit, static_argnames=["registered_variables", "config", "axis"])
def _weno_flux_transverse_native(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int,
    internal_energy_density=None,
    inflow_reference=None,
):
    """
    WENO flux reconstruction along y (``axis`` 1) or z (``axis`` 2).

    Reuses the x-direction flux by transposing the state so that the flux axis
    becomes the leading spatial axis (and swapping the x components of momentum
    and field with those along the axis), running ``_weno_flux_x_native``, then
    undoing both the transpose and the component swap on the resulting flux.

    Args:
        conserved_state: The conserved state array.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        axis: The flux axis, 1 (y) or 2 (z).
        internal_energy_density: The dual-energy ``g`` with the spatial shape
            of the state, or None.
        inflow_reference: Ideal MHD with ``weno_positivity_preserving``: the
            ``(B, sum S)`` pair of ``mhd_inflow_reference`` in the
            untransposed layout; None otherwise.

    Returns:
        The WENO interface fluxes along ``axis``.
    """
    permutation = _sweep_permutation(axis, config)
    sweep_state = jnp.transpose(conserved_state, permutation)

    # The dual-energy field is transposed the same way (it has no leading
    # variable axis).
    if internal_energy_density is not None:
        internal_energy_density = jnp.transpose(
            internal_energy_density, tuple(order - 1 for order in permutation[1:])
        )

    sweep_state = _swap_with_x_components(sweep_state, axis, config, registered_variables)
    flux = _weno_flux_x_native(
        sweep_state,
        params,
        config,
        registered_variables,
        internal_energy_density=internal_energy_density,
        inflow_reference=_reference_in_sweep_layout(
            inflow_reference,
            axis,
            config,
            registered_variables,
        ),
    )
    flux = jnp.transpose(flux, permutation)
    return _swap_with_x_components(flux, axis, config, registered_variables)


def _weno_flux_native_for_axis(axis: int):
    """Return the native-JAX WENO flux function for the given spatial axis."""
    if axis == 0:
        return _weno_flux_x_native
    return partial(_weno_flux_transverse_native, axis=axis)


def _native_tangent_sharded(axis: int, native):
    """
    Wrap the native-JAX tangent branch of a diffable Pallas WENO flux in the
    same multi-GPU ``shard_map`` + halo exchange as the Pallas primal
    (``_pallas_call_sharded``, halo 3 on the flux axis; a pass-through on one
    device).

    Without it the tangent -- and so the whole reverse-mode sweep -- runs the
    native WENO under GSPMD, where every periodic roll along the split axis
    becomes either an all-to-all reshard or one small ppermute per stencil
    shift, which makes the multi-GPU gradient slower than the single-GPU one.
    Inside the shard_map the rolls are local and only the halo moves.

    Args:
        axis: The spatial axis of the flux.
        native: ``native(state, params)``, or ``native(state, params,
            internal_energy_density)`` with the dual-energy ``g`` of shape
            ``(x, y, z)``, which then rides the halo exchange as a state-shaped
            ``(1, x, y, z)`` input.

    Returns:
        The wrapped callable, with the signature of ``native``.
    """
    halo_widths = [0, 0, 0]
    halo_widths[int(axis)] = 3

    def wrapped(state, params, *dual_energy):
        num_spatial_dims = state.ndim - 1
        halo = tuple(halo_widths[:num_spatial_dims])
        block_shape = (1,) * num_spatial_dims
        if dual_energy and dual_energy[0] is not None:
            internal_energy_density = dual_energy[0]

            def native_local(state_local, internal_energy_local):
                return native(state_local, params, internal_energy_local[0])

            return _pallas_call_sharded(
                native_local,
                state_inputs=(state, internal_energy_density[None]),
                halo=halo,
                block_shape=block_shape,
            )

        def native_local(state_local):
            return native(state_local, params)

        return _pallas_call_sharded(
            native_local,
            state_inputs=(state,),
            halo=halo,
            block_shape=block_shape,
        )

    return wrapped


def _hydro_pallas_axis_supported(conserved_state, axis: int, config: SimulationConfig):
    """Whether the hydro Pallas WENO kernel supports this state and axis
    (only the axes of the configured dimensionality)."""
    return _hydro_pallas_flux_supported(conserved_state, config) and (
        axis == 0
        or (axis == 1 and int(config.dimensionality) >= 2)
        or (axis == 2 and int(config.dimensionality) == 3)
    )


def _weno_flux_axis_dispatch(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int,
    internal_energy_density=None,
    inflow_reference=None,
):
    """
    Pick the Pallas flux for the supported equation set, falling back to
    the native per-axis JAX flux.

    Every Pallas path is wrapped through ``diffable_pallas_call`` (a
    ``jax.custom_jvp`` boundary: Pallas primal, native-JAX tangent).  A
    custom_jvp supports BOTH modes — forward-mode (``jax.jvp`` / ``jacfwd``)
    fires the tangent rule directly, and reverse-mode (``jax.grad`` /
    ``differentiation_mode == BACKWARDS``) is derived by JAX transposing that
    native tangent.  This differentiates w.r.t. the conserved STATE *and*
    ``params`` (so gradients w.r.t. physical parameters that enter the flux are
    non-zero), and the backward is standard native-JAX AD, which compiles fast.

    The native tangent also re-runs the native forward for its linearisation,
    so the reverse sweep runs at native-JAX speed. In reverse mode
    (``differentiation_mode == BACKWARDS``, ``backend_config.pallas_adjoint``)
    the ideal-gas hydro and ideal-MHD fluxes therefore take a Pallas adjoint
    instead (``pallas_vjp_call`` around ``_weno_flux_hydro_pallas_vjp`` /
    ``_weno_flux_mhd_pallas_vjp``: ``jax.vjp`` of the forward kernel's window
    function inside a Pallas kernel; the positivity-preserving recombination
    of MHD is differentiated in native JAX). It differentiates with respect to
    the state, the dual-energy ``g`` and the MHD inflow reference only
    (``params`` get a zero cotangent) and does not support forward mode, which
    is why it is tied to BACKWARDS.

    TODO: wrap the ideal- and isothermal-MHD tangents in
    ``_native_tangent_sharded`` like the hydro and dual-energy ones; they still
    run the native WENO under GSPMD on multi-GPU meshes.

    Args:
        conserved_state: The conserved state array.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        axis: The spatial axis of the flux.
        internal_energy_density: The dual-energy ``g`` with the spatial shape
            of the state, or None.
        inflow_reference: Ideal MHD with ``weno_positivity_preserving``: the
            axis-summed first-order inflow of every cell,
            ``mhd_inflow_reference``, for the joint per-cell limiting of the
            inflow faces; None otherwise.

    Returns:
        The WENO interface fluxes along ``axis``.
    """
    native_flux = _weno_flux_native_for_axis(axis)
    use_pallas_adjoint = (
        config.differentiation_mode == BACKWARDS and config.backend_config.pallas_adjoint
    )

    if internal_energy_density is not None:
        # Coupled dual-energy recovery: the hydro and MHD Pallas WENO kernels
        # carry the internal-energy field g as an extra (halo-exchanged) input
        # and apply the Bryan+95 switch in their pressure recovery.  Iso-MHD
        # has no energy equation, so g never arrives there.
        hydro_supported = _hydro_pallas_axis_supported(conserved_state, axis, config)
        mhd_supported = _mhd_pallas_flux_supported(conserved_state, config)
        if hydro_supported and use_pallas_adjoint:

            def pallas_forward(inputs, params):
                state, internal_energy = inputs
                return _weno_flux_hydro_pallas(
                    state,
                    params,
                    config,
                    registered_variables,
                    axis=axis,
                    internal_energy_density=internal_energy,
                )

            def pallas_backward(inputs, params, flux_cotangent):
                state, internal_energy = inputs
                return _weno_flux_hydro_pallas_vjp(
                    state,
                    flux_cotangent,
                    params,
                    config,
                    registered_variables,
                    axis=axis,
                    internal_energy_density=internal_energy,
                )

            return pallas_vjp_call(
                (conserved_state, internal_energy_density),
                params,
                pallas_forward=pallas_forward,
                pallas_backward=pallas_backward,
            )
        if mhd_supported and use_pallas_adjoint:

            def pallas_forward(inputs, params):
                state, internal_energy = inputs
                return _weno_flux_mhd_pallas(
                    state,
                    params,
                    config,
                    registered_variables,
                    axis=axis,
                    internal_energy_density=internal_energy,
                )

            def pallas_backward(inputs, params, flux_cotangent):
                state, internal_energy = inputs
                return _weno_flux_mhd_pallas_vjp(
                    state,
                    flux_cotangent,
                    params,
                    config,
                    registered_variables,
                    axis=axis,
                    internal_energy_density=internal_energy,
                )

            return pallas_vjp_call(
                (conserved_state, internal_energy_density),
                params,
                pallas_forward=pallas_forward,
                pallas_backward=pallas_backward,
            )
        if hydro_supported or mhd_supported:
            kernel = _weno_flux_hydro_pallas if hydro_supported else _weno_flux_mhd_pallas

            def pallas_branch(state, params, internal_energy):
                return kernel(
                    state,
                    params,
                    config,
                    registered_variables,
                    axis=axis,
                    internal_energy_density=internal_energy,
                )

            def native_branch(state, params, internal_energy):
                return native_flux(
                    state,
                    params,
                    config,
                    registered_variables,
                    internal_energy_density=internal_energy,
                )

            return diffable_pallas_call_n(
                (conserved_state, params, internal_energy_density),
                pallas_branch=pallas_branch,
                native_branch=_native_tangent_sharded(axis, native_branch),
            )
        return native_flux(
            conserved_state,
            params,
            config,
            registered_variables,
            internal_energy_density=internal_energy_density,
        )

    if _hydro_pallas_axis_supported(conserved_state, axis, config):
        # (the Pallas adjoint is that of the ideal-gas kernel)
        if use_pallas_adjoint and not is_barotropic(config):
            return pallas_vjp_call(
                conserved_state,
                params,
                pallas_forward=lambda state, params: _weno_flux_hydro_pallas(
                    state,
                    params,
                    config,
                    registered_variables,
                    axis=axis,
                ),
                pallas_backward=lambda state, params, flux_cotangent: _weno_flux_hydro_pallas_vjp(
                    state,
                    flux_cotangent,
                    params,
                    config,
                    registered_variables,
                    axis=axis,
                ),
            )

        def pallas_branch(state, params):
            return _weno_flux_hydro_pallas(
                state,
                params,
                config,
                registered_variables,
                axis=axis,
            )

        def native_branch(state, params):
            return native_flux(state, params, config, registered_variables)

        return diffable_pallas_call(
            conserved_state,
            params,
            pallas_branch=pallas_branch,
            native_branch=_native_tangent_sharded(axis, native_branch),
        )

    if _mhd_pallas_flux_supported(conserved_state, config):
        if use_pallas_adjoint:
            # The inflow reference (when present) is differentiable input of
            # the flux; its cotangents leave through the reference's own
            # boundary (``mhd_inflow_reference_dispatch``).
            reference = tuple(inflow_reference) if inflow_reference is not None else ()

            def pallas_forward(inputs, params):
                state, *reference_inputs = inputs
                return _weno_flux_mhd_pallas(
                    state,
                    params,
                    config,
                    registered_variables,
                    axis=axis,
                    inflow_reference=tuple(reference_inputs) if reference_inputs else None,
                )

            def pallas_backward(inputs, params, flux_cotangent):
                state, *reference_inputs = inputs
                cotangents = _weno_flux_mhd_pallas_vjp(
                    state,
                    flux_cotangent,
                    params,
                    config,
                    registered_variables,
                    axis=axis,
                    inflow_reference=tuple(reference_inputs) if reference_inputs else None,
                )
                return cotangents if reference_inputs else (cotangents,)

            return pallas_vjp_call(
                (conserved_state,) + reference,
                params,
                pallas_forward=pallas_forward,
                pallas_backward=pallas_backward,
            )
        if inflow_reference is not None:

            def pallas_branch(state, params, reference_state, speed_sum):
                return _weno_flux_mhd_pallas(
                    state,
                    params,
                    config,
                    registered_variables,
                    axis=axis,
                    inflow_reference=(reference_state, speed_sum),
                )

            def native_branch(state, params, reference_state, speed_sum):
                return native_flux(
                    state,
                    params,
                    config,
                    registered_variables,
                    inflow_reference=(reference_state, speed_sum),
                )

            return diffable_pallas_call_n(
                (conserved_state, params) + tuple(inflow_reference),
                pallas_branch=pallas_branch,
                native_branch=native_branch,
            )

        def pallas_branch(state, params):
            return _weno_flux_mhd_pallas(
                state,
                params,
                config,
                registered_variables,
                axis=axis,
            )

        def native_branch(state, params):
            return native_flux(state, params, config, registered_variables)

        return diffable_pallas_call(
            conserved_state,
            params,
            pallas_branch=pallas_branch,
            native_branch=native_branch,
        )

    if _mhd_iso_pallas_flux_supported(conserved_state, config):

        def pallas_branch(state, params):
            return _weno_flux_mhd_iso_pallas(
                state,
                params,
                config,
                registered_variables,
                axis=axis,
            )

        def native_branch(state, params):
            return native_flux(state, params, config, registered_variables)

        return diffable_pallas_call(
            conserved_state,
            params,
            pallas_branch=pallas_branch,
            native_branch=native_branch,
        )

    return native_flux(
        conserved_state,
        params,
        config,
        registered_variables,
        inflow_reference=inflow_reference,
    )


@partial(jax.jit, static_argnames=["registered_variables", "config", "axis"])
def _weno_flux(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int,
    internal_energy_density=None,
    inflow_reference=None,
):
    """
    WENO interface flux along ``axis`` (Pallas backend where supported, native
    JAX otherwise).

    Args:
        conserved_state: The conserved state array.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        axis: The flux axis, 0 (x), 1 (y) or 2 (z).
        internal_energy_density: The dual-energy ``g`` with the spatial shape
            of the state, or None.
        inflow_reference: Ideal MHD with ``weno_positivity_preserving``: the
            ``(B, sum S)`` pair of ``mhd_inflow_reference``; None otherwise.

    Returns:
        The WENO interface fluxes along ``axis``.
    """
    return _weno_flux_axis_dispatch(
        conserved_state,
        params,
        config,
        registered_variables,
        axis=axis,
        internal_energy_density=internal_energy_density,
        inflow_reference=inflow_reference,
    )
