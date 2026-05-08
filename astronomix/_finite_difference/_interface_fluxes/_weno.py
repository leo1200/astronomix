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
   and left eigenvector matrices R_{i+1/2} \\in R^{N_vars x N_char}
   and L_{i+1/2} \\in R^{N_char x N_vars}, as well
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
   into F_s^+ and F_s^- such that ∂_u F_s^+ only has non-negative
   eigenvalues, and ∂_u F_s^- only has non-positive eigenvalues.

   ΔF_s^+_{m+1/2} = 0.5 * (ΔF_s_{m+1/2} + alpha^s * Δq_s_{m+1/2}), m = i - 2, ..., i + 1
   ΔF_s^-_{m+1/2} = 0.5 * (ΔF_s_{m+1/2} - alpha^s * Δq_s_{m+1/2}), m = i - 1, ..., i + 2

5. WENO smoothness-weighted reconstruction.

Implementation notes
====================

The hot loop is structured around two ideas that minimize memory traffic:

* Each L row and R column is returned as a list of (scalar_field, var_index)
  pairs covering only its nonzero entries. We never materialize the full
  (num_vars, *spatial) eigenvector tensor — projection is a manual sum.

* The flux-projection differences ΔF and Δq are pre-computed once per
  direction (5 of each). Inside the mode loop we project differences
  directly, which saves one projection per mode compared to projecting six
  values and differencing in characteristic space.

* F_total is accumulated per touched conserved-variable component in scalar
  buffers, then stacked at the end. This avoids the (num_vars, *spatial)
  intermediate `R_col * Fs` tensor at every step of the loop.

For literature references, see:

 - High Order ENO and WENO Schemes for Computational Fluid Dynamics by Chi-Wang Shu (1997)
   (https://doi.org/10.1007/978-3-662-03882-6_5)

Concretely we implement the 5th-order WENO scheme as described in

- HOW-MHD: A High-Order WENO-Based Magnetohydrodynamic Code with a High-Order
  Constrained Transport Algorithm for Astrophysical Applications by Seo & Ryu 2023
  (https://arxiv.org/abs/2304.04360)
"""

from functools import partial, reduce
import jax
import jax.numpy as jnp

from astronomix._finite_difference._fluid_equations._eigen_hydro import (
    _eigen_all_lambdas_hydro,
    _eigen_lambdas_hydro,
    _eigen_L_pairs_hydro_from_blocks,
    _eigen_L_row_hydro,
    _eigen_L_row_hydro_from_blocks,
    _eigen_R_col_hydro,
    _eigen_R_col_hydro_from_blocks,
    _eigen_R_pairs_hydro_from_blocks,
    _eigenvector_building_blocks as _eigen_blocks_hydro,
)
from astronomix._finite_difference._fluid_equations._eigen_hydro_iso import (
    _eigen_all_lambdas_hydro_iso,
    _eigen_lambdas_hydro_iso,
    _eigen_L_pairs_hydro_iso_from_blocks,
    _eigen_L_row_hydro_iso,
    _eigen_L_row_hydro_iso_from_blocks,
    _eigen_R_col_hydro_iso,
    _eigen_R_col_hydro_iso_from_blocks,
    _eigen_R_pairs_hydro_iso_from_blocks,
    _eigenvector_building_blocks as _eigen_blocks_hydro_iso,
)
from astronomix._finite_difference._fluid_equations._eigen_mhd import (
    _eigen_all_lambdas,
    _eigen_lambdas,
    _eigen_L_pairs_from_blocks,
    _eigen_L_row,
    _eigen_L_row_from_blocks,
    _eigen_L_stack_from_blocks,
    _eigen_R_col,
    _eigen_R_col_from_blocks,
    _eigen_R_pairs_from_blocks,
    _eigen_R_stack_from_blocks,
    _eigenvector_building_blocks as _eigen_blocks_mhd,
)
from astronomix._finite_difference._fluid_equations._eigen_mhd_iso import (
    _eigen_all_lambdas_iso,
    _eigen_lambdas_iso,
    _eigen_L_pairs_iso_from_blocks,
    _eigen_L_row_iso,
    _eigen_L_row_iso_from_blocks,
    _eigen_R_col_iso,
    _eigen_R_col_iso_from_blocks,
    _eigen_R_pairs_iso_from_blocks,
    _eigenvector_building_blocks as _eigen_blocks_mhd_iso,
)
from astronomix._finite_difference._fluid_equations._fluxes import (
    _euler_flux_isothermal_x,
    _mhd_flux_isothermal_x,
    _mhd_flux_x,
)
from astronomix._fluid_equations._equations import primitive_state_from_conserved
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix.option_classes.simulation_config import (
    IDEAL_GAS,
    ISOTHERMAL,
    SimulationConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables


def _project_pairs(pairs, X):
    """Sum scalar_field * X[var_index] over (scalar_field, var_index) pairs."""
    return reduce(lambda acc, p: acc + p[0] * X[p[1]], pairs[1:], pairs[0][0] * X[pairs[0][1]])


def _alpha_max(lam):
    """alpha = max_{k=-3..2} |lam shifted by k| — chained jnp.maximum."""
    a = jnp.abs(_shift(lam,  2, axis=0))
    a = jnp.maximum(a, jnp.abs(_shift(lam,  1, axis=0)))
    a = jnp.maximum(a, jnp.abs(lam))
    a = jnp.maximum(a, jnp.abs(_shift(lam, -1, axis=0)))
    a = jnp.maximum(a, jnp.abs(_shift(lam, -2, axis=0)))
    a = jnp.maximum(a, jnp.abs(_shift(lam, -3, axis=0)))
    return a


def _weno_reconstruct(d0, d1, d2, d3, d4, dq0, dq1, dq2, dq3, dq4, amx, epsilon):
    """Smoothness-weighted scalar Fs from 5 dF / 5 dQ projections + alpha_max."""
    ap = 0.5 * (d0 + amx * dq0)
    bp = 0.5 * (d1 + amx * dq1)
    cp = 0.5 * (d2 + amx * dq2)
    dp_ = 0.5 * (d3 + amx * dq3)

    IS0p = 13.0 * (ap - bp) ** 2 + 3.0 * (ap - 3.0 * bp) ** 2
    IS1p = 13.0 * (bp - cp) ** 2 + 3.0 * (bp + cp) ** 2
    IS2p = 13.0 * (cp - dp_) ** 2 + 3.0 * (3.0 * cp - dp_) ** 2

    a0p = 1.0 / (epsilon + IS0p) ** 2
    a1p = 6.0 / (epsilon + IS1p) ** 2
    a2p = 3.0 / (epsilon + IS2p) ** 2
    asum_p = jnp.maximum(a0p + a1p + a2p, 1e-14)
    w0p = a0p / asum_p
    w2p = a2p / asum_p

    second = (
        w0p * (ap - 2.0 * bp + cp) / 3.0
        + (w2p - 0.5) * (bp - 2.0 * cp + dp_) / 6.0
    )

    am = 0.5 * (d4 - amx * dq4)
    bm = 0.5 * (d3 - amx * dq3)
    cm = 0.5 * (d2 - amx * dq2)
    dm = 0.5 * (d1 - amx * dq1)

    IS0m = 13.0 * (am - bm) ** 2 + 3.0 * (am - 3.0 * bm) ** 2
    IS1m = 13.0 * (bm - cm) ** 2 + 3.0 * (bm + cm) ** 2
    IS2m = 13.0 * (cm - dm) ** 2 + 3.0 * (3.0 * cm - dm) ** 2

    a0m = 1.0 / (epsilon + IS0m) ** 2
    a1m = 6.0 / (epsilon + IS1m) ** 2
    a2m = 3.0 / (epsilon + IS2m) ** 2
    asum_m = jnp.maximum(a0m + a1m + a2m, 1e-14)
    w0m = a0m / asum_m
    w2m = a2m / asum_m

    third = (
        w0m * (am - 2.0 * bm + cm) / 3.0
        + (w2m - 0.5) * (bm - 2.0 * cm + dm) / 6.0
    )

    return -second + third


def _touched_indices(config: SimulationConfig, registered_variables: RegisteredVariables):
    """Conserved-variable indices that R columns project back to.

    These are the components for which we accumulate WENO contributions; any
    index not in this list keeps the central-stencil F_interface flux (e.g.
    Bx in MHD, which is updated by the constrained-transport step).
    """
    rv = registered_variables
    indices = [rv.density_index]
    if config.dimensionality == 1 and not config.mhd:
        indices.append(rv.momentum_index)
    else:
        indices.append(rv.momentum_index.x)
    if config.dimensionality >= 2:
        indices.append(rv.momentum_index.y)
    if config.dimensionality == 3:
        indices.append(rv.momentum_index.z)
    if config.mhd:
        indices.append(rv.magnetic_index.y)
        indices.append(rv.magnetic_index.z)
    if config.equation_of_state == IDEAL_GAS:
        indices.append(rv.energy_index)
    return indices


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_x(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """5th-order WENO flux in the +x direction (axis=1 of conserved_state)."""

    epsilon = 1e-7
    rhomin = params.minimum_density
    pgmin = params.minimum_pressure
    gamma = params.gamma
    isothermal_sound_speed = params.isothermal_sound_speed

    # Cell-centered fluxes
    if config.equation_of_state == IDEAL_GAS:
        if config.mhd:
            F = _mhd_flux_x(
                conserved_state, rhomin, pgmin, gamma, config, registered_variables
            )
        else:
            F = _euler_flux(
                primitive_state_from_conserved(
                    conserved_state, gamma, config, registered_variables
                ),
                gamma,
                config,
                registered_variables,
                1,
            )
    else:  # ISOTHERMAL
        if config.mhd:
            F = _mhd_flux_isothermal_x(
                conserved_state, rhomin, isothermal_sound_speed, config, registered_variables
            )
        else:
            F = _euler_flux_isothermal_x(
                conserved_state, rhomin, isothermal_sound_speed, config, registered_variables
            )

    # Number of characteristic modes is static.
    if config.mhd:
        num_modes = 6 if config.equation_of_state == ISOTHERMAL else 7
    else:
        num_modes = config.dimensionality + (1 if config.equation_of_state == ISOTHERMAL else 2)

    # Central 5th-order stencil flux F_{i+1/2} (state-shape).
    F_interface = (1.0 / 12.0) * (
        -_shift(F, 1, axis=1) + 7.0 * F + 7.0 * _shift(F, -1, axis=1) - _shift(F, -2, axis=1)
    )

    if config.weno_low_memory:
        # ----- Memory-efficient path -----
        # Everything is computed inline in the ``lax.fori_loop`` body so
        # nothing is hoisted as resident state across iterations: per-mode
        # eigenvectors / eigenvalues, the F and Q shifts of the 6-cell
        # WENO stencil, and the WENO reconstruction itself. XLA CSEs the
        # eigen building-block computation between L_row, R_col and lambdas
        # within a single iteration; across iterations each iteration starts
        # from a clean slate so the working set never grows with mode count.
        if config.dimensionality == 3:
            proj_spec = "nxyz,nxyz->xyz"
            bcast_spec = "nxyz,xyz->nxyz"
        elif config.dimensionality == 2:
            proj_spec = "nxy,nxy->xy"
            bcast_spec = "nxy,xy->nxy"
        else:
            proj_spec = "nx,nx->x"
            bcast_spec = "nx,x->nx"

        def body(mode, F_current):
            # L_row + lambdas computed first; R_col is computed at the END
            # of the body (matching main's structure) so it doesn't share
            # the body's peak working set with L_row + the s_k / q_k
            # projections. This gives XLA the smallest per-iteration footprint.
            if config.equation_of_state == IDEAL_GAS:
                if config.mhd:
                    lambdas_center = _eigen_lambdas(
                        conserved_state, rhomin, pgmin, gamma, registered_variables, mode
                    )
                    L_row = _eigen_L_row(
                        conserved_state, rhomin, pgmin, gamma, registered_variables, mode
                    )
                else:
                    lambdas_center = _eigen_lambdas_hydro(
                        conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode
                    )
                    L_row = _eigen_L_row_hydro(
                        conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode
                    )
            else:
                if config.mhd:
                    lambdas_center = _eigen_lambdas_iso(
                        conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode
                    )
                    L_row = _eigen_L_row_iso(
                        conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode
                    )
                else:
                    lambdas_center = _eigen_lambdas_hydro_iso(
                        conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode
                    )
                    L_row = _eigen_L_row_hydro_iso(
                        conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode
                    )

            F_p2 = _shift(F,  2, axis=1)
            F_p1 = _shift(F,  1, axis=1)
            F_m1 = _shift(F, -1, axis=1)
            F_m2 = _shift(F, -2, axis=1)
            F_m3 = _shift(F, -3, axis=1)

            s0 = jnp.einsum(proj_spec, L_row, F_p2)
            s1 = jnp.einsum(proj_spec, L_row, F_p1)
            s2 = jnp.einsum(proj_spec, L_row, F)
            s3 = jnp.einsum(proj_spec, L_row, F_m1)
            s4 = jnp.einsum(proj_spec, L_row, F_m2)
            s5 = jnp.einsum(proj_spec, L_row, F_m3)

            q0 = jnp.einsum(proj_spec, L_row, _shift(conserved_state,  2, axis=1))
            q1 = jnp.einsum(proj_spec, L_row, _shift(conserved_state,  1, axis=1))
            q2 = jnp.einsum(proj_spec, L_row, conserved_state)
            q3 = jnp.einsum(proj_spec, L_row, _shift(conserved_state, -1, axis=1))
            q4 = jnp.einsum(proj_spec, L_row, _shift(conserved_state, -2, axis=1))
            q5 = jnp.einsum(proj_spec, L_row, _shift(conserved_state, -3, axis=1))

            amx = _alpha_max(lambdas_center)
            Fs = _weno_reconstruct(
                s1 - s0, s2 - s1, s3 - s2, s4 - s3, s5 - s4,
                q1 - q0, q2 - q1, q3 - q2, q4 - q3, q5 - q4,
                amx, epsilon,
            )

            # R_col last — keeps the working-set window narrow.
            if config.equation_of_state == IDEAL_GAS:
                if config.mhd:
                    R_col = _eigen_R_col(
                        conserved_state, rhomin, pgmin, gamma, registered_variables, mode
                    )
                else:
                    R_col = _eigen_R_col_hydro(
                        conserved_state, rhomin, pgmin, gamma, config, registered_variables, mode
                    )
            else:
                if config.mhd:
                    R_col = _eigen_R_col_iso(
                        conserved_state, rhomin, isothermal_sound_speed, registered_variables, mode
                    )
                else:
                    R_col = _eigen_R_col_hydro_iso(
                        conserved_state, rhomin, isothermal_sound_speed, config, registered_variables, mode
                    )
            return F_current + jnp.einsum(bcast_spec, R_col, Fs)

        return jax.lax.fori_loop(0, num_modes, body, F_interface)

    # ----- Performance path -----
    # Pre-differences (dF_k = F_shift_{k+1} - F_shift_k) are state-shape; each
    # mode projects them in characteristic space. This saves one projection
    # per mode versus projecting six absolute-value shifts.
    #
    # Eigenvector building blocks + all eigenvalues are hoisted out of the
    # mode loop so the per-mode work shares them. This is the explicit speed-
    # vs-memory trade vs the low_memory branch above (which recomputes per
    # iteration so nothing is resident).
    if config.equation_of_state == IDEAL_GAS:
        if config.mhd:
            blocks = _eigen_blocks_mhd(
                conserved_state, gamma, rhomin, pgmin, registered_variables
            )
            all_lambdas = _eigen_all_lambdas(
                conserved_state, rhomin, pgmin, gamma, registered_variables
            )

            def L_pairs_fn(mode):
                return _eigen_L_pairs_from_blocks(blocks, registered_variables, mode)

            def R_pairs_fn(mode):
                return _eigen_R_pairs_from_blocks(blocks, registered_variables, mode)
        else:
            blocks = _eigen_blocks_hydro(
                conserved_state, gamma, rhomin, pgmin, config, registered_variables
            )
            all_lambdas = _eigen_all_lambdas_hydro(
                conserved_state, rhomin, pgmin, gamma, config, registered_variables
            )

            def L_pairs_fn(mode):
                return _eigen_L_pairs_hydro_from_blocks(blocks, config, registered_variables, mode)

            def R_pairs_fn(mode):
                return _eigen_R_pairs_hydro_from_blocks(blocks, config, registered_variables, mode)
    else:  # ISOTHERMAL
        if config.mhd:
            blocks = _eigen_blocks_mhd_iso(
                conserved_state, isothermal_sound_speed, rhomin, registered_variables
            )
            all_lambdas = _eigen_all_lambdas_iso(
                conserved_state, rhomin, isothermal_sound_speed, registered_variables
            )

            def L_pairs_fn(mode):
                return _eigen_L_pairs_iso_from_blocks(blocks, registered_variables, mode)

            def R_pairs_fn(mode):
                return _eigen_R_pairs_iso_from_blocks(blocks, registered_variables, mode)
        else:
            blocks = _eigen_blocks_hydro_iso(
                conserved_state, isothermal_sound_speed, rhomin, config, registered_variables
            )
            all_lambdas = _eigen_all_lambdas_hydro_iso(
                conserved_state, rhomin, isothermal_sound_speed, config, registered_variables
            )

            def L_pairs_fn(mode):
                return _eigen_L_pairs_hydro_iso_from_blocks(blocks, config, registered_variables, mode)

            def R_pairs_fn(mode):
                return _eigen_R_pairs_hydro_iso_from_blocks(blocks, config, registered_variables, mode)

    F_p2 = _shift(F,  2, axis=1)
    F_p1 = _shift(F,  1, axis=1)
    F_m1 = _shift(F, -1, axis=1)
    F_m2 = _shift(F, -2, axis=1)
    F_m3 = _shift(F, -3, axis=1)
    dF0 = F_p1 - F_p2
    dF1 = F   - F_p1
    dF2 = F_m1 - F
    dF3 = F_m2 - F_m1
    dF4 = F_m3 - F_m2

    Q_p2 = _shift(conserved_state,  2, axis=1)
    Q_p1 = _shift(conserved_state,  1, axis=1)
    Q_m1 = _shift(conserved_state, -1, axis=1)
    Q_m2 = _shift(conserved_state, -2, axis=1)
    Q_m3 = _shift(conserved_state, -3, axis=1)
    dQ0 = Q_p1 - Q_p2
    dQ1 = conserved_state - Q_p1
    dQ2 = Q_m1 - conserved_state
    dQ3 = Q_m2 - Q_m1
    dQ4 = Q_m3 - Q_m2

    if config.mhd and config.equation_of_state == IDEAL_GAS:
        # JAX-native path: ``lax.fori_loop`` over modes; ``lax.switch``
        # dispatches the (K=7, *spatial) eigen helpers. dF / dQ are gathered
        # at the K touched indices ONCE and stacked along a new leading axis,
        # so each iteration runs ONE batched einsum per stencil-side. No
        # Python iteration in the body, no state-shape L_row / R_col
        # materialization.
        rv = registered_variables
        indices = jnp.array(
            [
                rv.density_index,
                rv.momentum_index.x,
                rv.momentum_index.y,
                rv.momentum_index.z,
                rv.magnetic_index.y,
                rv.magnetic_index.z,
                rv.energy_index,
            ]
        )
        # Stack dF/dQ at the touched indices: shape (5, K, *spatial).
        dF_stack = jnp.stack(
            [dF0[indices], dF1[indices], dF2[indices], dF3[indices], dF4[indices]],
            axis=0,
        )
        dQ_stack = jnp.stack(
            [dQ0[indices], dQ1[indices], dQ2[indices], dQ3[indices], dQ4[indices]],
            axis=0,
        )

        def body(mode, F_total):
            L_stack = _eigen_L_stack_from_blocks(blocks, mode)
            R_stack = _eigen_R_stack_from_blocks(blocks, mode)

            # Batched projection: contract K, keep j (= which difference).
            d_proj = jnp.einsum("k...,jk...->j...", L_stack, dF_stack)
            dq_proj = jnp.einsum("k...,jk...->j...", L_stack, dQ_stack)

            amx = _alpha_max(all_lambdas[mode])
            Fs = _weno_reconstruct(
                d_proj[0], d_proj[1], d_proj[2], d_proj[3], d_proj[4],
                dq_proj[0], dq_proj[1], dq_proj[2], dq_proj[3], dq_proj[4],
                amx, epsilon,
            )

            return F_total.at[indices].add(R_stack * Fs[None])

        return jax.lax.fori_loop(0, num_modes, body, F_interface)

    # ----- Fallback path (Python-unrolled) for non-MHD-ideal cases.
    # TODO: extend stack helpers to mhd_iso / hydro / hydro_iso so this branch
    # can be removed.
    touched = _touched_indices(config, registered_variables)
    F_acc = {idx: F_interface[idx] for idx in touched}
    touched_keys = list(F_acc.keys())

    for mode in range(num_modes):
        L_pairs = L_pairs_fn(mode)
        R_pairs = R_pairs_fn(mode)

        d0 = _project_pairs(L_pairs, dF0)
        d1 = _project_pairs(L_pairs, dF1)
        d2 = _project_pairs(L_pairs, dF2)
        d3 = _project_pairs(L_pairs, dF3)
        d4 = _project_pairs(L_pairs, dF4)
        dq0 = _project_pairs(L_pairs, dQ0)
        dq1 = _project_pairs(L_pairs, dQ1)
        dq2 = _project_pairs(L_pairs, dQ2)
        dq3 = _project_pairs(L_pairs, dQ3)
        dq4 = _project_pairs(L_pairs, dQ4)

        amx = _alpha_max(all_lambdas[mode])
        Fs = _weno_reconstruct(d0, d1, d2, d3, d4, dq0, dq1, dq2, dq3, dq4, amx, epsilon)

        for R_val, idx in R_pairs:
            F_acc[idx] = F_acc[idx] + R_val * Fs

        barriered = jax.lax.optimization_barrier(tuple(F_acc[k] for k in touched_keys))
        for k, v in zip(touched_keys, barriered):
            F_acc[k] = v

    num_vars = conserved_state.shape[0]
    components = [F_acc[i] if i in F_acc else F_interface[i] for i in range(num_vars)]
    return jnp.stack(components, axis=0)


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_y(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    if config.dimensionality == 2:
        qy = jnp.transpose(conserved_state, (0, 2, 1))
    elif config.dimensionality == 3:
        qy = jnp.transpose(conserved_state, (0, 2, 1, 3))

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

    Fy = _weno_flux_x(qy, params, config, registered_variables)

    if config.dimensionality == 2:
        Fy = jnp.transpose(Fy, (0, 2, 1))
    elif config.dimensionality == 3:
        Fy = jnp.transpose(Fy, (0, 2, 1, 3))

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
def _weno_flux_z(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    qz = jnp.transpose(conserved_state, (0, 3, 2, 1))

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

    Fz = _weno_flux_x(qz, params, config, registered_variables)

    Fz = jnp.transpose(Fz, (0, 3, 2, 1))

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
