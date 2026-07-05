"""Unified WENO interface-flux blending toward first-order Lax-Friedrichs.

Both robustness fix-ups in the FD WENO scheme are the SAME operation — blend the
high-order interface flux toward a first-order local Lax-Friedrichs (Rusanov)
flux,

    F_hat_{i+1/2} = (1 - w_{i+1/2}) F_WENO_{i+1/2} + w_{i+1/2} F_LLF_{i+1/2},

with a per-interface LLF weight ``w in [0, 1]``.  They differ ONLY in how that
weight is chosen — the *activation path*:

  * deep-void path (``config.positivity_config.deepvoid_blend``): ``w`` ramps from 1 at
    the density floor to 0 at ``blend_factor * minimum_density``.  Cures the
    high-Mach characteristic overshoot in the immediate neighbourhood of a deep
    void (see ``_deepvoid_blend_weight``).

  * positivity-preserving path (``config.positivity_config.preserving_flux``): ``w`` is
    the SMALLEST LLF fraction (largest WENO fraction) for which the LF-updated
    cell keeps both density and pressure above their floors — a Hu-Adams-Shu
    (2013) / Zalesak-FCT limiter (see ``_ppflux_blend_weight``).  Cures the
    WENO over-depletion / negative-pressure overshoot that crashes a violent
    self-gravity collapse.

Both paths share the single LLF flux ``_local_lax_friedrichs_flux`` (hydro and
isothermal/ideal MHD).  When both are enabled the unified blend takes the
stronger (max) weight, so the flux is as robust as either path demands.  This is
a native-JAX post-process on the assembled interface-flux array, applied before
the divergence; it does not touch the Pallas WENO kernel.  It is CT-safe for MHD
(CT rebuilds single-valued edge EMFs from whatever face fluxes it is given, so
div(B)=0 is preserved; blending toward LLF merely adds a localised magnetic
diffusivity at the trouble cell, and the normal-B flux is overwritten by CT).
"""

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _shift


# ---------------------------------------------------------------------------
# Shared first-order Lax-Friedrichs interface flux (hydro & MHD, iso & ideal)
# ---------------------------------------------------------------------------

def _local_lax_friedrichs_flux(conserved_state, axis, params, config,
                               registered_variables):
    """First-order local Lax-Friedrichs (Rusanov) interface flux along ``axis``.

    ``F_LLF[..., i]`` is the flux at interface ``i+1/2`` (cells ``i`` and ``i+1``),
    matching the WENO convention so the blended array feeds the existing
    ``-dt/dx (F_{i+1/2} - F_{i-1/2})`` divergence unchanged. Returns the full
    interface-flux array.
    """
    ndim = config.dimensionality
    di = registered_variables.density_index
    rhomin = params.minimum_density
    is_ideal = (config.equation_of_state == IDEAL_GAS)
    is_mhd = bool(config.mhd)

    if ndim == 1:
        mom_all = [registered_variables.velocity_index]
    else:
        mom_all = [
            registered_variables.velocity_index.x,
            registered_variables.velocity_index.y,
            registered_variables.velocity_index.z,
        ][:ndim]
    md = mom_all[axis]
    mom_others = [m for i, m in enumerate(mom_all) if i != axis]

    if is_mhd:
        B_all = [
            registered_variables.magnetic_index.x,
            registered_variables.magnetic_index.y,
            registered_variables.magnetic_index.z,
        ]
        Bd = B_all[axis]
        B_others = [B_all[i] for i in range(3) if i != axis]

    def R(a):
        return _shift(a, -1, axis=axis)

    def R_state(a):
        return _shift(a, -1, axis=axis + 1)

    rhoL = jnp.maximum(conserved_state[di], rhomin)
    rhoR = jnp.maximum(R(conserved_state[di]), rhomin)
    mdL = conserved_state[md]
    mdR = R(conserved_state[md])
    vdL = mdL / rhoL
    vdR = mdR / rhoR

    veL = [conserved_state[m] / rhoL for m in mom_others]
    veR = [R(conserved_state[m]) / rhoR for m in mom_others]

    if is_mhd:
        BdL = conserved_state[Bd]
        BdR = R(conserved_state[Bd])
        BeL = [conserved_state[b] for b in B_others]
        BeR = [R(conserved_state[b]) for b in B_others]
        b2L = BdL * BdL
        b2R = BdR * BdR
        for bl, br in zip(BeL, BeR):
            b2L = b2L + bl * bl
            b2R = b2R + br * br

    if is_ideal:
        gamma = params.gamma
        EL = conserved_state[registered_variables.energy_index]
        ER = R(EL)
        keL = 0.5 * (mdL * mdL) / rhoL
        keR = 0.5 * (mdR * mdR) / rhoR
        for ve in veL:
            keL = keL + 0.5 * rhoL * ve * ve
        for ve in veR:
            keR = keR + 0.5 * rhoR * ve * ve
        if is_mhd:
            pL = jnp.maximum((gamma - 1.0) * (EL - keL - 0.5 * b2L), params.minimum_pressure)
            pR = jnp.maximum((gamma - 1.0) * (ER - keR - 0.5 * b2R), params.minimum_pressure)
        else:
            pL = jnp.maximum((gamma - 1.0) * (EL - keL), params.minimum_pressure)
            pR = jnp.maximum((gamma - 1.0) * (ER - keR), params.minimum_pressure)
        cs2L = gamma * pL / rhoL
        cs2R = gamma * pR / rhoR
    else:
        cs = params.isothermal_sound_speed
        cs2L = cs * cs
        cs2R = cs * cs
        pL = cs2L * rhoL
        pR = cs2R * rhoR

    if is_mhd:
        def cfast(b2, rho, Bn, cs2):
            b2_over_rho = b2 / rho
            bn2_over_rho = (Bn * Bn) / rho
            disc = jnp.maximum((b2_over_rho + cs2) ** 2 - 4.0 * bn2_over_rho * cs2, 0.0)
            return jnp.sqrt(jnp.maximum(0.5 * (b2_over_rho + cs2 + jnp.sqrt(disc)), 0.0))
        cL = cfast(b2L, rhoL, BdL, cs2L)
        cR = cfast(b2R, rhoR, BdR, cs2R)
    else:
        cL = jnp.sqrt(cs2L)
        cR = jnp.sqrt(cs2R)

    alpha = jnp.maximum(jnp.abs(vdL) + cL, jnp.abs(vdR) + cR)

    qR = R_state(conserved_state)
    FL = jnp.zeros_like(conserved_state)
    FR = jnp.zeros_like(conserved_state)

    FL = FL.at[di].set(mdL)
    FR = FR.at[di].set(mdR)

    fmdL = mdL * vdL + pL
    fmdR = mdR * vdR + pR
    if is_mhd:
        fmdL = fmdL + 0.5 * b2L - BdL * BdL
        fmdR = fmdR + 0.5 * b2R - BdR * BdR
    FL = FL.at[md].set(fmdL)
    FR = FR.at[md].set(fmdR)

    for k, m in enumerate(mom_others):
        feL = mdL * veL[k]
        feR = mdR * veR[k]
        if is_mhd:
            feL = feL - BdL * BeL[k]
            feR = feR - BdR * BeR[k]
        FL = FL.at[m].set(feL)
        FR = FR.at[m].set(feR)

    if is_mhd:
        FL = FL.at[Bd].set(jnp.zeros_like(BdL))
        FR = FR.at[Bd].set(jnp.zeros_like(BdR))
        for k, b in enumerate(B_others):
            FL = FL.at[b].set(BeL[k] * vdL - BdL * veL[k])
            FR = FR.at[b].set(BeR[k] * vdR - BdR * veR[k])

    if is_ideal:
        ei = registered_variables.energy_index
        if is_mhd:
            vdotBL = vdL * BdL
            vdotBR = vdR * BdR
            for k in range(len(mom_others)):
                vdotBL = vdotBL + veL[k] * BeL[k]
                vdotBR = vdotBR + veR[k] * BeR[k]
            FL = FL.at[ei].set((EL + pL + 0.5 * b2L) * vdL - BdL * vdotBL)
            FR = FR.at[ei].set((ER + pR + 0.5 * b2R) * vdR - BdR * vdotBR)
        else:
            FL = FL.at[ei].set((EL + pL) * vdL)
            FR = FR.at[ei].set((ER + pR) * vdR)

    return 0.5 * (FL + FR) - 0.5 * alpha * (qR - conserved_state)


# ---------------------------------------------------------------------------
# Activation path 1: deep-void density ramp
# ---------------------------------------------------------------------------

def _deepvoid_blend_weight(conserved_state, axis, params, config,
                           registered_variables):
    """LLF weight that ramps from 1 at ``minimum_density`` to 0 at
    ``positivity_deepvoid_blend_factor * minimum_density`` (per interface, using
    the smaller of the two adjacent densities)."""
    di = registered_variables.density_index
    rhomin = params.minimum_density
    rhoL = jnp.maximum(conserved_state[di], rhomin)
    rhoR = jnp.maximum(_shift(conserved_state[di], -1, axis=axis), rhomin)
    rho_face = jnp.minimum(rhoL, rhoR)
    blend_thr = config.positivity_config.deepvoid_blend_factor * rhomin
    return jnp.clip((blend_thr - rho_face) / (blend_thr - rhomin), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Activation path 2: Hu-Adams-Shu / Zalesak positivity-preserving limiter
# ---------------------------------------------------------------------------

def _momentum_components(config, registered_variables):
    """Return the conserved-state momentum component indices for this run,
    truncated to the active dimensionality."""
    if config.dimensionality == 1:
        return [registered_variables.velocity_index]
    return [registered_variables.velocity_index.x,
            registered_variables.velocity_index.y,
            registered_variables.velocity_index.z][:config.dimensionality]


def _internal_energy_residual(U, config, registered_variables, e_floor):
    """q = rho*(E - e_floor) - 0.5|m|^2 (- 0.5*rho*|B|^2)  >= 0  <=>  p >= pgmin.
    Affine-in-t evaluation of the pressure constraint along the LF->WENO segment
    (quadratic for hydro, cubic for MHD — handled by direct evaluation)."""
    rho = U[registered_variables.density_index]
    m2 = sum(U[m] ** 2 for m in _momentum_components(config, registered_variables))
    res = rho * (U[registered_variables.energy_index] - e_floor) - 0.5 * m2
    if config.mhd:
        b2 = (U[registered_variables.magnetic_index.x] ** 2
              + U[registered_variables.magnetic_index.y] ** 2
              + U[registered_variables.magnetic_index.z] ** 2)
        res = res - 0.5 * rho * b2
    return res


def _ppflux_blend_weight(dF_weno, F_llf, conserved_state, axis, dtdx, params,
                         config, registered_variables):
    """LLF weight w = 1 - theta_keep, where theta_keep in [0,1] is the largest
    fraction of the antidiffusive flux ``A = F_WENO - F_LLF`` that keeps the
    LF-updated density (Zalesak) AND pressure (HAS) above their floors."""
    fa = axis
    va = axis + 1
    rhomin = params.minimum_density
    pgmin = params.minimum_pressure
    gamma = params.gamma
    di = registered_variables.density_index
    U = conserved_state

    A = dF_weno - F_llf

    # density limiter (Zalesak lower bound)
    A_rho = A[di]
    F_LF_rho = F_llf[di]
    rho_LF_new = U[di] - dtdx * (F_LF_rho - _shift(F_LF_rho, 1, axis=fa))
    P_minus = dtdx * (jnp.maximum(0.0, A_rho)
                      + jnp.maximum(0.0, -_shift(A_rho, 1, axis=fa)))
    Q_minus = jnp.maximum(rho_LF_new - rhomin, 0.0)
    R_minus = jnp.where(P_minus > 1e-30, jnp.minimum(1.0, Q_minus / P_minus), 1.0)
    theta_rho = jnp.where(A_rho >= 0.0, R_minus, _shift(R_minus, -1, axis=fa))

    # pressure limiter (ideal gas only; isothermal pressure is always positive)
    if config.equation_of_state == IDEAL_GAS:
        A1 = theta_rho[None, ...] * A
        U_LF = U - dtdx * (F_llf - _shift(F_llf, 1, axis=va))
        dU = -dtdx * (A1 - _shift(A1, 1, axis=va))
        e_floor = pgmin / (gamma - 1.0)
        c = _internal_energy_residual(U_LF, config, registered_variables, e_floor)
        q1 = _internal_energy_residual(U_LF + dU, config, registered_variables, e_floor)
        lo = jnp.zeros_like(c)
        hi = jnp.ones_like(c)
        for _ in range(30):  # bisect the per-cell admissible fraction
            mid = 0.5 * (lo + hi)
            qmid = _internal_energy_residual(
                U_LF + mid[None, ...] * dU, config, registered_variables, e_floor)
            ok = qmid >= 0.0
            lo = jnp.where(ok, mid, lo)
            hi = jnp.where(ok, hi, mid)
        t_cell = jnp.where(q1 >= 0.0, 1.0, lo)
        t_cell = jnp.where(c >= 0.0, t_cell, 0.0)
        theta_p = jnp.minimum(t_cell, _shift(t_cell, -1, axis=fa))
    else:
        theta_p = jnp.ones_like(theta_rho)

    theta_keep = theta_rho * theta_p
    return 1.0 - theta_keep


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def _blend_interface_flux(dF_weno, conserved_state, axis, dtdx, params, config,
                          registered_variables):
    """Blend the WENO interface flux toward LLF along ``axis``, combining the
    enabled activation paths (deep-void density ramp and/or FCT positivity).

    The shared LLF flux is computed once; the LLF weight is the max over the
    active paths (robust as either demands).  ``dtdx`` is the stage CFL factor
    ``dt_tilde / grid_spacing`` (used by the positivity path).  Returns
    ``dF_weno`` unchanged if neither path is enabled.
    """
    use_deepvoid = config.positivity_config.deepvoid_blend
    use_ppflux = config.positivity_config.preserving_flux
    if not (use_deepvoid or use_ppflux):
        return dF_weno

    F_llf = _local_lax_friedrichs_flux(
        conserved_state, axis, params, config, registered_variables)

    w = None
    if use_deepvoid:
        w = _deepvoid_blend_weight(
            conserved_state, axis, params, config, registered_variables)
    if use_ppflux:
        w_pp = _ppflux_blend_weight(
            dF_weno, F_llf, conserved_state, axis, dtdx, params, config,
            registered_variables)
        w = w_pp if w is None else jnp.maximum(w, w_pp)

    w = w[None, ...]
    return dF_weno * (1.0 - w) + F_llf * w


# Back-compat alias: the deep-void-only entry point (density ramp only). Kept so
# any external callers of the old name keep working; new code uses
# _blend_interface_flux with the activation paths selected via config.
def _deepvoid_llf_blend(dF_weno, conserved_state, axis, params, config,
                        registered_variables):
    """Deep-void-only LLF blend (density ramp). Back-compat entry point; new
    code uses ``_blend_interface_flux`` with the activation paths selected via
    config."""
    F_llf = _local_lax_friedrichs_flux(
        conserved_state, axis, params, config, registered_variables)
    w = _deepvoid_blend_weight(
        conserved_state, axis, params, config, registered_variables)[None, ...]
    return dF_weno * (1.0 - w) + F_llf * w
