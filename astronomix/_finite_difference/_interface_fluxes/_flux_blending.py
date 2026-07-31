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
                               registered_variables,
                               internal_energy_density=None):
    """First-order local Lax-Friedrichs (Rusanov) interface flux along ``axis``.

    ``F_LLF[..., i]`` is the flux at interface ``i+1/2`` (cells ``i`` and ``i+1``),
    matching the WENO convention so the blended array feeds the existing
    ``-dt/dx (F_{i+1/2} - F_{i-1/2})`` divergence unchanged. Returns the full
    interface-flux array.

    ``internal_energy_density`` (dual-energy ``g``), when given, switches the
    pressure recovery exactly like the WENO path does. Without it, the raw
    ``E - KE`` recovery is float32 cancellation garbage in cold KE-dominated
    cells, and a blend that activates there would inject fluxes built from a
    corrupted pressure — observed to blow up runs that blend broadly.
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
        eL = EL - keL
        eR = ER - keR
        if is_mhd:
            eL = eL - 0.5 * b2L
            eR = eR - 0.5 * b2R
        if internal_energy_density is not None:
            # Bryan+95 dual-energy switch, mirroring the WENO-side recovery
            eta = config.dual_energy_eta
            gL = internal_energy_density
            gR = _shift(internal_energy_density, -1, axis=axis)
            relL = (eL > eta * jnp.maximum(EL, 1e-30)) & (eL == eL)
            relR = (eR > eta * jnp.maximum(ER, 1e-30)) & (eR == eR)
            eL = jnp.where(relL, eL, gL)
            eR = jnp.where(relR, eR, gR)
        pL = jnp.maximum((gamma - 1.0) * eL, params.minimum_pressure)
        pR = jnp.maximum((gamma - 1.0) * eR, params.minimum_pressure)
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


def _hllc_flux(conserved_state, axis, params, config, registered_variables,
               internal_energy_density=None):
    """First-order HLLC interface flux (ideal-gas hydro) along ``axis``.

    Same convention and shape as :func:`_local_lax_friedrichs_flux`, and — with
    the Davis/Einfeldt wave-speed bounds used here — likewise positivity
    preserving under the CFL condition (Batten et al. 1997), so it is a valid
    low-order flux for the FCT limiter to blend toward.

    Why it matters: LLF smears the CONTACT wave at first order, so blending
    toward it erases cold dense condensations — precisely the structures a
    thermal instability builds (a two-phase medium imported from AthenaK was
    completely evaporated in 10 Myr through this path). HLLC restores the
    contact as an exact wave of the approximate Riemann fan, so the blend can
    still guarantee positivity without dissolving the cold phase.

    Only the ideal-gas hydro components are replaced; the LLF values are kept
    as the base for any other registered variables (and the whole flux falls
    back to LLF for MHD / isothermal, where this HLLC form does not apply).
    """
    F = _local_lax_friedrichs_flux(
        conserved_state, axis, params, config, registered_variables,
        internal_energy_density=internal_energy_density)
    if config.mhd or config.equation_of_state != IDEAL_GAS:
        return F

    ndim = config.dimensionality
    di = registered_variables.density_index
    ei = registered_variables.energy_index
    rhomin = params.minimum_density
    gamma = params.gamma

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

    def R(a):
        return _shift(a, -1, axis=axis)

    rhoL = jnp.maximum(conserved_state[di], rhomin)
    rhoR = jnp.maximum(R(conserved_state[di]), rhomin)
    mdL = conserved_state[md]
    mdR = R(mdL)
    vdL = mdL / rhoL
    vdR = mdR / rhoR
    veL = [conserved_state[m] / rhoL for m in mom_others]
    veR = [R(conserved_state[m]) / rhoR for m in mom_others]

    EL = conserved_state[ei]
    ER = R(EL)
    keL = 0.5 * (mdL * mdL) / rhoL
    keR = 0.5 * (mdR * mdR) / rhoR
    for ve in veL:
        keL = keL + 0.5 * rhoL * ve * ve
    for ve in veR:
        keR = keR + 0.5 * rhoR * ve * ve
    eL = EL - keL
    eR = ER - keR
    if internal_energy_density is not None:
        # same Bryan+95 switch as the LLF/WENO recoveries
        eta = config.dual_energy_eta
        gL = internal_energy_density
        gR = _shift(internal_energy_density, -1, axis=axis)
        relL = (eL > eta * jnp.maximum(EL, 1e-30)) & (eL == eL)
        relR = (eR > eta * jnp.maximum(ER, 1e-30)) & (eR == eR)
        eL = jnp.where(relL, eL, gL)
        eR = jnp.where(relR, eR, gR)
    pL = jnp.maximum((gamma - 1.0) * eL, params.minimum_pressure)
    pR = jnp.maximum((gamma - 1.0) * eR, params.minimum_pressure)
    cL = jnp.sqrt(gamma * pL / rhoL)
    cR = jnp.sqrt(gamma * pR / rhoR)

    # Davis/Einfeldt bounds: SL <= every wave <= SR
    SL = jnp.minimum(vdL - cL, vdR - cR)
    SR = jnp.maximum(vdL + cL, vdR + cR)

    dL = rhoL * (SL - vdL)
    dR = rhoR * (SR - vdR)
    denom = dL - dR
    tiny = jnp.finfo(conserved_state.dtype).tiny
    denom = jnp.where(jnp.abs(denom) > tiny, denom, jnp.sign(denom) * tiny + tiny)
    Sstar = (pR - pL + mdL * (SL - vdL) - mdR * (SR - vdR)) / denom

    def star_flux(rho, vd, ve_list, p, E, m_d, S, F_side_builder):
        gap = S - Sstar
        gap = jnp.where(jnp.abs(gap) > tiny, gap, jnp.sign(gap) * tiny + tiny)
        chi = (S - vd) / gap
        rho_s = rho * chi
        m_d_s = rho_s * Sstar
        sgap = S - vd
        sgap = jnp.where(jnp.abs(sgap) > tiny, sgap, jnp.sign(sgap) * tiny + tiny)
        E_s = chi * (E + (Sstar - vd) * (rho * Sstar + p / sgap))
        out = {di: rho_s, md: m_d_s, ei: E_s}
        for k, m in enumerate(mom_others):
            out[m] = rho_s * ve_list[k]
        return out

    UsL = star_flux(rhoL, vdL, veL, pL, EL, mdL, SL, None)
    UsR = star_flux(rhoR, vdR, veR, pR, ER, mdR, SR, None)

    UL = {di: rhoL, md: mdL, ei: EL}
    UR = {di: rhoR, md: mdR, ei: ER}
    for k, m in enumerate(mom_others):
        UL[m] = rhoL * veL[k]
        UR[m] = rhoR * veR[k]

    # physical fluxes on each side
    FLc = {di: mdL, md: mdL * vdL + pL, ei: (EL + pL) * vdL}
    FRc = {di: mdR, md: mdR * vdR + pR, ei: (ER + pR) * vdR}
    for k, m in enumerate(mom_others):
        FLc[m] = mdL * veL[k]
        FRc[m] = mdR * veR[k]

    for comp in [di, md, ei] + mom_others:
        f_star_L = FLc[comp] + SL * (UsL[comp] - UL[comp])
        f_star_R = FRc[comp] + SR * (UsR[comp] - UR[comp])
        f = jnp.where(
            SL >= 0.0, FLc[comp],
            jnp.where(Sstar >= 0.0, f_star_L,
                      jnp.where(SR >= 0.0, f_star_R, FRc[comp])))
        F = F.at[comp].set(f)
    return F


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
# Activation path 1b: cold-crush temperature ramp
# ---------------------------------------------------------------------------

def _coldcrush_blend_weight(conserved_state, axis, params, config,
                            registered_variables,
                            internal_energy_density=None):
    """LLF weight for radiatively crushed cells: interfaces that are both
    SUB-floor cold and CONVERGING.

    Two gates, both per interface:

    * temperature ramp — on the COLDER of the two adjacent cells' recovered
      ``p/rho``, ramping from 1 at (or below) the effective temperature
      floor ``minimum_specific_pressure`` down to 0 at
      ``coldcrush_blend_factor`` times it. Any interface with a cold side
      under compression gets the diffusive flux: cold-cold isothermal
      collapse AND the boundary faces of a cold dense clump being crushed
      by hot surroundings (the hero-4 failure mode — a hotter-side gate
      left exactly those faces unprotected). The price is that shock fronts
      advancing into cold ambient gas are handled at first order locally —
      the classic FOFC trade, and what the reference Athena SNR setups do.
    * convergence gate — the normal velocity must be compressive across the
      interface (``v_L > v_R``), ramped over the floor sound speed. The
      cold, freely-expanding ejecta core is divergent and never activates,
      so its seeded clump structure is not diffused away; the static cold
      ambient has no convergence and is untouched.
    """
    di = registered_variables.density_index
    gamma = params.gamma
    rhomin = params.minimum_density
    tfloor = params.minimum_specific_pressure  # p/rho at the floor temperature

    def R(a):
        return _shift(a, -1, axis=axis)

    rhoL = jnp.maximum(conserved_state[di], rhomin)
    rhoR = jnp.maximum(R(conserved_state[di]), rhomin)

    if config.dimensionality == 1:
        mom_all = [registered_variables.velocity_index]
    else:
        mom_all = [
            registered_variables.velocity_index.x,
            registered_variables.velocity_index.y,
            registered_variables.velocity_index.z,
        ][:config.dimensionality]
    keL = sum(conserved_state[m] ** 2 for m in mom_all) * 0.5 / rhoL
    keR = sum(R(conserved_state[m]) ** 2 for m in mom_all) * 0.5 / rhoR

    ei = registered_variables.energy_index
    EL = conserved_state[ei]
    ER = R(EL)
    eL = EL - keL
    eR = ER - keR
    if config.mhd:
        b2L = sum(conserved_state[b] ** 2 for b in (
            registered_variables.magnetic_index.x,
            registered_variables.magnetic_index.y,
            registered_variables.magnetic_index.z))
        eL = eL - 0.5 * b2L
        eR = eR - 0.5 * _shift(b2L, -1, axis=axis)

    if internal_energy_density is not None:
        # dual-energy switch: the raw e recovery is cancellation garbage in
        # exactly the cold cells this gate needs to classify
        eta = config.dual_energy_eta
        gL = internal_energy_density
        gR = _shift(internal_energy_density, -1, axis=axis)
        relL = (eL > eta * jnp.maximum(EL, 1e-30)) & (eL == eL)
        relR = (eR > eta * jnp.maximum(ER, 1e-30)) & (eR == eR)
        eL = jnp.where(relL, eL, gL)
        eR = jnp.where(relR, eR, gR)

    pL = jnp.maximum((gamma - 1.0) * eL, params.minimum_pressure)
    pR = jnp.maximum((gamma - 1.0) * eR, params.minimum_pressure)
    T_face = jnp.minimum(pL / rhoL, pR / rhoR)

    # temperature ramp on the colder side: 1 at (or below) the floor
    # temperature, 0 at factor * floor — any compressed cold side qualifies
    blend_thr = config.positivity_config.coldcrush_blend_factor * tfloor
    w_T = jnp.clip(
        (blend_thr - T_face) / jnp.maximum(blend_thr - tfloor, 1e-30), 0.0, 1.0
    )

    # convergence gate: compressive normal velocity, ramped over the floor
    # sound speed so it switches on smoothly
    ma = mom_all[axis] if config.dimensionality > 1 else mom_all[0]
    vdL = conserved_state[ma] / rhoL
    vdR = R(conserved_state[ma]) / rhoR
    c_floor = jnp.sqrt(gamma * jnp.maximum(tfloor, 1e-30))
    w_conv = jnp.clip((vdL - vdR) / c_floor, 0.0, 1.0)

    return w_T * w_conv


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
                          registered_variables, internal_energy_density=None):
    """Blend the WENO interface flux toward LLF along ``axis``, combining the
    enabled activation paths (deep-void density ramp and/or FCT positivity).

    The shared LLF flux is computed once; the LLF weight is the max over the
    active paths (robust as either demands).  ``dtdx`` is the stage CFL factor
    ``dt_tilde / grid_spacing`` (used by the positivity path).  Returns
    ``dF_weno`` unchanged if neither path is enabled.
    """
    use_deepvoid = config.positivity_config.deepvoid_blend
    use_ppflux = config.positivity_config.preserving_flux
    use_coldcrush = (config.positivity_config.coldcrush_blend
                     and config.equation_of_state == IDEAL_GAS)
    if not (use_deepvoid or use_ppflux or use_coldcrush):
        return dF_weno

    fallback = (_hllc_flux if config.positivity_config.blend_fallback_hllc
                else _local_lax_friedrichs_flux)
    F_llf = fallback(
        conserved_state, axis, params, config, registered_variables,
        internal_energy_density=internal_energy_density)

    w = None
    if use_deepvoid:
        w = _deepvoid_blend_weight(
            conserved_state, axis, params, config, registered_variables)
    if use_coldcrush:
        w_cc = _coldcrush_blend_weight(
            conserved_state, axis, params, config, registered_variables,
            internal_energy_density=internal_energy_density)
        w = w_cc if w is None else jnp.maximum(w, w_cc)
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
