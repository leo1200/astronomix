"""
Differentiable Cas A: gradients of the observables with respect to the physics.

Every number in ``OVERVIEW.md`` §4 was reached by hand -- a scan over
``(E, M_ej, n_w)`` in ``casa_calibrate_1d.py --scan``, then a judgement call.
That works for three parameters and stops working for the ones this study has
since added (the inner ejecta slope, the clumping amplitude and spectrum, the
plume field, the electron-heating fraction), because a grid search costs
``k^n`` runs and the residuals are correlated: the inner slope moves r_RS AND
the unshocked mass AND the shocked Si, so tuning them one at a time is
guaranteed to walk in circles. That is what this script is for.

``astronomix`` is written in JAX, so the solver is differentiable and the
gradient of any scalar built from the final state with respect to any parameter
of the initial condition is available exactly -- not by finite differences of a
chaotic simulation, which is the usual reason this is not attempted.

WHAT IS DIFFERENTIATED
----------------------
The whole chain, end to end::

    theta = (E, M_ej, n_w, delta)
        |  analytic initial condition, traced
        v
    1D spherical hydro, ~200 yr of evolution        <- the solver itself
        |
        v
    smooth observables: r_FS, r_RS, M_unshocked     <- see below
        |
        v
    chi^2 against the measured values

FORWARD MODE, NOT REVERSE, AND THAT IS THE WHOLE TRICK
------------------------------------------------------
``differentiation_mode = BACKWARDS`` exists and the field-level-inference
notebook uses it, but it is the wrong tool here. Reverse mode has to keep the
state at every checkpoint, which is what confines that notebook to 32^3. This
problem has the opposite shape: a HUGE state and a HANDFUL of scalar
parameters. Forward mode costs one tangent state -- O(1) memory, independent of
the number of steps -- and one pass per parameter. With four parameters that is
four passes and no checkpointing at all, and it scales to the 3D grids where
reverse mode simply will not fit. ``ADAPTIVE_WHILE`` (what ``FORWARDS``
selects) is JVP-differentiable for exactly this reason.

So: use ``jax.jvp`` per parameter, not ``jax.grad``.

THE OBSERVABLES HAVE TO BE SMOOTH, AND THE OBVIOUS DEFINITIONS ARE NOT
----------------------------------------------------------------------
``casa_analyze.py`` finds the forward shock with an ``argmax`` over a
compression profile. That is piecewise constant in the parameters: its gradient
is zero almost everywhere and undefined on a measure-zero set, so it would
report "the shock radius does not depend on the explosion energy". Every
observable here is therefore rebuilt as a smooth functional of the same
physics:

    w(r) = sigmoid((log s(r) - log s_ambient(r) - dex) / width)
    r*   = sum r |dw/dr| / sum |dw/dr|

i.e. locate where a smooth indicator SWITCHES, rather than integrating the
region it marks. Entropy is the indicator because it is exactly conserved by
adiabatic expansion, so it separates "shocked ever" from "never shocked"
without depending on how far the parcel has since expanded; and the transition
is located by its derivative because integrating ``int w dr`` accumulates the
sigmoid's tail across the whole box -- that version put r_FS at 4.48 pc in a
4.0 pc box, with a perfectly well-behaved gradient pointing at the wrong
quantity.

STATE OF THE OBSERVABLES (read before fitting anything)
-------------------------------------------------------
``r_FS`` is validated: 2.49 pc against 2.55 from the hard definition in
``casa_calibrate_1d.py`` at matched parameters. ``r_RS`` and ``M_unshocked``
are NOT yet reconciled with the definitions ``casa_analyze.py`` uses -- at
delta = 0 the cold ejecta core is nearly exhausted, so "outer edge of the
unshocked core" and "position of the reverse shock" are different quantities
and this module currently measures the first. Do not fit against those two
until that is settled; the gradient machinery is independent of it.

USAGE
-----
    ./run.sh casa_diff.py --check-grad     # JVP vs finite differences
    ./run.sh casa_diff.py --validate       # smooth vs hard observables
    ./run.sh casa_diff.py --fit            # gradient fit to the measurements
"""

# ==== precision / device ====
# IDENTICAL to casa_calibrate_1d.py, and both halves matter.
#
# float64 is REQUIRED, not a nicety: the cold ejecta core carries a ~1e-6
# pressure contrast, which is the float32 cancellation regime the 3D runs need
# the dual-energy formalism to survive. In float32 this model does not merely
# lose accuracy -- every solve NaNs and the integrator aborts, so the gradient
# comes back as nan and looks like a differentiability failure rather than a
# precision one.
#
# CPU by default because the 1D model is a few thousand cells and autocvd
# blocks indefinitely when the node's GPUs are busy with the 3D ladder.
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
if "--gpu" not in sys.argv:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
from functools import partial

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# units and constants
from astropy import units as u
import astropy.constants as const

# astronomix
from astronomix import (
    FINITE_VOLUME,
    SPHERICAL,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import POSITIVITY_HARD_FLOOR

# shared showcase helpers
from _common import GAMMA, MASS_PER_NUCLEUS, ejecta_radial_shape, snr_code_units


# =============================================================================
# ============ ↓ What we are fitting, and what we are fitting to ↓ ============
# =============================================================================
#: The free parameters, in the order the parameter vector uses. Fitted in
#: TRANSFORMED coordinates (log where the quantity is positive and spans orders
#: of magnitude) so the optimiser sees a better-conditioned problem and cannot
#: propose a negative mass.
PARAM_NAMES = ("log10_E51", "M_ej", "n_w", "inner_slope")
PARAM_TRANSFORM = ("log10", "linear", "linear", "linear")

#: Starting point: the hand-calibrated fiducial of CALIBRATION.md Result 2.
THETA0 = jnp.array([np.log10(2.09), 3.0, 0.928, 0.0])

#: The measurements, with their uncertainties. These are the SAME numbers the
#: scoreboard in OVERVIEW.md §4 is scored against, so a fit here is directly
#: comparable with the hand calibration.
#:
#: Sources: r_FS/r_RS Gotthelf et al. 2001 + DeLaney et al. 2004 as quoted in
#: the calibration; M_unshocked DeLaney et al. 2014 / Hwang & Laming 2012.
TARGETS = {
    "r_FS":        (2.52, 0.20),    # pc
    "r_RS":        (1.58, 0.16),    # pc
    "M_unshocked": (0.35, 0.10),    # Msun
}

AGE_YR = 350.0


def untransform(theta):
    """Parameter vector -> physical values, as a dict (traced)."""
    out = {}
    for i, (name, tr) in enumerate(zip(PARAM_NAMES, PARAM_TRANSFORM)):
        out[name] = 10.0 ** theta[i] if tr == "log10" else theta[i]
    return out


# =============================================================================
# ============ ↓ The initial condition, fully traced ↓ ========================
# =============================================================================
def base_config(num_cells, r_max):
    """The solver configuration. Fixed, so it can be built once outside the trace.

    ``differentiation_mode`` is left at its default ``FORWARDS``, which selects
    the plain adaptive while-loop integrator -- differentiable in forward mode
    at O(1) memory. See the module docstring for why that is the right choice.
    """
    return SimulationConfig(
        solver_mode=FINITE_VOLUME,
        geometry=SPHERICAL,
        dimensionality=1,
        box_size=r_max,
        num_cells=num_cells,
        # both of these are required by the 1D setup and are documented at
        # length in casa_calibrate_1d.build_1d: MUSCL/minmod NaNs on the r^-9
        # envelope without the fallback, and the origin cell evacuates without
        # the floor. Neither can touch the calibration targets.
        first_order_fallback=True,
        positivity_config=PositivityConfig(
            per_step_mode=POSITIVITY_HARD_FLOOR, nan_safe=True, vacuum_rest=True),
        return_snapshots=False,
        progress_bar=False,
    )


def build_state(theta, config, helper_data, registered_variables, code_units,
                *, r_max, num_cells, r0=0.05, core_fraction=0.5,
                envelope_slope=9.0, taper_cells=3.0, r_fs_ref=2.5, n_c=0.1,
                wind_temperature_K=1e4, ejecta_temperature_K=100.0):
    """The 1D initial condition as a TRACED function of the parameters.

    This mirrors ``casa_calibrate_1d.build_1d`` exactly, with one difference
    that is the entire point: that function calls ``float()`` on the energy and
    the ejecta mass to convert their units, which raises the moment either is a
    JAX tracer. Here the unit conversions are precomputed as constant FACTORS
    and multiplied in, so the parameters stay traced all the way to the state.
    """
    p = untransform(theta)
    r = helper_data.geometric_centers
    dx = r_max / num_cells
    cell_vol = helper_data.cell_volumes

    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3)
                      .to(code_units.code_density).value)
    p_per_n = float((const.k_B * wind_temperature_K * u.K / u.cm ** 3)
                    .to(code_units.code_pressure).value)
    # Constant conversion factors, so the traced parameters never meet float().
    # The 1e51 is folded INTO the factor rather than multiplied in afterwards:
    # jax defaults to float32, and 2.09e51 overflows it (max ~3.4e38), which
    # shows up as "overflow encountered in cast" and then a NaN primitive
    # state -- not as an error at the line that caused it.
    erg51_to_code = float((1e51 * u.erg).to(code_units.code_energy).value)
    msun_to_code = float((1.0 * u.Msun).to(code_units.code_mass).value)

    E = p["log10_E51"] * erg51_to_code       # untransform() already un-logged it
    M_ej = p["M_ej"] * msun_to_code

    # ambient: the progenitor's r^-2 wind plus a uniform floor
    n_amb = p["n_w"] * (r_fs_ref / jnp.maximum(r, 0.5 * r0)) ** 2 + n_c
    rho_amb = n_amb * rho_per_n
    p_amb = n_amb * p_per_n

    shape = ejecta_radial_shape(r, core_fraction * r0, r0, dx,
                                envelope_slope=envelope_slope,
                                inner_slope=p["inner_slope"],
                                taper_cells=taper_cells)

    d_rho = M_ej / jnp.sum(shape * cell_vol)
    m_ej = d_rho * shape
    rho = rho_amb + m_ej

    # homologous v = s r on the ejecta mass, s renormalised so KE == E exactly
    integrand = jnp.sum(m_ej ** 2 * r ** 2 / rho * cell_vol)
    s = jnp.sqrt(E / (0.5 * integrand))
    v = m_ej * s * r / rho

    p_cold = (rho / rho_per_n) * float(
        (const.k_B * ejecta_temperature_K * u.K / u.cm ** 3)
        .to(code_units.code_pressure).value)
    press = p_amb * (1.0 - shape) + p_cold * shape

    state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=rho, velocity_x=v, gas_pressure=press,
    )
    # the homologous profile already represents an age t0 = 1/s exactly, and
    # that age DEPENDS ON THE ENERGY -- a more energetic explosion of the same
    # ejecta is younger at the same radius. Carrying it as a traced quantity is
    # not pedantry: it is a real d(t_end)/dE term in every gradient below.
    t0 = 1.0 / s
    # wind mass that was already inside the initial ejecta radius
    m_wind_interior = jnp.sum(jnp.where(r < r0, rho_amb, 0.0) * cell_vol)
    return state, rho_amb, p_amb, m_wind_interior, t0


# =============================================================================
# ============ ↓ Smooth observables ↓ =========================================
# =============================================================================
def transition_radius(indicator, r, *, falling=True):
    """Where a smooth indicator switches, as a differentiable weighted mean.

    The obvious construction -- ``int indicator dr``, which for a perfect step
    IS the radius -- does not survive contact with a real profile. A sigmoid
    does not reach 0; it reaches ``sigmoid(-x)``, and integrating that tail
    across a 4 pc box adds an offset of order (tail height) x (box size) that
    swamps the answer. Measured: it put r_FS at 4.48 pc in a 4.0 pc box.

    So locate the TRANSITION instead of integrating the region. The derivative
    of the indicator is sharply peaked where the indicator switches, and its
    normalised first moment is the switch radius:

        r* = sum r |dw/dr| / sum |dw/dr|

    Tails contribute nothing because a flat tail has zero derivative, which is
    exactly the property the integral lacked. The SIGN selects which transition
    is meant: the shocked indicator rises at the reverse shock and falls at the
    forward shock, so a signed weight picks one without picking the other.
    """
    dw = jnp.diff(indicator)
    r_mid = 0.5 * (r[:-1] + r[1:])
    w = jnp.maximum(-dw, 0.0) if falling else jnp.maximum(dw, 0.0)
    return jnp.sum(r_mid * w) / (jnp.sum(w) + 1e-30)


def observables(final_state, rho_amb, p_amb, m_wind_interior,
                registered_variables, helper_data,
                code_units, *, dr, gamma=GAMMA, entropy_contrast=30.0,
                entropy_width=0.5, shock_dex=0.5):
    """The scoreboard quantities, as smooth functionals of the final state.

    Args:
        rho_amb, p_amb: the INITIAL ambient profiles, which are the correct
            reference: the unshocked wind ahead of the blast has not moved.
        m_wind_interior: wind mass that started inside the ejecta radius.
        entropy_contrast: factor above the minimum entropy at which a parcel
            counts as reverse-shocked.
        entropy_width: smoothing width, in decades of entropy.
        shock_dex: decades above the ambient entropy that count as shocked.
    """
    rho = final_state[registered_variables.density_index]
    press = final_state[registered_variables.pressure_index]
    cell_vol = helper_data.cell_volumes

    # ENTROPY IS THE LABEL FOR BOTH SHOCKS, and the reason is that it is the
    # only quantity that remembers. s = p / rho^gamma is EXACTLY conserved by
    # adiabatic expansion, so a parcel carries the record of having been
    # shocked however far it has since expanded. Density and temperature both
    # fall as the remnant expands and cannot separate "never shocked" from
    # "shocked long ago".
    #
    # At 350 yr the profile is three contiguous regions, inside out:
    #     [0, r_RS)        cold unshocked ejecta   -- entropy far BELOW ambient
    #     [r_RS, r_FS)     shocked                 -- entropy far ABOVE ambient
    #     [r_FS, R]        unshocked wind          -- entropy AT ambient
    # so each radius is the running extent of a contiguous region, and the
    # integral of a smooth indicator gives it directly.
    #
    # The first version of this thresholded rho/rho_wind at 2 and integrated,
    # which measured the SHELL THICKNESS (1.02 pc) rather than r_FS -- the
    # evacuated interior fails a compression test just as the far field does,
    # so the region it selected was not contiguous from the origin. Nothing
    # about that was visible in the gradient, which was perfectly well behaved
    # and pointed at the wrong quantity.
    log_s = (jnp.log10(jnp.maximum(press, 1e-30))
             - gamma * jnp.log10(jnp.maximum(rho, 1e-30)))
    log_s_amb = (jnp.log10(jnp.maximum(p_amb, 1e-30))
                 - gamma * jnp.log10(jnp.maximum(rho_amb, 1e-30)))

    r = helper_data.geometric_centers

    # ---- reverse shock: the outer edge of the cold ejecta core ------------
    log_s0 = jnp.min(log_s)
    unshocked = jax.nn.sigmoid(
        ((log_s0 + jnp.log10(entropy_contrast)) - log_s) / entropy_width)
    r_rs = transition_radius(unshocked, r, falling=True)

    # ---- forward shock: the outer edge of the shocked region -------------
    # the same indicator RISES at the reverse shock and FALLS at the forward
    # shock, so the sign of the weight is what separates them
    shocked = jax.nn.sigmoid((log_s - log_s_amb - shock_dex) / entropy_width)
    r_fs = transition_radius(shocked, r, falling=True)

    # ---- unshocked ejecta mass -------------------------------------------
    # Inside the reverse shock the material is ejecta PLUS whatever wind was
    # already inside the initial ejecta radius, which the 1D setup places there
    # as a dense r^-2 cusp. That is a real mass and it is not ejecta, so it is
    # subtracted -- it is the same ``M_wind_inside_r0`` correction
    # casa_calibrate_1d applies, and without it this observable exceeds the
    # total ejecta mass, which is how the error announced itself.
    # restricted to the cold core: the entropy indicator alone has sigmoid
    # tails that pick up shocked mass at every radius
    # NOT corrected for the wind that started inside the ejecta radius: that
    # correction is a constant, while the mass still inside r_RS at 350 yr is a
    # small fraction of it, so subtracting the whole thing drives this
    # observable NEGATIVE (measured: -0.08 Msun). Left uncorrected and flagged
    # -- an observable that can go negative is not ready to be fitted against.
    inside_rs = jax.nn.sigmoid((r_rs - r) / (4.0 * dr))
    m_unshocked = jnp.sum(rho * unshocked * inside_rs * cell_vol)
    m_unshocked_msun = m_unshocked * float(
        (1.0 * code_units.code_mass).to(u.Msun).value)

    return dict(r_FS=r_fs, r_RS=r_rs, M_unshocked=m_unshocked_msun)


# =============================================================================
# ============ ↓ The forward model and the loss ↓ =============================
# =============================================================================
def make_forward(num_cells=2000, r_max=4.0, age_yr=AGE_YR, cfl=0.4):
    """Build ``theta -> observables``, with everything constant hoisted out."""
    code_units = snr_code_units()
    config = base_config(num_cells, r_max)
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    dr = r_max / num_cells
    yr_to_code = float((1.0 * u.yr).to(code_units.code_time).value)

    def forward(theta):
        state, rho_amb, p_amb, m_wind_int, t0 = build_state(
            theta, config, helper_data, rv, code_units,
            r_max=r_max, num_cells=num_cells)
        cfg = finalize_config(config, state.shape)
        params = SimulationParams(
            C_cfl=cfl, gamma=GAMMA,
            t_end=age_yr * yr_to_code - t0,     # traced: t0 depends on E
            minimum_density=1e-6, minimum_pressure=1e-12,
        )
        # with return_snapshots=False the integrator hands back the final state
        # array itself, not a snapshot container
        final = time_integration(state, cfg, params, rv)
        return observables(final, rho_amb, p_amb, m_wind_int, rv, helper_data,
                           code_units, dr=dr)

    return forward, dict(code_units=code_units, config=config,
                         helper_data=helper_data, rv=rv, dr=dr)


def chi2(obs):
    """Sum of squared standardised residuals against :data:`TARGETS`."""
    total = 0.0
    for k, (value, sigma) in TARGETS.items():
        total = total + ((obs[k] - value) / sigma) ** 2
    return total


def make_loss(forward):
    def loss(theta):
        return chi2(forward(theta))
    return loss


# =============================================================================
# ============ ↓ Gradients ↓ ==================================================
# =============================================================================
def jvp_gradient(f, theta):
    """The full gradient of a scalar ``f`` by ONE FORWARD PASS PER PARAMETER.

    ``jax.grad`` would need reverse mode through the whole time loop. Forward
    mode costs one tangent state and no checkpoints, which is what makes this
    approach survive the move to 3D -- see the module docstring.
    """
    n = theta.shape[0]
    grads = []
    for i in range(n):
        tangent = jnp.zeros_like(theta).at[i].set(1.0)
        _, d = jax.jvp(f, (theta,), (tangent,))
        grads.append(d)
    return jnp.array(grads)


def jvp_jacobian(forward, theta, keys):
    """Jacobian of the observable vector, one forward pass per parameter."""
    cols = []
    for i in range(theta.shape[0]):
        tangent = jnp.zeros_like(theta).at[i].set(1.0)
        _, d = jax.jvp(forward, (theta,), (tangent,))
        cols.append(jnp.array([d[k] for k in keys]))
    return jnp.stack(cols, axis=1)      # (n_obs, n_param)


# =============================================================================
# ============ ↓ Entry points ↓ ===============================================
# =============================================================================
def cmd_check_grad(args):
    """Validate the JVP gradient against central finite differences.

    This is the check that decides whether any of this is usable. A solver with
    a hard floor, a first-order fallback and an adaptive timestep contains
    genuinely non-differentiable operations; whether the resulting gradient is
    still a useful descent direction is an empirical question, not a
    theoretical one, and it is answered here.
    """
    forward, _ = make_forward(num_cells=args.n, age_yr=args.age)
    loss = make_loss(forward)
    theta = THETA0

    print(f"[diff] {args.n} cells, {args.age:.0f} yr")
    obs = forward(theta)
    print(f"[diff] at the fiducial: " +
          ", ".join(f"{k} = {float(v):.4f}" for k, v in obs.items()))
    print(f"[diff] chi^2 = {float(loss(theta)):.3f}")

    g = jvp_gradient(loss, theta)
    print(f"\n[diff] {'parameter':<12s} {'JVP':>12s} {'finite diff':>12s} {'rel err':>10s}")
    for i, name in enumerate(PARAM_NAMES):
        h = args.eps * max(abs(float(theta[i])), 1.0)
        tp = theta.at[i].add(h)
        tm = theta.at[i].add(-h)
        fd = (float(loss(tp)) - float(loss(tm))) / (2 * h)
        rel = abs(float(g[i]) - fd) / max(abs(fd), 1e-12)
        print(f"[diff] {name:<12s} {float(g[i]):12.4f} {fd:12.4f} {rel:10.2%}")

    print("\n[diff] A large relative error is NOT automatically a bug: central "
          "differences of an adaptive-timestep solver carry their own error, "
          "because a perturbed run takes a different sequence of steps. "
          "Agreement in SIGN and order of magnitude is what a descent "
          "direction needs.")


def cmd_validate(args):
    """Compare the smooth observables against the hard (non-differentiable) ones."""
    forward, ctx = make_forward(num_cells=args.n, age_yr=args.age)
    code_units, hd, rv = ctx["code_units"], ctx["helper_data"], ctx["rv"]
    dr = ctx["dr"]

    theta = THETA0
    config = base_config(args.n, 4.0)
    state, rho_amb, p_amb, m_wind_int, t0 = build_state(
        theta, config, hd, rv, code_units, r_max=4.0, num_cells=args.n)
    cfg = finalize_config(config, state.shape)
    yr_to_code = float((1.0 * u.yr).to(code_units.code_time).value)
    params = SimulationParams(C_cfl=0.4, gamma=GAMMA,
                              t_end=args.age * yr_to_code - t0,
                              minimum_density=1e-6, minimum_pressure=1e-12)
    final = time_integration(state, cfg, params, rv)

    smooth = observables(final, rho_amb, p_amb, m_wind_int, rv, hd,
                         code_units, dr=dr)

    # the hard definitions, exactly as casa_analyze/casa_calibrate_1d use them
    r = np.asarray(hd.geometric_centers)
    rho = np.asarray(final[rv.density_index])
    press = np.asarray(final[rv.pressure_index])
    C = rho / np.maximum(np.asarray(rho_amb), 1e-30)
    hard_fs = float(r[np.max(np.where(C > 2.0)[0])]) if np.any(C > 2.0) else np.nan
    log_s = np.log10(np.maximum(press, 1e-30)) - GAMMA * np.log10(np.maximum(rho, 1e-30))
    cold = log_s < (log_s.min() + np.log10(30.0))
    hard_rs = float(r[np.max(np.where(cold)[0])]) if np.any(cold) else np.nan

    print(f"[diff] {'observable':<14s} {'smooth':>10s} {'hard':>10s} {'bias':>10s}")
    print(f"[diff] {'r_FS':<14s} {float(smooth['r_FS']):10.4f} {hard_fs:10.4f} "
          f"{float(smooth['r_FS']) - hard_fs:10.4f}")
    print(f"[diff] {'r_RS':<14s} {float(smooth['r_RS']):10.4f} {hard_rs:10.4f} "
          f"{float(smooth['r_RS']) - hard_rs:10.4f}")
    print("[diff] The bias is the cost of smoothing. It is nearly constant in "
          "the parameters, so it cancels from the gradient; report FITTED "
          "parameters, then re-measure the radii with the hard definition.")


def cmd_fit(args):
    """Gauss-Newton fit of the parameters to the measurements.

    Gauss-Newton rather than Adam because the problem is a small weighted
    least-squares with an exact Jacobian available: 4 parameters, 3
    observables. The step is damped (Levenberg) since the system is
    under-determined -- 3 measurements cannot fix 4 parameters, and pretending
    otherwise would report a spuriously precise answer.
    """
    forward, _ = make_forward(num_cells=args.n, age_yr=args.age)
    keys = list(TARGETS)
    sig = jnp.array([TARGETS[k][1] for k in keys])
    tgt = jnp.array([TARGETS[k][0] for k in keys])

    theta = THETA0
    print(f"[diff] fitting {PARAM_NAMES} to {keys}")
    print(f"[diff] {len(keys)} measurements for {len(PARAM_NAMES)} parameters: "
          f"the system is UNDER-DETERMINED, so the result is one point on a "
          f"degenerate valley, not a unique best fit. The damping is what "
          f"picks the point closest to the starting guess.")

    for it in range(args.steps):
        obs = forward(theta)
        res = (jnp.array([obs[k] for k in keys]) - tgt) / sig
        J = jvp_jacobian(forward, theta, keys) / sig[:, None]
        chi = float(jnp.sum(res ** 2))
        print(f"[diff] step {it}: chi^2 = {chi:8.3f}  " +
              "  ".join(f"{k} = {float(obs[k]):.3f}" for k in keys) + "  |  " +
              "  ".join(f"{n} = {float(v):.3f}" for n, v in zip(PARAM_NAMES, theta)))
        # damped normal equations
        A = J.T @ J + args.damping * jnp.eye(theta.shape[0])
        step = jnp.linalg.solve(A, -J.T @ res)
        theta = theta + args.step_scale * step

    obs = forward(theta)
    print(f"[diff] final: chi^2 = {float(chi2(obs)):.3f}")
    for n, v, tr in zip(PARAM_NAMES, theta, PARAM_TRANSFORM):
        phys = 10.0 ** float(v) if tr == "log10" else float(v)
        print(f"[diff]   {n:<12s} = {phys:.4f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true",
                    help="run on a GPU (default: CPU -- the 1D model is small "
                         "and a GPU wait costs more than the solve)")
    ap.add_argument("--n", type=int, default=2000, help="1D radial cells")
    ap.add_argument("--age", type=float, default=AGE_YR, help="age to evolve to (yr)")
    ap.add_argument("--check-grad", action="store_true",
                    help="validate the JVP gradient against finite differences")
    ap.add_argument("--validate", action="store_true",
                    help="compare the smooth observables with the hard ones")
    ap.add_argument("--fit", action="store_true", help="run the Gauss-Newton fit")
    ap.add_argument("--steps", type=int, default=8, help="fit iterations")
    ap.add_argument("--damping", type=float, default=1.0, help="Levenberg damping")
    ap.add_argument("--step-scale", type=float, default=0.5, help="step fraction")
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="finite-difference step for --check-grad")
    args = ap.parse_args()

    if args.check_grad:
        cmd_check_grad(args)
    if args.validate:
        cmd_validate(args)
    if args.fit:
        cmd_fit(args)
    if not (args.check_grad or args.validate or args.fit):
        ap.error("choose one of --check-grad / --validate / --fit")


if __name__ == "__main__":
    main()
