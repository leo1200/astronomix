"""
Finite-Difference Baseline Test for AD Gradients Through Shocks

The smooth sensitivity test in ``sensitivity.py`` validates JAX AD against the
exact analytic gradient of the linearized Euler equations -- it covers the
smooth, small-amplitude regime but cannot reach the shock regime where AD
must still produce a correct sub-gradient of the *discrete* cost J(theta).

This script provides the complementary check.  The standard Sod shock tube
with open boundaries is parameterized by six scalars

    theta = (rho_L, p_L, u_L, rho_R, p_R, u_R)

and the cost  J(theta) = 0.5 * integral( rho^2 + v^2 + p^2 ) dx  on the final
state is differentiated through the FV and FD solvers via AD, then compared
against a *finite-difference baseline* (central FD) on the same J(theta).

Why this is a sound complement to ``sensitivity.py``:

* The IC built with ``jnp.where(x < x_split, left, right)`` is exactly linear
  in theta.  Each cell value is one of the six components, with no smoothing
  or interpolation -- so initial-condition differentiability is clean.
* A fixed-dt integrator (``fixed_timestep=True``) is used to remove
  adaptive-CFL kinks.  Otherwise the step count is a piecewise-constant
  function of theta, creating O(dt) jumps in J that AD and FD resolve
  differently.  With dt fixed, the only remaining non-smoothness is from the
  limiters / Riemann sub-gradient switches.
* The axis-aligned (per-parameter) test exposes the sub-gradient ambiguity
  at kinks (limiter / Riemann switches).  For smooth directions at theta0
  AD and central FD agree to ~machine precision; on kinky directions they
  can disagree by O(1) of the sub-gradient gap -- which is consistent with
  AD being correct OR wrong by exactly that amount.  Central FD cannot
  arbitrate.
* The complementary random-direction directional-derivative check closes
  that gap.  The kinks of J live on codim-1 surfaces in theta-space, so a
  random unit vector v generically misses them; J is smooth along v at
  theta0 and central FD resolves  v . grad J  with O(h^2) truncation,
  approaching the round-off floor ~ eps^{2/3} ~ 1e-11.  Machine-precision
  agreement on a random slice does arbitrate AD correctness in the shock
  regime.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# =======================

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from astronomix import CARTESIAN
from astronomix import get_helper_data, time_integration, get_registered_variables
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FINITE_DIFFERENCE, FINITE_VOLUME, NATIVE_JAX, PALLAS,
    OPEN_BOUNDARY, SimulationConfig, finalize_config,
    BoundarySettings1D, GravityConfig, PositivityConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams


# ==============================================================================
# 1. Configuration and parameterized IC
# ==============================================================================
def make_config_and_params(N, L, t_end, num_timesteps, solver_mode, backend=NATIVE_JAX):
    """1D open-boundary config with a fixed timestep so J(theta) has no CFL kinks.

    A fixed dt is deliberate: under adaptive CFL the step count n(theta) =
    ceil(t_end / dt(theta)) is piecewise-constant in theta, so crossing a
    threshold injects an O(dt) jump into J(theta).  The AD gradient is unaffected
    (it differentiates the smooth branch), but the finite-difference baseline
    cannot tell such a jump from a real derivative, so the AD-vs-FD agreement
    degrades by 1-3 orders of magnitude and the random-direction arbitration no
    longer reaches round-off.  Fixing dt isolates the limiter / Riemann
    sub-gradient -- the only non-smoothness this test is meant to probe.

    With ``backend=PALLAS`` the FD forward runs the Pallas WENO kernel and the
    backward runs the native Pallas adjoint (the kernel transposes the periodic
    custom_roll stencil, correct for these ghost-cell boundaries too).  Block
    (4,1,1) divides the 1D shape.  At the shock the WENO gradient is
    FP-ill-conditioned, so FD (Pallas) AD can differ from FD (JAX) by a
    sub-gradient amount -- the kink-immune one-sided / random-direction checks
    below arbitrate.
    """
    config = SimulationConfig(
        solver_mode=solver_mode,
        geometry=CARTESIAN,
        progress_bar=False,
        differentiation_mode=BACKWARDS,
        backend=backend,
        pallas_block_shape=(4, 1, 1),
        pallas_use_triton=True,
        pallas_interpret=False,
        mhd=False,
        dimensionality=1,
        box_size=L,
        num_cells=N,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY,
            right_boundary=OPEN_BOUNDARY,
        ),
        return_snapshots=False,
        fixed_timestep=True,
        num_timesteps=num_timesteps,
    )
    params = SimulationParams(
        C_cfl=0.4,
        t_end=t_end,
        gamma=5/3,
        gravitational_constant=0.0,
    )
    return config, params


def build_initial_state(theta, config, registered_variables, helper_data, x_split):
    """
    Standard Sod-style IC parameterized by
        theta = (rho_L, p_L, u_L, rho_R, p_R, u_R).
    Cells with x < x_split get the left state, the rest the right state.
    """
    rho_L, p_L, u_L, rho_R, p_R, u_R = theta
    x = jnp.squeeze(helper_data.geometric_centers)
    left = x < x_split
    rho = jnp.where(left, rho_L, rho_R)
    u   = jnp.where(left, u_L,   u_R)
    p   = jnp.where(left, p_L,   p_R)
    return construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=rho, velocity_x=u, gas_pressure=p,
    )


# ==============================================================================
# 2. Forward cost + AD / FD gradients
# ==============================================================================
def cost(theta, config, params, registered_variables, helper_data, x_split):
    """J(theta) = 0.5 * sum_x ( rho^2 + v^2 + p^2 ) * dx on the final state."""
    state0 = build_initial_state(theta, config, registered_variables, helper_data, x_split)
    config_final = finalize_config(config, state0.shape)
    state_T = time_integration(state0, config_final, params, registered_variables)
    rho = state_T[registered_variables.density_index]
    v   = state_T[registered_variables.velocity_index]
    p   = state_T[registered_variables.pressure_index]
    dx = config.box_size / config.num_cells
    return 0.5 * jnp.sum(rho ** 2 + v ** 2 + p ** 2) * dx


def ad_gradient(theta, config, params, registered_variables, helper_data, x_split):
    fn = lambda th: cost(th, config, params, registered_variables, helper_data, x_split)
    return jax.grad(fn)(theta)


def fd_gradient(theta, h, jitted_cost):
    """Central finite difference: g_i = (J(theta+h e_i) - J(theta-h e_i)) / (2h)."""
    g = np.zeros(theta.size)
    for i in range(theta.size):
        ei = jnp.zeros_like(theta).at[i].set(1.0)
        Jp = float(jitted_cost(theta + h * ei))
        Jm = float(jitted_cost(theta - h * ei))
        g[i] = (Jp - Jm) / (2.0 * h)
    return g


def all_fd_gradients(theta, h, jitted_cost, J0):
    """
    Returns (central, forward, backward) FD gradient arrays for one h,
    using a single pair of J evaluations per parameter plus a shared J(theta).

    Forward FD approximates the right-derivative, backward FD the
    left-derivative.  At a sub-gradient kink, AD has picked one of these
    sides; central FD averages both and can never match AD there.  The
    pair therefore lets us run the kink-immune check

        rel_min_i = min(|g_AD_i - g_F_i|, |g_AD_i - g_B_i|) / |g_AD_i|

    which certifies "AD matches at least one one-sided slope at theta0".
    """
    g_c = np.zeros(theta.size)
    g_f = np.zeros(theta.size)
    g_b = np.zeros(theta.size)
    for i in range(theta.size):
        ei = jnp.zeros_like(theta).at[i].set(1.0)
        Jp = float(jitted_cost(theta + h * ei))
        Jm = float(jitted_cost(theta - h * ei))
        g_c[i] = (Jp - Jm) / (2.0 * h)
        g_f[i] = (Jp - J0) / h
        g_b[i] = (J0 - Jm) / h
    return g_c, g_f, g_b


def directional_fd_vs_ad(theta, g_ad, jitted_cost, h_values, n_dirs, seed):
    """
    Random-direction directional-derivative check.

    For each random unit vector v in R^{dim(theta)}, compare

        AD:   v . grad J_AD
        FD:   (J(theta + h v) - J(theta - h v)) / (2 h)

    The kinks in J(theta) (limiter / Riemann switches) live on codim-1
    surfaces in theta-space, so a random v generically misses them and J is
    smooth along v at theta.  Central FD then resolves the directional
    derivative with O(h^2) truncation and round-off floor at ~ eps^(2/3),
    which arbitrates AD correctness to machine precision -- something the
    axis-aligned check cannot do whenever an axis sits on a kink.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_dirs):
        v = rng.normal(size=theta.size)
        v = v / np.linalg.norm(v)
        v_j = jnp.asarray(v)
        ad_dd = float(np.dot(g_ad, v))
        fd_at_h, rel_at_h = {}, {}
        for h in h_values:
            Jp = float(jitted_cost(theta + h * v_j))
            Jm = float(jitted_cost(theta - h * v_j))
            fd = (Jp - Jm) / (2.0 * h)
            fd_at_h[h] = fd
            rel_at_h[h] = abs(fd - ad_dd) / (abs(ad_dd) + 1e-30)
        rows.append(dict(v=np.asarray(v), ad=ad_dd, fd=fd_at_h, rel=rel_at_h))
    return rows


# ==============================================================================
# 3. Driver
# ==============================================================================
def run_shock_tube_sensitivity_test():
    print(f"\n{'-'*60}\nShock-tube AD-vs-FD baseline test\n{'-'*60}")

    # Standard Sod state.
    theta0 = jnp.array([1.0, 1.0, 0.0, 0.125, 0.1, 0.0])
    param_names = ["rho_L", "p_L", "u_L", "rho_R", "p_R", "u_R"]

    N = 200
    L = 1.0
    x_split = 0.5
    t_end = 0.2          # canonical Sod evaluation time
    num_timesteps = 250  # fixed dt; conservative enough that the FD theta sweep stays CFL-stable

    h_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    # Extended sweep for the directional test -- random slices are smooth, so we
    # expect O(h^2) all the way down to the round-off floor (~eps^{2/3} ~ 1e-11).
    dir_h_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    n_random_dirs = 4

    backends = [
        ("FD (JAX)",    FINITE_DIFFERENCE, NATIVE_JAX),
        ("FD (Pallas)", FINITE_DIFFERENCE, PALLAS),
        ("FV (JAX)",    FINITE_VOLUME,     NATIVE_JAX),
    ]

    results = {}

    for label, solver_mode, backend in backends:
        print(f"\n=== {label} ===")
        config, params = make_config_and_params(N, L, t_end, num_timesteps, solver_mode, backend=backend)
        registered_variables = get_registered_variables(config)
        helper_data = get_helper_data(config)

        jitted_cost = jax.jit(
            lambda th: cost(th, config, params, registered_variables, helper_data, x_split)
        )
        jitted_ad = jax.jit(
            lambda th: ad_gradient(th, config, params, registered_variables, helper_data, x_split)
        )

        # Warm up the JITs and read off J(theta0) for a sanity check.
        J0 = float(jitted_cost(theta0))
        print(f"J(theta0) = {J0:.6e}")

        g_ad = np.asarray(jitted_ad(theta0))
        print(f"AD gradient:")
        for name, gi in zip(param_names, g_ad):
            print(f"   d J / d {name:<6s} = {gi: .6e}")

        fd_grads = {}
        fd_grads_f = {}
        fd_grads_b = {}
        rel_errs = {}
        rel_errs_minside = {}
        print("\nFinite-difference baseline (central + one-sided):")
        for h in h_values:
            g_c, g_f, g_b = all_fd_gradients(theta0, h, jitted_cost, J0)
            fd_grads[h] = g_c
            fd_grads_f[h] = g_f
            fd_grads_b[h] = g_b
            rel_errs[h] = float(np.linalg.norm(g_ad - g_c)
                                / (np.linalg.norm(g_ad) + 1e-30))
            # Min-side per parameter, then take vector norm of those residuals.
            rmin = np.minimum(np.abs(g_ad - g_f), np.abs(g_ad - g_b))
            rel_errs_minside[h] = float(np.linalg.norm(rmin)
                                        / (np.linalg.norm(g_ad) + 1e-30))
            print(f"   h = {h:.0e}  central: {rel_errs[h]:.3e}   "
                  f"min-one-sided: {rel_errs_minside[h]:.3e}")
            for name, gci, gfi, gbi in zip(param_names, g_c, g_f, g_b):
                print(f"      d J / d {name:<6s}  C={gci: .6e}  "
                      f"F={gfi: .6e}  B={gbi: .6e}")

        # Random-direction check: smooth slice -> machine-precision arbitration.
        print("\nRandom-direction directional derivative (v . grad J vs central FD):")
        dir_rows = directional_fd_vs_ad(
            theta0, g_ad, jitted_cost,
            h_values=dir_h_values, n_dirs=n_random_dirs, seed=0,
        )
        for k, row in enumerate(dir_rows):
            print(f"   dir {k}:  AD v.g = {row['ad']: .6e}")
            for h in dir_h_values:
                print(f"      h={h:.0e}  FD = {row['fd'][h]: .6e}  "
                      f"|FD-AD|/|AD| = {row['rel'][h]:.3e}")

        results[label] = dict(g_ad=g_ad,
                              fd=fd_grads, fd_f=fd_grads_f, fd_b=fd_grads_b,
                              rel_errs=rel_errs, rel_errs_minside=rel_errs_minside,
                              dir_rows=dir_rows)

    # ----- Plot 1: per-parameter AD vs FD bars + per-derivative relative error -
    # Only min-one-sided FD is shown (the kink-immune AD-correctness certificate);
    # central FD remains visible in the step-sweep plot below.
    os.makedirs("figures", exist_ok=True)
    fig, axs = plt.subplots(2, len(backends),
                            figsize=(7 * len(backends), 9),
                            sharex='col')
    if len(backends) == 1:
        axs = axs.reshape(2, 1)
    n_params = len(param_names)
    x_pos = np.arange(n_params)
    h_values_plot = [h for h in (1e-4, 1e-6) if h in h_values]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.8, len(h_values_plot)))
    tiny = 1e-30
    for col, (label, _, _) in enumerate(backends):
        g_ad = results[label]["g_ad"]

        # For each h, build the per-parameter min-one-sided gradient: pick
        # whichever of (forward, backward) is closer to AD on that axis.
        per_h_min_grad = {}
        for h in h_values_plot:
            g_f = results[label]["fd_f"][h]
            g_b = results[label]["fd_b"][h]
            choose_f = np.abs(g_ad - g_f) <= np.abs(g_ad - g_b)
            per_h_min_grad[h] = np.where(choose_f, g_f, g_b)

        # Top row: AD vs min-one-sided FD gradient values.
        ax_top = axs[0, col]
        for color, h in zip(cmap, h_values_plot):
            ax_top.scatter(x_pos, per_h_min_grad[h], color=color, s=80,
                           alpha=0.85, marker='^', edgecolors='black',
                           linewidths=0.6,
                           label=f"min one-sided FD  h={h:.0e}")
        ax_top.scatter(x_pos, g_ad, marker='x', s=180, color='black',
                       linewidths=2.5, label='AD')
        ax_top.set_title(f"{label}: AD vs min-one-sided FD per parameter")
        ax_top.set_ylabel(r'$\partial J / \partial \theta_i$')
        ax_top.grid(True, alpha=0.4)
        ax_top.axhline(0, color='gray', linewidth=0.7)
        ax_top.legend(fontsize=8, loc='best')

        # Bottom row: per-derivative relative error using the kink-immune
        # min(|g_AD - forward|, |g_AD - backward|) certificate.
        ax_bot = axs[1, col]
        for color, h in zip(cmap, h_values_plot):
            g_f = results[label]["fd_f"][h]
            g_b = results[label]["fd_b"][h]
            rel_min = np.minimum(np.abs(g_ad - g_f),
                                 np.abs(g_ad - g_b)) / (np.abs(g_ad) + tiny)
            ax_bot.scatter(x_pos, rel_min, color=color, s=80, alpha=0.85,
                           marker='^', edgecolors='black', linewidths=0.6,
                           label=f"min one-sided FD  h={h:.0e}")
        ax_bot.set_yscale('log')
        ax_bot.set_xticks(x_pos)
        ax_bot.set_xticklabels(param_names, rotation=30)
        ax_bot.set_ylabel(r'$|g_{\mathrm{AD},i} - g_{\mathrm{FD},i}| \, / \, |g_{\mathrm{AD},i}|$')
        ax_bot.set_title(f"{label}: per-derivative relative error")
        ax_bot.grid(True, which='both', alpha=0.4)
        ax_bot.legend(fontsize=8, loc='best')
    fig.tight_layout()
    out_path_1 = "figures/shock_tube_sensitivity.svg"
    fig.savefig(out_path_1, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path_1}")

    # ----- Plot 2: relative error vs h, min-one-sided FD only ----------------
    # One-sided FD has O(h) truncation error.  The min-one-sided AD residual
    # therefore tracks ~ h until round-off in the compiled J(theta) takes
    # over (~ eps_J / h, not visible in this h range).  The reference line
    # makes that scaling explicit; deviation from it would indicate kink
    # contamination or an AD bug.
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Distinct backend colours (matches the other figures): FD (JAX) light blue,
    # FD (Pallas) violet, FV (JAX) orange; FD (JAX) thick solid, FD (Pallas)
    # thinner dashed on top, so the two FD curves stay visible on overlap.
    def _stp(label):
        if label.startswith('FD'):
            if 'Pallas' in label:
                return dict(color='darkviolet', linestyle='--', linewidth=1.8,
                            marker='x', markersize=8, zorder=3)
            return dict(color='cornflowerblue', linestyle='-', linewidth=3.0,
                        marker='^', markersize=7, zorder=2)
        return dict(color='tab:orange', linestyle='-', linewidth=2.2,
                    marker='^', markersize=7, zorder=2)

    h_arr = np.array(h_values, dtype=float)
    for label, _, _ in backends:
        ys_min = np.array([results[label]["rel_errs_minside"][h] for h in h_values])
        ax.loglog(h_arr, ys_min, label=f"{label}: min one-sided FD", **_stp(label))

    # Reference O(h) (slope +1) truncation line, anchored at the tightest
    # min-one-sided value across backends to lie just under the data.
    h_anchor = min(h_values)
    y_anchor = min(results[label]["rel_errs_minside"][h_anchor]
                   for label, _, _ in backends)
    ax.loglog(h_arr, y_anchor * (h_arr / h_anchor),
              linestyle='--', color='gray', linewidth=1.5,
              label=r'$\propto h$ (one-sided FD truncation)')

    ax.set_xlabel("FD step size $h$")
    ax.set_ylabel(r'$\| \min_{F,B} | g_{\mathrm{AD}} - g_{\mathrm{FD}} | \|_2 \, / \, \| g_{\mathrm{AD}} \|_2$')
    ax.set_title("AD vs min-one-sided FD on the 1D Sod shock tube")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path_2 = "figures/shock_tube_sensitivity_step_sweep.svg"
    fig.savefig(out_path_2, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path_2}")

    # ----- Plot 3: random-direction directional derivative agreement ----------
    fig, axs = plt.subplots(1, len(backends),
                            figsize=(7 * len(backends), 5), sharey=True)
    if len(backends) == 1:
        axs = [axs]
    dir_cmap = plt.cm.plasma(np.linspace(0.15, 0.85, n_random_dirs))
    for ax, (label, _, _) in zip(axs, backends):
        for k, row in enumerate(results[label]["dir_rows"]):
            ys = [row["rel"][h] for h in dir_h_values]
            ax.loglog(dir_h_values, ys, marker='o', color=dir_cmap[k],
                      linewidth=1.8, label=f"random dir {k}")
        # Reference O(h^2) line anchored at h=1e-2.
        h_ref = np.array(dir_h_values, dtype=float)
        y0 = max((row["rel"][1e-2] for row in results[label]["dir_rows"]
                  if 1e-2 in row["rel"]), default=1e-4)
        ax.loglog(h_ref, y0 * (h_ref / 1e-2) ** 2,
                  linestyle='--', color='gray', linewidth=1.2,
                  label=r'$\propto h^2$')
        ax.set_xlabel("Central-FD step size $h$")
        ax.set_ylabel(r'$|v \cdot g_{\mathrm{AD}} - g_{\mathrm{FD}}^{(v)}| \, / \, |v \cdot g_{\mathrm{AD}}|$')
        ax.set_title(f"{label}: random-direction directional derivative")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    out_path_3 = "figures/shock_tube_sensitivity_random_directions.svg"
    fig.savefig(out_path_3, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path_3}")

    # ----- Summary -------------------------------------------------------------
    print(f"\n{'-'*60}\nSummary\n{'-'*60}")
    print("Per-axis (axis-aligned) test, central FD:")
    for label, _, _ in backends:
        best_h = min(results[label]["rel_errs"], key=results[label]["rel_errs"].get)
        print(f"  {label:<22s}: best h = {best_h:.0e}, "
              f"||g_AD - g_FD||/||g_AD|| = {results[label]['rel_errs'][best_h]:.3e}")
    print("Per-axis (axis-aligned) test, min-one-sided FD (kink-immune):")
    for label, _, _ in backends:
        re = results[label]["rel_errs_minside"]
        best_h = min(re, key=re.get)
        print(f"  {label:<22s}: best h = {best_h:.0e}, "
              f"||min-side residual||/||g_AD|| = {re[best_h]:.3e}")
    print("\nRandom-direction directional derivative (smooth-slice arbitration):")
    for label, _, _ in backends:
        best_per_dir = [min(row["rel"].values())
                        for row in results[label]["dir_rows"]]
        print(f"  {label:<22s}: best-of-h rel. err per direction "
              f"min = {min(best_per_dir):.3e}, max = {max(best_per_dir):.3e}  "
              f"({n_random_dirs} dirs)")


if __name__ == "__main__":
    run_shock_tube_sensitivity_test()
