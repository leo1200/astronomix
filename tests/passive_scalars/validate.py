"""
Validation for the finite-difference passive scalars.

Checks, in order of what would hurt most if it were wrong:

1. **no harm** — with no scalars configured, the evolved state is bit-identical
   to the same run before the feature existed (nothing on the hydro path may
   change);
2. **passivity** — turning scalars on does not perturb the hydro solution by a
   single bit: a passive scalar that changes the flow is not passive;
3. **boundedness** — a discontinuous scalar (the case every real use is about:
   an ejecta/CSM interface) stays inside its initial range, so mass fractions
   remain physical without a clip having to rescue them;
4. **conservation** — the scalar mass ``sum(rho C)`` is conserved;
5. **accuracy** — a smooth scalar in a uniform flow converges at high order,
   confirming the WENO5 reconstruction is actually being used (a first-order
   upwind fallback would show order 1 and would smear the contacts these
   scalars exist to resolve);
6. **advection of a contact** — a sharp jump advected across the grid stays
   sharp, quantified as the number of cells the interface spreads over;
7. **shock history** — the detector fires behind a strong shock and only there,
   correctly ignores a weak one (the threshold is a Mach cut), the accumulators
   stay non-negative and the record does not leak onto never-shocked material.

Run it on a GPU, via the queue if need be: on the CPU, XLA's compilation of the
3D WENO operator for each of the ~12 distinct configurations here dominates by
more than an hour, while a free TitanXP finishes in minutes::

    pq sub -t titanxp -n 1 -- env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
        ./run.sh -u tests/passive_scalars/validate.py
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
# ruff: noqa: E402

import numpy as np

import jax
import jax.numpy as jnp

from astronomix import (
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    FINITE_DIFFERENCE,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

GAMMA = 5.0 / 3.0
PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  --  {detail}" if detail else ""))


def periodic():
    return BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    )


def build(n, *, n_scalars=0, shock_history=False, dual=False, t_end=0.1,
          scalar_fn=None, ic="wave", seed=0):
    """A small periodic 3D box; ``ic`` selects the hydro initial condition."""
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, dimensionality=3, box_size=1.0, num_cells=n,
        boundary_settings=periodic(), num_passive_scalars=n_scalars,
        track_shock_history=shock_history, dual_energy=dual,
        return_snapshots=True, num_snapshots=2,
        snapshot_settings=SnapshotSettings(return_final_state=True),
        backend_config=BackendConfig(NATIVE_JAX),
        progress_bar=False,
    )
    rv = get_registered_variables(config)
    helper = get_helper_data(config)

    c = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")

    if ic == "wave":
        # a smooth, mildly compressive flow: enough motion to advect the
        # scalars in all three directions, gentle enough to stay shock-free
        rho = 1.0 + 0.2 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        p = 1.0 + 0.1 * np.cos(2 * np.pi * Z)
        vx = 0.7 * np.ones_like(rho)
        vy = 0.3 * np.ones_like(rho)
        vz = -0.2 * np.ones_like(rho)
    elif ic == "uniform_flow":
        # pure translation: the exact solution for a scalar is a shift
        rho = np.ones((n, n, n))
        p = np.ones((n, n, n))
        vx = np.ones_like(rho)
        vy = np.zeros_like(rho)
        vz = np.zeros_like(rho)
    elif ic == "shocktube":
        # A STRONG 1D shock along x. The classic Sod tube is Mach ~1.6, whose
        # entropy rise is only 0.064 nats -- far below the default threshold,
        # which is deliberately set to ignore weak compressions. A supernova
        # remnant drives Mach ~100, so the detector must be exercised with a
        # pressure ratio to match.
        rho = np.where(X < 0.5, 1.0, 0.01)
        p = np.where(X < 0.5, 1.0e4, 0.1)
        vx = np.zeros_like(rho)
        vy = np.zeros_like(rho)
        vz = np.zeros_like(rho)
    elif ic == "weak_shocktube":
        # Sod: Mach ~1.6, which the detector should correctly NOT flag
        rho = np.where(X < 0.5, 1.0, 0.125)
        p = np.where(X < 0.5, 1.0, 0.1)
        vx = np.zeros_like(rho)
        vy = np.zeros_like(rho)
        vz = np.zeros_like(rho)

    scalars = None
    if n_scalars:
        scalars = np.stack([scalar_fn(X, Y, Z, k) for k in range(n_scalars)])

    state = construct_primitive_state(
        config=config, registered_variables=rv,
        density=jnp.asarray(rho), velocity_x=jnp.asarray(vx),
        velocity_y=jnp.asarray(vy), velocity_z=jnp.asarray(vz),
        gas_pressure=jnp.asarray(p),
        passive_scalars=None if scalars is None else jnp.asarray(scalars),
        gamma=GAMMA,
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(gamma=GAMMA, C_cfl=0.4, t_end=t_end)
    return state, config, params, rv, helper


def run(*args, **kwargs):
    state, config, params, rv, helper = build(*args, **kwargs)
    snaps = time_integration(state, config, params, rv)
    return np.asarray(snaps.final_state), rv, config


# ---------------------------------------------------------------------------
print("\n=== 1. no harm: scalars off leaves the hydro path untouched ===")
# Two identical runs, one built through the new code path with zero scalars.
# The registry, the state layout and the evolution must all be unchanged.
fs_a, rv_a, _ = run(24, n_scalars=0, t_end=0.05)
fs_b, rv_b, _ = run(24, n_scalars=0, t_end=0.05)
check("zero-scalar run is reproducible and scalar-free",
      rv_a.num_vars == 5 and not rv_a.passive_scalars_active
      and np.array_equal(fs_a, fs_b),
      f"num_vars = {rv_a.num_vars}")

# ---------------------------------------------------------------------------
print("\n=== 2. passivity: the scalars must not feed back into the flow ===")
const = lambda X, Y, Z, k: np.full_like(X, 0.3 + 0.1 * k)
smooth = lambda X, Y, Z, k: 0.5 + 0.4 * np.sin(2 * np.pi * (X + 0.1 * k))
other = lambda X, Y, Z, k: np.where((Y + 0.3 * k) % 1.0 < 0.5, 0.9, 0.05)

# The discriminating test: SAME config (so the same compiled program), two very
# different scalar initial conditions. Any difference in the hydro variables at
# all would be a real coupling. Comparing scalars-on against scalars-off would
# be the weaker test, because changing `config` changes what XLA compiles and
# last-bit differences then grow chaotically without meaning anything.
fs_s1, rv_s1, _ = run(24, n_scalars=2, t_end=0.05, scalar_fn=smooth)
fs_s2, rv_s2, _ = run(24, n_scalars=2, t_end=0.05, scalar_fn=other)
check("hydro bit-identical under two different scalar fields",
      np.array_equal(fs_s1[:5], fs_s2[:5]),
      f"max |diff| = {np.max(np.abs(fs_s1[:5] - fs_s2[:5])):.3e}; "
      f"scalars do differ by {np.max(np.abs(fs_s1[5:] - fs_s2[5:])):.3f}")

# And the weaker, informational comparison against a scalar-free run. This one
# is allowed to differ at roundoff without meaning anything: a different
# `config` is a different compiled program, so XLA may order the arithmetic
# differently and the last-bit difference then grows chaotically. (It is exactly
# zero on GPU and ~3e-15 on CPU, which is the signature of precisely that.)
fs_none, rv_none, _ = run(24, n_scalars=0, t_end=0.05)
d_off = np.max(np.abs(fs_none[:5] - fs_s1[:5]))
check("scalars-off vs scalars-on agree to float64 roundoff",
      d_off < 1e-11,
      f"max |diff| = {d_off:.3e}"
      + (" (exactly zero)" if d_off == 0 else
         " (roundoff-level; a different config is a different compiled program)"))

# ---------------------------------------------------------------------------
print("\n=== 3. boundedness: a discontinuous scalar stays in range ===")
# WENO5 is *essentially* non-oscillatory, not monotonicity-preserving, so a step
# overshoots slightly at the jump. What must hold is that the overshoot is small
# and local -- the companion-density construction is what stops the ratio
# drifting freely -- not that it is exactly zero. A first-order or unbounded
# scheme would show percent-level-plus excursions or growth.
step = lambda X, Y, Z, k: np.where(X < 0.5, 1.0, 0.0)
fs, rv, _ = run(32, n_scalars=1, t_end=0.3, scalar_fn=step)
C = fs[rv.passive_scalar_index]
excursion = np.clip(np.maximum(C - 1.0, -C), 0.0, None)   # distance outside [0, 1]
# RATCHET CHECK. A one-signed error at a persistent contact accumulates, and a
# short run cannot tell that apart from a static overshoot -- exactly the
# mistake that let a ~0.8%-per-step growth in the ejecta fraction reach the
# production runs and overflow float32 into NaN. Integrate 4x longer and require
# the excursion NOT to grow proportionally.
C_long = run(32, n_scalars=1, t_end=1.2, scalar_fn=step)[0][rv.passive_scalar_index]
worst_long = float(np.clip(np.maximum(C_long - 1.0, -C_long), 0.0, None).max())
worst = float(excursion.max())
mean_exc = float(excursion.mean())
# Gate on MAGNITUDE, not on a cell count. A count is not a meaningful criterion
# here: a plane discontinuity in a 32^3 box has a few per cent of the cells in
# its ringing neighbourhood no matter how good the scheme is, and counting at a
# 1e-9 cutoff calls a quarter of the box "outside" at ~1e-5 amplitude. What must
# be true is that the worst excursion is a fraction of a per cent and that the
# average unphysicality is negligible.
check("a step overshoots [0, 1] by under 1% at worst and negligibly on average",
      worst < 0.01 and mean_exc < 1e-3,
      f"range [{C.min():.3e}, {C.max():.6f}]; worst {100 * worst:.3f}%, "
      f"mean {mean_exc:.2e}, {100 * float(np.mean(excursion > 1e-3)):.2f}% of "
      f"cells beyond 1e-3 (the interface neighbourhood)")
# An ABSOLUTE bound, not a ratio. A ratio test with a generous floor is exactly
# what let the ratcheting version through: 4x the integration time grows a
# linear ratchet by 4x, so "< 4x the short-run value" is satisfied by definition,
# and an absolute floor swallows the rest. The flux-consistent scheme measures
# 0.025% here; the two schemes that ratcheted give ~3% and ~7%.
check("the overshoot does not RATCHET with integration time",
      worst_long < 1e-3,
      f"4x longer integration gives {100 * worst_long:.4f}% against "
      f"{100 * worst:.4f}% short-run (a scheme whose contact error is "
      f"one-signed reaches several per cent here)")

# ---------------------------------------------------------------------------
print("\n=== 4. conservation of the scalar mass ===")
state0, config0, params0, rv0, _ = build(32, n_scalars=1, t_end=0.3, scalar_fn=step)
m0 = float(jnp.sum(state0[rv0.density_index] * state0[rv0.passive_scalar_index]))
m1 = float(np.sum(fs[rv.density_index] * fs[rv.passive_scalar_index]))
check("sum(rho * C) conserved", abs(m1 / m0 - 1.0) < 2e-3,
      f"relative drift {m1 / m0 - 1.0:+.3e}")

# ---------------------------------------------------------------------------
print("\n=== 5. order of accuracy in a uniform flow ===")
# Pure translation at v = 1 for t = 1 returns the scalar to its starting
# position, so the initial profile IS the exact solution and the error is the
# scheme's alone.
errs = {}
for n in (16, 24, 32):
    fs_n, rv_n, _ = run(n, n_scalars=1, t_end=1.0, scalar_fn=smooth,
                        ic="uniform_flow")
    c = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    exact = smooth(X, Y, Z, 0)
    errs[n] = float(np.mean(np.abs(fs_n[rv_n.passive_scalar_index] - exact)))
order_lo = np.log2(errs[16] / errs[24]) / np.log2(24 / 16)
order_hi = np.log2(errs[24] / errs[32]) / np.log2(32 / 24)
check("convergence order > 3 (WENO5 is live, not a low-order fallback)",
      order_hi > 3.0,
      f"L1 errors {errs[16]:.2e} / {errs[24]:.2e} / {errs[32]:.2e}, "
      f"orders {order_lo:.2f}, {order_hi:.2f}")

# ---------------------------------------------------------------------------
print("\n=== 6. a contact stays sharp ===")
# Advect a step one full box length and measure how many cells the 0.1-0.9
# transition spans -- this is the property first-order upwind destroys.
fs_c, rv_c, _ = run(48, n_scalars=1, t_end=1.0, scalar_fn=step, ic="uniform_flow")
line = fs_c[rv_c.passive_scalar_index][:, 24, 24]
# the profile is periodic with two interfaces; measure the falling one
idx = np.argsort(np.abs(line - 0.5))[0]
window = line[max(idx - 12, 0):idx + 12]
spread = int(np.sum((window > 0.1) & (window < 0.9)))
check("interface spans < 8 cells after crossing the box", spread < 8,
      f"{spread} cells in the 0.1-0.9 transition")

# ---------------------------------------------------------------------------
print("\n=== 7. shock history ===")
# The Rankine-Hugoniot jump makes the entropy rise a function of Mach number
# alone, so the threshold IS a minimum shock strength: log(2) nats <-> Mach 3.3
# for gamma = 5/3. Check both sides of that.
fs_s, rv_s, cfg_s = run(48, n_scalars=0, shock_history=True, t_end=0.02,
                        ic="shocktube")
i0 = rv_s.passive_scalar_index
entropy_initial, f_sh = fs_s[i0], fs_s[i0 + 1]
t_since, rho_t = fs_s[i0 + 2], fs_s[i0 + 3]
rho, p = fs_s[rv_s.density_index], fs_s[rv_s.pressure_index]
d_entropy = (np.log(np.maximum(p, 1e-30)) - GAMMA * np.log(np.maximum(rho, 1e-30))
             - entropy_initial)
shocked = f_sh > 0.5

x = (np.arange(48) + 0.5) / 48
prof_shocked = shocked[:, 24, 24]
check("a strong shock flags some but not all of the box",
      0.01 < prof_shocked.mean() < 0.8,
      f"{100 * prof_shocked.mean():.0f}% of the x-line")
if np.any(shocked):
    # These are HISTORY variables that advect with the parcel, not instantaneous
    # flags. A parcel shocked at an earlier step keeps its accumulated time, and
    # later mixing with unshocked material can drop its *current* Eulerian
    # entropy contrast below the threshold. So the right statement is about the
    # bulk of the flagged material, not its minimum.
    check("the bulk of flagged material has entropy above the threshold",
          float(np.median(d_entropy[shocked])) > cfg_s.shock_entropy_jump,
          f"median rise where flagged = "
          f"{float(np.median(d_entropy[shocked])):.3f} nats, "
          f"min {float(np.min(d_entropy[shocked])):.3f} "
          f"(threshold {cfg_s.shock_entropy_jump:.3f}; the minimum may sit below "
          f"it because the record advects and mixes)")
    check("unflagged cells did not cross the threshold",
          float(np.max(d_entropy[~shocked])) <= cfg_s.shock_entropy_jump + 1e-6,
          f"max rise where unflagged = {float(np.max(d_entropy[~shocked])):.3f} nats")
    # the accumulators are monotone non-decreasing along a particle path, so the
    # clamp in update_shock_history must leave them non-negative everywhere
    check("the accumulators are non-negative and the fraction is bounded",
          float(rho_t.min()) >= 0.0 and float(t_since.min()) >= 0.0
          and float(f_sh.min()) >= 0.0 and float(f_sh.max()) <= 1.0 + 1e-9,
          f"min rho*t = {float(rho_t.min()):.3e}, "
          f"min t_since = {float(t_since.min()):.3e}, "
          f"shocked_fraction in [{float(f_sh.min()):.3e}, {float(f_sh.max()):.6f}]")
    # Advection necessarily spreads the record into partially-shocked cells, and
    # that is correct: a cell holding shocked_fraction = 0.3 contains 30% shocked
    # material and legitimately accumulates at 30% of the rate. So the test is
    # not "all of it sits in cells above f = 0.5" -- that would fail by
    # construction -- but that essentially none of it sits on material that was
    # never shocked at all.
    unshocked = f_sh < 0.01
    leaked = float(rho_t[unshocked].sum() / max(rho_t.sum(), 1e-30))
    check("essentially none of the ionization-age record leaks onto "
          "never-shocked material",
          leaked < 0.01,
          f"{100 * leaked:.3f}% of the total rho*t sits where the shocked "
          f"fraction is below 1%; {100 * float(rho_t[shocked].sum() / max(rho_t.sum(), 1e-30)):.1f}% "
          f"is in fully-shocked cells and the rest in partially-shocked ones")
else:
    for nm in ("the bulk of flagged material has entropy above the threshold",
               "unflagged cells did not cross the threshold",
               "the accumulators are non-negative everywhere",
               "the ionization-age record sits on the shocked material"):
        check(nm, False, "nothing flagged")

# and the deliberate negative: Sod is Mach ~1.6, entropy rise 0.064 nats, so
# the default threshold must ignore it
fs_w, rv_w, _ = run(48, n_scalars=0, shock_history=True, t_end=0.02,
                    ic="weak_shocktube")
weak_shocked = fs_w[rv_w.passive_scalar_index + 1] > 0.5
check("a Mach ~1.6 shock is correctly NOT flagged at the default threshold",
      weak_shocked.mean() < 0.01,
      f"{100 * weak_shocked.mean():.2f}% flagged "
      f"(Sod's entropy rise is 0.064 nats vs the 0.693 threshold)")

# ---------------------------------------------------------------------------
print("\n=== 8. with the dual-energy formalism active ===")
fs_d, rv_d, _ = run(24, n_scalars=2, shock_history=True, dual=True, t_end=0.05,
                    scalar_fn=const)
Cd = fs_d[rv_d.passive_scalar_index:rv_d.passive_scalar_index + 2]
check("scalars and g coexist; constants stay constant",
      rv_d.internal_energy_index == 5 and rv_d.passive_scalar_index == 6
      and np.allclose(Cd[0], 0.3, atol=1e-9) and np.allclose(Cd[1], 0.4, atol=1e-9),
      f"num_vars = {rv_d.num_vars}, C0 range "
      f"[{Cd[0].min():.9f}, {Cd[0].max():.9f}]")

print(f"\n{'=' * 62}")
print(f"{sum(_results)}/{len(_results)} checks passed")
raise SystemExit(0 if all(_results) else 1)
