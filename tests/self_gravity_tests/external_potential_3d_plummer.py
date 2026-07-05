"""
3D external-potential well-balancedness test (no self-gravity), FV and FD.

A gas cloud is placed in *exact* hydrostatic equilibrium inside a fixed Plummer
potential and we check that the scheme keeps it approximately static -- the
canonical test for an external-gravity source term, representative of e.g. the
intracluster medium sitting in a fixed dark-matter potential.

Plummer potential (centered in the box):
    Phi(r) = -G M / sqrt(r^2 + a^2)

Isothermal stratification P = c^2 rho integrates grad P = -rho grad Phi exactly:
    rho(r) = rho0 * exp(-Phi(r)/c^2),   P(r) = c^2 * rho(r),   v = 0,
and since P/rho = c^2 everywhere the sound speed c_s = sqrt(gamma c^2) is uniform.

The discrete scheme is not exactly well balanced, so a truncation-level mismatch
between the pressure-gradient and gravity discretizations drives small spurious
velocities. The demonstration is that this spurious motion *converges away under
grid refinement*. For each solver we run a resolution sweep and require
    (1) density/pressure stay positive (no blow-up),
    (2) the spurious core Mach number stays small, and
    (3) refining the grid reduces it (convergence to the equilibrium).

Both the finite-volume (SIMPLE_SOURCE) and finite-difference
(SIMPLE_SOURCE) gravity couplings are exercised, purely off the external
potential through the config.gravity machinery (self_gravity is OFF).

Usage:
    python external_potential_3d_plummer.py            # default sweep [32, 64, 128]
    python external_potential_3d_plummer.py 16 32       # custom resolutions
"""

import os
import sys

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

from astronomix import CARTESIAN
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    SIMPLE_SOURCE,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    OPEN_BOUNDARY,
    SIMPLE_SOURCE,
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    SimulationConfig,
    StaticFloatVector,
    StaticIntVector,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.time_stepping.time_integration import time_integration

# ---------------------------------------------------------------------------
# problem constants
# ---------------------------------------------------------------------------
BOX = 8.0                # cubic box side length
A_PLUMMER = 1.0          # Plummer softening radius
GM = 1.0                 # G * M  (central/floor density ratio = exp(GM/(c^2 a)) ~ 2.7)
C2 = 1.0                 # isothermal stratification constant (P = C2 * rho)
RHO0 = 1.0               # density floor (far-field)
GAMMA = 5.0 / 3.0
T_END = 1.0              # ~1.3 core sound-crossing times (a / c_s)
R_CHECK = 2.0            # measure the core region (within 2 Plummer radii), away from boundaries

C_S = (GAMMA * C2) ** 0.5   # uniform sound speed

SOLVERS = [
    ("FV (simple source)", FINITE_VOLUME, SIMPLE_SOURCE),
    ("FD (simple source)", FINITE_DIFFERENCE, SIMPLE_SOURCE),
]

RESOLUTIONS = [int(a) for a in sys.argv[1:]] or [32, 64, 128]


def plummer_potential(r):
    return -GM / jnp.sqrt(r**2 + A_PLUMMER**2)


def run_case(N, solver_mode, gravity_version):
    _open = BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY)
    config = SimulationConfig(
        solver_mode=solver_mode,
        gravity_config=GravityConfig(
            self_gravity_version=gravity_version,
            external_potential=True,
        ),
        dimensionality=3,
        geometry=CARTESIAN,
        box_size=StaticFloatVector(BOX, BOX, BOX),
        num_cells=StaticIntVector(N, N, N),
        boundary_settings=BoundarySettings(x=_open, y=_open, z=_open),
        progress_bar=False,
    )

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    params = SimulationParams(t_end=T_END, gamma=GAMMA)

    centers = helper_data.geometric_centers            # (N, N, N, 3)
    cc = BOX / 2.0
    r = jnp.sqrt(
        (centers[..., 0] - cc) ** 2
        + (centers[..., 1] - cc) ** 2
        + (centers[..., 2] - cc) ** 2
    )

    phi = plummer_potential(r)
    rho_init = RHO0 * jnp.exp(-phi / C2)               # hydrostatic density
    p_init = C2 * rho_init                             # P = c^2 rho
    zero = jnp.zeros_like(r)

    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho_init,
        velocity_x=zero,
        velocity_y=zero,
        velocity_z=zero,
        gas_pressure=p_init,
    )

    # external potential on the bare grid (state-field shape, no ghosts)
    params = params._replace(gravitational_potential=phi)

    config = finalize_config(config, state.shape)
    assert config.gravity, "finalize_config did not turn on the master gravity flag"

    final_state = time_integration(state, config, params, registered_variables)

    vx = final_state[registered_variables.velocity_index.x]
    vy = final_state[registered_variables.velocity_index.y]
    vz = final_state[registered_variables.velocity_index.z]
    mach = jnp.sqrt(vx**2 + vy**2 + vz**2) / C_S

    rho_final = final_state[registered_variables.density_index]
    p_final = final_state[registered_variables.pressure_index]
    drho_rel = jnp.abs(rho_final - rho_init) / rho_init

    core = r < R_CHECK
    mach_core = float(jnp.max(jnp.where(core, mach, 0.0)))
    drho_core = float(jnp.max(jnp.where(core, drho_rel, 0.0)))
    positive = bool((rho_final > 0).all() and (p_final > 0).all())

    # z-midplane slices for visualization
    kz = N // 2
    slices = dict(
        rho=np.asarray(rho_final[:, :, kz]),
        drho=np.asarray(drho_rel[:, :, kz]),
        mach=np.asarray(mach[:, :, kz]),
        rho_init=np.asarray(rho_init[:, :, kz]),
    )
    return dict(
        N=N, mach_core=mach_core, drho_core=drho_core, positive=positive, slices=slices
    )


# ---------------------------------------------------------------------------
# run the refinement study for each solver
# ---------------------------------------------------------------------------
print(f"central/floor density ratio: {float(jnp.exp(GM / (C2 * A_PLUMMER))):.3f}")
print(f"uniform sound speed c_s = {C_S:.4f}")
print(f"resolutions: {RESOLUTIONS}\n")

all_results = {}   # label -> list of per-resolution result dicts
overall_ok = True
for label, solver_mode, version in SOLVERS:
    print(f"=== {label} ===")
    print(f"{'N':>5} {'dx':>9} {'max Mach (core)':>17} {'max drho/rho':>14} {'order':>7} {'pos':>5}")
    res = []
    prev = None
    for N in RESOLUTIONS:
        out = run_case(N, solver_mode, version)
        order = ""
        if prev is not None and out["mach_core"] > 0:
            pN, pmach = prev
            order = f"{float(jnp.log(pmach / out['mach_core']) / jnp.log(N / pN)):.2f}"
        print(f"{N:>5} {BOX / N:>9.4f} {out['mach_core']:>17.4e} {out['drho_core']:>14.4e} {order:>7} {str(out['positive']):>5}")
        res.append(out)
        prev = (N, out["mach_core"])
    all_results[label] = res

    all_positive = all(o["positive"] for o in res)
    mach_seq = [o["mach_core"] for o in res]
    drho_seq = [o["drho_core"] for o in res]
    small = mach_seq[-1] < 0.05
    converging = all(b < a for a, b in zip(mach_seq, mach_seq[1:])) and (drho_seq[-1] < drho_seq[0])
    solver_ok = all_positive and small and converging
    overall_ok = overall_ok and solver_ok
    print(f"  -> positive={all_positive}, small(finest={mach_seq[-1]:.2e}<0.05)={small}, "
          f"converging={converging}  [{'PASS' if solver_ok else 'FAIL'}]\n")


# ---------------------------------------------------------------------------
# plot 1: well-balancedness convergence (spurious Mach + density drift vs N)
# ---------------------------------------------------------------------------
fig, (ax_m, ax_d) = plt.subplots(1, 2, figsize=(11, 4.6))
for label, res in all_results.items():
    Ns = np.array([o["N"] for o in res], dtype=float)
    machs = np.array([o["mach_core"] for o in res])
    drhos = np.array([o["drho_core"] for o in res])
    p_m = -np.polyfit(np.log(Ns), np.log(machs), 1)[0]
    p_d = -np.polyfit(np.log(Ns), np.log(drhos), 1)[0]
    ax_m.loglog(Ns, machs, "o-", label=f"{label}  (order {p_m:.1f})")
    ax_d.loglog(Ns, drhos, "o-", label=f"{label}  (order {p_d:.1f})")

# reference slopes anchored at the coarsest point of the worst (largest) curve
Nref = np.array([RESOLUTIONS[0], RESOLUTIONS[-1]], dtype=float)
y0 = max(res[0]["mach_core"] for res in all_results.values())
for p, ls in ((2.0, ":"), (5.0, "--")):
    ax_m.loglog(Nref, y0 * (Nref / Nref[0]) ** (-p), color="gray", ls=ls, lw=1,
                alpha=0.6, label=f"slope -{int(p)}")
ax_m.set(xlabel="cells per dimension N", ylabel="max core Mach number",
         title="External-potential well-balancedness")
ax_d.set(xlabel="cells per dimension N", ylabel=r"max core $|\Delta\rho|/\rho$",
         title="Density drift")
for ax in (ax_m, ax_d):
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("3D Plummer hydrostatic equilibrium in a fixed external potential")
fig.tight_layout()
conv_path = os.path.join(FIG_DIR, "external_potential_3d_plummer_convergence.png")
fig.savefig(conv_path, dpi=150)
print(f"saved convergence figure -> {conv_path}")


# ---------------------------------------------------------------------------
# plot 2: equilibrium z-midplane slices (finest resolution per solver)
# ---------------------------------------------------------------------------
ext = [0.0, BOX, 0.0, BOX]
nrows = len(SOLVERS)
fig2, axes2 = plt.subplots(nrows, 3, figsize=(13.5, 4.3 * nrows), squeeze=False)
for i, (label, res) in enumerate(all_results.items()):
    fin = res[-1]
    sl = fin["slices"]
    panels = [
        ("final density", sl["rho"], "viridis", None),
        (r"$|\Delta\rho|/\rho$", sl["drho"], "magma", "log"),
        ("Mach number", sl["mach"], "inferno", "log"),
    ]
    for j, (title, arr, cmap, scale) in enumerate(panels):
        ax = axes2[i, j]
        if scale == "log":
            a = np.maximum(arr, arr.max() * 1e-4 + 1e-300)
            norm = LogNorm(vmin=a.max() * 1e-3, vmax=a.max())
            im = ax.imshow(a.T, origin="lower", extent=ext, cmap=cmap, norm=norm)
        else:
            im = ax.imshow(arr.T, origin="lower", extent=ext, cmap=cmap)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.add_patch(plt.Circle((BOX / 2, BOX / 2), R_CHECK, fill=False,
                                ec="cyan", ls="--", lw=0.8))
        ax.set_title(f"{label}, N={fin['N']}\n{title} (z-midplane)", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y")
fig2.suptitle("Equilibrium slices (dashed circle = core measurement region)")
fig2.tight_layout()
slice_path = os.path.join(FIG_DIR, "external_potential_3d_plummer_slices.png")
fig2.savefig(slice_path, dpi=150)
print(f"saved slices figure      -> {slice_path}")


print("\nRESULT:", "PASS" if overall_ok else "FAIL")
assert overall_ok, "3D Plummer hydrostatic-equilibrium test failed for some solver"
