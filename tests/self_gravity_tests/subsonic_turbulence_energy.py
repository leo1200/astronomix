"""Energy conservation in subsonic self-gravitating structure formation.

A gentler companion to the Evrard collapse: instead of a singular cold collapse,
we build a *physically developed* subsonic turbulent state by **driving** the box
with Ornstein-Uhlenbeck forcing (gravity off), then switch the forcing off and
turn self-gravity on and watch the closed system evolve. Once the forcing stops
the box is a closed system, so the total energy (internal + kinetic + 1/2 rho phi)
is conserved exactly by a perfect scheme. Self-gravity amplifies the turbulent
over-densities into mild structure (delta rho / rho ~ M^2), far less extreme than
Evrard's collapse.

Two phases per resolution:

* **Stir** (hydro only, no gravity): OU forcing drives the box from rest to a
  statistically stationary subsonic turbulent state (u_rms ~ M c_s). The forcing
  amplitude F0 is calibrated so the stationary Mach lands near ``MACH``; we use
  whatever amplitude the forcing settles at (no velocity renormalisation), so the
  achieved Mach varies mildly with resolution and is recorded alongside the data.
* **Measure** (self-gravity on, forcing off): the developed state seeds the
  closed self-gravitating run; we record the relative total energy error.

The same stirred state seeds all three schemes at a given resolution (the stir is
deterministic via ``random_seed``), so the conservation comparison is fair.

Two outputs:

1. ``subsonic_turbulence_energy_convergence.svg`` -- the final relative total
   energy error |E(t_end) - E(0)| / |E(0)| vs resolution for the three FD
   self-gravity source-term schemes (simple / flux-based / corrected
   flux-based).
2. ``subsonic_turbulence_structure.svg`` -- a visual of the structure that
   forms (initial vs final projected column density + a final density slice),
   for the corrected flux-based source scheme at the highest resolution.

Raw results are cached to ``subsonic_turbulence_energy.npz`` so the figures can
be regenerated without re-running the simulations.

Run from the repo root (PYTHONPATH so the local astronomix is picked up):

    PYTHONPATH=$(git rev-parse --show-toplevel) python tests/self_gravity_tests/subsonic_turbulence_energy.py
    PYTHONPATH=$(git rev-parse --show-toplevel) python tests/self_gravity_tests/subsonic_turbulence_energy.py --rerun
"""

# ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import argparse
from pathlib import Path
from typing import NamedTuple

import sys

import jax

# --- precision ---
# ``--double`` reruns in float64 to expose the true energy-conservation floor of
# the conservative schemes (in float32 their residual is dominated by ~1 machine
# epsilon of round-off per timestep, i.e. it grows with resolution). x64 must be
# enabled before any array is created, hence this argv check at import time.
DOUBLE = "--double" in sys.argv
if DOUBLE:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    NATIVE_JAX,
    PALLAS,
    BoundarySettings,
    BoundarySettings1D,
    SECOND_ORDER_CONSERVATIVE,
    FINITE_DIFFERENCE,
    FORWARDS,
    PERIODIC_BOUNDARY,
    SIMPLE_SOURCE,
    SnapshotSettings,
    FOURTH_ORDER_CONSERVATIVE,
    GravityConfig,
    PositivityConfig,
    finalize_config,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)

# In float32 the Pallas/Triton backend is used (~10x faster). In float64 we fall
# back to native JAX: the Pallas kernel does some math in reduced precision and
# floors at ~1e-11, masking the true ~1e-13 round-off floor we want to expose.
RUN_BACKEND = NATIVE_JAX if DOUBLE else PALLAS
_SUFFIX = "_fp64" if DOUBLE else ""
_PREC_LABEL = "float64" if DOUBLE else "float32"

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
DATA_FILE = HERE / f"subsonic_turbulence_energy{_SUFFIX}.npz"
FIG_CONVERGENCE = FIG_DIR / f"subsonic_turbulence_energy_convergence{_SUFFIX}.svg"
FIG_STRUCTURE = FIG_DIR / "subsonic_turbulence_structure.svg"

# --- physical / numerical parameters ---------------------------------------
RESOLUTIONS = [32, 48, 64, 96, 128]
BOX_SIZE = 1.0
RHO_0 = 1.0
GAMMA = 5 / 3
SOUND_SPEED = 1.0
MACH = 0.5                       # target stationary RMS Mach number
V_RMS = MACH * SOUND_SPEED
K_PEAK_MODE = 2                  # injection at wavelength ~ L / 2
G_GRAV = 2 * jnp.pi             # mildly super-Jeans (lambda_J ~ 0.7 L)
T_CROSS = (BOX_SIZE / 2) / V_RMS
# Measurement window: long enough that the turbulent over-densities actually
# collapse (so the non-conservative simple source accumulates a significant
# energy error, ~1-5%), but kept below ~1.0 t_cross where the deep collapse NaNs
# the conservative flux schemes (no positivity floor is used, by design, since a
# floor would itself violate energy conservation). 0.9 t_cross is crash-free at
# every resolution 32^3-128^3 with margin.
T_END = 0.9 * T_CROSS
SEED = 0

# --- driven-stirring (phase 1) parameters ----------------------------------
# OU forcing drives the box from rest to a stationary subsonic turbulent state
# (gravity off). FORCING_AMPLITUDE was calibrated at N=48 so the stationary
# u_rms ~ MACH c_s (u_rms scales ~ sqrt(F0); F0=0.65 -> M~0.5); per the
# "use the forcing's stationary amplitude" choice we do NOT renormalise, so the
# achieved Mach varies mildly with resolution and is recorded in the npz.
FORCING_K = K_PEAK_MODE * 2 * jnp.pi / BOX_SIZE   # OU peak wavenumber (physical)
FORCING_CORR_TIME = T_CROSS                       # ~ one eddy turnover
FORCING_AMPLITUDE = 0.65
T_STIR = 3 * T_CROSS             # reach the stationary plateau (settles by ~1.8)

# Consistent scheme naming / styling, matching the other gravity figures.
SCHEME_LABELS = {
    SIMPLE_SOURCE: "FD, simple source",
    SECOND_ORDER_CONSERVATIVE: "FD, flux-based source",
    FOURTH_ORDER_CONSERVATIVE: "FD, corrected flux-based source",
}


class Scheme(NamedTuple):
    self_gravity_version: int
    marker: str
    linestyle: str
    color: str

    @property
    def label(self):
        return SCHEME_LABELS[self.self_gravity_version]


SCHEMES = [
    Scheme(SIMPLE_SOURCE, "o", ":", "C0"),
    Scheme(SECOND_ORDER_CONSERVATIVE, "s", "-.", "C1"),
    Scheme(FOURTH_ORDER_CONSERVATIVE, "^", "--", "C2"),
]
STRUCTURE_SCHEME = FOURTH_ORDER_CONSERVATIVE


_PERIODIC = BoundarySettings(
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)


def make_stir_config(num_cells):
    """Hydro-only config driven by OU forcing (gravity off): phase 1."""
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        progress_bar=False,
        positivity_config=PositivityConfig(default_positivity_protection=False),
        mhd=False,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        differentiation_mode=FORWARDS,
        boundary_settings=_PERIODIC,
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True, ou_forcing=True
        ),
        random_seed=SEED,
        return_snapshots=False,
        backend=RUN_BACKEND,
        pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True,
        pallas_interpret=False,
    )


def make_measure_config(num_cells, self_gravity_version, want_states):
    """Self-gravitating, unforced closed-system config: phase 2."""
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        progress_bar=False,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=self_gravity_version,
        ),
        positivity_config=PositivityConfig(default_positivity_protection=False),
        mhd=False,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        differentiation_mode=FORWARDS,
        boundary_settings=_PERIODIC,
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(
            return_states=want_states,
            return_final_state=True,
            return_total_energy=True,
            return_internal_energy=True,
            return_kinetic_energy=True,
            return_gravitational_energy=True,
        ),
        num_snapshots=40,
        backend=RUN_BACKEND,
        pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True,
        pallas_interpret=False,
    )


def stir_state(num_cells):
    """Phase 1: drive the box from rest with OU forcing (no gravity) to a
    stationary subsonic turbulent state. Returns ``(rho, vx, vy, vz, p, mach)``
    extracted from the final primitive state, ready to seed the measurement."""
    config = make_stir_config(num_cells)
    registered_variables = get_registered_variables(config)

    rho = RHO_0 * jnp.ones((num_cells, num_cells, num_cells))
    zero = jnp.zeros_like(rho)
    # ideal-gas pressure for the target (adiabatic) sound speed c_s^2 = gamma p / rho
    p = RHO_0 * SOUND_SPEED**2 / GAMMA * jnp.ones_like(rho)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=zero,
        velocity_y=zero,
        velocity_z=zero,
        gas_pressure=p,
    )
    config = finalize_config(config, initial_state.shape)

    params = SimulationParams(
        t_end=T_STIR,
        C_cfl=0.4,
        gamma=GAMMA,
        turbulent_forcing_params=TurbulentForcingParams(
            correlation_time=FORCING_CORR_TIME,
            forcing_wavenumber=FORCING_K,
            forcing_amplitude=FORCING_AMPLITUDE,
        ),
        minimum_density=1e-4,
        minimum_pressure=1e-5,
    )

    final = jax.block_until_ready(
        time_integration(initial_state, config, params, registered_variables)
    )
    rv = registered_variables
    sx, sy, sz = rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z
    stirred = (
        final[rv.density_index],
        final[sx],
        final[sy],
        final[sz],
        final[rv.pressure_index],
    )
    v_rms = float(jnp.sqrt(jnp.mean(final[sx] ** 2 + final[sy] ** 2 + final[sz] ** 2)))
    mach = v_rms / SOUND_SPEED
    return stirred, mach


def measure_case(num_cells, stirred, self_gravity_version, want_states=False):
    """Phase 2: from the stirred state, run the closed self-gravitating system
    (forcing off) and return the energy snapshots."""
    config = make_measure_config(num_cells, self_gravity_version, want_states)
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    rho, vx, vy, vz, p = stirred
    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=vx,
        velocity_y=vy,
        velocity_z=vz,
        gas_pressure=p,
    )
    config = finalize_config(config, initial_state.shape)

    params = SimulationParams(
        t_end=T_END,
        C_cfl=0.4,
        gamma=GAMMA,
        gravitational_constant=G_GRAV,
        minimum_density=1e-4,
        minimum_pressure=1e-5,
    )

    snapshots = jax.block_until_ready(
        time_integration(initial_state, config, params, registered_variables)
    )
    return snapshots, helper_data, registered_variables


def run_and_cache():
    energy_errors = {v: [] for v in SCHEME_LABELS}
    achieved_mach = []
    stirred_top = None  # reuse the highest-res stirred state for the structure visual

    for num_cells in RESOLUTIONS:
        # Phase 1: drive once per resolution; all schemes share this state.
        stirred, mach = stir_state(num_cells)
        achieved_mach.append(mach)
        print(f"  N={num_cells:4d}  stirred to Mach = {mach:.3f}", flush=True)
        if num_cells == RESOLUTIONS[-1]:
            stirred_top = stirred

        # Phase 2: closed self-gravitating run for each scheme.
        for scheme in SCHEMES:
            snapshots, _, _ = measure_case(
                num_cells, stirred, scheme.self_gravity_version
            )
            total = snapshots.total_energy
            rel_err = float(jnp.abs(total[-1] - total[0]) / jnp.abs(total[0]))
            energy_errors[scheme.self_gravity_version].append(rel_err)
            print(
                f"  N={num_cells:4d}  {scheme.label:32s}  |dE|/|E| = {rel_err:.6e}",
                flush=True,
            )

    save_kwargs = {
        "resolutions": np.array(RESOLUTIONS),
        "achieved_mach": np.array(achieved_mach),
    }
    for version, errs in energy_errors.items():
        save_kwargs[f"energy_error_{version}"] = np.array(errs)

    # The structure visual is precision-independent, so only compute it for the
    # default float32 run (skipped in --double to save the extra 128^3 run).
    if not DOUBLE:
        print(f"  structure run: N={RESOLUTIONS[-1]} {SCHEME_LABELS[STRUCTURE_SCHEME]}",
              flush=True)
        snaps, helper_data, regvars = measure_case(
            RESOLUTIONS[-1], stirred_top, STRUCTURE_SCHEME, want_states=True
        )
        states = np.asarray(snaps.states)
        density_series = states[:, regvars.density_index]  # (T, N, N, N)
        save_kwargs["structure_resolution"] = np.array(RESOLUTIONS[-1])
        save_kwargs["structure_time"] = np.asarray(snaps.time_points)
        save_kwargs["structure_density_initial"] = density_series[0]
        save_kwargs["structure_density_final"] = density_series[-1]

    np.savez_compressed(DATA_FILE, **save_kwargs)
    print(f"Saved -> {DATA_FILE}")


def plot_convergence():
    data = np.load(DATA_FILE)
    resolutions = data["resolutions"]
    mach_label = (
        f"\\mathcal{{M}}\\approx{float(np.mean(data['achieved_mach'])):.2f}"
        if "achieved_mach" in data
        else f"\\mathcal{{M}}={MACH}"
    )

    # The non-conservative simple source (~1e-3) sits decades above the
    # flux-based schemes (~1e-6), so a shared y-axis flattens the per-class
    # trends. Split into two panels by conservation class, each free to scale.
    by_version = {s.self_gravity_version: s for s in SCHEMES}
    panels = [
        ("non-conservative source", [SIMPLE_SOURCE]),
        ("flux-based (conservative) sources", [SECOND_ORDER_CONSERVATIVE, FOURTH_ORDER_CONSERVATIVE]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, (subtitle, versions) in zip(axes, panels):
        for v in versions:
            scheme = by_version[v]
            ax.plot(
                resolutions,
                data[f"energy_error_{v}"],
                marker=scheme.marker,
                linestyle=scheme.linestyle,
                color=scheme.color,
                linewidth=2.0,
                label=scheme.label,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(list(resolutions))
        ax.set_xticklabels([str(int(r)) for r in resolutions])
        ax.minorticks_off()
        ax.set_xlabel("number of cells per dimension $N$")
        ax.set_title(subtitle)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel(r"final relative total energy error $|\Delta E| / |E_0|$")

    fig.suptitle(
        f"Driven subsonic self-gravitating turbulence (${mach_label}$, {_PREC_LABEL}): "
        "energy conservation vs resolution"
    )
    fig.tight_layout()
    fig.savefig(FIG_CONVERGENCE)
    print(f"Saved -> {FIG_CONVERGENCE}")


def plot_structure():
    data = np.load(DATA_FILE)
    resolution = int(data["structure_resolution"])
    rho_i = data["structure_density_initial"]
    rho_f = data["structure_density_final"]
    t_final = float(data["structure_time"][-1])
    mach_label = (
        f"\\mathcal{{M}}\\approx{float(data['achieved_mach'][-1]):.2f}"
        if "achieved_mach" in data
        else f"\\mathcal{{M}}={MACH}"
    )

    # column density (projection along z) and a mid-plane slice of the final state
    col_i = rho_i.sum(axis=2) * (BOX_SIZE / resolution)
    col_f = rho_f.sum(axis=2) * (BOX_SIZE / resolution)
    slice_f = rho_f[:, :, resolution // 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    extent = (0, BOX_SIZE, 0, BOX_SIZE)

    col_vmin = min(col_i.min(), col_f.min())
    col_vmax = max(col_i.max(), col_f.max())
    im0 = axes[0].imshow(col_i.T, origin="lower", extent=extent, cmap="magma",
                         vmin=col_vmin, vmax=col_vmax)
    axes[0].set_title("initial column density")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(col_f.T, origin="lower", extent=extent, cmap="magma",
                         vmin=col_vmin, vmax=col_vmax)
    axes[1].set_title(f"final column density ($t = {t_final:.2f}$)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(
        slice_f.T, origin="lower", extent=extent, cmap="viridis",
        norm=LogNorm(vmin=max(slice_f.min(), 1e-3), vmax=slice_f.max()),
    )
    axes[2].set_title("final density slice ($z = L/2$)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(
        f"Driven subsonic self-gravitating structure formation "
        f"(${mach_label}$, $N = {resolution}^3$, "
        f"{SCHEME_LABELS[STRUCTURE_SCHEME]})"
    )
    fig.tight_layout()
    fig.savefig(FIG_STRUCTURE, dpi=200)
    print(f"Saved -> {FIG_STRUCTURE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="re-run simulations")
    parser.add_argument(
        "--double",
        action="store_true",
        help="run in float64 (native backend) to expose the round-off floor",
    )
    args = parser.parse_args()

    if args.rerun or not DATA_FILE.exists():
        run_and_cache()
    else:
        print(f"Using cached results in {DATA_FILE} (pass --rerun to recompute).")
    plot_convergence()
    if not DOUBLE:
        plot_structure()


if __name__ == "__main__":
    main()
