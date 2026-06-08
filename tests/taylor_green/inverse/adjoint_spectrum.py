"""
Stage 1 of the inverse-modeling study: the scale-resolved adjoint
(gradient) energy spectrum vs. horizon on the 3D compressible
Taylor-Green vortex.

This is the "clean first cut" that ``init_optim_theory.md`` Section 8
recommends *before* committing to the full single- vs multiple-shooting
inverse problem:

    "A clean first cut is the scale-resolved adjoint-spectrum experiment
     on the Taylor-Green vortex before committing to the full inverse
     problem."

**What it shows.** Single shooting computes the gradient of a *large
scale* observable by propagating the adjoint backward through the entire
turbulent trajectory. The theory predicts that adjoint grows and piles
up at high k (local rate ~1/tau_eta, the Kolmogorov time), so the
large-scale gradient is swamped by amplified small-scale content within
a fraction of a turnover. Multiple shooting caps the back-prop to a
single short segment, so the per-segment gradient stays large-scale.

**The experiment.** Take a fixed large-scale (low-k) co-vector ``w``
(here the analytic TGV velocity pattern, which lives at |k|=sqrt(3)).
Define the linear observable

    J(u0) = < w , u(T) >        (inner product of w with the final velocity)

so that its gradient with respect to the initial velocity field,

    g_T = dJ/du0 = (dF_T/du0)^T w,

is exactly the adjoint field at t=0 seeded by the large-scale
observation w and pulled back through the horizon T. We evaluate g_T at
the base trajectory for a sweep of horizons T and look at the
kinetic-energy-style spectrum E_g(k) of g_T:

    * small T  == one multiple-shooting segment  -> spectrum stays low-k
    * large T  == single shooting over the whole horizon -> spectrum
                  migrates / piles up at high k

We run the sweep around two base states:

  1. LAMINAR start -- the smooth from-rest TGV. The flow is laminar until
     t~3.5, so the *total* gradient norm stays flat (dominated by the
     decaying large-scale band) and only the high-k band grows
     exponentially underneath; the total norm turns up only once the
     small-scale band overtakes the large-scale one (~T=4). This is the
     honest, physically meaningful inverse-problem regime.

  2. TURBULENT restart -- seed the adjoint from an already-turbulent
     state (the TGV evolved to t_turb near peak dissipation). Now the
     unstable subspace is populated from the start, so even the *total*
     gradient norm blows up ~e^{lambda T} with no laminar lag. Recovering
     an IC from such a start is not physically sensible, but it is the
     clean demonstration of the gradient pathology -- and the short-T end
     of the very same curve is exactly the per-segment gradient multiple
     shooting would see, i.e. the fix.

Hydro finite-difference (WENO) scheme, single GPU, reverse-mode AD
through ``time_integration`` (``differentiation_mode = BACKWARDS`` with
checkpointing). Note: the Pallas backend falls back to native JAX on the
backward pass, so the gradient is native-JAX either way.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.analysis_helpers.energy_spectrum import (
    LOG_BINNING,
    vector_field_energy_spectrum,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    finalize_config,
)


# ---------------------------------------------------------------------------
#  Parameters
# ---------------------------------------------------------------------------

NUM_CELLS = 64
BOX_SIZE = 2.0 * jnp.pi
V0 = 1.0
RHO0 = 1.0
MA0 = 0.3
GAMMA = 5.0 / 3.0
C_CFL = 0.4
NUM_CHECKPOINTS = 32

# Horizons in eddy-turnover units (t_c = 1/(V0 k0) = 1).
HORIZONS_LAMINAR = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
# Turbulent restart: grows fast from t=0, so a shorter sweep suffices.
HORIZONS_TURB = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
T_TURB = 8.0  # time to evolve the TGV to before seeding the turbulent restart

# band split for the low-k / high-k gradient norm (integer mode number)
K_SPLIT = 4.0

OUTPUT_DIR = Path(__file__).parent / "figures"
DATA_DIR = Path(__file__).parent / "data"


def build_config_params():
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS,
        mhd=False,
        backend=NATIVE_JAX,           # backward pass is native JAX regardless
        differentiation_mode=BACKWARDS,
        num_checkpoints=NUM_CHECKPOINTS,
        enforce_positivity=False,     # avoid non-smooth clamps biasing the adjoint
        progress_bar=False,
        dimensionality=3,
        num_cells=NUM_CELLS,
        box_size=BOX_SIZE,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        return_snapshots=False,
    )
    params = SimulationParams(C_cfl=C_CFL, gamma=GAMMA, t_end=1.0)
    return config, params


def tgv_velocity(helper_data):
    """Analytic TGV velocity field (purely large-scale, |k| = sqrt(3))."""
    coords = helper_data.geometric_centers
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    vx = V0 * jnp.sin(x) * jnp.cos(y) * jnp.cos(z)
    vy = -V0 * jnp.cos(x) * jnp.sin(y) * jnp.cos(z)
    vz = jnp.zeros_like(x)
    return jnp.stack([vx, vy, vz])


def background_pressure_field(shape):
    p0 = RHO0 * V0**2 / (GAMMA * MA0**2)
    return p0 * jnp.ones(shape)


def spectral_centroid(k, ek):
    """Energy-weighted mean wavenumber <k> = sum(k E) / sum(E)."""
    return float(jnp.sum(k * ek) / jnp.sum(ek))


def run_sweep(config, params, registered_variables, base_velocity,
              base_density, base_pressure, seed_w, horizons, label):
    """Compute the adjoint spectrum of J=<w,u(T)> vs horizon for a base state.

    Returns a dict of results and produces a 2x2 diagnostic figure.
    """
    vidx = registered_variables.velocity_index

    def observable(velocity, t_end):
        state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=base_density,
            velocity_x=velocity[0],
            velocity_y=velocity[1],
            velocity_z=velocity[2],
            gas_pressure=base_pressure,
        )
        local_params = params._replace(t_end=t_end)
        final_state = time_integration(state, config, local_params, registered_variables)
        return jnp.sum(
            seed_w[0] * final_state[vidx.x]
            + seed_w[1] * final_state[vidx.y]
            + seed_w[2] * final_state[vidx.z]
        )

    npz_path = DATA_DIR / f"adjoint_spectrum_{label}.npz"
    if npz_path.exists():
        # decouple plotting from the expensive sweep: replot from cache
        print(f"\n=== Adjoint-spectrum sweep [{label}] -- loading cached "
              f"{npz_path.name} ===")
        d = jnp.load(npz_path)
        Ts = [float(t) for t in d["horizons"]]
        centroids = [float(c) for c in d["centroids"]]
        lows = [float(x) for x in d["low_k_norm"]]
        highs = [float(x) for x in d["high_k_norm"]]
        totals = [float(x) for x in d["total_norm"]]
        spectra = {T: (d[f"k_T{T}"], d[f"eg_T{T}"]) for T in Ts}
    else:
        grad_observable = jax.jit(jax.grad(observable))
        spectra = {}
        centroids, lows, highs, totals = [], [], [], []
        print(f"\n=== Adjoint-spectrum sweep [{label}], N={NUM_CELLS}^3 ===")
        for T in horizons:
            g = grad_observable(base_velocity, jnp.asarray(T, dtype=base_velocity.dtype))
            g.block_until_ready()
            if bool(jnp.any(jnp.isnan(g))):
                print(f"  T={T:>5}: NaN in gradient -- skipping")
                continue
            k, eg = vector_field_energy_spectrum(g[0], g[1], g[2], binning=LOG_BINNING)
            k_mode = k / (2.0 * jnp.pi)
            lo = float(jnp.sum(eg[k_mode < K_SPLIT]))
            hi = float(jnp.sum(eg[k_mode >= K_SPLIT]))
            cen = spectral_centroid(k_mode, eg)
            spectra[T] = (k_mode, eg)
            centroids.append(cen); lows.append(lo); highs.append(hi); totals.append(lo + hi)
            print(f"  T={T:>5}: ||g||^2={lo + hi:.3e}  low-k={lo:.3e}  "
                  f"high-k={hi:.3e}  <k>={cen:.2f}")
        Ts = list(spectra.keys())
        jnp.savez(
            npz_path,
            horizons=jnp.array(Ts),
            centroids=jnp.array(centroids),
            low_k_norm=jnp.array(lows),
            high_k_norm=jnp.array(highs),
            total_norm=jnp.array(totals),
            **{f"k_T{T}": spectra[T][0] for T in spectra},
            **{f"eg_T{T}": spectra[T][1] for T in spectra},
        )

    highs_a = jnp.array(highs)
    # exponential fit of the high-k band over the turbulent part of the sweep
    fit_mask = jnp.array(Ts) >= (3.0 if label == "laminar" else 1.0)
    lam = None
    if int(jnp.sum(fit_mask)) >= 2:
        lam = float(jnp.polyfit(jnp.array(Ts)[fit_mask], jnp.log(highs_a[fit_mask]), 1)[0])
        print(f"  high-k band exp rate lambda ~ {lam:.2f} / t_c  "
              f"(e-folding {1.0 / lam:.2f} t_c)")

    # ---- 2x2 figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    cmap = plt.cm.viridis
    n = len(spectra)
    for i, (T, (k_mode, eg)) in enumerate(spectra.items()):
        color = cmap(i / max(n - 1, 1))
        axes[0, 0].loglog(k_mode, eg, "-o", ms=3, color=color, label=f"T={T}")
        axes[0, 1].loglog(k_mode, eg / jnp.sum(eg), "-o", ms=3, color=color, label=f"T={T}")
    axes[0, 0].set(xlabel="k (mode number)", ylabel=r"$E_g(k)$",
                   title="Adjoint spectrum (raw)")
    axes[0, 0].set_xlim(1, NUM_CELLS // 2); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xlabel="k (mode number)", ylabel=r"$E_g(k)/\sum E_g$",
                   title="Adjoint spectrum (normalized): migration to high k")
    axes[0, 1].set_xlim(1, NUM_CELLS // 2); axes[0, 1].legend(fontsize=8)
    axes[0, 1].axvline(K_SPLIT, color="gray", ls=":", alpha=0.6)

    axes[1, 0].plot(Ts, centroids, "-o")
    if label == "laminar":
        axes[1, 0].axvline(3.5, color="k", ls="--", alpha=0.5, label="laminar->turbulent (~3.5)")
        axes[1, 0].legend(fontsize=8)
    axes[1, 0].set(xlabel="horizon T / t_c", ylabel=r"spectral centroid $\langle k\rangle$",
                   title="Adjoint pile-up at small scales vs horizon")
    axes[1, 0].grid(alpha=0.3)

    # band-resolved norm panel (answers: where is the exponential?)
    axes[1, 1].semilogy(Ts, totals, "-ko", label="total ||g||^2")
    axes[1, 1].semilogy(Ts, lows, "-o", color="C0", label=f"low-k (k<{K_SPLIT:g})")
    axes[1, 1].semilogy(Ts, highs, "-o", color="C3", label=f"high-k (k>={K_SPLIT:g})")
    if lam is not None:
        Tf = jnp.array(Ts)[fit_mask]
        c0 = float(jnp.log(highs_a[fit_mask][0]) - lam * float(Tf[0]))
        axes[1, 1].semilogy(Tf, jnp.exp(c0 + lam * Tf), "k--", alpha=0.6,
                            label=f"~ exp({lam:.2f} T)")
    axes[1, 1].set(xlabel="horizon T / t_c", ylabel="gradient energy",
                   title="Band-resolved gradient norm: exp growth lives at high k")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3, which="both")

    title = ("LAMINAR start: total norm masked by decaying large scales; "
             "high-k band grows exponentially underneath"
             if label == "laminar" else
             "TURBULENT restart: high-k adjoint grows from t=0 (no laminar lag); "
             "total still large-scale-dominated for a large-scale observable")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"adjoint_spectrum_{label}.png", dpi=200)
    plt.close(fig)
    print(f"  figure -> {OUTPUT_DIR / f'adjoint_spectrum_{label}.png'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    config, params = build_config_params()
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    u0 = tgv_velocity(helper_data)
    density = RHO0 * jnp.ones((NUM_CELLS, NUM_CELLS, NUM_CELLS))
    pressure = background_pressure_field(density.shape)

    probe_state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=density, velocity_x=u0[0], velocity_y=u0[1], velocity_z=u0[2],
        gas_pressure=pressure,
    )
    config = finalize_config(config, probe_state.shape)

    # large-scale co-vector: normalized analytic TGV velocity pattern (low-k)
    w = u0 / jnp.sqrt(jnp.sum(u0**2))

    # (1) laminar from-rest TGV start
    run_sweep(config, params, registered_variables, u0, density, pressure,
              w, HORIZONS_LAMINAR, "laminar")

    # (2) turbulent restart: evolve the TGV to t_turb (near peak dissipation)
    print(f"\nGenerating turbulent base state by evolving TGV to t={T_TURB} ...")
    turb_state = time_integration(
        probe_state, config, params._replace(t_end=T_TURB), registered_variables
    )
    turb_velocity = jnp.stack([
        turb_state[registered_variables.velocity_index.x],
        turb_state[registered_variables.velocity_index.y],
        turb_state[registered_variables.velocity_index.z],
    ])
    turb_density = turb_state[registered_variables.density_index]
    turb_pressure = turb_state[registered_variables.pressure_index]
    run_sweep(config, params, registered_variables, turb_velocity,
              turb_density, turb_pressure, w, HORIZONS_TURB, "turbulent")


if __name__ == "__main__":
    main()
