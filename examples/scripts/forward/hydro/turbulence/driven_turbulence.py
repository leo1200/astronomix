"""Driven subsonic hydrodynamic turbulence in astronomix (WENO5 finite difference).

Half of a two-code convergence study: this script drives isothermal, solenoidally
forced turbulence in a unit periodic box with the 5th-order FD/WENO solver, while
``athenak_turb.py`` drives the *same* physical setup with AthenaK's 2nd-order
PLM + HLLC finite-volume solver. ``spectra.py`` then computes time-averaged
kinetic-energy spectra for both with one shared estimator.

The script runs the TurbGen driving of the code now: solenoidal, a flat band
``nlow <= n <= nhigh`` in mode number n = k L / 2pi, Ornstein-Uhlenbeck with the
turnover time ``L / (k_driving v_rms)`` as correlation time, the amplitude
adjusted automatically towards the target velocity dispersion. The earlier
term-by-term match with AthenaK's ``turb_driver`` (its ``k^-(expo+2)/2``
envelope, ``tcorr`` and the exact injection ``dedt * dt``) was made with the
forcing module before the streamline branch (tag ``pre-streamline``); the
stored ``dedt`` is NaN now and ``tcorr`` is the correlation time of the driving.

The two codes draw independent random driving realisations, so the comparison is
statistical: spectra are averaged over snapshots in the stationary window.

    PYTHONPATH=$(git rev-parse --show-toplevel) \
        python examples/scripts/forward/hydro/turbulence/driven_turbulence.py --n 128
"""

# ==== GPU selection ====
import os
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
import sys
import time as walltime
from pathlib import Path

# numerics
import numpy as np
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FINITE_DIFFERENCE,
    ISOTHERMAL,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)
from astronomix import TurbulenceDrivingConfig, TurbulenceDrivingParams
from astronomix._modules._turbulence_driving import BAND

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _spectral import reduce_snapshots

DATA_DIR = Path(__file__).resolve().parent / "data"
SCRATCH_DIR = Path("/export/data/lstorcks/turb_spectra")

#: Shared physical setup (see athenak_turb.py — these must stay in sync).
BOX_SIZE = 1.0          # unit box
RHO0 = 1.0              # uniform initial density
CS = 1.0                # isothermal sound speed; Mach = v_rms / CS


# -------------------------------------------------------------
# ============ ↓ Solver configuration ↓ =======================
# -------------------------------------------------------------
def build_config(args):
    """The 5th-order FD/WENO configuration for the driven box."""
    periodic = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    )
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        geometry=CARTESIAN,
        mhd=False,
        equation_of_state=ISOTHERMAL,
        box_size=BOX_SIZE,
        num_cells=args.n,
        boundary_settings=periodic,
        # Subsonic isothermal turbulence never approaches vacuum; leaving the
        # positivity machinery off keeps the scheme unconditionally 5th order,
        # which is exactly what this study is measuring.
        positivity_config=PositivityConfig(),
        turbulence_driving_config=TurbulenceDrivingConfig(
            turbulence_driving=True,      # temporally correlated, like AthenaK
            spectral_form=BAND,           # flat nlow..nhigh band
            k_min=args.nlow,
            k_max=args.nhigh,
            random_seed=args.seed,
            amplitude_auto_adjust=True,
        ),
        return_snapshots=True,
        num_snapshots=args.nsnap,
        snapshot_settings=SnapshotSettings(
            return_states=True,
            return_kinetic_energy=True,
            return_total_mass=True,
        ),
        progress_bar=True,
    )
# -------------------------------------------------------------
# ============ ↑ Solver configuration ↑ =======================
# -------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=128, help="cells per dimension")
    p.add_argument("--mach", type=float, default=0.3,
                   help="target turbulent Mach number (the target velocity dispersion of the driving)")
    p.add_argument("--nlow", type=int, default=1, help="driving band, low mode number")
    p.add_argument("--nhigh", type=int, default=2, help="driving band, high mode number")
    p.add_argument("--nturn", type=float, default=10.0,
                   help="run length in large-eddy turnover times L / (2 v_rms)")
    p.add_argument("--nsnap", type=int, default=41, help="snapshots over the whole run")
    # astronomix forms dt = C_cfl * dx / (lambda_x + lambda_y + lambda_z) while
    # AthenaK uses dt = cfl_number * min_i(dx / lambda_i), so C_cfl = 0.9 here is
    # the same effective time step as AthenaK's cfl_number = 0.3 in 3D.
    p.add_argument("--cfl", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--x64", action="store_true", help="run in double precision")
    p.add_argument("--weighted", action="store_true",
                   help="reduce the sqrt(rho)-weighted spectrum instead of the plain one")
    p.add_argument("--save-states", action="store_true",
                   help="also dump the raw snapshot cubes to scratch")
    p.add_argument("--tag", type=str, default="", help="suffix for the output file name")
    args = p.parse_args()

    if args.x64:
        jax.config.update("jax_enable_x64", True)

    v_target = args.mach * CS
    # The correlation time of the driving is its turnover time L / (k_driving v_target).
    k_driving = 0.5 * (args.nlow + args.nhigh)
    tcorr = BOX_SIZE / (k_driving * v_target)
    t_turnover = BOX_SIZE / (2.0 * v_target)
    t_end = args.nturn * t_turnover

    config = build_config(args)
    rv = get_registered_variables(config)

    shape = (args.n,) * 3
    zeros = jnp.zeros(shape)
    state = construct_primitive_state(
        config=config, registered_variables=rv,
        density=jnp.full(shape, RHO0),
        velocity_x=zeros, velocity_y=zeros, velocity_z=zeros,
        gas_pressure=jnp.full(shape, RHO0 * CS ** 2),   # isothermal p = rho cs^2
    )
    config = finalize_config(config, state.shape)

    params = SimulationParams(
        C_cfl=args.cfl,
        t_end=t_end,
        isothermal_sound_speed=CS,
        turbulence_driving_params=TurbulenceDrivingParams(
            velocity=v_target, k_driving=k_driving, solenoidal_weight=1.0
        ),
    )

    print(f"[astronomix-turb] N={args.n}^3  target Mach={args.mach}  cs={CS}")
    print(f"[astronomix-turb] TurbGen driving  tcorr={tcorr:.4f}  band n=[{args.nlow},{args.nhigh}]")
    print(f"[astronomix-turb] t_turnover={t_turnover:.4f}  t_end={t_end:.4f} "
          f"({args.nturn} turnovers, {args.nsnap} snapshots)")

    t0 = walltime.perf_counter()
    result = time_integration(state, config, params, rv, sharding=None)
    jax.block_until_ready(result)
    runtime = walltime.perf_counter() - t0

    # Reduce to per-snapshot spectra here, while the states are still in memory:
    # 41 snapshots of a 256^3 box are ~11 GB of raw cubes but only a few kB of
    # spectra, and the averaging window is chosen later in analysis.
    states = np.asarray(result.states)            # (nsnap, nvar, N, N, N)
    times = np.asarray(result.time_points)

    def snapshots():
        for i, t in enumerate(times):
            yield (float(t),
                   states[i, rv.density_index].astype(np.float64),
                   states[i, rv.velocity_index.x].astype(np.float64),
                   states[i, rv.velocity_index.y].astype(np.float64),
                   states[i, rv.velocity_index.z].astype(np.float64))

    reduced = reduce_snapshots(snapshots(), weighted=args.weighted)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"n{args.n}"
    out = DATA_DIR / f"astronomix_{tag}.npz"
    np.savez_compressed(
        out, **reduced,
        n=args.n, box_size=BOX_SIZE, cs=CS, dedt=np.nan, tcorr=tcorr,
        nlow=args.nlow, nhigh=args.nhigh, weighted=args.weighted,
        t_turnover=t_turnover, runtime=runtime,
        num_iterations=int(result.num_iterations),
        label="astronomix (WENO5 FD, 5th order)",
    )
    v_rms = reduced["v_rms"]
    print(f"[astronomix-turb] stationary Mach (last half) = "
          f"{v_rms[len(v_rms) // 2:].mean() / CS:.4f}")
    print(f"[astronomix-turb] runtime {runtime:.1f} s over {int(result.num_iterations)} steps "
          f"-> {out}")

    if args.save_states:
        # Raw cubes go to scratch (the repo filesystem has no room for them);
        # only needed for slice images and re-analysis with a different estimator.
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        raw = SCRATCH_DIR / f"astronomix_{tag}_states.npz"
        np.savez(
            raw, times=times,
            rho=states[:, rv.density_index].astype(np.float32),
            vx=states[:, rv.velocity_index.x].astype(np.float32),
            vy=states[:, rv.velocity_index.y].astype(np.float32),
            vz=states[:, rv.velocity_index.z].astype(np.float32),
        )
        print(f"[astronomix-turb] raw states -> {raw}")


if __name__ == "__main__":
    main()
