"""Driven subsonic hydrodynamic turbulence in astronomix (WENO5 finite difference).

Half of a two-code convergence study: this script drives isothermal, solenoidally
forced turbulence in a unit periodic box with the 5th-order FD/WENO solver, while
``athenak_turb.py`` drives the *same* physical setup with AthenaK's 2nd-order
PLM + HLLC finite-volume solver. ``spectra.py`` then computes time-averaged
kinetic-energy spectra for both with one shared estimator.

The forcing is matched to AthenaK's ``turb_driver`` term-by-term:

    - discrete driving band ``nlow <= n <= nhigh`` in mode number n = k L / 2pi
      (``banded_spectrum=True``), with the isotropic ``k^-(expo+2)/2`` envelope,
    - solenoidal projection of the forcing field,
    - Ornstein-Uhlenbeck update with ``fcorr = exp(-dt / tcorr)``,
    - exact energy injection ``dedt * dt`` per step via the same quadratic
      normalisation (``ou_exact_injection=True``).

Because the box has unit volume, astronomix's ``energy_injection_rate`` (a total
rate) and AthenaK's ``dedt`` (a rate per unit volume) are numerically the same
number, so both codes are given the identical ``--dedt``.

The two codes draw independent random forcing realisations, so the comparison is
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
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)

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
BOX_SIZE = 1.0          # unit box, so dedt (per volume) == E_inj (total)
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
        first_order_fallback=False,       # pure high-order flux, no FOFC
        box_size=BOX_SIZE,
        num_cells=args.n,
        boundary_settings=periodic,
        # Subsonic isothermal turbulence never approaches vacuum; leaving the
        # positivity machinery off keeps the scheme unconditionally 5th order,
        # which is exactly what this study is measuring.
        positivity_config=PositivityConfig(),
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True,
            ou_forcing=True,              # temporally correlated, like AthenaK
            ou_exact_injection=True,      # inject exactly dedt * dt per step
            banded_spectrum=True,         # discrete nlow..nhigh band
            vacuum_protection=False,
        ),
        return_snapshots=True,
        num_snapshots=args.nsnap,
        snapshot_settings=SnapshotSettings(
            return_states=True,
            return_kinetic_energy=True,
            return_total_mass=True,
        ),
        random_seed=args.seed,
        progress_bar=True,
    )
# -------------------------------------------------------------
# ============ ↑ Solver configuration ↑ =======================
# -------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=128, help="cells per dimension")
    p.add_argument("--mach", type=float, default=0.3,
                   help="target turbulent Mach number (sets the default dedt)")
    p.add_argument("--dedt", type=float, default=-1.0,
                   help="energy injection rate; <0 uses the v_rms^3 / (2 L) estimate")
    p.add_argument("--tcorr", type=float, default=-1.0,
                   help="OU correlation time; 0 = white-in-time driving; "
                        "<0 uses L / (2 pi v_rms)")
    p.add_argument("--nlow", type=int, default=1, help="driving band, low mode number")
    p.add_argument("--nhigh", type=int, default=2, help="driving band, high mode number")
    p.add_argument("--expo", type=float, default=5.0 / 3.0, help="driving envelope exponent")
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
    # AthenaK's turb.athinput convention, dedt ~ v_rms^3 / (2 L), which puts
    # astronomix at Mach ~0.31-0.32 at every resolution in this ladder — its
    # injection really is the requested rate, so no per-N calibration is needed
    # on this side (AthenaK's is; see calibrate_athenak.py).
    dedt = args.dedt if args.dedt > 0 else v_target ** 3 / (2.0 * BOX_SIZE)
    if args.tcorr < 0:
        tcorr = BOX_SIZE / (2.0 * np.pi * v_target)
    elif args.tcorr == 0.0:
        # White-in-time driving: exp(-dt / tcorr) underflows to 0, so the OU
        # field is a fresh draw each step, matching AthenaK's tcorr <= 1e-6
        # branch. Kept as an option for the driving diagnostics; note it did NOT
        # rescue AthenaK's resolution dependence (check_athenak_driving.py finds
        # a 66% Mach spread under white driving, worse than OU's 22%).
        tcorr = 1e-30
    else:
        tcorr = args.tcorr
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
        turbulent_forcing_params=TurbulentForcingParams(
            energy_injection_rate=dedt,      # unit box volume => same as AthenaK dedt
            correlation_time=tcorr,
            forcing_nlow=args.nlow,
            forcing_nhigh=args.nhigh,
            forcing_expo=args.expo,
        ),
    )

    print(f"[astronomix-turb] N={args.n}^3  target Mach={args.mach}  cs={CS}")
    print(f"[astronomix-turb] dedt={dedt:.6g}  tcorr={tcorr:.4f}  band n=[{args.nlow},{args.nhigh}] "
          f"expo={args.expo:.4f}")
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
        n=args.n, box_size=BOX_SIZE, cs=CS, dedt=dedt, tcorr=tcorr,
        nlow=args.nlow, nhigh=args.nhigh, expo=args.expo, weighted=args.weighted,
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
