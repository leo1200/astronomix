"""Driven subsonic MHD turbulence in astronomix: the small-scale dynamo, with
resolution.

Half of a two-code convergence study. This script drives the ICM case of
``paper_turbulence.py`` (isothermal, ``M_turb ~ 0.5``, initial ``beta_p = 1e6``,
i.e. a seed field so weak that everything the magnetic field does afterwards is
dynamo) with the 5th-order FD/WENO solver and constrained transport, while
``athenapk_turb.py`` drives the *same* physical setup with AthenaPK's 2nd-order
(PLM + VL2) and 3rd-order (PPM + RK3) finite-volume GLM-MHD solver. Both codes
are reduced through ``_mhd_spectral.snapshot_spectra``.

The physical setup is byte-for-byte the ICM case already in this directory
(``data/paper_ICM_N128.npz``): unit periodic box, ``rho0 = 1``, isothermal sound
speed ``a = 1 / M_turb = 2``, ``B0 = sqrt(2 a^2 rho0 / beta) = 2.83e-3`` along z,
OU forcing with ``F0 = 3.5``, ``tau = 0.5``, ``k_f = 3 pi``, run for 30 crossing
times ``t_cross = L/2 / v_rms``. That is long enough to pass through the
kinematic growth phase and reach dynamo saturation at 128^3.

Snapshots are reduced *in flight* (spectra and scalars computed on the GPU, only
the reduced arrays crossing to the host), so the memory cost is independent of
the number of snapshots and the ladder can be pushed to 256^3 and beyond -- the
``return_states=True`` path used by ``paper_turbulence.py`` would need 36 GB of
snapshot buffer at 256^3.

    PYTHONPATH=$(git rev-parse --show-toplevel) \
        python examples/scripts/forward/mhd/turbulence/dynamo_convergence.py --n 128
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
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)

# astronomix functions
from astronomix._spatial_operators._differencing import _interface_field_divergence
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    initialize_interface_fields,
    time_integration,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_spectral import SCALAR_NAMES, SPECTRUM_NAMES, snapshot_spectra, shell_numbers

DATA_DIR = Path(__file__).resolve().parent / "data"
SCRATCH_DIR = Path("/export/data/lstorcks/mhd_dynamo")

#: Shared physical setup — athenapk_turb.py must stay in sync with these.
BOX_SIZE = 1.0          # unit periodic box
RHO0 = 1.0              # uniform initial density
L_INJ = 0.5             # injection scale, used for t_cross = L_INJ / v_rms
VRMS_TARGET = 1.0       # the normalisation drives v_rms ~ 1, so a = 1 / M_turb


# -------------------------------------------------------------
# ============ ↓ Solver configuration ↓ =======================
# -------------------------------------------------------------
def build_config(args, num_snapshots):
    """The 5th-order FD/WENO + constrained-transport configuration."""
    periodic = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    )
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        geometry=CARTESIAN,
        mhd=True,
        equation_of_state=ISOTHERMAL,
        box_size=BOX_SIZE,
        num_cells=args.n,
        boundary_settings=periodic,
        # Subsonic turbulence never approaches vacuum, so the positivity
        # machinery stays off and the scheme is unconditionally 5th order --
        # which is what this study measures. (The supersonic ISM case of
        # paper_turbulence.py needs it; this one does not.)
        positivity_config=PositivityConfig(),
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True,
            ou_forcing=True,              # temporally correlated, like AthenaPK
            vacuum_protection=False,
        ),
        # In-flight reduction: no snapshot buffer, so the memory cost does not
        # grow with num_snapshots (see module docstring).
        return_snapshots=False,
        activate_snapshot_callback=True,
        num_snapshots=num_snapshots,
        random_seed=args.seed,
        progress_bar=False,               # our own per-snapshot progress line
    )
# -------------------------------------------------------------
# ============ ↑ Solver configuration ↑ =======================
# -------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=128, help="cells per dimension")
    p.add_argument("--mturb", type=float, default=0.5, help="target turbulent Mach number")
    p.add_argument("--beta", type=float, default=1e6, help="initial plasma beta")
    p.add_argument("--seed-field", choices=("uniform", "sin"), default="uniform",
                   help="uniform: B0 along z, the HOW-MHD ICM configuration. Its "
                        "mean is exactly conserved in a periodic box, so the "
                        "early growth is tangling of a mean field rather than a "
                        "dynamo eigenmode -- see the README. sin: "
                        "B_x = sqrt(2) B0 sin(2 pi z / L), zero net flux at the "
                        "same magnetic energy, matching AthenaPK's b_config = 2.")
    p.add_argument("--tcross", type=float, default=40.0, help="run length in crossing times")
    p.add_argument("--F0", type=float, default=3.5, help="OU forcing amplitude")
    p.add_argument("--tau", type=float, default=0.5, help="OU correlation time")
    p.add_argument("--kf", type=float, default=3.0 * np.pi,
                   help="OU peak wavenumber (mode number n = k L / 2pi = 1.5)")
    p.add_argument("--cfl", type=float, default=1.5, help="astronomix C_cfl")
    p.add_argument("--nsnap", type=int, default=81,
                   help="snapshots over the whole run. The in-flight reduction "
                        "is not free (see the README's runtime section), so "
                        "timing runs use a small number")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--x64", action="store_true",
                   help="run in double precision (the default is x32, which is "
                        "what the ladder uses; this is the control that shows "
                        "the precision is not what sets the dynamo)")
    p.add_argument("--dealias", action="store_true",
                   help="form the transfer spectra on a 3/2-refined grid, so "
                        "the quadratic terms carry no aliases into the kept "
                        "shells (Orszag). Control for the aliasing systematic.")
    p.add_argument("--cell-average", action="store_true",
                   help="box-filter the state before reducing it, i.e. store "
                        "what a finite-volume code would have stored. Control "
                        "for the FD-vs-FV representation difference.")
    p.add_argument("--transfer", action="store_true",
                   help="also record the ideal spectral transfer, which turns "
                        "the spectra into a measurement of the scheme's "
                        "scale-dependent numerical diffusivity (see "
                        "make_dissipation_figure.py). Roughly doubles the "
                        "per-snapshot reduction cost.")
    p.add_argument("--scalars-only", action="store_true",
                   help="skip the spectra in the snapshot callback and record "
                        "only the scalar diagnostics. The spectra dominate the "
                        "per-snapshot cost (7.4 s at 256^3 against ~0.1 s for "
                        "the scalars), so this is what makes a high-cadence "
                        "E_B(t) affordable -- which the growth-rate fit needs, "
                        "and which AthenaPK gets for free from its .hst file.")
    p.add_argument("--slice-series", action="store_true",
                   help="keep the mid-plane magnetic-energy slice at every "
                        "snapshot, not just the last one (for the animation)")
    p.add_argument("--save-slices", action="store_true",
                   help="also store mid-plane slices of the last snapshot")
    p.add_argument("--tag", type=str, default="", help="suffix for the output file name")
    p.add_argument("--outdir", type=str, default=str(DATA_DIR))
    args = p.parse_args()

    if args.x64:
        jax.config.update("jax_enable_x64", True)

    a = 1.0 / args.mturb                       # isothermal sound speed
    P_thermal = a ** 2 * RHO0
    B0 = float(np.sqrt(2.0 * P_thermal / args.beta))
    t_cross = L_INJ / VRMS_TARGET
    t_end = args.tcross * t_cross

    config = build_config(args, args.nsnap)
    rv = get_registered_variables(config)
    params = SimulationParams(
        C_cfl=args.cfl,
        isothermal_sound_speed=a,
        t_end=t_end,
        turbulent_forcing_params=TurbulentForcingParams(
            forcing_amplitude=args.F0,
            correlation_time=args.tau,
            forcing_wavenumber=args.kf,
        ),
        minimum_density=1e-4,
        minimum_pressure=1e-6,
    )

    # Let the array dtype follow the x64 flag rather than pinning float32, so
    # --x64 really runs in double precision end to end.
    shape = (args.n,) * 3
    zeros = jnp.zeros(shape)
    if args.seed_field == "uniform":
        Bx, By, Bz = zeros, zeros, jnp.full(shape, B0)
    else:
        # B_x = sqrt(2) B0 sin(2 pi z / L): zero net flux, mean energy 0.5 B0^2
        # (the sqrt(2) is what keeps the energy equal to the uniform case), and
        # divergence-free for free since it varies only along z. Cell-centred z.
        z = (jnp.arange(args.n) + 0.5) * (BOX_SIZE / args.n)
        Bx = jnp.sqrt(2.0) * B0 * jnp.sin(2.0 * jnp.pi * z / BOX_SIZE)[None, None, :]
        Bx = jnp.broadcast_to(Bx, shape).astype(zeros.dtype)
        By, Bz = zeros, zeros
    bx_face, by_face, bz_face = initialize_interface_fields(Bx, By, Bz)
    state = construct_primitive_state(
        config=config, registered_variables=rv,
        density=jnp.full(shape, RHO0),
        velocity_x=zeros, velocity_y=zeros, velocity_z=zeros,
        magnetic_field_x=Bx, magnetic_field_y=By, magnetic_field_z=Bz,
        interface_magnetic_field_x=bx_face,
        interface_magnetic_field_y=by_face,
        interface_magnetic_field_z=bz_face,
    )
    config = finalize_config(config, state.shape)
    ng = config.num_ghost_cells

    print(f"[astronomix N={args.n}] isothermal a={a} beta0={args.beta:g} B0={B0:.4g} "
          f"seed={args.seed_field} "
          f"F0={args.F0} tau={args.tau} kf={args.kf:.3f} C_cfl={args.cfl} "
          f"t_end={t_end} ({args.tcross} t_cross)", flush=True)

    # -------------------------------------------------------------
    # ========= ↓ In-flight snapshot reduction (callback) ↓ ========
    # -------------------------------------------------------------
    di = rv.density_index
    vxi, vyi, vzi = rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z
    bxi, byi, bzi = rv.magnetic_index.x, rv.magnetic_index.y, rv.magnetic_index.z
    fxi, fyi, fzi = (rv.interface_magnetic_field_index.x,
                     rv.interface_magnetic_field_index.y,
                     rv.interface_magnetic_field_index.z)
    records = {"t": [], "scalars": [], "spectra": [], "lam": [], "divb_face": [],
               "slices": [], "keep": True}

    def reduce_cb(time, state, registered_variables):
        interior = (slice(ng, -ng) if ng else slice(None),) * 3
        rho = state[di][interior]
        vx, vy, vz = (state[i][interior] for i in (vxi, vyi, vzi))
        bx, by, bz = (state[i][interior] for i in (bxi, byi, bzi))
        dx = BOX_SIZE / args.n
        scalars, spectra = snapshot_spectra(
            rho, vx, vy, vz, bx, by, bz, a, dx, spectra=not args.scalars_only,
            transfer=args.transfer, dealias=args.dealias,
            cell_average_input=args.cell_average)

        # One mid-plane magnetic-energy slice per snapshot, for the animation.
        # 61 snapshots of a 256^2 float32 plane is 16 MB, so this can just be
        # carried in the reduced file rather than needing the raw dumps back.
        z_mid = rho.shape[2] // 2
        eb_slice = (0.5 * (bx[:, :, z_mid] ** 2 + by[:, :, z_mid] ** 2
                           + bz[:, :, z_mid] ** 2)) if args.slice_series \
            else jnp.zeros((1, 1), dtype=rho.dtype)

        # The constraint constrained transport actually preserves is the 6th-order
        # interface divergence of the *staggered* field, not the centred
        # difference of the cell-centred field that the shared `rel_divB`
        # measures. Both are reported, with the same normalisation so they sit
        # on one axis: the face measure is the honest statement about CT, and
        # the gap between them is what the face -> centre interpolation costs.
        bxf, byf, bzf = (state[i][interior] for i in (fxi, fyi, fzi))
        div_face = _interface_field_divergence(bxf, byf, bzf, dx)
        abs_b = jnp.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        divb_face = jnp.mean(jnp.where(
            abs_b > 0.0,
            0.5 * jnp.sqrt(3.0) * dx * jnp.abs(div_face) / jnp.maximum(abs_b, 1e-300),
            0.0))

        # Sum of the per-axis maximum fast-magnetosonic signal speeds. astronomix
        # sets dt = C_cfl * dx / (lam_x + lam_y + lam_z), so recording this lets
        # the step count be reconstructed on the host without a second run.
        b2_over_rho = (bx ** 2 + by ** 2 + bz ** 2) / jnp.maximum(rho, 1e-12)
        lam = 0.0
        for v, b in ((vx, bx), (vy, by), (vz, bz)):
            ca2 = b ** 2 / jnp.maximum(rho, 1e-12)
            s = a ** 2 + b2_over_rho
            c_fast = jnp.sqrt(0.5 * (s + jnp.sqrt(jnp.maximum(s ** 2 - 4.0 * a ** 2 * ca2, 0.0))))
            lam = lam + jnp.max(jnp.abs(v) + c_fast)

        def _host(t, sc, sp, lm, dvf, ebs):
            if not records["keep"]:
                return
            sc = np.asarray(sc)
            records["t"].append(float(t))
            records["scalars"].append(sc)
            records["spectra"].append(np.asarray(sp))
            records["lam"].append(float(lm))
            records["divb_face"].append(float(dvf))
            records["slices"].append(np.asarray(ebs, dtype=np.float32))
            named = dict(zip(SCALAR_NAMES, sc))
            print(f"  t/tc={float(t) / t_cross:6.2f}  M={named['mach']:.3f}  "
                  f"E_K={named['E_K']:.4f}  E_B={named['E_B']:.3e}  "
                  f"<|B|>={named['mean_absB']:.4e}  "
                  f"E_B/E_K={named['E_B'] / max(named['E_K'], 1e-30):.3e}", flush=True)

        jax.debug.callback(_host, time, scalars, spectra, lam, divb_face,
                           eb_slice)
    # -------------------------------------------------------------
    # ========= ↑ In-flight snapshot reduction (callback) ↑ ========
    # -------------------------------------------------------------

    # Warm-up call: same config (so the same compiled executable is reused) with
    # a t_end of a few steps, purely to move JIT compilation out of the timed
    # region. Its callbacks are discarded.
    records["keep"] = False
    t0 = walltime.time()
    warm_params = params._replace(t_end=1e-4)
    jax.block_until_ready(time_integration(state, config, warm_params, rv, reduce_cb))
    # block_until_ready waits for the *output*, not for the host callbacks the
    # snapshot recorder queues, so without this barrier a warm-up callback could
    # still land after `keep` is flipped back on and inject a spurious record.
    jax.effects_barrier()
    t_compile = walltime.time() - t0
    records["keep"] = True
    print(f"[astronomix N={args.n}] compile + warm-up: {t_compile:.1f} s", flush=True)

    t0 = walltime.time()
    final_state = time_integration(state, config, params, rv, reduce_cb)
    jax.block_until_ready(final_state)
    t_run = walltime.time() - t0
    jax.effects_barrier()          # every snapshot record has landed on the host
    print(f"[astronomix N={args.n}] run: {t_run:.1f} s", flush=True)

    times = np.asarray(records["t"])
    order = np.argsort(times)
    times = times[order]
    scalars = np.asarray(records["scalars"])[order]
    spectra = np.asarray(records["spectra"])[order]
    lam = np.asarray(records["lam"])[order]
    divb_face = np.asarray(records["divb_face"])[order]
    eb_slices = np.asarray(records["slices"])[order]

    # Step count from the recorded signal speeds: n_steps = int dt' / dt(t') with
    # dt = C_cfl * dx / lam_sum. Accurate to the snapshot sampling of lam_sum.
    dx = BOX_SIZE / args.n
    inv_dt = lam / (args.cfl * dx)
    n_steps = float(np.trapezoid(inv_dt, times)) if len(times) > 1 else float("nan")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"N{args.n}"
    path = out / f"astronomix_{tag}.npz"
    payload = dict(
        code="astronomix", label="astronomix WENO5+CT", scheme="WENO5 / SSP-RK / CT",
        N=args.n, tag=tag, t_wall=t_run, t_compile=t_compile,
        n_steps_estimated=n_steps, zone_updates_per_s=args.n ** 3 * n_steps / t_run,
        finite_volume=False,
        times=times, t_over_tc=times / t_cross, t_cross=t_cross,
        n_shell=shell_numbers(args.n), spectra=spectra, x64=args.x64,
        lam_sum=lam, cfl=args.cfl, rel_divB_face=divb_face,
        a=a, B0=B0, beta0=args.beta, mturb=args.mturb, rho0=RHO0,
        seed_field=args.seed_field,
        F0=args.F0, tau=args.tau, kf=args.kf, seed=args.seed,
        scalar_names=np.array(SCALAR_NAMES), spectrum_names=np.array(SPECTRUM_NAMES),
        **{name: scalars[:, i] for i, name in enumerate(SCALAR_NAMES)},
    )
    if args.slice_series:
        payload["EB_slice_series"] = eb_slices
    if args.save_slices:
        # time_integration returns the *unpadded* state (unlike the padded one
        # the snapshot callback sees), so no ghost slicing here.
        z = args.n // 2
        fin = np.asarray(final_state)
        payload["rho_slice"] = fin[di][:, :, z]
        payload["EB_slice"] = 0.5 * sum(fin[i][:, :, z] ** 2 for i in (bxi, byi, bzi))
        payload["EK_slice"] = 0.5 * fin[di][:, :, z] * sum(
            fin[i][:, :, z] ** 2 for i in (vxi, vyi, vzi))
    np.savez_compressed(path, **payload)
    print(f"[astronomix N={args.n}] wrote {path}  "
          f"(t_wall={t_run:.1f} s, ~{n_steps:.0f} steps)", flush=True)


if __name__ == "__main__":
    main()
