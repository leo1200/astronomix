#!/usr/bin/env python
"""Generate the 4 density cubes for the field-level-inference panel figure.

Loads the optimized 64^3 MHD state and evolves it (turbulence OFF) to four
times:

    t = 0  (initial optimized state)
    t = f_pre  * t_end   (shortly before the loss time)
    t = t_end            (the target / loss time)
    t = f_post * t_end   (shortly after)

The loss time t_end is recovered WITHOUT the expensive forced-turbulence run:
we scan evolution times and pick the one whose z-projection best *correlates*
with the target logo (the velocity was optimized so the logo appears exactly at
t_end).  Correlation -- not MSE -- is used because the logo's contrast is only
~1%, so an MSE of the projection is dominated by the overall mean amplitude and
mis-detects the (narrow) logo peak.  The scan is two-stage (coarse bracket then
fine refine) since the peak is only ~1-2% of t_end wide.

Default device is GPU via autocvd (CPU compute is against cluster rules here).
Saves a single npz (4 density cubes [N,N,N] + times + target image).
Intended env: astx (jax 0.10.2 + astronomix) or jf1uids.
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--resolution", type=int, default=64)
ap.add_argument("--state", type=str, default="best_state.npy")
ap.add_argument("--out", type=str, default="panel_snaps.npz")
ap.add_argument("--f-pre", type=float, default=0.995)
ap.add_argument("--f-post", type=float, default=1.005)
ap.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"],
                help="cpu is against cluster rules on this box; use gpu (autocvd).")
ap.add_argument("--backend", type=str, default="pallas", choices=["pallas", "native"])
ap.add_argument("--scan-lo", type=float, default=0.4)
ap.add_argument("--scan-hi", type=float, default=1.8)
ap.add_argument("--scan-n", type=int, default=15)
ap.add_argument("--refine-n", type=int, default=41,
                help="fine-scan samples spanning +/- one coarse step around the peak.")
ap.add_argument("--t-end", type=float, default=0.0,
                help="if >0, skip the correlation scan and use this loss time directly "
                     "(needed at 256³ where each evolution is minutes).")
args = ap.parse_args()

if args.device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    from autocvd import autocvd
    autocvd(num_gpus=1)

# ruff: noqa: E402
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image

from astronomix import (SimulationConfig, SimulationParams, get_registered_variables,
                        construct_primitive_state, time_integration, CodeUnits)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    PALLAS, finalize_config, GravityConfig, PositivityConfig,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astropy import units as u
import astropy.constants as c


def load_image(path, height, width):
    img = jnp.array(Image.open(path).convert("L"))
    img = 1.0 - img / 255.0
    h, w = img.shape
    pad_h = (-h) % height
    pad_w = (-w) % width
    pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2))
    img = jnp.pad(img, pad)
    hp, wp = img.shape
    result = img.reshape(height, hp // height, width, wp // width).mean(axis=(1, 3))
    return 1.0 + (result - 0.01) / (1.0 - 0.01) * 0.01


def main():
    resolution = args.resolution
    gamma = 5 / 3
    box_size = 1.0
    print(f"device={args.device} backend={args.backend} jax={jax.__version__} "
          f"platform={jax.default_backend()}")

    backend_kw = {}
    if args.backend == "pallas":
        backend_kw = dict(backend=PALLAS, pallas_block_shape=(4, 4, 8),
                          pallas_use_triton=True, pallas_interpret=False)

    config = SimulationConfig(
        positivity_config=PositivityConfig(default_positivity_protection=True),
        solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
        donate_state=False, dimensionality=3,
        box_size=box_size, num_cells=resolution,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=False),
        boundary_settings=BoundarySettings(
            *[BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY,
                                 right_boundary=PERIODIC_BOUNDARY) for _ in range(3)]),
        **backend_kw,
    )
    rv = get_registered_variables(config)

    code_length = 3 * u.parsec
    code_mass = 100 * u.M_sun
    code_velocity = 100 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)

    n_h = 2
    rho_0 = n_h * c.m_p / u.cm**3
    p_0 = 3e4 * u.K / u.cm**3 * c.k_B

    params = SimulationParams(
        C_cfl=0.8, dt_max=0.1, gamma=gamma,
        minimum_density=(1e-2 * rho_0).to(code_units.code_density).value,
        minimum_pressure=(1e-2 * p_0).to(code_units.code_pressure).value,
    )

    # a config.finalize needs a representative state shape
    sh = (resolution,) * 3
    zero = jnp.zeros(sh)
    B_0 = (13.5 * u.microgauss / c.mu0**0.5).to(code_units.code_magnetic_field).value
    bxb, byb, bzb = initialize_interface_fields(jnp.ones(sh) * B_0, zero, zero)
    probe = construct_primitive_state(
        config=config, registered_variables=rv, density=jnp.ones(sh),
        velocity_x=zero, velocity_y=zero, velocity_z=zero, gas_pressure=jnp.ones(sh),
        magnetic_field_x=jnp.ones(sh) * B_0, magnetic_field_y=zero, magnetic_field_z=zero,
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb)
    config = finalize_config(config, probe.shape)

    best_state = jnp.asarray(np.load(args.state))
    di = rv.density_index
    vi = rv.velocity_index

    target = load_image("logo.png", resolution, resolution)
    target = target / jnp.sum(target) * jnp.sum(best_state[di])
    tgt_zm = target - jnp.mean(target)

    # crude crossing-time estimate from the optimized velocity to centre the scan
    rms_v = float(jnp.sqrt(jnp.mean(best_state[vi.x]**2 + best_state[vi.y]**2 + best_state[vi.z]**2)))
    t_cross = box_size / max(rms_v, 1e-12)
    t_center = 0.5 * t_cross
    print(f"rms_v(best)={rms_v:.4f}  t_cross~{t_cross:.4f}  t_center~{t_center:.4f}")

    def evolve(t):
        if t <= 0.0:
            return best_state[di]
        fin = time_integration(best_state, config, params._replace(t_end=float(t)), rv)
        return fin[di]

    def logo_corr(t):
        # Normalised correlation of the z-projection with the (mean-subtracted)
        # logo.  The logo's contrast is only ~1%, so a plain MSE of the
        # projection is dominated by the overall mean amplitude and is a poor
        # detector (it picks coarse-grid artefacts, not the logo).  The
        # optimised velocity makes the projection *correlate* sharply with the
        # logo at exactly the loss time, so we detect that peak instead.
        proj = jnp.sum(evolve(t), axis=2)
        pz = proj - jnp.mean(proj)
        return float(jnp.sum(pz * tgt_zm)
                     / jnp.sqrt(jnp.sum(pz * pz) * jnp.sum(tgt_zm * tgt_zm)))

    # Fixed t_end bypasses the (expensive) scan — essential at high resolution
    # (256³) where each evolution is minutes, so a ~56-point scan is infeasible.
    # The loss time is crossing_time/2 ≈ 0.5/rms_v, which the inverse run already
    # reports; pass it with --t-end.
    if args.t_end > 0.0:
        t_end = args.t_end
        print(f"==> using fixed t_end={t_end:.4f} (scan skipped)")
        times = {"init": 0.0, "pre": args.f_pre * t_end, "target": t_end, "post": args.f_post * t_end}
        cubes = {}
        for name, t in times.items():
            cubes[name] = np.asarray(evolve(t), dtype=np.float32)
            print(f"  {name}: t={t:.4f}  rho[{cubes[name].min():.4g},{cubes[name].max():.4g}]")
        np.savez(args.out,
                 init=cubes["init"], pre=cubes["pre"], target=cubes["target"], post=cubes["post"],
                 t_init=0.0, t_pre=times["pre"], t_target=t_end, t_post=times["post"],
                 f_pre=args.f_pre, f_post=args.f_post,
                 target_image=np.asarray(target, dtype=np.float32), N=resolution)
        print(f"wrote {args.out}")
        return

    # Two-stage scan for the loss time.  The logo peak is narrow (FWHM ~1-2% of
    # t_end), so a single coarse grid can step right over it: first a coarse
    # correlation scan to bracket the peak, then a fine refine spanning +/- one
    # coarse step around it.
    coarse_ts = np.linspace(args.scan_lo * t_center, args.scan_hi * t_center, args.scan_n)
    coarse_c = [logo_corr(t) for t in coarse_ts]
    for t, cval in zip(coarse_ts, coarse_c):
        print(f"  coarse t={t:.4f}  corr={cval:+.4f}")
    t0 = float(coarse_ts[int(np.argmax(coarse_c))])
    dt = float(coarse_ts[1] - coarse_ts[0])
    fine_ts = np.linspace(t0 - dt, t0 + dt, args.refine_n)
    fine_c = [logo_corr(t) for t in fine_ts]
    for t, cval in zip(fine_ts, fine_c):
        print(f"  fine   t={t:.4f}  corr={cval:+.4f}")
    t_end = float(fine_ts[int(np.argmax(fine_c))])
    print(f"==> recovered t_end={t_end:.4f}  (corr={max(fine_c):.4f})")

    times = {"init": 0.0, "pre": args.f_pre * t_end, "target": t_end, "post": args.f_post * t_end}
    cubes = {}
    for name, t in times.items():
        dens = evolve(t)
        cubes[name] = np.asarray(dens, dtype=np.float32)
        print(f"  {name}: t={t:.4f}  rho[{cubes[name].min():.4g},{cubes[name].max():.4g}]")

    np.savez(args.out,
             init=cubes["init"], pre=cubes["pre"], target=cubes["target"], post=cubes["post"],
             t_init=0.0, t_pre=times["pre"], t_target=t_end, t_post=times["post"],
             f_pre=args.f_pre, f_post=args.f_post,
             target_image=np.asarray(target, dtype=np.float32), N=resolution)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
