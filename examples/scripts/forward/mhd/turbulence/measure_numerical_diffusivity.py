"""Measure a scheme's numerical resistivity and viscosity directly, shell by shell.

Neither code has an explicit diffusivity, so the usual move is to infer one from
a dissipation scale. This measures it instead, with no convention and no fit
band, by removing everything except the dissipation:

* **resistivity** -- a random solenoidal magnetic field of tiny amplitude in a
  motionless box. With ``v = 0`` the induction equation has zero right-hand side
  analytically, and the field is weak enough (``|B| ~ 1e-6``) that the Lorentz
  force cannot start any motion within the run, so *every* change in ``E_B(n)``
  is truncation error. Then ``eta(n) = -d ln E_B(n)/dt / (2 k_n^2)``.
* **viscosity** -- the same trick on the velocity: a random solenoidal ``v`` of
  amplitude ``1e-4`` with no field and no driving, so the nonlinear term is
  ``1e-8`` and the decay of ``E_v(n)`` is again pure truncation error, giving
  ``nu(n) = -d ln E_v(n)/dt / (2 k_n^2)``.

What comes out is not one number but a *function of scale*, which is the point:
a p-th order scheme has ``eta(k) ~ |lambda| dx (k dx)^(p-1)``, so raising the
order does not lower the dissipation uniformly -- it pushes it towards the grid
scale and leaves the resolved range far cleaner.

Two things this measurement establishes and one caveat it carries:

* The measurable range is narrow *because* the scheme is high order. Only the
  top ~1.5 octaves decay at all on any affordable timescale; below that the
  dissipation is too small to measure by decay. For a 2nd-order scheme the
  measurable range would extend much further down, which is the same statement
  as "it dissipates in the resolved range".
* **WENO's dissipation is solution-dependent, by orders of magnitude.** Fitting
  the same run over ``t < 0.5`` gives ``eta ~ 2e-6`` at the Nyquist shell and
  over ``t < 30`` gives ``4e-10``: the white-noise initial field is maximally
  non-smooth, the smoothness indicators respond with heavy dissipation, and once
  the field has been smoothed the weights relax to the linear optimal ones and
  the dissipation collapses. So there is no single ``eta`` even at fixed ``k`` --
  what is quoted here is the late-time, smooth-field (linear) value.
* Consequently this number is a *lower bound* on what a scheme does to real
  turbulence, which is rough and intermittent. The effective diffusivities in
  `make_reynolds_figure.py`, measured on the actual turbulent state, are the ones
  to compare between codes; this measurement explains their k-dependence.

    python measure_numerical_diffusivity.py --n 128
    python measure_numerical_diffusivity.py --n 128 --field velocity
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
from pathlib import Path

# numerics
import numpy as np
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import (
    CARTESIAN, FINITE_DIFFERENCE, ISOTHERMAL,
)

# astronomix containers
from astronomix import (
    BoundarySettings, BoundarySettings1D, PositivityConfig, SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state, finalize_config, get_registered_variables,
    initialize_interface_fields, time_integration,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_spectral import shell_numbers, shell_spectrum

HERE = Path(__file__).resolve().parent
BOX_SIZE = 1.0
RHO0 = 1.0
CS = 2.0                      # the sound speed of the dynamo study


# -------------------------------------------------------------
# ============ ↓ A flat-spectrum solenoidal field ↓ ===========
# -------------------------------------------------------------
def solenoidal_white_field(key, n_cells, amplitude, shell=0):
    """Random divergence-free vector field, flat in shell energy.

    ``shell > 0`` band-limits the field to that single shell. That is the
    measurement that means something: a single-shell field is *smooth*, so the
    WENO smoothness indicators sit at their linear optimal weights and the decay
    rate is the scheme's linear numerical dissipation at that ``k`` -- the
    quantity the modified-equation analysis predicts, and the one a decaying-wave
    test (Rembiasz et al.) measures. The broadband version is maximally rough and
    measures something else entirely (see the module docstring).
    """
    k1 = 2.0 * jnp.pi * jnp.fft.fftfreq(n_cells, d=BOX_SIZE / n_cells)
    kx, ky, kz = k1[:, None, None], k1[None, :, None], k1[None, None, :]
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k = jnp.sqrt(k2)

    key, s1, s2 = jax.random.split(key, 3)
    noise = (jax.random.normal(s1, (3, n_cells, n_cells, n_cells))
             + 1j * jax.random.normal(s2, (3, n_cells, n_cells, n_cells)))
    # Shell energy ~ k^2 |f_k|^2, so |f_k| ~ 1/k gives a flat E(n).
    k_safe = jnp.where(k > 0, k, 1.0)
    f = noise / k_safe
    if shell > 0:
        band = jnp.abs(jnp.rint(k / (2.0 * jnp.pi)) - shell) < 0.5
        f = jnp.where(band[None], f, 0.0)
    f = f.at[:, 0, 0, 0].set(0.0)

    # Project out the compressible part: exactly divergence-free by construction.
    div = (kx * f[0] + ky * f[1] + kz * f[2]) / jnp.where(k2 > 0, k2, 1.0)
    f = jnp.stack([f[0] - kx * div, f[1] - ky * div, f[2] - kz * div])

    real = jnp.stack([jnp.real(jnp.fft.ifftn(c)) for c in f])
    rms = jnp.sqrt(jnp.mean(jnp.sum(real ** 2, axis=0)))
    return key, real * (amplitude / rms)
# -------------------------------------------------------------
# ============ ↑ A flat-spectrum solenoidal field ↑ ===========
# -------------------------------------------------------------


def build_config(n_cells, num_snapshots, seed):
    periodic = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    )
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, dimensionality=3, geometry=CARTESIAN,
        mhd=True, equation_of_state=ISOTHERMAL, box_size=BOX_SIZE,
        num_cells=n_cells, boundary_settings=periodic,
        positivity_config=PositivityConfig(),
        return_snapshots=False, activate_snapshot_callback=True,
        # Linear snapshot cadence. Log spacing would sample every shell's decay
        # equally well, but `use_specific_snapshot_timepoints` clips the step to
        # land on each snapshot exactly and the first log intervals are short
        # enough to collapse dt to zero.
        num_snapshots=num_snapshots, random_seed=seed, progress_bar=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--field", choices=("magnetic", "velocity"), default="magnetic")
    p.add_argument("--t-end", type=float, default=-1.0,
                   help="<0 auto-scales as 30 * (64 / N): at fixed n/N the decay "
                        "rate grows with N, so a coarser grid needs longer")
    p.add_argument("--nsnap", type=int, default=60)
    p.add_argument("--shell", type=int, default=0,
                   help="band-limit the initial field to this shell. >0 is the "
                        "measurement to trust: a single-shell field is smooth, "
                        "so what decays is the scheme's LINEAR dissipation at "
                        "that k. 0 = broadband (rough field, see the docstring)")
    p.add_argument("--amplitude", type=float, default=-1.0,
                   help="<0 uses 1e-6 for B and 1e-4 for v: small enough that "
                        "the Lorentz force / the nonlinear term cannot act")
    p.add_argument("--cfl", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--outdir", type=str, default=str(HERE / "data" / "diffusivity"))
    args = p.parse_args()

    amp = args.amplitude if args.amplitude > 0 else (
        1e-6 if args.field == "magnetic" else 1e-4)
    # At fixed n/N the decay rate grows with N, so a coarser grid needs longer.
    t_end = args.t_end if args.t_end > 0 else 30.0 * (64.0 / args.n)
    config = build_config(args.n, args.nsnap, args.seed)
    rv = get_registered_variables(config)
    params = SimulationParams(C_cfl=args.cfl, isothermal_sound_speed=CS,
                              t_end=t_end, minimum_density=1e-4,
                              minimum_pressure=1e-6)

    shape = (args.n,) * 3
    zeros = jnp.zeros(shape)
    key, field = solenoidal_white_field(jax.random.PRNGKey(args.seed), args.n, amp,
                                        shell=args.shell)
    if args.field == "magnetic":
        Bx, By, Bz = field
        vx, vy, vz = zeros, zeros, zeros
    else:
        vx, vy, vz = field
        Bx, By, Bz = zeros, zeros, zeros
    bxf, byf, bzf = initialize_interface_fields(Bx, By, Bz)
    state = construct_primitive_state(
        config=config, registered_variables=rv, density=jnp.full(shape, RHO0),
        velocity_x=vx, velocity_y=vy, velocity_z=vz,
        magnetic_field_x=Bx, magnetic_field_y=By, magnetic_field_z=Bz,
        interface_magnetic_field_x=bxf, interface_magnetic_field_y=byf,
        interface_magnetic_field_z=bzf,
    )
    config = finalize_config(config, state.shape)

    di = rv.density_index
    vi = (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)
    bi = (rv.magnetic_index.x, rv.magnetic_index.y, rv.magnetic_index.z)
    idx = bi if args.field == "magnetic" else vi
    other = vi if args.field == "magnetic" else bi
    records = {"t": [], "E": [], "leak": []}

    def cb(time, st, registered_variables):
        fx, fy, fz = (st[i] for i in idx)
        E = shell_spectrum(fx, fy, fz)
        # How much has leaked into the field that was supposed to stay zero:
        # the check that this really is a passive decay and not dynamics.
        leak = jnp.sqrt(jnp.mean(sum(st[i] ** 2 for i in other)))

        def host(t, e, lk):
            records["t"].append(float(t))
            records["E"].append(np.asarray(e))
            records["leak"].append(float(lk))
        jax.debug.callback(host, time, E, leak)

    print(f"[diffusivity N={args.n} {args.field}] amplitude={amp:g} "
          f"t_end={t_end:.4g} in {args.nsnap} snapshots", flush=True)
    jax.block_until_ready(time_integration(state, config, params, rv, cb))
    jax.effects_barrier()

    t = np.asarray(records["t"])
    order = np.argsort(t)
    t, E = t[order], np.asarray(records["E"])[order]
    leak = np.asarray(records["leak"])[order]
    n = shell_numbers(args.n).astype(float)

    # eta(n) from the per-shell decay: ln E(n, t) = ln E(n, 0) - 2 eta k^2 t.
    k = 2.0 * np.pi * n / BOX_SIZE
    # Fit each shell only where it is actually decaying: below a 10% drop the
    # signal is round-off, below 1e-4 of the initial energy the shell has been
    # wiped out and what is left is leakage from its neighbours.
    diff = np.full_like(n, np.nan)
    n_fit = np.zeros_like(n, dtype=int)
    for j in range(1, len(n)):
        y = E[:, j]
        if y[0] <= 0:
            continue
        frac = y / y[0]
        good = (frac <= 0.9) & (frac >= 1e-4) & (y > 0)
        if good.sum() >= 4:
            slope = np.polyfit(t[good], np.log(y[good]), 1)[0]
            diff[j] = -slope / (2.0 * k[j] ** 2)
            n_fit[j] = int(good.sum())

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_shell{args.shell}" if args.shell > 0 else ""
    path = out / f"astronomix_{args.field}_N{args.n}{tag}.npz"
    np.savez_compressed(path, N=args.n, field=args.field, amplitude=amp,
                        shell=args.shell,
                        times=t, n_shell=n, E_shell=E, diffusivity=diff,
                        n_fit=n_fit, leak=leak, t_end=t_end, cs=CS,
                        box=BOX_SIZE, cfl=args.cfl)
    measured = np.isfinite(diff)
    print(f"[diffusivity N={args.n} {args.field}] leak {leak[0]:.2e} -> {leak[-1]:.2e} "
          f"(must stay far below the amplitude for this to be a passive decay)")
    print(f"[diffusivity N={args.n} {args.field}] measurable on {measured.sum()} of "
          f"{len(n) - 1} shells; wrote {path}", flush=True)
    if measured.sum():
        for j in np.where(measured)[0][::max(1, measured.sum() // 8)]:
            print(f"    n={n[j]:5.0f}  n/N={n[j] / args.n:5.3f}  "
                  f"diffusivity={diff[j]:.3e}")


if __name__ == "__main__":
    main()
