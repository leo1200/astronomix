"""Diagnose WHERE/HOW the conservative gravity scheme crashes: per-snapshot
min pressure, max |v|, max density, and the location (core vs void) of the
pressure minimum, for the conservative (weno) vs simple scheme."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import argparse
import numpy as np
import jax.numpy as jnp
from astronomix.option_classes.simulation_config import (
    SIMPLE_SOURCE, FOURTH_ORDER_CONSERVATIVE,
)
from _collapse_lib import run_collapse, GAMMA

SCHEMES = {"simple": SIMPLE_SOURCE, "weno": FOURTH_ORDER_CONSERVATIVE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="weno")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--t_end", type=float, default=1.2)
    ap.add_argument("--e0", type=float, default=0.05)
    ap.add_argument("--pp_flux", type=int, default=0)
    args = ap.parse_args()

    snaps, helper, rv = run_collapse(
        args.N, SCHEMES[args.scheme], t_end=args.t_end, initial_energy=args.e0,
        pp_flux=bool(args.pp_flux), want_states=True,
    )
    states = np.asarray(snaps.states)
    t = np.asarray(snaps.time_points)
    r = np.asarray(helper.r)  # radius field (padded shape == state field)
    di, pi = rv.density_index, rv.pressure_index
    vx, vy, vz = rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z

    print(f"scheme={args.scheme} N={args.N} e0={args.e0} adapt={args.adapt}")
    print(f"{'snap':>4} {'t':>7} {'finite':>6} {'rho_max':>10} {'p_min':>11} "
          f"{'|v|_max':>10} {'r@pmin':>8} {'rho@pmin':>10} {'r@vmax':>8} {'rho@vmax':>10}")
    for s in range(states.shape[0]):
        st = states[s]
        rho = st[di]; p = st[pi]
        finite = bool(np.all(np.isfinite(st)))
        # dead/unfilled snapshot buffer (run already crashed): everything zero
        if finite and float(np.max(np.abs(st))) == 0.0:
            print(f"{s:>4} {t[s]:>7.3f}  <dead/unfilled buffer; crash was earlier>")
            break
        speed = np.sqrt(st[vx]**2 + st[vy]**2 + st[vz]**2)
        pmin = np.nanmin(p)
        pidx = np.unravel_index(
            np.nanargmin(np.where(np.isfinite(p), p, np.inf)), p.shape)
        vidx = np.unravel_index(
            np.nanargmax(np.where(np.isfinite(speed), speed, -np.inf)), speed.shape)
        print(f"{s:>4} {t[s]:>7.3f} {str(finite):>6} {np.nanmax(rho):>10.3f} "
              f"{pmin:>11.3e} {np.nanmax(speed):>10.2f} {r[pidx]:>8.3f} "
              f"{rho[pidx]:>10.3e} {r[vidx]:>8.3f} {rho[vidx]:>10.3e}", flush=True)
        if not finite:
            for name, arr in [("rho", rho), ("p", p), ("vx", st[vx]),
                              ("vy", st[vy]), ("vz", st[vz])]:
                nbad = int(np.sum(~np.isfinite(arr)))
                if nbad:
                    print(f"      NaN/Inf in {name}: {nbad} cells", flush=True)
            break


if __name__ == "__main__":
    main()
