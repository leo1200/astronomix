"""Sweep gravity scheme / positivity options on the cold Evrard collapse.

For each (scheme, N) combination report whether it stayed finite, when it
crashed, and the energy-conservation error. The positivity-preserving flux
limiter (--pp_flux) is the reconstruction-level cure; --stage_mode picks a
per-stage positivity mode for comparison.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
# --double must enable x64 BEFORE jax is first imported
if "--double" in sys.argv:
    import jax
    jax.config.update("jax_enable_x64", True)

from astronomix.option_classes.simulation_config import (
    SIMPLE_SOURCE, SECOND_ORDER_CONSERVATIVE, FOURTH_ORDER_CONSERVATIVE,
    POSITIVITY_NONE, POSITIVITY_HARD_FLOOR, POSITIVITY_REDISTRIBUTE,
    POSITIVITY_CONSERVATIVE, NATIVE_JAX, PALLAS,
)
from _collapse_lib import run_collapse, diagnose

SCHEMES = {
    "simple": SIMPLE_SOURCE,
    "second": SECOND_ORDER_CONSERVATIVE,
    "fourth": FOURTH_ORDER_CONSERVATIVE,
}

POS = {"none": POSITIVITY_NONE, "floor": POSITIVITY_HARD_FLOOR,
       "redist": POSITIVITY_REDISTRIBUTE, "cons": POSITIVITY_CONSERVATIVE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[64])
    ap.add_argument("--scheme", default="fourth", choices=list(SCHEMES))
    ap.add_argument("--t_end", type=float, default=1.2)
    ap.add_argument("--e0", type=float, default=0.05)
    ap.add_argument("--pp_flux", type=int, default=0,
                    help="positivity-preserving WENO flux limiter (the fix)")
    ap.add_argument("--stage_mode", choices=list(POS), default="none")
    ap.add_argument("--protect", type=int, default=0,
                    help="default_positivity_protection (read-only clamps + floors)")
    ap.add_argument("--double", action="store_true",
                    help="run in float64 on the native backend (x64 set at import)")
    args = ap.parse_args()
    backend = NATIVE_JAX if args.double else PALLAS

    print(f"Evrard collapse  scheme={args.scheme}  e0={args.e0}  t_end={args.t_end}  "
          f"pp_flux={args.pp_flux}  stage_mode={args.stage_mode}  protect={args.protect}")
    print(f"{'N':>4} {'crashed':>8} {'crash_t':>8} {'t_final':>8} {'nfin':>7} "
          f"{'max_relE':>10} {'final_relE':>11}")
    for N in args.N:
        snaps, _, _ = run_collapse(
            N, SCHEMES[args.scheme], t_end=args.t_end, initial_energy=args.e0,
            backend=backend, pp_flux=bool(args.pp_flux),
            protect=bool(args.protect), per_stage_mode=POS[args.stage_mode],
        )
        d = diagnose(snaps)
        ct = f"{d['crash_t']:.3f}" if d["crash_t"] is not None else "-"
        print(f"{N:>4} {str(d['crashed']):>8} {ct:>8} {d['t_final']:>8.3f} "
              f"{d['n_finite']:>3}/{d['n_total']:<3} "
              f"{d['max_rel_err']:>10.2e} {d['final_rel_err']:>11.2e}", flush=True)


if __name__ == "__main__":
    main()
