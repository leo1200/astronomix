"""Reproduce the Evrard-collapse crash baseline for the conservative FD
self-gravity schemes at low resolution."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
from astronomix.option_classes.simulation_config import (
    SIMPLE_SOURCE, SECOND_ORDER_CONSERVATIVE, FOURTH_ORDER_CONSERVATIVE,
)
from _collapse_lib import run_collapse, diagnose

SCHEMES = {
    "simple": SIMPLE_SOURCE,
    "flux": SECOND_ORDER_CONSERVATIVE,
    "weno": FOURTH_ORDER_CONSERVATIVE,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--schemes", nargs="+", default=["simple", "weno"])
    ap.add_argument("--t_end", type=float, default=1.2)
    ap.add_argument("--e0", type=float, default=0.05)
    args = ap.parse_args()

    print(f"Evrard collapse  e0={args.e0}  t_end={args.t_end}")
    print(f"{'scheme':>8} {'N':>4} {'crashed':>8} {'crash_t':>8} "
          f"{'t_final':>8} {'nfin':>6} {'max_relE':>10} {'final_relE':>11}")
    for scheme in args.schemes:
        for N in args.N:
            snaps, _, _ = run_collapse(N, SCHEMES[scheme], t_end=args.t_end,
                                       initial_energy=args.e0)
            d = diagnose(snaps)
            ct = f"{d['crash_t']:.3f}" if d["crash_t"] is not None else "-"
            print(f"{scheme:>8} {N:>4} {str(d['crashed']):>8} {ct:>8} "
                  f"{d['t_final']:>8.3f} {d['n_finite']:>3}/{d['n_total']:<2} "
                  f"{d['max_rel_err']:>10.2e} {d['final_rel_err']:>11.2e}",
                  flush=True)


if __name__ == "__main__":
    main()
