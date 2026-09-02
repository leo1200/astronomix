"""Figures and convergence metrics for the astronomix / AthenaK spectral study.

Reads the ``spectra.npz`` written by ``spectra.py`` and produces

    figures/turbulence_spectra.png      compensated spectra, both codes, all N
    figures/turbulence_convergence.png  effective resolution + self-convergence
    (and prints the metric table that the two figures visualise)

Two things are being measured:

  *Effective resolution* -- the shell number ``n_1/2`` at which the compensated
  spectrum ``n^(5/3) E(n)`` has fallen to half its plateau value. Everything
  above ``n_1/2`` is numerical dissipation rather than physics, so ``n_1/2 / N``
  is the fraction of the grid a scheme actually turns into resolved turbulence.

  *Self-convergence* -- the relative L2 difference between a run and the next
  resolution up of the same code, over the shells they share. A scheme whose
  spectrum is converging shows this shrinking with N.

    python examples/scripts/forward/hydro/turbulence/make_figures.py
"""

# general
import argparse
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

ASTRO_COLOR = "#1f77b4"
ATHENA_COLOR = "#d62728"
HLLE_COLOR = "#ff9e4a"

SERIES = {
    "astronomix": (ASTRO_COLOR, "astronomix WENO5 (5th order)"),
    "athenak": (ATHENA_COLOR, "AthenaK PLM+Roe (2nd order)"),
    "athenak_hlle": (HLLE_COLOR, "AthenaK PLM+HLLE (control)"),
}
#: One line style per resolution, shared by both codes.
STYLE = {64: dict(ls=":", lw=1.6, alpha=0.75),
         128: dict(ls="--", lw=1.9, alpha=0.9),
         256: dict(ls="-", lw=2.3, alpha=1.0),
         512: dict(ls="-", lw=2.8, alpha=1.0)}


def classify(key):
    """Series label for a run key.

    The Riemann-solver control run shares its code and resolution with a
    production run, so it has to be its own series or it would overwrite it.
    """
    if key.startswith("astronomix"):
        return "astronomix"
    if key.endswith("_hlle"):
        return "athenak_hlle"
    return "athenak"


def load(path):
    """Unflatten the ``key|field`` layout that spectra.py writes."""
    raw = np.load(path, allow_pickle=True)
    runs = {}
    for key in [str(k) for k in raw["run_keys"]]:
        runs[key] = {f.split("|", 1)[1]: raw[f]
                     for f in raw.files if f.startswith(key + "|")}
        runs[key]["code"] = classify(key)
        runs[key]["n"] = int(runs[key]["n"])
    return runs, bool(raw["weighted"])


# -------------------------------------------------------------
# ============ ↓ Convergence metrics ↓ ========================
# -------------------------------------------------------------
def compensated(run):
    """``n^(5/3) E(n)`` on the shells that carry meaningful statistics.

    Shell 0 is the (removed) mean flow and the first shells are the driving
    band itself; the Nyquist shell is truncated by the FFT grid, so the useful
    range starts above the driving band and stops at ``N/2``.
    """
    n = np.asarray(run["n_shell"], dtype=float)
    E = np.asarray(run["E_mean"], dtype=float)
    good = n >= 1
    return n[good], (n[good] ** (5.0 / 3.0)) * E[good]


#: Reference band for the inertial-range level: just above the driving band
#: (n <= 2) and well inside the resolved range even at N = 64, so the reference
#: is the same physical scales for every run in the ladder.
REF_BAND = (3, 6)


def plateau_level(run, band=REF_BAND):
    """Inertial-range level of the compensated spectrum, averaged over ``band``.

    Anchoring to a *fixed* shell band rather than to the maximum keeps the
    reference comparable across codes and resolutions: a high-order scheme can
    develop a bottleneck bump near its dissipation scale, and taking the maximum
    would silently move the reference onto that bump.
    """
    n, C = compensated(run)
    sel = (n >= band[0]) & (n <= band[1])
    return float(C[sel].mean())


def threshold_shell(run, frac=0.5, band=REF_BAND):
    """Shell where the compensated spectrum falls to ``frac`` of the plateau.

    ``frac = 0.5`` marks the onset of the dissipation range; the schemes are only
    ~1.5x apart there because both spectra are still rolling over gently. The
    deeper ``frac = 0.25`` sits where the curves have genuinely separated, so
    both are reported.

    Takes the *last* crossing, so a bottleneck bump that lifts the spectrum back
    above the threshold does not truncate the estimate early. Log-log
    interpolation between the bracketing shells gives a sub-shell value.
    """
    n, C = compensated(run)
    target = frac * plateau_level(run, band)
    sel = n >= band[0]
    n, C = n[sel], C[sel]
    above = np.nonzero(C >= target)[0]
    if len(above) == 0:
        return float(n[0])
    j = int(above[-1])
    if j + 1 >= len(n):
        return float(n[-1])           # never falls to the threshold on this grid
    n0, n1 = np.log(n[j]), np.log(n[j + 1])
    c0, c1 = np.log(C[j]), np.log(C[j + 1])
    return float(np.exp(n0 + (np.log(target) - c0) * (n1 - n0) / (c1 - c0)))


def half_power_shell(run, band=REF_BAND):
    return threshold_shell(run, 0.5, band)


def self_convergence(run_lo, run_hi, n_lo=3):
    """Relative L2 difference in spectrum *shape* between two resolutions.

    Each spectrum is divided by its own plateau level first. Without that
    normalisation the metric is dominated by a large-scale amplitude offset
    rather than by convergence: AthenaK seeds its driver identically at every
    resolution (``rstate.idum = -1``, not settable from the input file) while
    astronomix draws a fresh forcing realisation per grid, so the raw difference
    charges astronomix for realisation scatter that AthenaK does not pay and
    makes the 2nd-order code look better converged than it is.

    Compared over the shells the two runs share, from just above the driving
    band up to the Nyquist shell of the coarser run.
    """
    n_a, E_a = np.asarray(run_lo["n_shell"]), np.asarray(run_lo["E_mean"])
    n_b, E_b = np.asarray(run_hi["n_shell"]), np.asarray(run_hi["E_mean"])
    n_max = min(n_a.max(), n_b.max())
    sel_a = (n_a >= n_lo) & (n_a <= n_max)
    sel_b = (n_b >= n_lo) & (n_b <= n_max)
    a = E_a[sel_a] / plateau_level(run_lo)
    b = E_b[sel_b] / plateau_level(run_hi)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
# -------------------------------------------------------------
# ============ ↑ Convergence metrics ↑ ========================
# -------------------------------------------------------------


def ratio_figure(runs, out):
    """Per-shell power ratio at equal N, and astronomix N vs AthenaK 2N.

    The ratio panel is the least-processed statement of the result: no threshold,
    no fitted plateau, just how much more power the 5th-order scheme retains at
    each scale. The right panel asks the practical question — does doubling the
    2nd-order code's grid buy back the difference?
    """
    fig, (ax_ratio, ax_equiv) = plt.subplots(1, 2, figsize=(13, 5.0))

    by = {}
    for r in runs.values():
        by.setdefault(r["code"], {})[r["n"]] = r
    astro, athena = by.get("astronomix", {}), by.get("athenak", {})

    for N in sorted(set(astro) & set(athena)):
        a, k = astro[N], athena[N]
        n = np.asarray(a["n_shell"], dtype=float)
        Ea, Ek = np.asarray(a["E_mean"]), np.asarray(k["E_mean"])
        # Propagate the snapshot-to-snapshot standard errors into the ratio, so
        # it is visible that the separation is far outside the noise.
        ea, ek = np.asarray(a["E_err"]), np.asarray(k["E_err"])
        sel = (n >= 1) & (Ek > 0)
        ratio = Ea[sel] / Ek[sel]
        rel = ratio * np.sqrt((ea[sel] / np.maximum(Ea[sel], 1e-300)) ** 2
                              + (ek[sel] / np.maximum(Ek[sel], 1e-300)) ** 2)
        style = STYLE.get(N, dict(ls="-", lw=2.0))
        ax_ratio.plot(n[sel], ratio, color=ASTRO_COLOR, label=f"$N={N}$", **style)
        ax_ratio.fill_between(n[sel], ratio - rel, ratio + rel,
                              color=ASTRO_COLOR, alpha=0.2, lw=0)

    ax_ratio.axhline(1.0, color="k", lw=1.0, ls=":")
    ax_ratio.set_xscale("log")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlabel(r"shell number $n = kL/2\pi$")
    ax_ratio.set_ylabel(r"$E_{\rm astronomix}(n) \, / \, E_{\rm AthenaK}(n)$")
    ax_ratio.set_title("retained power at equal resolution")
    ax_ratio.legend(fontsize=8)
    ax_ratio.grid(alpha=0.25, which="both")

    for N in sorted(astro):
        if 2 * N not in athena:
            continue
        a, k = astro[N], athena[2 * N]
        for r, color, lab in ((a, ASTRO_COLOR, f"astronomix {N}$^3$"),
                              (k, ATHENA_COLOR, f"AthenaK {2 * N}$^3$")):
            nc, C = compensated(r)
            ax_equiv.loglog(nc, C, color=color, label=lab, **STYLE.get(r["n"],
                                                                       dict(ls="-", lw=2.0)))
    ax_equiv.set_xlabel(r"shell number $n = kL/2\pi$")
    ax_equiv.set_ylabel(r"$n^{5/3} E(n)$")
    ax_equiv.set_title("does doubling the 2nd-order grid catch up?")
    ax_equiv.legend(fontsize=8)
    ax_equiv.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[figures] wrote {out}")


def spectra_figure(runs, weighted, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax_raw, ax_comp = axes

    short = {"astronomix": "astronomix WENO5", "athenak": "AthenaK PLM+Roe",
             "athenak_hlle": "AthenaK PLM+HLLE"}
    for key in sorted(runs, key=lambda k: (runs[k]["code"], runs[k]["n"])):
        r = runs[key]
        color = SERIES[r["code"]][0]
        style = STYLE.get(r["n"], dict(ls="-", lw=2.0))
        n = np.asarray(r["n_shell"], dtype=float)
        E = np.asarray(r["E_mean"], dtype=float)
        sel = n >= 1
        label = f"{short[r['code']]} {r['n']}$^3$"
        ax_raw.loglog(n[sel], E[sel], color=color, label=label, **style)
        nc, C = compensated(r)
        ax_comp.loglog(nc, C, color=color, label=label, **style)

    n_ref = np.array([3.0, 30.0])
    ax_raw.loglog(n_ref, 3e-4 * (n_ref / 3.0) ** (-5.0 / 3.0), color="k",
                  lw=1.0, ls="-.", label=r"$k^{-5/3}$")
    ax_raw.set_xlabel(r"shell number $n = kL/2\pi$")
    ax_raw.set_ylabel(r"$E(n)$" + (r"  ($\sqrt{\rho}\,v$)" if weighted else r"  ($v$)"))
    ax_raw.set_title("kinetic energy spectrum")
    ax_raw.legend(fontsize=8, ncol=2)
    ax_raw.grid(alpha=0.25, which="both")

    ax_comp.set_xlabel(r"shell number $n = kL/2\pi$")
    ax_comp.set_ylabel(r"$n^{5/3} E(n)$")
    ax_comp.set_title("compensated — flat where the cascade is resolved")
    ax_comp.grid(alpha=0.25, which="both")

    fig.suptitle("Driven subsonic isothermal turbulence — same driving band, "
                 "same $\\mathcal{M} \\approx 0.32$ in both codes", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[figures] wrote {out}")


def convergence_figure(runs, metrics, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_eff, ax_conv = axes

    for code, (color, name) in SERIES.items():
        pts = sorted((m["n"], m["n_half"]) for m in metrics if m["code"] == code)
        if pts:
            Ns, halves = zip(*pts)
            # The slope is the headline number: an exponent of 1 means the scheme
            # resolves a fixed fraction of the grid however far you refine, while
            # anything below 1 means that fraction shrinks as you refine.
            if len(Ns) >= 2:
                slope = np.polyfit(np.log(Ns), np.log(halves), 1)[0]
                name = f"{name}  ($\\propto N^{{{slope:.2f}}}$)"
            ax_eff.plot(Ns, halves, "o-", color=color, label=name)
        conv = sorted((m["n"], m["self_conv"]) for m in metrics
                      if m["code"] == code and m["self_conv"] is not None)
        if conv:
            Ns, cs = zip(*conv)
            ax_conv.plot(Ns, cs, "o-", color=color, label=name)

    # Reference: a scheme resolving a fixed fraction of the grid scales as N.
    all_n = sorted({m["n"] for m in metrics})
    if all_n:
        ax_eff.plot(all_n, [n / 4.0 for n in all_n], "k:", lw=1.0, label=r"$N/4$")
    ax_eff.set_xscale("log", base=2)
    ax_eff.set_yscale("log", base=2)
    ax_eff.set_xlabel("cells per dimension $N$")
    ax_eff.set_ylabel(r"$n_{1/2}$  (half-plateau shell)")
    ax_eff.set_title("effective resolution")
    ax_eff.legend(fontsize=8)
    ax_eff.grid(alpha=0.25, which="both")

    ax_conv.set_xscale("log", base=2)
    ax_conv.set_yscale("log")
    ax_conv.set_xlabel("cells per dimension $N$")
    ax_conv.set_ylabel(r"$\|E_N - E_{2N}\| \, / \, \|E_{2N}\|$")
    ax_conv.set_title("self-convergence of the spectrum")
    ax_conv.legend(fontsize=8)
    ax_conv.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[figures] wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spectra", type=str, default="spectra.npz")
    args = p.parse_args()

    runs, weighted = load(DATA_DIR / args.spectra)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    by_code = {}
    for key, r in runs.items():
        by_code.setdefault(r["code"], {})[r["n"]] = r

    metrics = []
    for code, per_n in by_code.items():
        for N, r in sorted(per_n.items()):
            hi = per_n.get(2 * N)
            metrics.append(dict(
                code=code, n=N,
                mach=float(r["mach"]),
                n_half=threshold_shell(r, 0.5),
                n_quarter=threshold_shell(r, 0.25),
                self_conv=self_convergence(r, hi) if hi is not None else None,
                runtime=float(r["runtime"]),
                n_avg=int(r["n_avg"]),
            ))

    print(f"\n{'code':<13} {'N':>5} {'Mach':>7} {'n_1/2':>7} {'n_1/4':>7} "
          f"{'n_1/4/N':>8} {'||E_N-E_2N||':>13} {'runtime[s]':>11} {'snaps':>6}")
    for m in sorted(metrics, key=lambda m: (m["code"], m["n"])):
        conv = f"{m['self_conv']:.4f}" if m["self_conv"] is not None else "-"
        print(f"{m['code']:<13} {m['n']:>5} {m['mach']:>7.4f} {m['n_half']:>7.2f} "
              f"{m['n_quarter']:>7.2f} {m['n_quarter'] / m['n']:>8.4f} {conv:>13} "
              f"{m['runtime']:>11.1f} {m['n_avg']:>6}")

    # The headline number: at equal N, how much further into the cascade does
    # the 5th-order scheme reach, and what grid would the 2nd-order one need?
    astro = {m["n"]: m["n_quarter"] for m in metrics if m["code"] == "astronomix"}
    athena = {m["n"]: m["n_quarter"] for m in metrics if m["code"] == "athenak"}
    shared = sorted(set(astro) & set(athena))
    if shared:
        print("\neffective-resolution ratio (astronomix / AthenaK) at equal N, "
              "at the quarter-plateau shell:")
        for N in shared:
            print(f"  N={N:<5d}  n_1/4: {astro[N]:6.2f} vs {athena[N]:6.2f}  "
                  f"-> {astro[N] / athena[N]:.2f}x")

        print("\npeak power ratio E_astronomix/E_AthenaK at equal N:")
        for N in shared:
            a = runs[f"astronomix_n{N}"]; k = runs[f"athenak_n{N}"]
            n = np.asarray(a["n_shell"], dtype=float)
            Ea, Ek = np.asarray(a["E_mean"]), np.asarray(k["E_mean"])
            sel = (n >= 3) & (n <= N / 2) & (Ek > 0)
            ratio = Ea[sel] / Ek[sel]
            j = int(np.argmax(ratio))
            print(f"  N={N:<5d}  max {ratio[j]:.2f}x at n={int(n[sel][j])}"
                  f"   (ratio at n=N/8: {Ea[N // 8] / Ek[N // 8]:.2f}x)")

        # How much grid would the 2nd-order scheme need to reach the 5th-order
        # scheme's dissipation scale? Interpolate n_1/2(N) as a power law
        # through the AthenaK ladder and invert it.
        Ns = np.array(sorted(athena), dtype=float)
        if len(Ns) >= 2:
            hs = np.array([athena[int(N)] for N in Ns])
            slope, intercept = np.polyfit(np.log(Ns), np.log(hs), 1)
            print(f"\nAthenaK n_1/2 scales as N^{slope:.2f}; "
                  "equivalent grid to match astronomix:")
            for N in sorted(astro):
                need = float(np.exp((np.log(astro[N]) - intercept) / slope))
                print(f"  astronomix {N}^3 (n_1/2={astro[N]:.1f})  ~  "
                      f"AthenaK {need:.0f}^3   ({need / N:.2f}x the linear grid, "
                      f"{(need / N) ** 3:.1f}x the cells)")

    print("\ncost at equal N (wall clock on one A100, 10 turnovers):")
    rt = {(m["code"], m["n"]): m["runtime"] for m in metrics}
    for N in shared:
        a, b = rt[("astronomix", N)], rt[("athenak", N)]
        print(f"  N={N:<5d}  astronomix {a:8.1f} s   AthenaK {b:8.1f} s   "
              f"-> {a / b:.1f}x")

    # The question a user actually faces: for a fixed wall-clock budget, which
    # scheme resolves more of the cascade? Extrapolate AthenaK's cost to the grid
    # it needs to match astronomix's dissipation scale, using its own measured
    # runtime scaling rather than the ideal N^4.
    if len(shared) >= 2:
        Nk = np.array(sorted(athena), dtype=float)
        hk = np.array([athena[int(N)] for N in Nk])
        s_res, i_res = np.polyfit(np.log(Nk), np.log(hk), 1)
        tk = np.array([rt[("athenak", int(N))] for N in Nk])
        s_cost, i_cost = np.polyfit(np.log(Nk), np.log(tk), 1)
        print(f"\nequal-QUALITY cost (AthenaK runtime scales as N^{s_cost:.2f} "
              f"as measured, not the ideal N^4):")
        if len(Nk) < 3:
            print("  WARNING: cost exponent fitted from only "
                  f"{len(Nk)} resolutions, and the smallest grids underutilise the "
                  "GPU, so the exponent is biased low and the extrapolated AthenaK "
                  "cost is an UNDERestimate. Treat as indicative until N=256 lands.")
        for N in sorted(astro):
            need = float(np.exp((np.log(astro[N]) - i_res) / s_res))
            cost = float(np.exp(i_cost + s_cost * np.log(need)))
            verdict = "astronomix cheaper" if rt[("astronomix", N)] < cost else "AthenaK cheaper"
            print(f"  astronomix {N}^3 ({rt[('astronomix', N)]:.0f} s)  vs  "
                  f"AthenaK {need:.0f}^3 ({cost:.0f} s)  ->  {verdict}")

    spectra_figure(runs, weighted, FIG_DIR / "turbulence_spectra.png")
    convergence_figure(runs, metrics, FIG_DIR / "turbulence_convergence.png")
    ratio_figure(runs, FIG_DIR / "turbulence_ratio.png")


if __name__ == "__main__":
    main()
