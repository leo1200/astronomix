"""Animate the magnetic (and kinetic) energy spectrum through the dynamo.

The single figure in which the whole story is visible: the magnetic spectrum
holds a fixed *shape* while its amplitude grows through the kinematic phase --
that is what "eigenmode" means and it is why a growth rate is definable at all --
and then, as the field reaches equipartition, the peak migrates to larger scales
and the shape stops being self-similar. Doing it for two codes at once shows the
difference the static figures compress into one number: the scheme with the lower
numerical resistivity keeps magnetic energy at higher ``n`` and climbs faster.

    python make_spectrum_movie.py --data data/reynolds --n 256
    python make_spectrum_movie.py --data data --n 256 --out figures/dynamo_movie.gif

Writes a GIF (Pillow is always available; ffmpeg is not on this cluster).
"""

# general
import argparse
import sys
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_metrics import E_MAG, E_V, load_runs, spectra_of
from make_convergence_figures import SERIES, series_of

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(HERE / "data"))
    p.add_argument("--n", type=int, default=0,
                   help="resolution to animate; 0 uses the finest available")
    p.add_argument("--exclude", nargs="*", default=("ppm",))
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--frames", type=int, default=140)
    p.add_argument("--out", default="")
    args = p.parse_args()

    runs = load_runs(args.data, skip=("smoke", "calib", *args.exclude))
    runs = [r for r in runs if np.asarray(r["spectra"]).shape[0] > 1]
    if not runs:
        raise SystemExit(f"no runs with spectra in {args.data}")
    target = args.n or max(int(r["N"]) for r in runs)
    runs = [r for r in runs if int(r["N"]) == target]
    if not runs:
        raise SystemExit(f"no runs at N={target}")

    # One shared frame grid: the runs have different snapshot cadences, so every
    # curve is interpolated in *time* onto a common set of frames rather than
    # animating snapshot index, which would desynchronise the codes.
    t_end = min(float(np.asarray(r["t_over_tc"]).max()) for r in runs)
    frames = np.linspace(0.0, t_end, args.frames)

    curves = []
    for run in runs:
        t = np.asarray(run["t_over_tc"])
        spec = spectra_of(run)
        n = np.asarray(run["n_shell"], dtype=float)
        good = n >= 1
        color, label = SERIES[series_of(run)]
        curves.append(dict(
            t=t, n=n[good], color=color, label=label,
            E_mag=spec[:, E_MAG][:, good], E_v=spec[:, E_V][:, good],
            E_B=np.asarray(run["E_B"]), E_K=np.asarray(run["E_K"]),
        ))

    fig, (ax, axs, axt) = plt.subplots(
        1, 3, figsize=(16.5, 5.2), gridspec_kw=dict(width_ratios=(1.5, 1.5, 1.0)))

    n_max = max(c["n"].max() for c in curves)
    for a, ylab, title_txt in (
            (ax, r"$E_B(n)$", "magnetic spectrum (absolute)"),
            (axs, r"$E_B(n)\,/\,\sum_n E_B(n)$", "the same, normalised: "
             "a fixed shape here IS the eigenmode")):
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlim(1, n_max)
        a.set_xlabel(r"mode number $n = k L / 2\pi$")
        a.set_ylabel(ylab)
        a.set_title(title_txt, fontsize=10)
        a.grid(alpha=0.25, which="both")
    axs.set_ylim(2e-5, 0.5)

    for c in curves:
        # The saturated kinetic spectrum, faint and static, as the backdrop the
        # magnetic spectrum grows into.
        ax.plot(c["n"], c["E_v"][-1], color=c["color"], lw=1.0, alpha=0.22, ls="--")
        c["line"], = ax.plot([], [], color=c["color"], lw=2.4, label=c["label"])
        c["shape"], = axs.plot([], [], color=c["color"], lw=2.4)
        axt.semilogy(c["t"], np.maximum(c["E_B"], 1e-30), color=c["color"], lw=1.8)
        c["dot"], = axt.plot([], [], "o", color=c["color"], ms=7)
    ax.legend(fontsize=8, loc="lower left",
              title=f"N = {target}   (dashed: saturated $E_v(n)$)", title_fontsize=8)

    axt.set_xlabel(r"$t / t_{\rm cross}$")
    axt.set_ylabel(r"$E_B$")
    axt.set_title("magnetic energy", fontsize=10)
    axt.grid(alpha=0.25)
    axt.set_xlim(0, t_end)
    marker = axt.axvline(0.0, color="0.4", lw=1.0)
    title = fig.suptitle("")

    def _index(c, t_now):
        j = int(np.searchsorted(c["t"], t_now, side="right") - 1)
        return max(0, min(j, len(c["t"]) - 1))

    def update(i):
        t_now = frames[i]
        peak = 0.0
        for c in curves:
            j = _index(c, t_now)
            E = np.maximum(c["E_mag"][j], 1e-300)
            c["line"].set_data(c["n"], E)
            c["shape"].set_data(c["n"], E / max(E.sum(), 1e-300))
            c["dot"].set_data([c["t"][j]], [max(c["E_B"][j], 1e-30)])
            peak = max(peak, float(E.max()))
        # Follow the growth: a fixed six-decade window under the current peak,
        # otherwise the twelve decades this run spans squash the shape flat.
        if peak > 0:
            ax.set_ylim(peak * 1e-6, peak * 3.0)
        marker.set_xdata([t_now, t_now])
        ratios = "    ".join(
            f"{c['label'].split()[0]}: $E_B/E_K$ = "
            f"{c['E_B'][_index(c, t_now)] / max(c['E_K'][_index(c, t_now)], 1e-30):.1e}"
            for c in curves)
        title.set_text(f"Magnetic energy spectrum through the dynamo    "
                       f"$t/t_{{\\rm cross}}$ = {t_now:5.2f}        {ratios}")
        return ([c["line"] for c in curves] + [c["shape"] for c in curves]
                + [c["dot"] for c in curves] + [marker])

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(args.out) if args.out else HERE / "figures" / f"dynamo_spectrum_movie_N{target}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=80)
    plt.close(fig)
    print(f"wrote {out}  ({len(frames)} frames, {len(curves)} runs at N={target})")


if __name__ == "__main__":
    main()
