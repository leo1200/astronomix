"""
Regenerate all scaling figures from the standardised (NPZ + JSON) result files
under ``pytests/scaling_results/``.  Re-running the (expensive) simulations is
NOT required -- this reads only the saved data, so plots can be restyled freely
by editing ``STYLE`` / ``SOLVER_STYLE`` below.

    python pytests/scaling_results/plot_scaling.py
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

# --------------------------------------------------------------------------
# Shared style -- edit here to restyle every figure at once.
# --------------------------------------------------------------------------
STYLE = dict(figsize=(8, 6), grid_alpha=0.25, marker_size=6, linewidth=2.0, dpi=150)
SOLVER_STYLE = {
    "FV (JAX)": dict(color="tab:orange", marker="o", linestyle="-"),
    "FD (JAX)": dict(color="cornflowerblue", marker="o", linestyle="-"),
    "FD (Pallas)": dict(color="darkviolet", marker="s", linestyle="--"),
}
MB = 1024 ** 2
GB = 1024 ** 3


def _load(npz_path):
    data = dict(np.load(npz_path, allow_pickle=True))
    meta_path = npz_path[:-4] + ".json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    return data, meta


def _style_for(label):
    return SOLVER_STYLE.get(label, dict(marker="o", linestyle="-"))


def _finish(ax, fig, path, title):
    ax.grid(True, which="both", ls="-", alpha=STYLE["grid_alpha"])
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=STYLE["dpi"])
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------
# Single-GPU: runtime + memory vs cell count, grouped by setup.
# --------------------------------------------------------------------------
def plot_single_gpu():
    files = sorted(glob.glob(os.path.join(HERE, "single_gpu", "*.npz")))
    by_setup = {}
    for f in files:
        data, meta = _load(f)
        by_setup.setdefault(meta.get("setup", "?"), []).append((data, meta))
    for setup, runs in by_setup.items():
        gpu = runs[0][1].get("gpu_model", "?")
        # runtime per step vs cells
        fig, ax = plt.subplots(figsize=STYLE["figsize"])
        for data, meta in runs:
            cells = np.asarray(data["cells"], float)
            rt = np.asarray(data["runtime"], float)
            it = np.asarray(data["iterations"], float)
            per_step = rt / np.where(it > 0, it, np.nan)
            m = np.isfinite(per_step)
            ax.loglog(cells[m], per_step[m], label=meta.get("solver_label", "?"),
                      markersize=STYLE["marker_size"], linewidth=STYLE["linewidth"],
                      **_style_for(meta.get("solver_label", "")))
        ax.set_xlabel("number of cells")
        ax.set_ylabel("wall-clock per timestep (s)")
        _finish(ax, fig, os.path.join(FIG, f"single_gpu_{setup}_runtime.png"),
                f"{setup}: single-GPU time/step ({gpu})")
        # total memory vs cells
        fig, ax = plt.subplots(figsize=STYLE["figsize"])
        for data, meta in runs:
            cells = np.asarray(data["cells"], float)
            tot = np.asarray(data["total_bytes"], float) / GB
            m = tot > 0
            ax.loglog(cells[m], tot[m], label=meta.get("solver_label", "?"),
                      markersize=STYLE["marker_size"], linewidth=STYLE["linewidth"],
                      **_style_for(meta.get("solver_label", "")))
        ax.set_xlabel("number of cells")
        ax.set_ylabel("compiled total memory (GiB)")
        _finish(ax, fig, os.path.join(FIG, f"single_gpu_{setup}_memory.png"),
                f"{setup}: single-GPU memory ({gpu})")


# --------------------------------------------------------------------------
# Block-shape sweep: runtime + memory vs block shape.
# --------------------------------------------------------------------------
def plot_block_shape():
    for f in sorted(glob.glob(os.path.join(HERE, "block_shape", "*.npz"))):
        data, meta = _load(f)
        blocks = ["x".join(map(str, b)) for b in np.asarray(data["block_shape"])]
        rt = np.asarray(data["runtime"], float)
        mem = np.asarray(data["total_bytes"], float) / MB
        x = np.arange(len(blocks))
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        a1.bar(x, rt, color="darkviolet", alpha=0.8)
        a1.set_xticks(x); a1.set_xticklabels(blocks, rotation=45, ha="right")
        a1.set_ylabel("runtime (s)"); a1.set_title("runtime vs block shape")
        a1.grid(True, axis="y", alpha=STYLE["grid_alpha"])
        a2.bar(x, mem, color="teal", alpha=0.8)
        a2.set_xticks(x); a2.set_xticklabels(blocks, rotation=45, ha="right")
        a2.set_ylabel("compiled total memory (MiB)"); a2.set_title("memory vs block shape")
        a2.grid(True, axis="y", alpha=STYLE["grid_alpha"])
        gpu = meta.get("gpu_model", "?"); N = meta.get("N", "?")
        fig.suptitle(f"pallas_block_shape sweep (hydro FD Pallas, grid 2*{N}^3, {gpu})")
        fig.tight_layout()
        out = os.path.join(FIG, os.path.basename(f)[:-4] + ".png")
        fig.savefig(out, dpi=STYLE["dpi"]); plt.close(fig); print("wrote", out)
        # report best
        if np.any(np.isfinite(rt)):
            best = blocks[int(np.nanargmin(rt))]
            print(f"  [block_shape] fastest = {best} ({np.nanmin(rt):.3f}s)")


# --------------------------------------------------------------------------
# Strong scaling: speedup vs N from the run_strong_scaling NPZ format.
# --------------------------------------------------------------------------
def plot_strong():
    for f in sorted(glob.glob(os.path.join(HERE, "strong_scaling", "*_strong_scaling.npz"))):
        data, _ = _load(f)
        meta_path = f.replace("_strong_scaling.npz", "_strong_scaling.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        N = np.asarray(data["N_values"], float)
        G = int(data["num_gpus"]) if "num_gpus" in data else 0
        slugs = sorted({k.split("__")[0] for k in data if "__" in k})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        for slug in slugs:
            r1 = np.asarray(data[f"{slug}__runtime_1"], float)
            rN = np.asarray(data[f"{slug}__runtime_N"], float)
            sp = r1 / rN
            a1.loglog(N, r1, marker="o", label=f"{slug} 1 GPU")
            a1.loglog(N, rN, marker="s", linestyle="--", label=f"{slug} {G} GPU")
            a2.plot(N, sp, marker="o", linewidth=STYLE["linewidth"], label=slug)
        a2.axhline(G, color="k", ls="--", alpha=0.6, label=f"ideal ({G}x)")
        a1.set_xlabel("N (grid 2N x N x N)"); a1.set_ylabel("runtime (s)")
        a1.set_title("runtime"); a1.grid(True, alpha=STYLE["grid_alpha"]); a1.legend(fontsize=8)
        a2.set_xscale("log"); a2.set_xlabel("N (grid 2N x N x N)")
        a2.set_ylabel(f"speedup T1/T{G}"); a2.set_title("strong-scaling speedup")
        a2.grid(True, alpha=STYLE["grid_alpha"]); a2.legend(fontsize=8)
        gpu = meta.get("gpu_model", "?"); setup = meta.get("setup", os.path.basename(f))
        fig.suptitle(f"{setup}: strong scaling 1 vs {G} GPU ({gpu})")
        fig.tight_layout()
        out = os.path.join(FIG, os.path.basename(f)[:-4] + ".png")
        fig.savefig(out, dpi=STYLE["dpi"]); plt.close(fig); print("wrote", out)


# --------------------------------------------------------------------------
# Weak scaling: runtime, throughput, efficiency vs GPU count.
# --------------------------------------------------------------------------
def plot_weak():
    runs = []
    for f in sorted(glob.glob(os.path.join(HERE, "weak_scaling", "*.npz"))):
        data, meta = _load(f)
        runs.append((int(data["G"]), data, meta))
    if not runs:
        return
    runs.sort(key=lambda t: t[0])
    G = np.array([r[0] for r in runs], float)
    rt = np.array([float(r[1]["runtime"]) for r in runs])
    cps = np.array([float(r[1]["cells_per_s_per_gpu"]) for r in runs])
    cells = np.array([float(r[1]["total_cells"]) for r in runs])
    gpu = runs[0][2].get("gpu_model", "?")
    eff = rt[0] / rt  # weak-scaling efficiency: T(1)/T(G), ideal = 1

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    axs[0].plot(G, rt, marker="o", color="darkviolet", linewidth=STYLE["linewidth"])
    axs[0].set_xlabel("GPUs"); axs[0].set_ylabel("runtime (s)")
    axs[0].set_title("weak-scaling runtime (ideal: flat)")
    for g, c in zip(G, cells):
        axs[0].annotate(f"{c:.2e}", (g, rt[list(G).index(g)]), fontsize=7)
    axs[1].plot(G, cps, marker="s", color="teal", linewidth=STYLE["linewidth"])
    axs[1].set_xlabel("GPUs"); axs[1].set_ylabel("cells / s / GPU")
    axs[1].set_title("throughput per GPU")
    axs[2].plot(G, eff, marker="o", color="firebrick", linewidth=STYLE["linewidth"])
    axs[2].axhline(1.0, color="k", ls="--", alpha=0.6, label="ideal")
    axs[2].set_xlabel("GPUs"); axs[2].set_ylabel("efficiency T(1)/T(G)")
    axs[2].set_title("weak-scaling efficiency"); axs[2].legend()
    for a in axs:
        a.set_xscale("log", base=2); a.set_xticks(G); a.set_xticklabels([int(x) for x in G])
        a.grid(True, alpha=STYLE["grid_alpha"])
    fig.suptitle(f"Multi-node weak scaling: hydro FD Pallas ({gpu}), "
                 f"largest grid {cells.max():.2e} cells")
    fig.tight_layout()
    out = os.path.join(FIG, "weak_scaling_hydro_pallas.png")
    fig.savefig(out, dpi=STYLE["dpi"]); plt.close(fig); print("wrote", out)


def plot_weak_efficiency():
    """Standalone headline figure: weak-scaling efficiency vs GPU count."""
    runs = []
    for f in sorted(glob.glob(os.path.join(HERE, "weak_scaling", "*.npz"))):
        data, meta = _load(f)
        runs.append((int(data["G"]), data, meta))
    if not runs:
        return
    runs.sort(key=lambda t: t[0])
    G = np.array([r[0] for r in runs], float)
    rt = np.array([float(r[1]["runtime"]) for r in runs])
    cells = np.array([float(r[1]["total_cells"]) for r in runs])
    gpu = runs[0][2].get("gpu_model", "?")
    eff = rt[0] / rt  # weak-scaling efficiency: T(1)/T(G), ideal = 1

    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax.plot(G, eff * 100, marker="o", color="firebrick",
            markersize=STYLE["marker_size"] + 2, linewidth=STYLE["linewidth"],
            label="measured")
    ax.axhline(100.0, color="k", ls="--", alpha=0.6, label="ideal (100%)")
    for g, e in zip(G, eff):
        ax.annotate(f"{e*100:.0f}%", (g, e * 100), textcoords="offset points",
                    xytext=(0, 8), fontsize=9, ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(G); ax.set_xticklabels([int(x) for x in G])
    ax.set_xlabel("number of GPUs")
    ax.set_ylabel("weak-scaling efficiency  T(1)/T(G)  (%)")
    ax.set_ylim(0, 110)
    n_nodes = int(max(G) // 4) if max(G) >= 4 else 1
    _finish(ax, fig, os.path.join(FIG, "weak_efficiency_hydro.png"),
            f"Weak-scaling efficiency: hydro FD-Pallas ({gpu})\n"
            f"per-GPU grid fixed, up to {int(max(G))} GPUs / {n_nodes} nodes, "
            f"largest grid {cells.max():.2e} cells")


if __name__ == "__main__":
    plot_single_gpu()
    plot_block_shape()
    plot_strong()
    plot_weak()
    plot_weak_efficiency()
    print("All figures in", FIG)
