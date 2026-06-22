"""
Size the weak-scaling per-GPU block from MEASURED single-GPU memory.

Reads the single-GPU FD(Pallas) hydro result (total compiled bytes vs cells),
fits an asymptotic bytes/cell, and prints the largest per-GPU block that fits a
given GPU memory budget, plus suggested (BX, BY, BZ) for the X-sharded weak
ladder and the resulting global grids.

    python pytests/scaling_results/size_weak.py [tag]
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "h200"

# Prefer a same-tag FD(Pallas) hydro single-GPU result; fall back to any.
cands = sorted(glob.glob(os.path.join(HERE, "single_gpu", f"hydro_{tag}*fd_pallas.npz")))
if not cands:
    cands = sorted(glob.glob(os.path.join(HERE, "single_gpu", "hydro_*fd_pallas.npz")))
if not cands:
    print("No hydro FD(Pallas) single-GPU result found yet.")
    sys.exit(1)

f = cands[-1]
d = np.load(f, allow_pickle=True)
meta = json.load(open(f[:-4] + ".json")) if os.path.exists(f[:-4] + ".json") else {}
cells = np.asarray(d["cells"], float)
total = np.asarray(d["total_bytes"], float)
temp = np.asarray(d["temp_bytes"], float)
m = (total > 0) & np.isfinite(total)
cells, total, temp = cells[m], total[m], temp[m]

print(f"source: {os.path.basename(f)}  gpu={meta.get('gpu_model','?')}  "
      f"block={meta.get('pallas_block_shape')}  integrator={meta.get('time_integrator')}")
print(f"{'cells':>14} {'total_GiB':>10} {'B/cell':>8}")
for c, t in zip(cells, total):
    print(f"{int(c):>14} {t/1024**3:>10.2f} {t/c:>8.1f}")

# Asymptotic bytes/cell from the largest grid measured.
b_per_cell = float(total[-1] / cells[-1])
print(f"\nasymptotic total bytes/cell ~ {b_per_cell:.1f}")

for gpu_name, gib in [("H200", 141), ("H100", 94)]:
    budget = gib * 0.85 * 1024**3  # leave ~15% headroom
    max_cells = budget / b_per_cell
    print(f"\n== {gpu_name} ({gib} GiB, 85% usable) -> max ~{max_cells:.3e} cells/GPU ==")
    # X-sharded: per-GPU (BX, BY, BZ); choose BX=128, BY=BZ square-ish.
    bx = 128
    bybz = (max_cells / bx) ** 0.5
    # round down to a multiple of 128 for block divisibility
    m128 = int(bybz // 128) * 128
    per_gpu = bx * m128 * m128
    print(f"   suggested per-GPU block (BX,BY,BZ) = (128, {m128}, {m128}) "
          f"= {per_gpu:.3e} cells (~{per_gpu*b_per_cell/1024**3:.1f} GiB)")
    for G in [1, 2, 4, 8, 16]:
        gx = bx * G
        print(f"     G={G:>2}: global ({gx}, {m128}, {m128}) = {gx*m128*m128:.3e} cells")
