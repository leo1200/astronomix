# Multi-node scaling on HoreKa — working recipe

A self-contained recipe for running astronomix across **multiple nodes** (one
JAX process per GPU, `jax.distributed` rendezvous) so a future session can pick
up directly. Read `../../HOREKA.md` first for cluster/partition/filesystem rules.

> **Status honesty (2026-06-29).** The single-process multi-GPU path (one host,
> up to 8 GPUs via `autocvd`) is **verified on hardware**: sharded FD-Pallas is
> bit-exact vs 1 GPU at G=1/2/4 (see `probe_h100.sh` / `_mgpu_pallas_probe.py`).
> The **true multi-process / multi-node** path below is **designed, coded, and
> every known bug fixed — but NOT yet confirmed end-to-end on hardware**: the
> `dist_sanity` job has been queue-bound for ~6 days and never ran. Treat the
> `dist_sanity` step as the gate that must pass before trusting a big run.

---

## 0. TL;DR — the fastest correct path

1. **If you only need ≤ 8 GPUs: avoid multi-node entirely.** Run single-process
   multi-GPU on one node (`weak_teal_singlenode.sh`, or `--gpus N` on the weak
   driver). No `jax.distributed`, no rendezvous, no NCCL-clique risk. This is the
   verified path. Use it whenever the science fits on one node.
2. **If you genuinely need > 8 GPUs (multi-node):**
   - First submit `dist_sanity_h100.sh` (2 nodes, 8 procs, ~1 min). It must
     print `PASS distributed rendezvous`. **Do not run a big job until it does.**
   - Then submit `weak_h100_2node.sh` (G=8) and `weak_h100_4node.sh` (G=16 →
     2048³). These reuse the exact same launch contract the sanity check proved.

---

## 1. The launch model (one process per GPU)

JAX (≥ 0.4) auto-detects Slurm and assigns **one device per process**. We launch
one task per GPU with `srun`:

```bash
srun --ntasks=<G> --ntasks-per-node=4 --gpu-bind=none python driver.py
```

- `<G>` = total GPUs = `nodes × 4` (HoreKa H100/H200 nodes have 4 GPUs each).
- The SBATCH header must request matching resources:
  ```bash
  #SBATCH --nodes=<nodes>
  #SBATCH --ntasks-per-node=4
  #SBATCH --gres=gpu:4
  ```
- **`srun --ntasks` MUST equal the job's total task count.** A mismatch is the
  single most common failure (see §6, "GetKeyValue timeout").

---

## 2. The two non-negotiable code rules

### Rule A — bootstrap distributed BEFORE importing astronomix

Importing `astronomix` **creates the JAX backend** at import time (some option
`NamedTuple`s have `jnp.array` defaults). `jax.distributed.initialize()` must run
*before* the backend exists. So a multi-node driver must, in this exact order:

```python
import os, jax                      # raw jax only — NOT astronomix yet
# (optional autocvd here for single-process mode; see §4)
if _multi:                          # SLURM_NTASKS > 1
    jax.distributed.initialize(local_device_ids=_local_ids)   # see Rule B
jax.config.update("jax_use_shardy_partitioner", False)        # §3
jax.config.update("jax_enable_x64", False)
import astronomix...                # ONLY now is the backend-creating import safe
```

If you import astronomix (or call `jax.devices()`, or even some
`jax.config.update()` calls) before `initialize()`, each process eagerly grabs
all local GPUs → invalid device ordinal → **5-minute topology-gather deadlock**.

The canonical implementation is `pytests/weak_scaling_hydro.py` (top of file).
`astronomix/parallel/distributed.py::init_distributed()` is a safe wrapper that
no-ops single-process and tolerates a pre-bootstrapped client.

### Rule B — GPU binding: `--gpu-bind=none` + select by `SLURM_LOCALID`

Two Slurm GPU-binding modes, and only one works for intra-node NCCL:

| Launch flag | Each task sees | Pick local device | Intra-node NCCL P2P |
| --- | --- | --- | --- |
| `--gpus-per-task=1` (cgroup-bound) | ONE GPU as ordinal 0 | `[0]` | **BREAKS** |
| `--gpu-bind=none` (all-visible) | ALL node GPUs | `[SLURM_LOCALID]` | works |

**Always launch with `--gpu-bind=none`** and have each rank select its own GPU:

```python
cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
localid = int(os.environ.get("SLURM_LOCALID", "0"))
local_ids = [localid] if len(cvd) > 1 else [0]   # robust to either mode
jax.distributed.initialize(local_device_ids=local_ids)
```

This was hard-won (commits `97be721`, `c882ae2`, `dee97f0`): under
`--gpus-per-task=1` the cgroup hides peer GPUs and NCCL P2P can't build the
intra-node clique.

---

## 3. Environment: JAX 0.9, GSPMD (not Shardy), Auto mesh axes

`pytests/runners/_env.sh` encodes all of this — **source it from every runner**
(`source "$REPO/pytests/runners/_env.sh"`). The settled choices and *why*:

- **Use the `astrojax09` env (JAX 0.9.2), not `astrocu12` (JAX 0.10).**
  JAX 0.10 breaks the sharded Pallas path (NaN / partitioner errors). `_env.sh`
  defaults `ASTRO_ENV=astrojax09`. The sharding API is identical across 0.9/0.10,
  so no version-adaptive code is needed — only the Pallas bug differs.
- **`export JAX_USE_SHARDY_PARTITIONER=false`.** JAX 0.10 defaults to the Shardy
  partitioner, which rejects this codebase's integer mesh-axis names and its
  `with_sharding_constraint` usage. Force the GSPMD partitioner the code was
  written for. (You'll see benign `GSPMD ... going to be deprecated` warnings —
  ignore them.)
- **Mesh must use Auto axis types.** JAX ≥ 0.10 defaults `jax.make_mesh` to
  *Explicit* axes, which breaks `with_sharding_constraint`. The mesh builders in
  `_benchmark_utils.py` (`_build_sharding`) force `AxisType.Auto`. If you add new
  mesh-creation sites, do the same.
- **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`, `PREALLOCATE=false`** — leave headroom
  on shared queues; the strong-scaling OOM guard reads the resulting
  `bytes_limit`.

`astronomix` must be **editable-installed** in the env (the cloned env captured
conda pkgs only, not the pip editable link):
```bash
micromamba activate astrojax09
pip install -e . --no-build-isolation     # poetry-core backend must be present
python -c "import astronomix, jax; print(jax.__version__)"   # from a NON-repo cwd
```

---

## 4. Sharding-aware initial conditions (never materialize the full grid)

The largest runs (2048³ on 16 GPUs) can't fit the global grid on one host. The IC
factory `build_sound_wave_state_sharded` (in
`astronomix/test_setups/hydrodynamics/sound_wave3D.py`) builds the state
**already globally-sharded** — each process only materializes its own shard.

- Mesh shards the **X axis only**: `mesh_shape = (1, G, 1, 1)` over axes
  `(VAR, X, Y, Z)`; global grid `= (bx * G, by, bz)`.
- Box is set numerically equal to the grid → uniform spacing `h = 1` at every
  rung. This is a throughput benchmark, so wave periodicity is irrelevant.
- Verified to match the reference `setup_sound_wave` to ~5e-23.

The driver (`weak_scaling_hydro.py`) wires this together; you normally don't touch
the factory.

### Single-process multi-GPU (the verified ≤ 8-GPU path)

Same driver, but pass `--gpus N` and launch **without** srun multi-tasking. It
calls `autocvd(num_gpus=N)` before importing jax and builds the same sharded
state in one process — no `jax.distributed`. Use `weak_teal_singlenode.sh`.

---

## 5. Concrete runbook

```bash
cd /hkfs/home/project/hk-project-pai00101/hd_bn306/astronomix

# 1. GATE: prove the rendezvous (2 nodes, 8 procs, ~1 min). MUST pass first.
sbatch pytests/runners/dist_sanity_h100.sh
#   -> grep the .out for "PASS distributed rendezvous". If it hangs/timeouts,
#      STOP and debug (§6) — do not burn a big multi-node slot.

# 2. Multi-node weak rungs (after the gate passes):
sbatch pytests/runners/weak_h100_2node.sh   # G=8,  global 1024 x 2048 x 2048
sbatch pytests/runners/weak_h100_4node.sh   # G=16, global 2048 x 2048 x 2048 (headline)

# tunables (env vars, defaults shown): BX=128 BY=2048 BZ=2048 STEPS=10 DT=0.4 BLK=4,4,8
# e.g.  BY=1024 BZ=1024 sbatch pytests/runners/weak_h100_4node.sh   (smaller footprint)
```

Per-GPU footprint at `(bx,by,bz)=(128,2048,2048)` and ~60 B/cell is ~32 GiB/GPU
— fits H100 (94 GB) and H200. Each weak rung writes one NPZ to
`pytests/scaling_results/weak_scaling/`; regenerate figures with
`pytests/scaling_results/plot_scaling.py`.

Partitions: multi-node H100 = `accelerated-h100`; multi-node H200 (Ruby) =
`accelerated-h200` (4 GPUs/node). The single 8-GPU node is `accelerated-h200-8`
(Teal, **single node only** — use it for the ≤8-GPU single-process path, not
multi-node). All have been heavily contended; expect long queue waits.

---

## 6. Known failure modes → fixes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `GetKeyValue timed out` / rendezvous waits for N procs that never arrive | `srun --ntasks` ≠ job task count (e.g. `--ntasks=2` inside a 4-task job → JAX waits for 4) | Make `srun --ntasks` == `nodes × 4`; match the SBATCH header. |
| 5-min hang at startup, "invalid device ordinal" | astronomix (or `jax.devices()`) imported before `jax.distributed.initialize()` | Rule A: bootstrap distributed first, import astronomix last. |
| Intra-node NCCL P2P fails to form a clique | `--gpus-per-task=1` cgroup binding hides peer GPUs | Rule B: launch `--gpu-bind=none`, select `[SLURM_LOCALID]`. |
| NaN results from sharded Pallas | JAX 0.10 sharded-Pallas bug | Use `astrojax09` (JAX 0.9.2). |
| Partitioner rejects integer mesh-axis names | JAX 0.10 Shardy default | `JAX_USE_SHARDY_PARTITIONER=false` (in `_env.sh`). |
| `with_sharding_constraint` error about Explicit axes | JAX 0.10 Explicit mesh default | Build mesh with `AxisType.Auto`. |
| Job idles to the SLURM timeout after an OOM, no checkpoint | A rank OOMs → survivors hang in NCCL `Acquire clique` rendezvous (collective waits hang, not fail-fast) | Don't launch a run that won't fit. For strong scaling the 1-GPU baseline guard in `run_strong_scaling` already skips doomed baselines; for weak runs, size `(bx,by,bz)` so per-GPU ≤ ~32 GiB. |
| `ImportError` on orbax / `DeviceLocalLayout` at astronomix import | orbax-checkpoint incompatible with the pinned JAX | The orbax import is guarded lazy (broad `except`); checkpointing is optional for scaling. If it resurfaces, widen the guard in `setup_helpers`. |

---

## 7. File map

- `astronomix/parallel/distributed.py` — `init_distributed()` bootstrap wrapper.
- `pytests/_dist_sanity.py` — minimal rendezvous + `process_allgather` check.
- `pytests/weak_scaling_hydro.py` — the multi-node weak driver (canonical
  bootstrap ordering; supports both srun-multiprocess and `--gpus` single-proc).
- `astronomix/test_setups/hydrodynamics/sound_wave3D.py` —
  `build_sound_wave_state_sharded` IC factory.
- `pytests/runners/_env.sh` — shared env (JAX 0.9 env, GSPMD, mem fraction, GPU
  logger). Source it from every runner.
- `pytests/runners/dist_sanity_h100.sh`, `weak_h100_2node.sh`,
  `weak_h100_4node.sh` — the multi-node job scripts.
- `pytests/runners/weak_teal_singlenode.sh` — the verified ≤8-GPU single-process
  alternative.

---

## 8. First thing to do in a fresh session

```bash
squeue -u $USER -o "%.10i %.16j %.8T %R"        # are the multi-node jobs still queued?
```
If `dist_sanity` has run, read its `.out`:
- **PASS** → trust the launch contract; proceed to the weak rungs / scale up.
- **hang/timeout** → walk §6 top-to-bottom; the rendezvous is the gate, not the
  science.
If still queued (the persistent reality), either keep waiting or fall back to the
verified single-node ≤8-GPU path for real numbers now.
