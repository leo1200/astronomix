---
name: multinode-runs
description: Set up, launch, or debug multi-GPU and multi-node (Slurm/srun) JAX runs of astronomix on HoreKa — distributed init, GPU/NCCL binding, sharding, and the sharding-aware initial condition. Use when the user asks to run/scale a simulation across multiple GPUs or nodes, writes or edits a Slurm runner for a sharded run, or hits errors like "invalid device ordinal", a hang at the NCCL/topology exchange, "initialize() must be called before any JAX calls", "not the same on each process", NaNs under sharding, or "connection refused" between ranks. Encodes the fixes already baked into the repo so we don't rediscover them.
---

# multinode-runs — reliable multi-GPU / multi-node astronomix on HoreKa

This skill is the checklist + failure map for running astronomix sharded
across GPUs and nodes. Every gotcha below **already has a fix in the repo** —
the job is to reuse it, not rediscover it. Reference implementations:

- `astronomix/parallel/distributed.py` — distributed bootstrap helper.
- `pytests/weak_scaling_hydro.py` — canonical multi-node driver (init ordering).
- `pytests/_dist_sanity.py` — minimal rendezvous + collective sanity check.
- `pytests/runners/_env.sh` — env activation + Slurm/GSPMD flags + GPU logger.
- `pytests/runners/weak_h100_2node.sh`, `weak_h100_4node.sh` — runner templates.
- `astronomix/time_stepping/time_integration.py` (~line 165) — replicated params.
- `astronomix/test_setups/hydrodynamics/sound_wave3D.py` — sharded IC factory.

## Launch model

One **process per GPU** under `srun` (JAX distributed auto-detects Slurm).
`N` GPUs across `nodes` nodes → `--ntasks=N --ntasks-per-node=4` (H100/H200
nodes have 4 GPUs; the Teal `accelerated-h200-8` node has 8). Sharding is a
1D domain decomposition along `x` (mesh `(1, G, 1, 1)`).

## Environment (non-obvious)

- Activate with **`micromamba activate`, not `conda`** — `conda activate`
  silently falls back to system `/usr/bin/python` and imports fail confusingly.
  `_env.sh` uses `micromamba` and defaults to the `astrojax09` env.
- **jax 0.9.2** (env `astrojax09`) is the validated stack for the Pallas
  multi-GPU path. jax 0.10 works for native-FD sharding but pulls in the
  Shardy/Explicit-mesh changes below; keep the flags set on either version.
- astronomix must be **editable-installed** (`pip install -e .`) so new
  subpackages (`astronomix/parallel/`) and edits are visible when running
  from `pytests/`. A stale site-packages copy silently shadows the repo.
- matplotlib lives in the env (3.11 in `astrojax09`); regenerate figures with
  `micromamba activate astrojax09 && python pytests/scaling_results/plot_scaling.py`.

## The gotchas (each already fixed — keep it that way)

1. **GPU binding / NCCL (the multi-node killer).** With
   `srun --gpus-per-task=1`, Slurm cgroup-binds each task to one GPU that
   appears as ordinal 0, so intra-node NCCL P2P can't see peer GPUs →
   `invalid device ordinal` and a ~5-minute topology-exchange **deadlock**.
   Fix: launch with **`srun --gpu-bind=none`** (all node GPUs visible to every
   task) and select this rank's GPU with **`local_device_ids=[SLURM_LOCALID]`**.
   Make it robust to either binding: `[localid]` if `len(CUDA_VISIBLE_DEVICES)>1`
   else `[0]` (see `_dist_sanity.py::_local_device_ids`).

2. **Distributed init ordering.** `jax.distributed.initialize()` must run
   **before any JAX call that creates the backend** — and **importing
   astronomix creates the backend** (NamedTuple `jnp.array` defaults). So a
   driver must call `initialize()` (raw `jax`, no astronomix) *first*, then
   import astronomix. Do **not** call `jax.config.update()` before
   `initialize()` — it can trigger backend init. Pattern in
   `weak_scaling_hydro.py` (argparse → `import jax` → `initialize()` →
   `config.update` → import astronomix).

3. **Redundant re-init.** A driver that calls `initialize()` directly and then
   also calls `init_distributed()` will re-init. Guard for it: treat an
   already-up client (`jax.process_count() > 1`) as benign. JAX phrases the
   error differently per version ("already"/"only once" on 0.9, "must be
   called before any JAX calls" on 0.10) — match on the *state*, not the
   string. Handled in `distributed.py::init_distributed`.

4. **JAX 0.10 sharding defaults.** `jax.make_mesh` defaults to **Explicit**
   axis types (breaks `with_sharding_constraint`, which needs **Auto**), and
   the default **Shardy** partitioner rejects the integer mesh-axis names the
   code uses. Fix: build the mesh with `axis_types=(AxisType.Auto,)*4` and set
   **`JAX_USE_SHARDY_PARTITIONER=false`** (exported in `_env.sh`; also
   `jax.config.update("jax_use_shardy_partitioner", False)` after init). The
   "GSPMD is deprecated" warning that follows is expected and harmless.

5. **Multi-host replicated `device_put`.** Putting a param leaf onto a
   fully-replicated cross-host `NamedSharding` asserts every process passed an
   identical value, but it gathers each host's value as the canonical **fp32**
   and compares it to the raw **fp64** Python scalar (e.g. `gamma = 5/3`),
   failing with *"not the same on each process"* on dtype alone. Fix:
   `jax.device_put(jnp.asarray(leaf), replicated)` so all hosts present fp32
   (in `time_integration.py`, guarded by `sharding is not None`).

6. **Build-then-reshard memory cap.** Building the full IC on one device and
   then `device_put`-ing to the mesh caps *even multi-GPU* runs at
   **single-device** memory, and OOMs on rank 0. For large grids use the
   **sharding-aware IC factory** that materializes only each rank's shard
   (`build_sound_wave_state_sharded`) so the full grid never lives on one host.

7. **Pallas + sharding is NOT broken.** If a sharded Pallas run returns NaN,
   suspect the *test*, not the backend: a CFL-unstable `dt` makes the
   reference NaN too. On jax 0.9.2, sharded Pallas matches the 1-GPU result
   bit-for-bit (`_mgpu_pallas_probe.py`). Always check finiteness with a
   stable `dt` before blaming the halo exchange.

## Runner hygiene (bit us, now standard)

- **Source `_env.sh` by absolute path**, not `$(dirname "$0")/_env.sh` —
  under Slurm `$0` is the spool copy and the include is not found, so the env
  never activates and the job runs from the wrong cwd with system python.
- **Propagate srun's exit code**: `srun ... ; rc=$?; ...; exit $rc`. Never end
  a runner with an unconditional `echo DONE` — a failed srun then reports
  success and masks the real error (this hid two failed multi-node runs).
- **Match `--ntasks` to the job's task count.** `srun --ntasks=2` inside a
  4-task allocation makes JAX wait for 4 processes and hang at rendezvous.

## Validation ladder (run in order; cheap → expensive)

1. `_dist_sanity.py` on **1 node / dev queue, 4 procs** — reproduces the
   intra-node NCCL P2P path fast; expect `allgather = [0. 1. 2. 3.]` and
   `PASS`.
2. `_dist_sanity.py` on **2 nodes** — confirms inter-node network NCCL.
3. The real **2-node** run (e.g. `weak_h100_2node.sh`) — first true sharded
   integration; check a result NPZ was written, not just exit 0.
4. Scale to **4 nodes** last. Editable install means queued jobs pick up code
   fixes at runtime, so validate small before the big slot runs.

## Failure quick-reference

| Symptom | Cause | Fix |
|---|---|---|
| `invalid device ordinal`, ~5 min hang | `--gpus-per-task=1` cgroup binding | `--gpu-bind=none` + `local_device_ids=[SLURM_LOCALID]` |
| `initialize() must be called before any JAX calls` | astronomix imported (backend up) before `initialize()` | bootstrap distributed before importing astronomix |
| `not the same on each process` | fp64 python scalar vs fp32 gather in replicated `device_put` | `jnp.asarray(leaf)` first |
| `connection refused` between ranks | rank 0 crashed (coordinator down) — look above it | fix the real first-rank traceback |
| NaN under sharding | CFL-unstable `dt` (reference also NaN) | use a stable `dt`, check finiteness |
| rank grabs all 4 GPUs / OOM on rank 0 | full-grid build before reshard | sharding-aware IC factory |
| job "DONE" but no results | runner masked srun failure | propagate `rc=$?`; read `.err` |

## HoreKa queue reality

- `dev_accelerated-h100` (1 node, ≤1 h) is the **reliable** validation queue
  (~2 h wait). Prod `accelerated-h100/-h200` multi-node and the Teal
  `accelerated-h200-8` node are heavily contended (hours–days) — validate on
  dev first, submit prod early, let it land unattended.
- Strong scaling needs the **1-GPU baseline**, so its top resolution is capped
  by what fits on one GPU (independent of `G`). Predict-and-skip a doomed
  baseline (see `run_strong_scaling::_baseline_fits`) rather than letting a
  1-GPU OOM wedge the NCCL clique until the walltime.
