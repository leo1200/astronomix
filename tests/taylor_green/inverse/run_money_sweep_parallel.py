"""
Parallel dispatcher for the single-vs-multiple-shooting recovery ensemble.

Runs many independent recovery_worker.py jobs (one per (truth seed, horizon,
m)) across the free GPUs of a node -- one job pinned per GPU at a time, the
next job dispatched as soon as a GPU frees. Pure Python (no JAX); the workers
do the GPU work. Resumable: jobs whose output .npz already exists are skipped.

Configure via environment (all optional):
    SWEEP_GPUS    comma list of GPU ids to use, e.g. "0,1,2,3,4,5,6,7,8,9".
                  If unset, auto-detects GPUs with < SWEEP_MEM_MB used.
    SWEEP_MEM_MB  free-memory threshold for auto-detect (default 1500)
    SWEEP_SEEDS   number of random truth seeds (default 5)
    SWEEP_TOBS    comma list of horizons (default "1,2,3,4")
    SWEEP_M       comma list of shooting splits (default "1,4")
    SWEEP_N       cells per dim (default 32)
    SWEEP_STEPS   max Adam steps per job (default 150)
    SWEEP_INIT_AMP  init-perturbation amplitude (default 0 = cold start)

Run on the node:  python run_money_sweep_parallel.py
Then aggregate:   python aggregate_money_v2.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
WORKER = HERE / "recovery_worker.py"
OUTDIR = HERE / "data" / "ensemble"


def _detect_via_gpustat(mem_mb):
    """Free GPUs via `gpustat --json`: no running processes AND low memory."""
    out = subprocess.check_output(["gpustat", "--json"], text=True,
                                  stderr=subprocess.DEVNULL)
    data = json.loads(out)
    free = []
    for g in data["gpus"]:
        used = float(g.get("memory.used", 0))
        procs = g.get("processes", []) or []
        if used < mem_mb and len(procs) == 0:
            free.append(int(g["index"]))
    return sorted(free)


def _detect_via_nvidia_smi(mem_mb):
    """Free GPUs via nvidia-smi: low memory AND no compute processes."""
    gq = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    idx_mem, uuid_of = {}, {}
    for line in gq.strip().splitlines():
        idx, uuid, used = (x.strip() for x in line.split(","))
        idx_mem[int(idx)] = float(used)
        uuid_of[uuid] = int(idx)
    busy_idx = set()
    try:
        pq = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid",
             "--format=csv,noheader"], text=True)
        for line in pq.strip().splitlines():
            u = line.strip()
            if u in uuid_of:
                busy_idx.add(uuid_of[u])
    except Exception:
        pass  # no processes / query unsupported
    return sorted(i for i, used in idx_mem.items()
                  if used < mem_mb and i not in busy_idx)


def detect_free_gpus(mem_mb):
    """All free GPUs, preferring gpustat (process-aware), else nvidia-smi."""
    for name, fn in (("gpustat", _detect_via_gpustat),
                     ("nvidia-smi", _detect_via_nvidia_smi)):
        try:
            free = fn(mem_mb)
            print(f"GPU auto-detect via {name}: free = {free}")
            return free
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  {name} detection failed ({e}); trying next")
    print("no GPU query tool available; defaulting to GPU 0")
    return [0]


def main():
    outdir = HERE / os.environ.get("SWEEP_OUTDIR", "data/ensemble")
    outdir.mkdir(parents=True, exist_ok=True)

    gpus_env = os.environ.get("SWEEP_GPUS", "").strip()
    mem_mb = float(os.environ.get("SWEEP_MEM_MB", 1500))
    if gpus_env:
        gpus = [int(x) for x in gpus_env.split(",")]
    else:
        gpus = detect_free_gpus(mem_mb)
    if not gpus:
        print("no free GPUs found"); sys.exit(1)

    n_seeds = int(os.environ.get("SWEEP_SEEDS", 5))
    horizons = [float(x) for x in os.environ.get("SWEEP_TOBS", "1,2,3,4").split(",")]
    m_list = [int(x) for x in os.environ.get("SWEEP_M", "1,4").split(",")]
    mu_list = [float(x) for x in os.environ.get("SWEEP_MU", "10").split(",")]
    k_cut = float(os.environ.get("SWEEP_KCUT", 4.0))
    n_cells = int(os.environ.get("SWEEP_N", 32))
    steps = int(os.environ.get("SWEEP_STEPS", 150))
    init_amp = float(os.environ.get("SWEEP_INIT_AMP", 0.0))
    opt = os.environ.get("SWEEP_OPT", "lbfgs").lower()

    # build the job list; schedule longest-horizon first so the slow jobs
    # start early and the tail is short. mu only matters for m>1 (single
    # shooting is mu-independent), so m=1 runs once at the first mu.
    jobs = []
    for T in horizons:
        for m in m_list:
            mus = [mu_list[0]] if m == 1 else mu_list
            for mu in mus:
                for s in range(n_seeds):
                    out = outdir / f"rec_T{T}_m{m}_mu{mu}_k{k_cut}_s{s}_{opt}.npz"
                    if out.exists():
                        continue
                    jobs.append({"T": T, "m": m, "mu": mu, "s": s, "out": str(out)})
    jobs.sort(key=lambda j: (-j["T"], -j["m"]))

    print(f"GPUs: {gpus}  |  {len(jobs)} jobs in {outdir} "
          f"({n_seeds} seeds x {len(horizons)} horizons x {len(m_list)} m x "
          f"{len(mu_list)} mu), N={n_cells}, opt={opt}, steps<={steps}, init_amp={init_amp}")

    running = {}      # gpu -> (Popen, job)
    queue = list(jobs)
    free = list(gpus)
    t_start = time.time()
    done = 0

    def launch(gpu, job):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.update({
            "JOB_TOBS": str(job["T"]), "JOB_M": str(job["m"]),
            "JOB_MU": str(job["mu"]), "JOB_KCUT": str(k_cut),
            "JOB_TRUTH_SEED": str(job["s"]), "JOB_N": str(n_cells),
            "JOB_STEPS": str(steps), "JOB_INIT_AMP": str(init_amp),
            "JOB_OPT": opt, "JOB_OUT": job["out"],
        })
        log = outdir / f"log_T{job['T']}_m{job['m']}_mu{job['mu']}_s{job['s']}_{opt}.txt"
        fh = open(log, "w")
        p = subprocess.Popen([sys.executable, str(WORKER)], env=env, stdout=fh, stderr=fh)
        p._logfh = fh
        print(f"  [launch gpu {gpu}] T={job['T']} m={job['m']} mu={job['mu']} seed={job['s']}")
        return p

    last_detect = time.time()
    while queue or running:
        # elastically pick up GPUs that free up later (auto-detect mode only)
        if not gpus_env and queue and time.time() - last_detect > 60:
            last_detect = time.time()
            try:
                for g in detect_free_gpus(mem_mb):
                    if g not in running and g not in free:
                        free.append(g)
                        print(f"  [+gpu {g}] now free, added to pool")
            except Exception:
                pass
        # dispatch to any free GPU
        while free and queue:
            gpu = free.pop(0)
            job = queue.pop(0)
            running[gpu] = (launch(gpu, job), job)
        # poll
        time.sleep(5)
        for gpu, (p, job) in list(running.items()):
            if p.poll() is not None:
                p._logfh.close()
                ok = p.returncode == 0
                done += 1
                status = "ok" if ok else f"FAILED(rc={p.returncode})"
                el = time.time() - t_start
                print(f"  [reap  gpu {gpu}] T={job['T']} m={job['m']} mu={job['mu']} "
                      f"seed={job['s']} {status}  ({done}/{len(jobs)}, {el/3600:.2f}h)")
                del running[gpu]
                free.append(gpu)

    print(f"ALL DONE in {(time.time()-t_start)/3600:.2f}h; outputs in {outdir}")


if __name__ == "__main__":
    main()
