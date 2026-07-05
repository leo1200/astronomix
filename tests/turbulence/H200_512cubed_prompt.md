# H200 prompt: 512³ isothermal-MHD turbulence (HOW-MHD Fig.14/15 high-res column)

The H200 node shares the same storage as the main node, so there is **nothing to
clone and no files to transfer** — work directly in the existing `refactor`
worktree and write the output npz into its `data_fig14/`. Paste the block below
to an agent on the H200 node.

512³ does not fit on a 40 GB A100 (~35 GB + remat headroom); a single 140 GB
H200 fits both runs on one GPU.

---

You are on a node with NVIDIA H200 GPUs, on shared storage with the main node.
Run two 512³ isothermal-MHD turbulence simulations and write the outputs into
the existing repo's `data_fig14/`. Do **not** clone, checkout, pull, or commit —
just run.

**0. Go to the right folder and verify the branch (do not change it).**
```bash
cd /export/home/lstorcks/agent-home/astronomix-refactor-port
git rev-parse --abbrev-ref HEAD          # MUST print: refactor
test -f tests/turbulence/paper_turbulence.py || { echo "WRONG FOLDER"; exit 1; }
```
If the branch is not `refactor` or the file is missing, stop — you are in the
wrong place. This worktree is pinned to `refactor`; the path above is the only
correct one. Do not `git checkout`/`pull`/`commit` (the main node is working here).

**1. Use the project's existing Python env (jax 0.10, `astx`); don't reinstall.**
```bash
# activate however this project's astx env is normally activated, then:
python -c "import jax; print(jax.devices())"   # must list an H200
```

**2. Grab one truly-free GPU via autocvd** (single-GPU runs — no sharding/NCCL):
```bash
# prefix each run with: autocvd -- python ...
# (autocvd waits for a fully idle GPU; never take a GPU another user holds)
```

**3. Run both cases** (from the repo root; outputs land in `tests/turbulence/data_fig14/`):
```bash
cd tests/turbulence

# --- ISM: hypersonic, M_turb ~ 10, beta_p = 0.1 ---
# Mass-conserving REDISTRIBUTE positivity + velocity cap + vacuum protection
# (the robust, validated stack for the hypersonic case).
autocvd -- python paper_turbulence.py \
  --tag ISM_N512 --outdir data_fig14 \
  --eos iso --N 512 --mturb 10 --beta 0.1 \
  --F0 3.5 --cfl 1.5 --tcross 5 --nsnap 6 \
  --stage_mode redist --rhomin 0.02 --vmaxcap 50 --protect 1 \
  2>&1 | tee data_fig14/ISM_N512.log

# --- ICM: subsonic, M_turb ~ 0.5, beta_p = 1e6 ---
autocvd -- python paper_turbulence.py \
  --tag ICM_N512 --outdir data_fig14 \
  --eos iso --N 512 --mturb 0.5 --beta 1e6 \
  --F0 3.5 --cfl 1.5 --tcross 5 --nsnap 6 \
  --stage_mode redist --rhomin 0.02 --vmaxcap 50 --protect 0 \
  2>&1 | tee data_fig14/ICM_N512.log
```

**4. Verify each run before declaring done** — each prints a line like
`[ISM_N512] stationary(last 1/3): M_turb=... first_bad_snap=-1 (all finite)`:
- ISM: `M_turb` ≈ **9–11** (target 10) and `first_bad_snap=-1`.
- ICM: `M_turb` ≈ **0.45–0.6** (target 0.5) and `first_bad_snap=-1`.
- If `first_bad_snap` is not −1 (a NaN appeared), the case went unstable —
  re-run it with `--cfl 1.0` (and only if still unstable, `--rhomin 0.05`).
  Do **not** weaken positivity: keep `--stage_mode redist --vmaxcap 50`.

**5. Done — no transfer needed.** The files
`tests/turbulence/data_fig14/paper_ISM_N512.npz` and `paper_ICM_N512.npz`
are now on the shared storage. Report the two stationary M_turb values and that
both have `first_bad_snap=-1`.

**Notes**
- Forward sims to ~5 crossing times; expect a few hours per run on one H200.
  Lower `--tcross 3` to finish faster (still saturated).
- Single-GPU on H200 — no `NCCL_*` flags, no `sharding`.
- Runs use the v_rms≈1 normalisation; the figure scripts convert E_B to the
  paper's a=1 units, so do not change `--norm`.

---

Back on the main node, once the two npz exist, re-run `python make_fig14.py`
and `python make_fig15.py` in `tests/turbulence/` — the 2×2 hi-res column and
the spectra overlay upgrade from 448³ to 512³ automatically.
