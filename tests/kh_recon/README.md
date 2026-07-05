# Single vs multiple shooting for KH initial-condition reconstruction

A fair, **Adam-only** comparison of two optimisation *formulations* for recovering the
unknown initial perturbation of a 2D Kelvin–Helmholtz flow from its terminal velocity
field. The forward model is the differentiable astronomix finite-difference hydro
solver, run with the **Pallas WENO backend and its reverse-mode adjoint**.

Everything needed to run the experiment and make the figure is in one script:
**`kh_shooting_study.py`** (self-contained — only depends on the astronomix library).

## The comparison

Identical Adam optimiser; only the formulation differs.

- **Single shooting** — optimise the 10-D mode-space control `c` through one
  integration over the whole horizon `T`:  `J(c) = ½‖H(Φ_T(s0(c))) − y‖²`.
  The gradient is the product of all segment Jacobians; at strong nonlinear folding it
  is hypersensitive and Adam stalls / diverges.
- **Multiple shooting (best version)** — lift `M−1` independent interior states and
  impose continuity with an **augmented Lagrangian** (λ-update + ρ-ramp), Adam on the
  inner merit. Each gradient flows through only one short segment of length `h = T/M`,
  so it stays well-conditioned where the full-horizon gradient is useless. The schedule
  closes the defects while remaining optimizable (a fixed penalty cannot).

Both run for the same total Adam-step budget; single uses a cosine-decay LR, MS a
constant inner LR with the ρ-ramp as its schedule.

## Headline result (N=256, M=8, 16 cold inits per cell)

The winning formulation **flips with the amount of nonlinear folding** — see
`figures/study_2x3_N256.png` (rows = horizon, cols = terminal loss `J` / IC error /
continuity defect).

| horizon | fold | single (median) | MS (median) | winner |
|---------|------|-----------------|-------------|--------|
| **T = 20 t_g** | ×~15 | ic **0.0024**, recovers 16/16 | ic 0.020, recovers 16/16 | **single** — beats MS on IC 15/16, on J 16/16 (≈9–30× better) |
| **T = 60 t_g** | ×~40 | ic **2.46**, recovers 0/16 (diverges) | ic **0.079**, recovers 9/16 | **multiple** — beats single 16/16 on both; J ~3 orders lower |

So the multiple-shooting advantage is real but **regime-specific**: it is an
*optimisation* advantage that appears only once the single-shooting horizon is too
chaotic to differentiate through. At short horizon single shooting wins outright (and
more precisely). Continuity also closes far better at short horizon (median MS defect
0.16 at T=20 vs 1.47 at T=60).

The earlier exploration that established this (fixed-μ penalty MS, an N=64/256 ×
T-sweep, and the 400-step vs iteration-matched single-shooting check) confirmed the
same crossover; the matched re-run only made the long-horizon gap *larger* (single’s
median IC rose from 1.58 to 2.46 when given the full step budget — extra Adam steps
push its garbage gradient further from the truth).

## Reproduce

```bash
ROOT=/export/home/lstorcks/agent-home/astronomix-refactor-port   # the refactor worktree
PY=/export/home/lstorcks/.local/share/mamba/envs/jclone/bin/python3   # jax 0.6.2 + optax

# fan 16 inits x {T=20, T=60} across GPUs 0,1,4 (resumable: skips finished cells)
for T in 20 60; do for i in $(seq 0 15); do
  g=$(( i % 3 )); [ $g -eq 2 ] && g=4
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$ROOT $PY kh_shooting_study.py run --horizon $T --init $i &
done; wait; done

PYTHONPATH=$ROOT $PY kh_shooting_study.py agg  --horizon 60          # paired Wilson-CI stats
PYTHONPATH=$ROOT $PY kh_shooting_study.py plot --horizons 20 60      # -> figures/study_2x3_N256.png

# 2x4 reconstruction example (true/optimized initial state, observed/reconstructed final
# state) for the best successful init per horizon -- single@T=20, multiple@T=60.
# Needs a GPU once to forward-sim (fields are cached for re-plots).
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=$ROOT $PY kh_shooting_study.py recon --horizons 20 60
```

Cost (Pallas reverse, this box): single ≈ 7.9 s/step, MS ≈ 5.4 s/step at N=256 T=60;
~4 h/init, ~2.4 h/init at T=20.

## Files

- `kh_shooting_study.py` — the experiment + figures (one script; `run` / `agg` / `plot` / `recon`).
- `data/study_N256_T{20,60}_i*.npz` — per-init results (convergence traces + finals).
- `figures/study_2x3_N256.png` — convergence/comparison figure (rows = horizon, cols = J / IC / defect).
- `figures/recon_2x5_N256.png` — reconstruction example (rows = horizon, cols = true /
  initialization / optimized initial state `v_y'`, observed / reconstructed final state
  `omega_z`); fields cached in `data/recon_2x4_fields.npz`.
