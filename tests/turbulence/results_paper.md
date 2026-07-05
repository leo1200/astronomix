# HOW-MHD turbulence tests reproduced (Seo & Ryu 2023, §3.9–3.10)

Faithful reproduction of the three HOW-MHD turbulence tests (arXiv:2304.04360),
driven with **Ornstein–Uhlenbeck** forcing (vs the paper's white-in-time driving)
and the **Pallas** backend throughout (isothermal- and ideal-gas-MHD WENO kernels).

## The paper's tests
| test | EoS | M_turb | β_p | duration | spectral window |
|------|-----|--------|-----|----------|-----------------|
| ISM (mol. clouds) | isothermal | ≈10  | 0.1  | 5 t_cross  | 2.5–5 t_cross |
| ICM (clusters)    | isothermal | ≈0.5 | 10⁶  | 30 t_cross | 15–30 t_cross |
| comparison        | iso **and** adiabatic | ≈1 | 1 | 4 t_cross | 1.5–2.5 t_cross |

with ρ₀=1, uniform guide field B₀ from β_p (β=2P_th/B₀²), solenoidal forcing
|δv_k|²∝k⁶exp(−8k/k_exp), k_exp=4π/L₀, L_inj=L₀/2, box [−0.5,0.5]³ periodic.

## Our normalisation
We keep driven v_rms≈1 and set a=1/M_turb, B₀=√(2P_th/β) (P_th=a²ρ₀ iso, P₀ adiabatic).
This is physically identical to the paper's a=1/v_rms=M_turb normalisation for every
dimensionless diagnostic (M_turb, β, spectral slopes, density contrast). OU forcing
peak wavenumber k_f=0.75·k_exp=3π matches the paper's injection spectrum.

## Files
- `paper_turbulence.py` — runner (one case/call); args `--mturb --beta --eos {iso,adiabatic}
  --N --tcross --F0 --cfl --rhomin --vmax --tag`. Computes per-snapshot density/kinetic/
  magnetic spectra + (M_turb, (ρ−ρ₀)_rms, E_K, E_B) time series → `data_paper/paper_<tag>.npz`.
- `make_fig_paper.py` — Figs 14/15/16/17/18 analogues (`ISM_TAG=… python make_fig_paper.py`).
- Figures: `figures/fig_paper_{isothermal_spectra,isothermal_slices,cmp_timeseries,cmp_spectra,cmp_slices}.png`.

## Runs (Pallas, single A100 via autocvd)
| tag | N | M_turb (meas.) | β | stable window | notes |
|-----|---|----------------|---|---------------|-------|
| **ISM_N128_M10_stable** | 128 | ≈10.9 | 0.1 | **0–5 t_cross (full)** | **cfl 1.5, rhomin 0.02 (paper values), prot+positivity** |
| ICM_N128      | 128 | ≈0.6           | 10⁶ | 0–30 t_cross (full) | cfl 1.5 |
| CMPiso_N128   | 128 | ≈1.25 (sat.)   | 1   | 0–4 t_cross (full)  | cfl 1.0 |
| CMPadia_N128  | 128 | ≈1.1→0.7 (decays) | 1 | 0–4 t_cross (full)  | cfl 1.0, γ=5/3 |

## Results
- **ICM, comparison (iso + adiabatic): fully stable** over the paper durations.
  ICM magnetic energy grows from E_B,0≈4e−6 to ≈0.16 — strong small-scale **dynamo**.
- **Iso-vs-adiabatic time evolution reproduces Fig. 16 exactly:** isothermal saturates at
  M_turb≈1.2; **adiabatic peaks ≈1.1 then decays to ≈0.7** because shock heating raises the
  sound speed (no cooling → no saturation). At 1.5≤t/t_cross≤2.5 both are M≈1 and their
  spectra (Fig. 18) and slices (Fig. 17) coincide — matching the paper.
- **Spectra (Fig. 15/18):** density/kinetic/magnetic all show ~k⁻⁵/³ inertial ranges;
  ICM is Kolmogorov-like, ISM density spectrum is flatter (supersonic).
- **Slices (Fig. 14):** ISM (M≈10) shows shock-dominated magnetic structure with voids;
  ICM (M≈0.5) shows tangled **flux ribbons** — exactly the paper's described morphologies.

## Stability of the hypersonic ISM run — ROOT CAUSE FOUND & FIXED

M_turb≈10 hypersonic isothermal turbulence is the stability-critical test and was the
source of the original "stability problems." Comparing against the HOW-MHD Fortran source
(`prot.f`, `forc.f`, `eigenst.f` — downloaded from the paper site) revealed the real
cause:

- The paper applies its conservative **vacuum-protection redistribution `prot`** (sum ρ
  and momentum over the valid 3×3×3 neighbours, set the sub-threshold cell to that
  average, clip velocity) **once per step, right after forcing** (`forc.f:441`). It never
  hard-floors the evolved state; `rhopmin` is only a read-only clamp inside the eigen-
  decomposition (`eigenst.f`: `rho=max(qone,rhopmin)`) — astronomix already mirrors this.
- **In astronomix, `_vacuum_protection` (the `prot` port) was only wired into the
  white-forcing path — the OU forcing path never called it.** So my OU runs had *no*
  conservative redistribution and leaned entirely on the crude per-substage density floor,
  which is exactly why they needed restrictive limits.

**Fix:** added the `_vacuum_protection` call to `_apply_ou_forcing`
(`astronomix/_physics_modules/_turbulent_forcing/_turbulent_forcing.py`), matching the
white-forcing path and `forc.f`.

**Result (verified, M_turb≈10):** with `prot` now active, the recipe that reproduces the
paper's own settings is stable —
- **prot (`vacuum_protection`) + per-RK-substage `enforce_positivity`, rhopmin=0.02,
  CFL=1.5** (all the paper's values) → **fully stable** for the whole 5 t_cross at 64³ AND
  128³ (M_turb=10.9, (ρ−ρ₀)_rms=2.1, no collapse), 128³ wall time ~3.5 min.
- The per-substage `enforce_positivity` (called inside each SSPRK stage) is the **CFL
  lever**: prot-only is stable only to CFL≈0.4 (stochastic collapse above that, since the
  OU realisation matters); prot **+** per-substage positivity is robust to CFL=1.5.
- Either piece alone fails: prot-only at high CFL collapses; per-substage floor alone (no
  prot — my pre-fix runs) collapses. **Both together** = the paper's stability.

So "improvements in the positivity / redistribution mechanism" reduced to: **wire the
existing conservative redistribution (`prot`) into the OU path.** No new floor tuning,
no reduced CFL. Production ISM run upgraded to `ISM_N128_M10_stable` (CFL 1.5, rhopmin
0.02, fully stable). The OU forcing amplitude follows v_rms∝√F0.

Not committed (user manages git). Library change: `_apply_ou_forcing` now calls
`_vacuum_protection` when `config.turbulent_forcing_config.vacuum_protection` is set.
