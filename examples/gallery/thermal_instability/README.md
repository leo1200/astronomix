# Thermal instability, and astronomix against AthenaK

A cross-code study: run the Guo, Kim & Stone (2025) thermally-unstable ISM box
with astronomix's high-order FD/WENO scheme, and compare it against AthenaK on
the same physical setup. It lived in `../supernova_showcase/` because it grew
out of that work — the Cas A blast needs the same cooling curve and the same
turbulent driving — and was moved here so each directory has one subject.

| file | what it is |
|---|---|
| `casa_ti_phase.py` | The TI box itself: uniform n_p = 1 cm⁻³ at 471 K in the GKS unit system (pc / Myr / μ m_u cm⁻³), AthenaK-exact ISM cooling + heating (KI2002 + Schure SPEX + CGOLS, `hrate = 5e-26`), solenoidal white-in-time driving. The box heats to the warm branch (~6600 K) and TI condenses the cold (~180 K) phase over tens of Myr. |
| `casa_ti_snr.py` | A supernova in that medium, in the same units, bit-comparable to the reference (unlike the tapered/renormalised `snr_sedov.py` next door). |
| `bench_ti_components.py` | Step-cost breakdown: cooling vs conduction vs forcing vs the WENO flux. |
| `prof_ti.sh` | The same profile as a run script (implicit vs explicit cooling, conduction on/off). |
| `compare_codes.py` | astronomix vs AthenaK on the same state and time — PDFs, phase fractions, phase diagrams. Statistical by necessity: the two drivers draw independent realisations. |
| `compare_slices.py` | Density and temperature slices, before and after the SN, colour scales shared across codes. |
| `compare_fig5.py` | The Guo, Kim & Stone Figure 5 panel layout (ρΛ, n, v_φ), astronomix over AthenaK. |
| `athenapk_ref/` | The AthenaK/AthenaPK input decks these were matched against. Note the `ti`/`snr` decks are **GKS-fork** decks: `<turb_init1>`, `turb_flag`, `turb_count` and `mu_h` are fork-only, and the mainline equivalent is `<turb_driving>` + `<hydro_srcterms>`. |

## Running

The solver environment and its wrapper live with the Cas A showcase, and so does
`_common.py` — one solver configuration serving both studies rather than two
copies drifting apart. `_tipaths.py` puts it on `sys.path`; import it before
`_common`.

```bash
cd examples/gallery/thermal_instability
../supernova_showcase/run.sh casa_ti_phase.py --n 128 --t-end 50 --nsnap 26 \
    --save-state /export/data/lstorcks/supernova_showcase/casa_ti_n128.npz
python compare_codes.py <astronomix.npz> /export/data/lstorcks/athenak_ref/ti128_prod
```

AthenaK reference runs are in `/export/data/lstorcks/athenak_ref/` (`ti128_prod`
is the production 128³ pair). Figures go to `figures/` in this directory.

## Where the findings are

This directory holds the code, not a write-up. The measurements — the cooling
curve cross-validation, the `dedt` normalisation trap in the mainline driver,
the forcing-amplitude underflow at L = 64 pc, and the conduction convention
mismatch (`alpha_iso` is a diffusivity, so χ is density-independent) — are
recorded in the commit history and in the session handoffs, not here. If this
study is picked up again, write them down first.
