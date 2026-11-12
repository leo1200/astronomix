# Upwind dissipation of the constrained-transport EMFs

`FDSolverConfig.ct_upwind_dissipation` adds to every edge EMF of the
finite-difference CT the jumps of the two interface-field components that meet
at the edge, reconstructed to the edge from both sides at fifth order, times
half the local fast signal speed across the jump (the upwind CT of Londrillo &
Del Zanna 2004, applied to the evolved interface fields):

    Omega_z += -(a_y / 2) (B_x^+y - B_x^-y) + (a_x / 2) (B_y^+x - B_y^-x)   (and cyclically)

with `a_d = |v_d| + c_fast,d` the largest over the four cells around the edge.
The increments are the curl of these EMFs with the CT's own staggered
derivative, so `div B = 0` is kept exactly; the jumps are `O(dx^5)` on smooth
fields (`astronomix/_finite_difference/_magnetic_update/_constrained_transport.py:
ct_upwind_dissipation_rhs`). The strength kappa is
`FDSolverConfig.ct_upwind_strength`, 0.03 by default (1 is the full upwind
form); the runners take `--ct-upwind [KAPPA]`, the bare flag meaning 0.03. In
1D and 2D the jumps and differences along the collapsed axes drop out; in 3D
on the Pallas backend the term runs as two kernels
(`_constrained_transport_pallas.py`), which reproduce the native increments to
round-off (float64: 4e-15) and cost 12 % per step of the ISM setup at 512^3 in
float32 (the native form: 20 %).

## Why

The CT of Seo & Ryu (2023) builds the edge EMFs by central high-order
interpolation; its interface fields are damped only through the WENO fluxes of
the cell-centred field, which is itself a central sixth-order interpolation of
the interface fields. Near the Nyquist wavenumber that leaves too little
damping: a **uniform field oblique to the grid** lets grid-scale perturbations
grow, independently of beta and of the CFL number (it is a property of the
spatial discretisation), at a rate that scales like `v_A / dx`. Per time step
the growth is fastest in the cells that set the step, i.e. where the Alfven
speed is largest. Replacing only the face-to-centre interpolation by a two-point
average removes the instability but makes the scheme second order; changing the
CT derivative, the centre-to-face interpolation or the point-to-average step
does not help. The upwind EMF dissipation removes it at fifth order.

## Tests

`oblique_field_stability.py`: uniform isothermal state, `rho = 1`, `|B| = 1`
along `--direction`, `beta = 1e-4`, random velocity of rms `1e-6`, `N x N x 8`
periodic slab, float64, WENO-Z + positivity-preserving WENO, SSPRK4,
`C_cfl = 0.75`, 5 Alfven crossing times. All perturbation modes of the exact
problem are neutral.

| direction, N 256 | CT as before: max \|v.b\| 4.8e-6 -> | e-folding / t_A | with the term, kappa 1 | e-folding / t_A |
|---|---|---|---|---|
| (1, 1) | 37 | +7.3 | 1.6e-7 | -0.10 |
| (2, 1) | 2.0e-3 | +3.4 | 1.7e-7 | -0.06 |
| (1, 0) | 1.5e-7 | -0.05 | | |

kappa 0.3, 0.1, 0.03 and 0.01 stabilise (1, 1) and (2, 1) as well (end values
of max |v.b| as for kappa 1).

The CT variants of Seo & Ryu (2023) do not remove it (N 256, float64): their CT4
(fourth-order staggered derivative, sixth-order face-to-centre interpolation)
grows like CT6 ((1, 1): +7.0 / t_A, (2, 1): +3.5); CT4 with a fourth-order
face-to-centre interpolation grows more slowly ((1, 1): +3.7, (2, 1): +2.2) and
converges at fourth order (CPAW orders 4.68, 4.32); only a two-point
face-to-centre interpolation is stable, at second order.

Without the term, (2, 1) is stable at N 128, grows at +3.4 / t_A at N 256 and
+6 to +9 / t_A at N 512; the growing perpendicular velocity sits near
`(k_x, k_y) = (15, 125)` of N 256. beta 1e-2 and 1 grow as fast as beta 1e-4,
and so does `C_cfl = 0.4`.

**Accuracy** (oblique circularly polarised Alfven wave, isothermal, one period,
L1 error of (v, B), float32):

| N | as before | with the term, kappa 1 |
|---|---|---|
| 8 | 1.30e-2 | 1.87e-2 |
| 16 | 4.01e-4 (order 5.01) | 6.15e-4 (order 4.93) |
| 32 | 1.41e-5 (order 4.83) | 2.02e-5 (order 4.93) |

Both reach the float32 floor at N 64.

**Supersonic ISM turbulence** (`examples/scripts/forward/mhd/turbulence`,
M = 10, beta = 0.1, isothermal, 512^3, no density floor, no Alfven ceiling):
restarted from the state at 3.33 crossing times, the scheme without the term
evacuates strongly magnetised cells within 0.06 crossing times (min rho
2.5e-6, max |v| 393, Alfven speed 763, time step / 25); with the term the same
stretch to 3.45 crossing times keeps min rho >= 1.6e-4, max |v| <= 17 and the
time step within a factor two.

**Dissipation** (`examples/scripts/forward/mhd/turbulence/dynamo_convergence.py`,
float64, kinematic dynamo from a zero-net-flux seed; `make_mechanism_table.py`
estimator at matched `E_B/E_K = 0.01`, the fixed windows agree). Subsonic:
`M ~ 0.6`, seed beta 1e6, without positivity-preserving WENO. Mach 10:
`--pp --mturb 10 --beta 1e4 --rhomin 1e-10 --cfl 0.5`. Resistivity relative to
the CT as before, at N 64 / 128:

| strength | subsonic eta_eff | subsonic Pm | Mach 10 eta_eff | Mach 10 Pm |
|---|---|---|---|---|
| 0 | 1 / 1 | 1.16 / 1.17 | 1 / 1 | 1.49 / 2.00 |
| 0.03 | 1.05 / 1.02 | 1.13 / 1.13 | 0.97 / 1.00 | 1.56 / 1.88 |
| 0.1 | 1.12 / 1.09 | 1.03 / 1.06 | 1.06 / 0.96 | 1.53 / 1.98 |
| 1 | 2.26 / 2.31 | 0.51 / 0.48 | 1.19 / 1.18 | 1.30 / 1.61 |

The viscosity is unchanged. At N 256 (subsonic) the full form raises
`eta_eff` 2.5-fold (Pm 1.22 -> 0.47). The Mach 10, N 128, strength 0.03 pair
ran at `C_cfl = 0.4` (its own reference: Pm 1.90): at 0.5 that run lost
positivity, see below.

**Positivity at Mach 10.** The positivity-preserving WENO is positivity
preserving for `C_cfl <= 0.754` with SSPRK4 only while no stage's splitting
speeds exceed those the step was sized with. In the weak-field Mach 10 runs a
near-vacuum cell (the one that sets the time step) loses mass faster than
momentum within a single step, so `v = m / rho` roughly doubles per stage and
the condition `sum_d lambda_FE (alpha_L + alpha_R) <= 1` reaches 1.7-3 by the
second and third stages; the density turns negative and the step blows up. At
`C_cfl = 0.75` all three tested realisations fail within 1.6-3 crossing times;
at 0.5, 1 of 8 runs failed. This is independent of the CT and of this switch.
`PositivityConfig.step_retry` repeats such a step at half the step size: from
the last state before two captured failures the plain step ends at min rho
-1.6e4 and -64, the retried one at 8.5e-8 and 1.7e-7; over N 64 runs to 6
crossing times at `C_cfl = 0.75`, 8 of 8 realisations survive with it against
3 of 8 without, with 0 to 4 retried steps in about 3000
(`pytests/hydrodynamics/test_step_retry.py` checks the mechanism on a 1D
near-vacuum rarefaction).

```bash
python oblique_field_stability.py --direction 1,1 --N 256                 # as before
python oblique_field_stability.py --direction 1,1 --N 256 --ct-upwind 1   # with the upwind EMFs
JAX_PLATFORMS=cpu python -m pytest pytests/mhd/test_ct_upwind_dissipation.py
```
