# Self-gravity of the VL2 scheme: Athena++'s coupling

AthenaPK, whose VL2 scheme astronomix reproduces, has no Poisson self-gravity.
The scheme comes from Athena++ (Stone et al. 2020), which has FFT and multigrid
self-gravity, so astronomix couples self-gravity to VL2 as Athena++ does
(`self_gravity_version=ATHENA_SECOND_ORDER`, selected automatically for VL2):

```python
config = SimulationConfig(
    solver_mode=FINITE_VOLUME,
    solver_config=FVSolverConfig(time_integrator=VL2),
    gravity_config=GravityConfig(
        self_gravity=True,
        self_gravity_version=ATHENA_SECOND_ORDER,
        poisson_discrete_laplacian=True,      # Athena++'s FFT kernel
        # time_centred_energy_source=True,    # Mullen, Hanawa & Gammie (2021)
    ),
)
```

## What Athena++ does (commit `ae17c53c`)

* **Source terms, no gravity fluxes.** `src/hydro/srcterms/self_gravity.cpp`
  (`HydroSourceTerms::SelfGravity`, l. 32-94) adds, per axis, with
  `dpl = -(phi_i - phi_{i-1})`, `dpr = -(phi_{i+1} - phi_i)` and the stage's
  `dt = beta * Delta t`:
  * momentum (l. 47): `M_i += dt/(2 dx) * rho_i * (dpl + dpr)`, the
    cell-centred `rho grad(phi)` (average of the two face accelerations);
  * energy (l. 49-50): `E_i += dt/(2 dx) * (F_{i-1/2} dpl + F_{i+1/2} dpr)`
    with `F = flux[X1DIR](IDN)`, the stage's Riemann mass flux; this is
    `-div(F phi_face) + phi_i div(F)` with `phi_face = (phi_i + phi_{i+1})/2`
    (the form before commit `5982564c`).

  The gravitational stress-tensor fluxes (`src/hydro/gravity_fluxes.cpp`,
  `Hydro::AddGravityFlux`, with the `grav_mean_rho` Jeans-swindle term) were
  removed on 2020-12-31 (commits `640fdde1`, `1d264a9d`, following Mullen et
  al.); the `-gflux` switch existed only in between. There is no switch today.
* **When.** `TimeIntegratorTaskList::AddSourceTerms` (`time_integrator.cpp`
  l. 1890-1912) runs after the flux update (`SRC_TERM` depends on `INT_HYD`,
  l. 977-979), with `ph->w` (the stage's initial primitive state) and
  `ph->flux` (the stage's fluxes: donor cell in the VL2 predictor, PLM in the
  corrector, l. 1694-1708). The Poisson equation is solved after every stage
  from the updated conserved density (`main.cpp` l. 512-519,
  `fft_gravity.cpp` l. 75; first solve `mesh.cpp` l. 1668-1672), so the
  predictor uses `phi(rho^n)`, the corrector `phi(rho^{n+1/2})`: two solves
  per step.
* **Green's function.** `FFTGravity::ApplyKernel` (`fft_gravity.cpp`
  l. 105-148) is called with mode 0, the eigenvalues of the second-order
  discrete Laplacian, `sum_d (2 cos(k_d dx) - 2)/dx^2` (mode 1, `-k^2`, is
  unused); `k = 0` is set to zero. The multigrid solver inverts the same
  7-point Laplacian (`mg_gravity.cpp` l. 275-294).
* **Mullen, Hanawa & Gammie (2021).** With the discrete-Laplacian potential
  the cell-centred momentum source is the divergence of a discrete stress
  tensor (local momentum conservation, curl-free `g`); this part is in
  Athena++. Their energy source charges the mass fluxes with the
  time-averaged gravity, `(g^(0) + g^(1))/2` in the predictor and
  `(g^(0) + g^(2))/2` in the corrector, which conserves total energy to
  round-off; Athena++ did not adopt it ("The energy source term is not
  conservative in this version", `self_gravity.cpp` l. 26-29). astronomix
  offers it as `time_centred_energy_source=True` (one more Poisson solve per
  step).

The coupling runs natively; the fused Pallas VL2 kernels do not write out the
face mass fluxes the energy source needs.

## Agreement with Athena++

`compare_to_athenapp.py` runs Athena++ (built with
`src/pgen/astronomix_state.cpp`, which reads astronomix's initial state; see
`/e/project1/astronomix/storcks1/athenapp_build/README_build.txt`) and
astronomix from the same initial state, C_cfl = 0.3, one period
(`results/comparison_{gpu,cpu}.json`). L1 differences relative to the L1 norm
of the perturbation (density, pressure) or of the field (velocities), largest
over the variables:

| case | cycles | astronomix vs Athena++ | spectral kernel vs Athena++ | truncation error |
|---|---|---|---|---|
| Jeans wave 64x32x32, amplitude 1e-6 | 73 / 73 | 2.7e-9 | 2.5e-4 | 3.8e-2 |
| Jeans wave 64x32x32, amplitude 1e-2 | 73 / 73 | 2.8e-13 | 2.5e-4 | 5.4e-2 |
| slab 32^3, epsilon 0.3 | 250 / 250 | 2.0e-14 | 7.9e-4 | 4.3e-2 |

The codes agree to round-off (the 1e-9 of the 1e-6 wave is 1e-15 absolute),
on the GPU and on the CPU alike. With the spectral Green's function instead of
Athena++'s the results move by 0.1-5 % of the truncation error. The
regression test `pytests/self_gravity/vl2_athenapp_regression.py` keeps two
small Athena++ runs and checks the conservation properties.

## In the paper's self-gravity tests

`examples/scripts/forward/self_gravity/` runs the VL2 scheme (HLLC,
C_cfl = 0.3, discrete-Laplacian kernel) next to the finite-difference
couplings (GH200, float64):

* Jeans wave (amplitude 1e-6, N = 16-256): second order, local orders
  2.1, 1.5, 2.1, 2.1, 2.1, 2.1 (fit 1.86); L1 = 1.35e-9 at N = 256.
* Slab (epsilon = 0.3, N = 16-256): local orders 1.5, 1.9, 2.0, 2.0, 2.1, 2.1
  (fit 1.84); L1 = 1.30e-4 at N = 256.
* Cold Evrard collapse, 128^3, t = 1.2: the run needs AthenaPK's floors
  (dfloor 1e-5, pfloor 3e-6); without them it fails before t = 0.05, as the
  gravitational work charged to the cold ambient cells at the sphere's edge
  exceeds their internal energy. Largest relative total-energy error
  4.3e-3 with Athena++'s energy source, 1.2e-4 with the time-centred one
  (1.6e-4 with 25 instead of 121 snapshots, which change the step sequence);
  the latter is round-off (1e-16) until the pressure floor first acts at
  t ~ 0.15, and drops to 6.7e-5 with floors (1e-6, 1e-8) (the depth of the
  negative-pressure excursions, not the floor value, sets it). The profiles at
  t = 0.8 agree with the finite-difference couplings (shock at r = 0.155-0.162,
  central density 112-113 against 116).
* Time-step dependence (32^3, fixed steps, t = 1.2; `evrard_checks.py`, which
  also runs the floor checks above): with Athena++'s energy
  source the error of the mild collapse (e0 = 0.2) falls as dt^1.0 (1.1e-3 at
  dt = 4.8e-3 to 6.9e-5 at 3e-4); the corrector charges its fluxes with the
  potential of the donor-cell half-step density, which differs from the
  average of the potentials at t^n and t^{n+1} by O(dt dx). With the
  time-centred source the error is round-off at every dt (2e-15 to 4e-14).
  In the cold collapse the floors dominate (time-centred: 4.1e-4 for every dt).
