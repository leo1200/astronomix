# Configuration: schemes and presets

A simulation is set up by two containers: the `SimulationConfig` holds the static choices (they are part of the
compiled program), the `SimulationParams` the numbers that can change, and be differentiated, without a new
compilation (CFL number, end time, adiabatic index, floors, module coefficients). `finalize_config` checks a
configuration and derives the rest (ghost cells, boundary handling, backend).

The solver is chosen by `solver_mode` (`FINITE_DIFFERENCE`, the default, or `FINITE_VOLUME`) and its settings by
`solver_config`, an `FDSolverConfig` or an `FVSolverConfig` (left at `None`, `finalize_config` fills in the
default container of the mode and rejects a container of the other one). A preset is a complete container;
single settings are changed with `_replace`:

```python
from astronomix import (
    ATHENAPK_VL2, FINITE_VOLUME, HLLC, SimulationConfig,
)

config = SimulationConfig(
    solver_mode=FINITE_VOLUME,
    solver_config=ATHENAPK_VL2._replace(riemann_solver=HLLC),  # hydrodynamics
    dimensionality=3,
    num_cells=128,
)
```

| Preset | Mode | Scheme | Use with |
|---|---|---|---|
| `FD_PAPER` | `FINITE_DIFFERENCE` | WENO-Z flux splitting, SSPRK(5,4), HOW-MHD constrained transport with the upwind dissipation of the edge fields | ideal gas, isothermal or polytropic; `C_cfl` up to 1.5; add `weno_positivity_preserving=True` at Mach numbers of two and above |
| `FD_RIEMANN_ISOTHERMAL` | `FINITE_DIFFERENCE` | WENO face states, HLLEM-type Riemann flux, SSPRK(3,3), flux-form constrained transport | `ISOTHERMAL` or `POLYTROPIC`, MHD in 3D or hydrodynamics in 1D to 3D; `C_cfl = 1.0` |
| `FD_RIEMANN_IDEAL_GAS` | `FINITE_DIFFERENCE` | the same with the bounded HLLD flux (HLLC in hydrodynamics) and SSPRK(5,4) | `IDEAL_GAS`; `C_cfl = 1.0` |
| `ATHENAPK_VL2` | `FINITE_VOLUME` | VL2, van Leer-limited PLM, HLLD, GLM cleaning, first-order flux correction | `riemann_solver=HLLC` for hydrodynamics; `C_cfl = 0.3` in 3D, `glm_alpha = 0.1` |
| `BEATTIE_REFERENCE` | `FINITE_VOLUME` | the scheme of Beattie et al. (2025): split MUSCL-Hancock, HLL5R, Powell term in the induction equation, parabolic cleaning | `ISOTHERMAL`, `mhd=True`; `C_cfl = 0.8` |

The options of the containers (time integrators `SSPRK54`, `SSPRK33`, `LSRK4` for FD and `RK2_SSP`,
`MUSCL_HANCOCK`, `VL2` for FV; limiters; Riemann solvers; divergence control `GLM`, `CONSTRAINED_TRANSPORT`,
`INDUCTION_SOURCE`) are documented field by field in
`astronomix/option_classes/simulation_config.py`. Physics modules have containers of their
own (`gravity_config`, `diffusion_config`, `passive_scalar_config`, `wind_config`, `cooling_config`,
`turbulence_driving_config`), as do the positivity safeguards (`positivity_config`) and the backend
(`backend_config`). Scripts written for the configuration before these containers are migrated by
[`tools/migrate_config.py`](https://github.com/leo1200/astronomix/blob/main/tools/migrate_config.py); see the [migration note](migration.md).

## Equations of state

`equation_of_state` is `IDEAL_GAS` (the default; `gamma` in the parameters), `ISOTHERMAL` (`p = c_s^2 rho`,
`isothermal_sound_speed`) or `POLYTROPIC` (`p = K rho^Gamma`, `polytropic_constant` and `polytropic_exponent`).
The isothermal and the polytropic gas are one barotropic law in the code: the state has no energy variable. Both
FD schemes carry it (with their kernels); the FV integrators carry it with the Riemann solvers that need no energy
jump (`HLL`, `LAX_FRIEDRICHS`, `HLL5R`), so `ATHENAPK_VL2` with HLLD or HLLC is an ideal-gas scheme (the paper runs
it with `gamma = 1.0001` for a nearly isothermal gas), and `BEATTIE_REFERENCE` runs the isothermal and the ideal
gas. A polytropic exponent of one reproduces the isothermal gas to round-off.

## Physics modules

The modules run in every scheme family where they apply (a barotropic gas has no energy equation, so the modules
that act on the energy need the ideal gas; `finalize_config` says which combination it rejects). Explicit sources
and diffusive fluxes enter the stages of the integrator (one function, `_modules/_explicit_sources.py`),
operator-split updates run once per step after the solver (`_modules/_per_step_updates.py`).

- self-gravity (`GravityConfig(self_gravity=True)`, FFT Poisson solver; periodic or isolated boundaries) and
  external potentials;
- radiative cooling (`CoolingConfig`: power-law and piecewise power-law curves, tables, explicit or implicit
  update);
- thermal conduction, viscosity and resistivity (`DiffusionConfig`; the coefficients are parameters);
- passive scalars (`PassiveScalarConfig(num_scalars=...)`, optionally bounded, advected with their own
  sub-steps);
- stellar winds (`WindConfig`; in 3D in every scheme, on the spherical 1D grid in finite volume);
- turbulence driving (`TurbulenceDrivingConfig(turbulence_driving=True)` with
  `SimulationParams(turbulence_driving_params=TurbulenceDrivingParams(velocity=...))`): a port of TurbGen
  (Ornstein-Uhlenbeck modes with band, parabola or power-law spectra, solenoidal weight, automatic amplitude
  adjustment) that runs on the GPU inside the time loop without host round trips.

## Pallas kernels and device meshes

`BackendConfig(backend=...)` selects `NATIVE_JAX` (array form), `PALLAS` (fused GPU kernels) or
`OPTIMAL_BACKEND` (the default: Pallas on GPUs of compute capability 8.0 and above, the array form otherwise).
The kernels give the results of the array form to round-off and carry their own derivatives where reverse mode
needs them. `pallas_block_shape` sets the block of a kernel (a power of two that divides the padded grid);
`pallas_interpret=True` runs the kernels in the interpreter on the CPU (for tests).

For several GPUs, put the state on a device mesh and hand the sharding to `time_integration`:

```python
mesh = jax.make_mesh((1, 2, 2, 1), (VARAXIS, XAXIS, YAXIS, ZAXIS))
sharding = NamedSharding(mesh, PartitionSpec(VARAXIS, XAXIS, YAXIS, ZAXIS))
state = jax.device_put(initial_state, sharding)
final_state = time_integration(state, config, params, registered_variables, sharding=sharding)
```

Every kernel then runs on the local block of each device inside a `shard_map`, with a halo exchange between
neighbouring devices; the array form is partitioned by XLA. See
`examples/scripts/multi_gpu` for building a large initial state directly on the
devices and for runs over several nodes.
