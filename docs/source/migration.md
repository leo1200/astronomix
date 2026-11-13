# Migration from version 0.6

The version after 0.6 (git tag `pre-streamline`) reorganised the configuration and removed options without users.
No compatibility layer is kept: an old keyword raises a `TypeError` when the `SimulationConfig` is built, so no
script runs with a silently ignored option. `tools/migrate_config.py` rewrites scripts and notebooks:

```bash
python tools/migrate_config.py rewrite my_scripts/ --dry-run   # report what would change
python tools/migrate_config.py rewrite my_scripts/             # rewrite in place
python tools/migrate_config.py check my_scripts/               # list what is left
```

## The configuration

The settings of the solver moved from `SimulationConfig` into `solver_config`, an `FVSolverConfig` (for
`solver_mode=FINITE_VOLUME`) or an `FDSolverConfig` (for `solver_mode=FINITE_DIFFERENCE`, the default). Named
presets are complete containers (`ATHENAPK_VL2`, `BEATTIE_REFERENCE`, `FD_PAPER`, `FD_RIEMANN_ISOTHERMAL`,
`FD_RIEMANN_IDEAL_GAS`; see the README).

| Old `SimulationConfig` keyword | New place |
|---|---|
| `time_integrator` | `solver_config.time_integrator`; `MUSCL` is `MUSCL_HANCOCK`, `RK4_SSP` is `SSPRK54`, `RK4_LSRK` is `LSRK4` |
| `limiter`, `riemann_solver`, `glm_extended_source`, `first_order_flux_correction` | `FVSolverConfig`, same names |
| `first_order_fallback=True` | `FVSolverConfig(reconstruction=DONOR_CELL)` |
| `vl2_constrained_transport=True` | `FVSolverConfig(divergence_control=CONSTRAINED_TRANSPORT)` |
| `split` (`SPLIT`, `UNSPLIT`) | removed: the integrator fixes it (`MUSCL_HANCOCK` is split, `RK2_SSP` and `VL2` are not) |
| `weno_z`, `weno_admissible_face_state`, `weno_positivity_preserving`, `weno_ad_frozen_weights`, `ct_upwind_dissipation`, `ct_upwind_strength`, `dual_energy` | `FDSolverConfig`, same names |
| `weno_epsilon`, `dual_energy_eta` | module constants |
| `num_passive_scalars`, `passive_scalar_bounds`, `passive_scalar_cfl`, `max_passive_scalar_substeps` | `passive_scalar_config=PassiveScalarConfig(num_scalars=..., bounds=..., cfl=..., max_substeps=...)` |
| `diffusion` | `diffusion_config=DiffusionConfig(viscosity=...)` |
| `viscosity_type`, `resistivity`, `thermal_conduction`, `conduction_density_weighted`, `conduction_order` | `DiffusionConfig`, same names |
| `turbulent_forcing_config`, `random_seed` | `turbulence_driving_config=TurbulenceDrivingConfig(...)` (the seed is `TurbulenceDrivingConfig.random_seed`); the parameters are `SimulationParams(turbulence_driving_params=TurbulenceDrivingParams(...))`. The driving is a port of TurbGen, a different model from the old forcing: there is no translation of the old options |
| `ad_remat` | `rematerialize_stages=True` for the stage form; the other forms and `ad_remat_chunks` are removed |
| `halo_exchange="ppermute"` / `"pair_all_to_all"` (`BackendConfig`) | the constants `PPERMUTE` / `PAIR_ALL_TO_ALL` |

`BoundarySettings`, `GravityConfig`, `SnapshotSettings`, `WindConfig`, `CoolingConfig` and `BackendConfig` keep
their fields. `PositivityConfig` lost `per_step_specific_floor`, `coldcrush_blend` and `coldcrush_blend_factor`.

## New options

- `FDSolverConfig(scheme=RIEMANN_FLUX)`: the finite-difference scheme with WENO face states and a Riemann flux
  (`riemann_solver`: `RIEMANN_BY_EOS`, `HLLEM_T`, `HLLD_BOUNDED`, `HLL`), MHD in 3D and hydrodynamics in 1D to
  3D; the time integrator `SSPRK33`.
- `equation_of_state=POLYTROPIC` with `SimulationParams(polytropic_constant=..., polytropic_exponent=...)`.
- `FVSolverConfig`: the solver `HLL5R`, the divergence control `INDUCTION_SOURCE` and `parabolic_cleaning`
  (the reference scheme `BEATTIE_REFERENCE`).
- `SimulationConfig(per_step_update=...)` with `SimulationParams(per_step_update_params=...)`: a user function
  applied once per step (the neural-network force, the CNN corrector and the frame tracking of the mixing-layer
  example use it; they moved from the library into their examples).

## Removed

- options: cosmic rays (`cosmic_ray_config`, `cosmic_ray_params`) and the shock finder; the implicit MHD scheme
  (`numerical_precision`, `fv_magnetic_integrator`, `IMPLICIT_MIDPOINT`, `IMPLICIT_EULER`,
  `SINGLE_PRECISION`, `DOUBLE_PRECISION`); the differentiation options of the Cas A study (`ad_remat_chunks`,
  `ad_scalar_lean`, `ad_shock_arrival`, `ad_shock_arrival_damping`, `ad_shock_latch_compression_width`,
  `ad_shock_latch_entropy_width`, `ad_shock_latch_transport`, `ad_smooth_shock_latch`,
  `ad_tangent_llf_cold_factor`, `passive_scalar_substep_loop` with `SUBSTEPS_*`); the shock history
  (`track_shock_history`, `shock_entropy_jump`); the cold-crush blend and the specific-pressure floor
  (`minimum_specific_pressure`); `frame_tracking`, `neural_net_force_config`, `cnn_mhd_corrector_config` and
  their parameters (now examples); never-set fields (`reconstruction_order`, `state_struct`, `host_helper_data`,
  `weno_epsilon_relative`, `use_max_adaptive_timestep`);
- constants: the limiters `OSHER`, `VAN_ALBADA`, `VAN_ALBADA_PP`; the solvers `HLLC_LM`, `HYBRID_HLLC`,
  `AM_HLLC`; the geometry `CYLINDRICAL`;
- functions: `solver_mode_to_string`.

## Private modules

Scripts that import from the private modules (names with a leading underscore) may meet these changes:

| Old | New |
|---|---|
| `_finite_volume/_state_evolution/` (`evolve_state.py`, `_van_leer_*.py`, `_pallas_evolve.py`, the reconstruction and limiter files) | `_finite_volume/_evolve_state.py`, `_integrators.py`, `_stage.py`, `_stage_pallas.py`, `_reconstruction.py`, `_riemann/`, `_divergence/` |
| `_weno._weno_flux_x`, `_weno_flux_y`, `_weno_flux_z` | `_weno._weno_flux(..., axis=0, 1, 2)` |
| `_weno._weno_flux_y_native`, `_weno_flux_z_native` | `_weno._weno_flux_transverse_native(..., axis=1, 2)`, or `_weno_flux_native_for_axis(axis)` |
| `_ssprk._ssprk4_with_ct`, `_ssprk._ssprk4_hydro` | `_ssprk._ssprk_with_ct`, `_ssprk._ssprk_hydro` (they also run SSPRK(3,3)) |
| `_modules/_time_integrator_sources.py`, `_modules/_iteration_level_updates.py` | `_modules/_explicit_sources.py` (`explicit_sources`), `_modules/_per_step_updates.py` (`per_step_updates`) |
| `_modules/_turbulent_forcing/` | `_modules/_turbulence_driving/` |
| `_finite_difference/_state_evolution/_evolve_state.py` | `_finite_difference/_evolve_state.py` |
| `_finite_difference/_timestep_estimation/_timestep_estimator.py`, `_finite_volume/_timestep_estimation/_timestep_estimator.py` | `_finite_difference/_timestep.py`, `_finite_volume/_timestep.py` |
| `_fluid_equations/_eigen_hydro_iso.py`, `_eigen_mhd_iso.py` | `_fluid_equations/_eigen_hydro_baro.py`, `_eigen_mhd_baro.py` (isothermal and polytropic) |
