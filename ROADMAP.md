# Roadmap

Where the next studies start in this repository, what is already in place for
them, and what is known to be open. The detailed notes live next to the code
they describe; this file only points to them.

## 1. AthenaPK's second-order scheme in astronomix against AthenaPK

**In place.** `time_integrator=VL2` (with `HLLD`, GLM cleaning, first-order flux
correction and floors) re-implements AthenaPK's default scheme, native and Pallas,
single- and multi-GPU (`_pallas_call_sharded`).
[`examples/scripts/validation/athenapk_vl2`](examples/scripts/validation/athenapk_vl2)
holds the AthenaPK runner, the 19-case comparison (round-off agreement, the scale
being AthenaPK's own GPU-vs-CPU difference), the CP Alfvén convergence study and
the A100 benchmark (Pallas 9.25 ms per step at 2 x 128³ in double precision
against AthenaPK's 25.4 ms). `pytests/mhd/vl2_athenapk_regression.py` guards it.

**Done since** (paper revision, branch `paper-revision`): AthenaPK builds for the
JUPITER GH200 nodes in double and single precision
(`/e/project1/astronomix/storcks1/athenapk_build/`, `jupiter_athenapk.sh`), the
single-GPU comparison on GH200 in both precisions (`benchmark.py`,
`fv_athenapk_single_precision.py`; VL2 Pallas 2.8x faster per step than AthenaPK
in double, 6.9x in single precision at N = 128), and strong/weak scaling of VL2 to
128 GPUs (`examples/scripts/scaling/`).

**Next.** Scaling against AthenaPK's own multi-GPU scaling; the VL2 Pallas kernels
on multiple nodes are communication bound (FV weak-scaling efficiency 58-65 % at
128 GPUs, at eight to nine times the FD throughput per GPU).

## 2. Cas A at high resolution

**In place.** The calibrated pipeline, the X-ray fit and the 4D-Var moved to their own
repository, [casanova](https://github.com/leo1200/casanova) (2026-10-20, with the history
of `examples/gallery/supernova_showcase`), and the JAX observation model to
[observistix](https://github.com/leo1200/observistix); start with casanova's `OVERVIEW.md`,
`ROADMAP.md` and `HANDOFF_LARGER_CLUSTER.md`. The scripts run on the
positivity-preserving WENO (`weno_positivity_preserving=True`, set by
`_common.make_fd_config`), whose gradients are finite in single precision. Its
`run.sh` imports the astronomix and observistix checkouts next to it.

**On JUPITER** (casanova's `jupiter/README.md`):
the observation data rebuilt from the public Chandra archive (no CIAO on aarch64:
`--responses tables`), the ICs from Orlando's delivery, gates G0-G5 measured,
the hydro WENO Pallas adjoint back (`BackendConfig.pallas_adjoint`),
deterministic scatters made cheap, `casa_4dvar` across nodes (`CASA_DISTRIBUTED=1`),
a 128³ refit and the 512³ 4D-Var on 8 GH200 (198-835 s per gradient, <= 66 GB/GPU),
which improves the held-out 2019/2022 score by 17.2 % (128³: 12.3 %).

**Next**: the CIAO responses (built on an x86 machine), a longer refit and the
Bayesian stage of the hand-off on the 512³ analysis; a separate frozen-theta switch
(P3); the MHD WENO Pallas adjoint; the native-vs-Pallas dual-energy forward
difference in kinetic-dominated cells. Callers outside the repository still pass
the removed `--positivity` flag.

## 3. Second-order FV against fifth-order FD: spectra, dynamo, Prandtl number

**In place.** [`examples/scripts/forward/mhd/turbulence`](examples/scripts/forward/mhd/turbulence)
(start with `README.md` and `DYNAMO_MECHANISM.md`): the small-scale dynamo with
astronomix WENO5/WENO-Z + CT against AthenaPK PLM/PPM/WENO-Z + GLM, in-flight
spectra and transfer functions, the numerical viscosity and resistivity measured
at matched E_B/E_K (Pm_num is a scheme constant: CT ~1.21, every GLM scheme
0.48-0.63), explicit-diffusivity calibration ladders, AthenaPK-style few-modes
forcing (`TurbulentForcingConfig.forcing_modes` of the forcing module before the
TurbGen driving, tag `pre-streamline`). The hydro counterpart against
AthenaK is in `examples/scripts/forward/hydro/turbulence`.

**Done since** (paper revision): the study with astronomix's own VL2 scheme
(`dynamo_convergence.py --scheme {fd,vl2}`, data in `data/fd_fv/`,
`make_fd_fv_reynolds_figure.py`): with the same forcing and diagnostics, FD
(WENO-Z + CT) has Pm 1.16-1.22 and FV (VL2 + GLM) 0.61-0.63 at N = 64-256 (VL2
reproduces the earlier AthenaPK PLM numbers to 1-3 %); Re and Rm grow as N^1.2.

**Next.** Compare against the literature (e.g. arXiv:astro-ph/0109497) and with
explicit diffusivities at fixed Pm.

## Known open issues

- Reflective boundaries applied to the stacked face magnetic field or velocity
  (`MAGNETIC_FIELD_ONLY` / `VELOCITY_ONLY`) negate the component with index equal
  to the array axis rather than the normal component, and mirror face-centred
  values like cell-centred ones; this matters for finite-difference MHD with
  reflective walls.
- `BackendConfig(pallas_ct=True)` is not differentiable (the CT Pallas kernels
  have no native tangent wrapper); use `pallas_ct=False` for gradients.
- Self-gravity with the isothermal equation of state is not rejected by
  `finalize_config`, although the energy source then has no energy variable.
- The multi-GPU fast path of the LSRK4 + CT step (kept x halo, fused CT parts) is
  used only without positivity-preserving WENO and dual energy; porting it to
  the SSPRK4 integrator and to the PP data flow is open.
- The pairwise `all_to_all` halo exchange of the multi-GPU scaling work
  (esovetkin/optim_scaling, 256483e) is now merged and the default for
  multi-device meshes (`BackendConfig.halo_exchange = "pair_all_to_all"`; the
  shard_map of the exchange runs under string axis names, which JAX needs for
  `axis_index_groups`). It needs ~10 % of the device memory for NCCL (the
  sharded runners set a 90 % XLA pool); `"ppermute"` restores the ring exchange.
- `_as_3tuple_block_shape` (`_pallas_helpers.py`) clamps the Pallas block to the
  padded extent, which can give a non-power-of-two block that Triton cannot lower
  (e.g. 1D finite difference with open boundaries at N = 64); a block shape that
  does not divide the grid silently falls back to the native backend.
- `analysis_helpers/_exposed_rhs.py` does not support fully periodic
  (`PERIODIC_ROLL`) layouts.
- Multi-node runs should use one process per node driving its four GPUs
  (`PROCESS_LAYOUT=node`, the runners' default): one process per GPU is ~30 %
  slower per step across nodes, for a reason not yet identified. At 128 GPUs
  the NCCL buffers need more than the 8 % of device memory that the default
  92 % XLA pool leaves free (`XLA_PYTHON_CLIENT_MEM_FRACTION` 0.85-0.90, and
  `NCCL_NVLS_ENABLE=0 NCCL_BUFFSIZE=1048576` for FD MHD).
