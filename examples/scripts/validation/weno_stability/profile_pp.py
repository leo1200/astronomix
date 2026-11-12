"""Component timing of the ideal-MHD FD step with and without PP-WENO.

Each component is jitted on its own, warmed up, then timed as the median of
``--reps`` runs with ``block_until_ready``; the compiled executable's peak
temporary memory is reported next to it. Input: a real turbulence state.

    WENO_VARIANT=pp python examples/scripts/validation/weno_stability/profile_pp.py --state /export/data/.../negp_M20_c0375_lastpositive.npy
"""
import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--state", required=True)
parser.add_argument("--n", type=int, default=0, help="crop to n^3 (0 = full)")
parser.add_argument("--reps", type=int, default=10)
parser.add_argument("--variants", default="off,pairs,joint")
parser.add_argument("--components", default="step,sweep,kernel,recombine,reference,ct")
args = parser.parse_args()

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)

# ruff: noqa: E402
import numpy as np
import jax
import jax.numpy as jnp

from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    PALLAS,
    PERIODIC_BOUNDARY,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
    finalize_config,
    get_helper_data,
    get_registered_variables,
)
from astronomix._fluid_equations._equations_mhd import conserved_state_from_primitive_mhd
from astronomix._finite_difference._interface_fluxes import _weno as weno_module
from astronomix._finite_difference._interface_fluxes import _weno_pallas as pallas_module
from astronomix._finite_difference._interface_fluxes import _weno_positivity as pp
from astronomix._finite_difference._interface_fluxes._weno_positivity_pallas import mhd_inflow_reference_dispatch
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    _constrained_transport_rhs_from_slices, update_cell_center_fields,
)
from astronomix._finite_difference._time_integrators import _ssprk as ssprk_module
from astronomix.option_classes.simulation_config import IDEAL_GAS

GAMMA = 5.0 / 3.0
primitive = np.load(args.state)
if args.n:
    primitive = primitive[:, :args.n, :args.n, :args.n]
primitive = jnp.asarray(primitive, dtype=jnp.float32)
n = primitive.shape[-1]
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)


def make(pp_on):
    config = SimulationConfig(
        equation_of_state=IDEAL_GAS,
        dimensionality=3,
        num_cells=n,
        box_size=1.0,
        mhd=True,
        backend_config=BackendConfig(backend=PALLAS),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        positivity_config=PositivityConfig(clamp_in_estimates=False),
        solver_config=FDSolverConfig(weno_positivity_preserving=pp_on),
    )
    config = finalize_config(config, primitive.shape)
    params = SimulationParams(C_cfl=1.5, gamma=GAMMA, minimum_density=1e-30, minimum_pressure=1e-30)
    return config, params, get_registered_variables(config)


def timed(name, fn, *fn_args):
    compiled = jax.jit(fn).lower(*fn_args).compile()
    try:
        memory = compiled.memory_analysis()
        temp_gb = memory.temp_size_in_bytes / 1e9
    except Exception:  # noqa: BLE001
        temp_gb = float("nan")
    jax.block_until_ready(compiled(*fn_args))
    times = []
    for _ in range(args.reps):
        start = time.perf_counter()
        jax.block_until_ready(compiled(*fn_args))
        times.append(time.perf_counter() - start)
    median = float(np.median(times)) * 1e3
    spread = (float(np.percentile(times, 90)) - float(np.percentile(times, 10))) * 1e3
    print(f"  {name:38s} {median:9.2f} ms  (p90-p10 {spread:6.2f})  temp {temp_gb:6.2f} GB", flush=True)
    return median


results = {}
for variant in args.variants.split(","):
    config, params, rv = make(variant != "off")
    helper_data = get_helper_data(config)
    q = conserved_state_from_primitive_mhd(primitive[:-3], GAMMA, rv)
    faces = tuple(primitive[k] for k in (-3, -2, -1))
    q = update_cell_center_fields(q, *faces, config, rv)
    dt = 1e-4
    use_reference = variant == "joint"
    print(f"[{variant}] N={n}", flush=True)
    components = args.components.split(",")

    if variant == "pairs":
        # the per-axis form: the integrator passes no reference
        original_reference = ssprk_module.mhd_inflow_reference
        ssprk_module.mhd_inflow_reference = lambda *a, **k: None

    if "step" in components:
        def step(q_, bx, by, bz):
            return ssprk_module._ssprk4_with_ct(q_, bx, by, bz, GAMMA, config.grid_spacing, dt, params,
                                                helper_data, config, rv)
        # the step donates its inputs: hand it fresh copies each call
        results[(variant, "step")] = timed(
            "full SSPRK(5,4)+CT step", lambda q_, bx, by, bz: step(q_ + 0.0, bx + 0.0, by + 0.0, bz + 0.0),
            q, *faces)
    if variant == "pairs":
        ssprk_module.mhd_inflow_reference = original_reference

    reference = mhd_inflow_reference_dispatch(q, params, config, rv) if use_reference else None
    if "reference" in components and variant != "off":
        results[(variant, "reference")] = timed(
            "inflow reference pre-pass", lambda q_: mhd_inflow_reference_dispatch(q_, params, config, rv), q)
    if "sweep" in components:
        for axis in range(3):
            flux_fn = [weno_module._weno_flux_x, weno_module._weno_flux_y, weno_module._weno_flux_z][axis]
            if use_reference:
                results[(variant, f"sweep{axis}")] = timed(
                    f"WENO sweep axis {axis} (kernel+recombine)",
                    lambda q_, r, w: flux_fn(q_, params, config, rv, inflow_reference=(r, w)), q, *reference)
            else:
                results[(variant, f"sweep{axis}")] = timed(
                    f"WENO sweep axis {axis} (kernel+recombine)", lambda q_: flux_fn(q_, params, config, rv), q)
    if variant != "off" and ("kernel" in components or "recombine" in components):
        # split the Pallas path: the kernel alone, and the array recombination alone
        captured = {}
        original = pallas_module.positivity_preserving_interface_flux

        def capture(*a, **k):
            captured["args"] = a
            captured["kwargs"] = k
            return a[3]  # skip the recombination: plus split flux as a stand-in output

        pallas_module.positivity_preserving_interface_flux = capture
        if "kernel" in components:
            for axis in range(3):
                results[(variant, f"kernel{axis}")] = timed(
                    f"Pallas split kernel axis {axis}",
                    lambda q_: pallas_module._weno_flux_mhd_pallas_local(q_, params, config, rv, axis=axis), q)
        pallas_module.positivity_preserving_interface_flux = original
        if "recombine" in components:
            for axis in range(3):
                pallas_module.positivity_preserving_interface_flux = capture
                jax.block_until_ready(pallas_module._weno_flux_mhd_pallas_local(
                    q, params, config, rv, axis=axis, inflow_reference=reference))
                pallas_module.positivity_preserving_interface_flux = original
                a = captured["args"]
                arrays = (a[0], a[1], a[2], a[3], a[4])

                def recombine(s, f, c, pf, mf, r=None, w=None, axis=axis):
                    zero = jnp.zeros_like(s)
                    return pp.positivity_preserving_interface_flux(
                        s, f, c, pf, mf, zero, zero, params, config, rv, axis=axis,
                        inflow_reference=None if r is None else (r, w))
                if use_reference:
                    results[(variant, f"recombine{axis}")] = timed(
                        f"array recombination axis {axis}", recombine, *arrays, *reference)
                else:
                    results[(variant, f"recombine{axis}")] = timed(
                        f"array recombination axis {axis}", recombine, *arrays)
    if "ct" in components:
        slices = tuple(jnp.ones_like(q[0]) * 1e-3 for _ in range(6))
        results[(variant, "ct")] = timed(
            "CT rhs from slices",
            lambda q_, *sl: _constrained_transport_rhs_from_slices(q_, *sl, 0.1, 0.1, 0.1, config, rv), q, *slices)
    jax.clear_caches()

print("SUMMARY " + " ".join(f"{k[0]}:{k[1]}={v:.1f}" for k, v in results.items()), flush=True)
