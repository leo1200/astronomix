"""
Agreement of the Pallas and the native JAX backend on the forward tests of the
paper.

Every case runs the setup of the paper script (at a smaller resolution) once
with the Pallas and once with the native JAX backend, in double precision, and
prints the largest difference of the results relative to their largest value.
The backend is switched without editing the scripts: their module-level
``PALLAS`` constant is pointed at ``NATIVE_JAX``, and for the scripts that leave
the backend to ``OPTIMAL_BACKEND`` the GPU check that resolves it to Pallas is
answered with "no". The scripts' ``_common`` modules share a name, so every case
runs in a process of its own.

    sbatch --export=ALL,RUN_DIR=examples/scripts/validation pytests/runners/gpu1.sh \\
        python pallas_native_agreement.py
"""

# general
import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORWARD = HERE.parent / "forward"

#: case name -> (script directory, description)
CASES = {
    "sound_wave_fd": ("hydro", "3D sound wave, FD (WENO-Z), N = 16"),
    "sound_wave_fv": ("hydro", "3D sound wave, FV (VL2, HLLC), N = 16"),
    "sod_fd": ("hydro", "Sod shock tube, FD (WENO-Z), N = 200"),
    "sod_fv": ("hydro", "Sod shock tube, FV (VL2, HLLC), N = 200"),
    "double_blast_fd": ("hydro", "double blast, FD (WENO-Z), N = 400"),
    "double_blast_fv": ("hydro", "double blast, FV (VL2, HLLC), N = 400"),
    "sedov_fd": ("hydro", "3D Sedov, FD (WENO-Z), N = 32"),
    "sedov_fv": ("hydro", "3D Sedov, FV (VL2, HLLC), N = 32"),
    "alfven_fd": ("mhd", "3D CP Alfven wave, FD (WENO-Z, CT), N = 16"),
    "alfven_fd_pp_uct": ("mhd", "3D CP Alfven wave, FD + PP + upwind EMF, N = 16"),
    "alfven_fv": ("mhd", "3D CP Alfven wave, FV (VL2, HLLD, GLM), N = 16"),
    "orszag_tang_fd": ("mhd", "2D Orszag-Tang, FD (WENO-Z, CT), N = 64"),
    "orszag_tang_fv": ("mhd", "2D Orszag-Tang, FV (VL2, HLLD, GLM), N = 64"),
    "mhd_blast_fd": ("mhd", "3D MHD blast, FD (WENO-Z, CT), N = 32"),
    "mhd_blast_fd_short": ("mhd", "3D MHD blast to t = 0.002 (a tenth), FD (WENO-Z, CT), N = 32"),
    "mhd_blast_fv": ("mhd", "3D MHD blast, FV (VL2, HLLD, FOFC), N = 32"),
    "jeans_fd": ("self_gravity", "Jeans wave, FD flux-based corrected, N = 32"),
    "slab_fd": ("self_gravity", "slab advection, FD flux-based corrected, N = 32"),
    "evrard_fd": ("self_gravity", "Evrard collapse to t = 0.4, FD flux-based corrected + PP, N = 32"),
}


def relative_difference(first, second):
    """The largest difference of two results relative to their largest magnitude.

    Args:
        first: A number, an array, a tuple/list or a dict of these (or a JAX
            pytree of arrays).
        second: The same structure from the other backend.

    Returns:
        ``max |first - second| / max |first|`` over all numeric leaves.
    """
    import jax
    import numpy as np

    if isinstance(first, dict):
        keys = sorted(set(first) & set(second))
        first = [first[key] for key in keys]
        second = [second[key] for key in keys]
    leaves_first = [np.asarray(leaf, dtype=float) for leaf in jax.tree_util.tree_leaves(first)
                    if np.issubdtype(np.asarray(leaf).dtype, np.number)]
    leaves_second = [np.asarray(leaf, dtype=float) for leaf in jax.tree_util.tree_leaves(second)
                     if np.issubdtype(np.asarray(leaf).dtype, np.number)]
    difference = max(float(np.max(np.abs(a - b))) for a, b in zip(leaves_first, leaves_second))
    scale = max(float(np.max(np.abs(a))) for a in leaves_first)
    return difference / scale


def use_native_backend(module_names):
    """Make the following runs use the native JAX backend.

    Args:
        module_names: The script modules whose ``PALLAS`` constant selects the
            backend.
    """
    from astronomix.option_classes import simulation_config
    from astronomix.option_classes.simulation_config import NATIVE_JAX

    simulation_config.gpu_compute_capability_at_least_80 = lambda: False
    for name in module_names:
        module = sys.modules[name]
        if hasattr(module, "PALLAS"):
            module.PALLAS = NATIVE_JAX


def run_case(case):
    """Run one case with both backends and print the relative difference."""
    import jax

    jax.config.update("jax_enable_x64", True)
    directory, description = CASES[case]
    sys.path.insert(0, str(FORWARD / directory))
    scratch = Path(tempfile.mkdtemp(prefix="pallas_native_"))

    from astronomix.option_classes.simulation_config import (
        FINITE_DIFFERENCE,
        FINITE_VOLUME,
        HLLC,
        HLLD,
        NATIVE_JAX,
        PALLAS,
        VL2,
        StaticIntVector,
    )
    from astronomix import (
        BackendConfig,
        SimulationConfig,
        SimulationParams,
        get_helper_data,
        get_registered_variables,
        time_integration,
    )

    def run_with(backend):
        """The case's result with one backend."""
        if case.startswith("sound_wave"):
            from astronomix.test_setups.hydrodynamics.sound_wave3D import setup_sound_wave
            from astronomix.option_classes.simulation_config import StaticFloatVector

            solver = (dict(solver_mode=FINITE_VOLUME, time_integrator=VL2, riemann_solver=HLLC)
                      if case.endswith("fv") else dict(weno_z=True))
            config = SimulationConfig(
                backend_config=BackendConfig(backend=backend),
                box_size=StaticFloatVector(3.0, 1.5, 1.5),
                num_cells=StaticIntVector(32, 16, 16),
                dimensionality=3,
                progress_bar=False,
                **solver,
            )
            state, config, params = setup_sound_wave(
                config, SimulationParams(C_cfl=0.3 if case.endswith("fv") else 1.5)
            )
            return time_integration(state, config, params, get_registered_variables(config))
        if case.startswith("sod"):
            from astronomix.test_setups.hydrodynamics.shock_tube1D import setup_sod_shock_tube

            solver = (dict(solver_mode=FINITE_VOLUME, time_integrator=VL2, riemann_solver=HLLC)
                      if case.endswith("fv") else dict(weno_z=True))
            config = SimulationConfig(
                backend_config=BackendConfig(backend=backend),
                dimensionality=1,
                num_cells=200,
                progress_bar=False,
                **solver,
            )
            registered_variables = get_registered_variables(config)
            state, config, params = setup_sod_shock_tube(
                config,
                registered_variables,
                SimulationParams(C_cfl=0.5 if case.endswith("fv") else 0.75),
                get_helper_data(config),
            )
            return time_integration(state, config, params, registered_variables)
        if case.startswith("alfven"):
            from astronomix.test_setups.mhd.alfven_wave3D import setup_cp_alfven_wave

            if case.endswith("fv"):
                solver = dict(solver_mode=FINITE_VOLUME, time_integrator=VL2, riemann_solver=HLLD)
                cfl = 0.3
            elif case.endswith("pp_uct"):
                solver = dict(weno_z=True, weno_positivity_preserving=True, ct_upwind_dissipation=True)
                cfl = 0.75
            else:
                solver = dict(weno_z=True)
                cfl = 1.5
            config = SimulationConfig(
                backend_config=BackendConfig(backend=backend),
                num_cells=StaticIntVector(32, 16, 16),
                progress_bar=False,
                **solver,
            )
            state, config, params = setup_cp_alfven_wave(config, SimulationParams(C_cfl=cfl))
            return time_integration(state, config, params, get_registered_variables(config))
        if case.startswith("double_blast"):
            module = importlib.import_module("double_blast")
            if backend == NATIVE_JAX:
                use_native_backend(["double_blast"])
            mode = FINITE_VOLUME if case.endswith("fv") else FINITE_DIFFERENCE
            return module.simulate(mode, HLLC, 400)
        if case.startswith("sedov"):
            sys.argv = [sys.argv[0], "--res", "32"]
            module = importlib.import_module("sedov_blast")
            if backend == NATIVE_JAX:
                use_native_backend(["sedov_blast"])
            mode = FINITE_VOLUME if case.endswith("fv") else FINITE_DIFFERENCE
            return module.simulate(mode, HLLC, 32, True)
        if case.startswith("orszag_tang"):
            module = importlib.import_module("orszag_tang")
            if backend == NATIVE_JAX:
                use_native_backend(["orszag_tang"])
            mode = FINITE_VOLUME if case.endswith("fv") else FINITE_DIFFERENCE
            return module.simulate(mode, 64)[0]
        if case.startswith("mhd_blast"):
            module = importlib.import_module("mhd_blast")
            module.DATA_DIR = scratch
            if case.endswith("short"):
                module.T_END = 0.002  # a tenth of the test's end time, for both runs
            if backend == NATIVE_JAX:
                use_native_backend(["mhd_blast"])
            return module.run_and_cache("fv_vl2_hlld" if case.endswith("fv") else "fd_mhd", 32)
        if case in ("jeans_fd", "slab_fd"):
            name = "jeans_convergence" if case == "jeans_fd" else "slab_convergence"
            module = importlib.import_module(name)
            common = importlib.import_module("_common")
            original = common.solver_options

            def solver_options(scheme):
                options = original(scheme)
                options["backend_config"] = BackendConfig(backend=backend)
                return options

            module.solver_options = solver_options
            function = module.jeans_l1_error if case == "jeans_fd" else module.slab_l1_error
            return function(32, common.CORRECTED_SCHEME)
        if case == "evrard_fd":
            collapse = importlib.import_module("_collapse")
            snapshots, _, _ = collapse.run_collapse(
                32, collapse.CollapseScheme(), t_end=0.4, backend=backend, num_snapshots=8
            )
            return snapshots
        raise ValueError(case)

    pallas = run_with(PALLAS)
    native = run_with(NATIVE_JAX)
    print(f"RESULT {case:20s} {relative_difference(pallas, native):.2e}  {description}", flush=True)


def main():
    """Run every case (each in a process of its own) and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=sorted(CASES), default=None)
    arguments = parser.parse_args()
    if arguments.case is not None:
        run_case(arguments.case)
        return
    for case in CASES:
        result = subprocess.run(
            [sys.executable, __file__, "--case", case], capture_output=True, text=True
        )
        lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT")]
        if lines:
            print(lines[-1][len("RESULT "):], flush=True)
        else:
            print(f"{case:20s} FAILED: {result.stderr.strip().splitlines()[-1:]}", flush=True)


if __name__ == "__main__":
    main()
