"""
Athena++ versus astronomix: the self-gravity coupling of the VL2 scheme.

Runs the same self-gravitating problems with Athena++ (VL2, PLM, HLLC, FFT
self-gravity with the discrete-Laplacian kernel; CPU) and with astronomix's VL2
scheme and its ``ATHENA_SECOND_ORDER`` coupling, from the same initial
primitive state written by astronomix, and compares the final primitive states.
The cases are the Jeans wave of the paper (64 x 32 x 32 cells, amplitudes 1e-6
and 1e-2, one wave period) and the advected equilibrium slab (32^3 cells, one
advection period), all at C_cfl = 0.3.

For each case the script reports the cycle counts and, per variable, the L1
difference to Athena++ relative to the L1 norm of the field's perturbation
(density, pressure: deviation from the mean; velocities: the field itself),
for astronomix with Athena++'s discrete-Laplacian Green's function and with
the spectral one, together with astronomix's own error against the exact
solution (the initial state after one period) in the same normalisation: the
truncation error the differences are to be compared with.

The Athena++ binary is built with ``src/pgen/astronomix_state.cpp`` (see
``/e/project1/astronomix/storcks1/athenapp_build/README_build.txt``).

    python compare_to_athenapp.py --athena <path to athena> [--work-dir <dir>]

Results are written to ``results/comparison_<platform>.json``. With
``--regression-data`` the script instead stores two small Athena++ runs
(initial state, final state, cycles) in
``pytests/self_gravity/data/athenapp_vl2_gravity`` for the regression test
``pytests/self_gravity/vl2_athenapp_regression.py``.
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
import json
import subprocess
import time
from pathlib import Path

# typing
from typing import NamedTuple

# jax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    HLLC,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    VL2,
)
from astronomix.option_classes.simulation_config import ATHENA_SECOND_ORDER

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FVSolverConfig,
    GravityConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)
from astronomix.option_classes.simulation_config import (
    StaticFloatVector,
    StaticIntVector,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
REGRESSION_DATA_DIR = HERE.parents[3] / "pytests" / "self_gravity" / "data" / "athenapp_vl2_gravity"

GAMMA = 5 / 3
COURANT_NUMBER = 0.3
VARIABLE_NAMES = ("density", "velocity_x", "velocity_y", "velocity_z", "pressure")


class Case(NamedTuple):
    """One comparison problem."""

    name: str
    problem: str
    num_cells: tuple
    box_size: tuple
    amplitude: float
    gravitational_constant: float


CASES = (
    Case("jeans_amplitude_1e-6", "jeans", (64, 32, 32), (np.pi, np.pi / 2, np.pi / 2), 1e-6, 1 / (4 * np.pi)),
    Case("jeans_amplitude_1e-2", "jeans", (64, 32, 32), (np.pi, np.pi / 2, np.pi / 2), 1e-2, 1 / (4 * np.pi)),
    Case("slab_32", "slab", (32, 32, 32), (3 * np.pi,) * 3, 0.3, 1 / (4 * np.pi)),
)


#: Small cases for the regression test: a nonlinear Jeans wave and the slab.
REGRESSION_CASES = (
    Case("jeans_16", "jeans", (16, 8, 8), (np.pi, np.pi / 2, np.pi / 2), 1e-2, 1 / (4 * np.pi)),
    Case("slab_16", "slab", (16, 16, 16), (3 * np.pi,) * 3, 0.3, 1 / (4 * np.pi)),
)


# -------------------------------------------------------------
# =================== ↓ Initial conditions ↓ ==================
# -------------------------------------------------------------


def jeans_initial_state(coordinates, case):
    """
    The Jeans linear wave of the paper (``jeans_convergence.py``): wave vector
    (2, 4, 4), rho_B = 1, c_s^2 = 1, 4 pi G = 1.

    Args:
        coordinates: The cell centres, shape ``(3, nx, ny, nz)``.
        case: The comparison case (amplitude).

    Returns:
        ``(primitives, period)``: the primitive fields (rho, vx, vy, vz, p)
        and the wave period.
    """
    wave_vector = jnp.array([2.0, 4.0, 4.0])
    wavenumber_squared = jnp.sum(wave_vector**2)
    frequency = jnp.sqrt(wavenumber_squared - 4 * jnp.pi * case.gravitational_constant)
    phase = sum(wave_vector[axis] * coordinates[axis] for axis in range(3))
    perturbation = case.amplitude * jnp.sin(phase)
    density = 1.0 + perturbation
    velocities = [frequency * wave_vector[axis] / wavenumber_squared * perturbation for axis in range(3)]
    pressure = 1.0 / GAMMA + perturbation
    return (density, *velocities, pressure), float(2 * jnp.pi / frequency)


def slab_initial_state(coordinates, case):
    """
    The advected self-gravitating equilibrium slab of the paper
    (``slab_convergence.py``): k = (2/3, 2/3, 2/3), v = (0.6, 0.6, 0.6),
    p_0 = 6, G = 1 / (4 pi).

    Args:
        coordinates: The cell centres, shape ``(3, nx, ny, nz)``.
        case: The comparison case (amplitude epsilon).

    Returns:
        ``(primitives, period)``: the primitive fields and the advection
        period.
    """
    wave_vector = jnp.array([2 / 3, 2 / 3, 2 / 3])
    advection_velocity = jnp.array([0.6, 0.6, 0.6])
    epsilon = case.amplitude
    phase = sum(wave_vector[axis] * coordinates[axis] for axis in range(3))
    wavenumber_squared = jnp.sum(wave_vector**2)
    density = 1.0 + epsilon * jnp.cos(phase) + epsilon**2 / 3 * jnp.cos(2 * phase)
    pressure = 6.0 + 4 * jnp.pi * case.gravitational_constant * epsilon / wavenumber_squared * (
        (1 - epsilon**2 / 12) * jnp.cos(phase)
        + epsilon / 3 * jnp.cos(2 * phase)
        + epsilon**2 / 12 * jnp.cos(3 * phase)
        + epsilon**3 / 144 * jnp.cos(4 * phase)
    )
    velocities = [advection_velocity[axis] * jnp.ones_like(density) for axis in range(3)]
    period = 2 * jnp.pi / jnp.sum(wave_vector * advection_velocity)
    return (density, *velocities, pressure), float(period)


# -------------------------------------------------------------
# =================== ↑ Initial conditions ↑ ==================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ====================== ↓ The two codes ↓ ====================
# -------------------------------------------------------------


def astronomix_setup(case, discrete_laplacian):
    """
    The VL2 configuration of a case and its initial state.

    Args:
        case: The comparison case.
        discrete_laplacian: Whether the Poisson solve uses the discrete
            Laplacian (Athena++'s kernel) or the spectral one.

    Returns:
        ``(config, params, registered_variables, initial_state)``.
    """
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        backend_config=BackendConfig(backend=NATIVE_JAX),
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(time_integrator=VL2, riemann_solver=HLLC),
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=ATHENA_SECOND_ORDER,
            poisson_discrete_laplacian=discrete_laplacian,
        ),
        mhd=False,
        dimensionality=3,
        box_size=StaticFloatVector(*[float(size) for size in case.box_size]),
        num_cells=StaticIntVector(*case.num_cells),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_states=False, return_final_state=True),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    coordinates = jnp.transpose(helper_data.geometric_centers, (3, 0, 1, 2))
    initial_state_function = jeans_initial_state if case.problem == "jeans" else slab_initial_state
    primitives, period = initial_state_function(coordinates, case)
    params = SimulationParams(
        C_cfl=COURANT_NUMBER,
        t_end=period,
        gamma=GAMMA,
        gravitational_constant=case.gravitational_constant,
    )
    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=primitives[0],
        velocity_x=primitives[1],
        velocity_y=primitives[2],
        velocity_z=primitives[3],
        gas_pressure=primitives[4],
    )
    config = finalize_config(config, initial_state.shape)
    return config, params, registered_variables, initial_state


def run_astronomix(case, discrete_laplacian):
    """
    Run a case with astronomix's VL2 scheme.

    Args:
        case: The comparison case.
        discrete_laplacian: Whether to use Athena++'s discrete-Laplacian kernel.

    Returns:
        ``(initial primitives, final primitives, cycles, period)`` with the
        primitives as a ``(5, nx, ny, nz)`` array.
    """
    config, params, registered_variables, initial_state = astronomix_setup(case, discrete_laplacian)
    snapshots = time_integration(initial_state, config, params, registered_variables)
    final_state = jax.block_until_ready(snapshots.final_state)
    return np.asarray(initial_state), np.asarray(final_state), int(snapshots.num_iterations), float(params.t_end)


def athena_input(case, period, initial_state_file, final_state_file):
    """
    The Athena++ input file of a case.

    Args:
        case: The comparison case.
        period: The end time.
        initial_state_file: The raw initial primitive state.
        final_state_file: Where Athena++ writes the final primitive state.

    Returns:
        The input file as a string.
    """
    lines = [
        "<job>",
        f"problem_id = {case.name}",
        "<time>",
        f"cfl_number = {COURANT_NUMBER!r}",
        "nlim = -1",
        f"tlim = {period!r}",
        "integrator = vl2",
        "xorder = 2",
        "ncycle_out = 100000",
        "<mesh>",
    ]
    for axis, (cells, size) in enumerate(zip(case.num_cells, case.box_size), start=1):
        lines += [
            f"nx{axis} = {cells}",
            f"x{axis}min = 0.0",
            f"x{axis}max = {float(size)!r}",
            f"ix{axis}_bc = periodic",
            f"ox{axis}_bc = periodic",
        ]
    lines.append("<meshblock>")
    lines += [f"nx{axis} = {cells}" for axis, cells in enumerate(case.num_cells, start=1)]
    lines += [
        "<hydro>",
        f"gamma = {GAMMA!r}",
        "<problem>",
        f"four_pi_G = {4 * np.pi * case.gravitational_constant!r}",
        f"initial_state = {initial_state_file}",
        f"final_state = {final_state_file}",
    ]
    return "\n".join(lines) + "\n"


def run_athena(athena_binary, case, initial_primitives, period, work_directory):
    """
    Run a case with Athena++ from astronomix's initial primitive state.

    Args:
        athena_binary: The Athena++ executable.
        case: The comparison case.
        initial_primitives: The initial primitive state ``(5, nx, ny, nz)``.
        period: The end time.
        work_directory: The directory for the input and state files.

    Returns:
        ``(final primitives, cycles, seconds)``.
    """
    case_directory = Path(work_directory) / case.name
    case_directory.mkdir(parents=True, exist_ok=True)
    initial_state_file = case_directory / "initial_state.bin"
    final_state_file = case_directory / "final_state.bin"
    np.ascontiguousarray(initial_primitives, dtype="<f8").tofile(initial_state_file)
    input_file = case_directory / "athinput"
    input_file.write_text(athena_input(case, period, initial_state_file, final_state_file))

    start = time.perf_counter()
    subprocess.run(
        [str(athena_binary), "-i", str(input_file)],
        cwd=case_directory,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    seconds = time.perf_counter() - start

    values = np.fromfile(final_state_file, dtype="<f8")
    cycles = int(values[0])
    final_primitives = values[2:].reshape((5, *case.num_cells))
    return final_primitives, cycles, seconds


# -------------------------------------------------------------
# ====================== ↑ The two codes ↑ ====================
# -------------------------------------------------------------


def perturbation_relative_l1(state, reference, reference_scale):
    """
    Per-variable L1 difference relative to the L1 norm of the reference's
    perturbation (density and pressure: deviation from the mean; velocities:
    the field).

    Args:
        state: The compared primitive state ``(5, nx, ny, nz)``.
        reference: The reference primitive state.
        reference_scale: The primitive state that sets the normalisation.

    Returns:
        A dict of the relative L1 difference per variable.
    """
    differences = {}
    for index, name in enumerate(VARIABLE_NAMES):
        scale_field = reference_scale[index]
        if name in ("density", "pressure"):
            scale_field = scale_field - scale_field.mean()
        scale = np.sum(np.abs(scale_field))
        differences[name] = float(np.sum(np.abs(state[index] - reference[index])) / scale)
    return differences


def compare_case(athena_binary, case, work_directory):
    """
    Run one case with both codes and compare.

    Args:
        athena_binary: The Athena++ executable.
        case: The comparison case.
        work_directory: The directory for Athena++'s files.

    Returns:
        A dict of the comparison metrics.
    """
    initial, discrete_final, discrete_cycles, period = run_astronomix(case, discrete_laplacian=True)
    _, spectral_final, spectral_cycles, _ = run_astronomix(case, discrete_laplacian=False)
    athena_final, athena_cycles, athena_seconds = run_athena(
        athena_binary,
        case,
        initial,
        period,
        work_directory,
    )
    # The velocity of the slab is uniform; its scale is the field itself.
    return dict(
        case=case.name,
        cycles=dict(athena=athena_cycles, astronomix_discrete=discrete_cycles, astronomix_spectral=spectral_cycles),
        athena_seconds=athena_seconds,
        discrete_kernel_vs_athena=perturbation_relative_l1(discrete_final, athena_final, initial),
        spectral_kernel_vs_athena=perturbation_relative_l1(spectral_final, athena_final, initial),
        discrete_kernel_truncation_error=perturbation_relative_l1(discrete_final, initial, initial),
        athena_truncation_error=perturbation_relative_l1(athena_final, initial, initial),
    )


def write_regression_data(athena_binary, work_directory):
    """
    Run the small regression cases with Athena++ and store them.

    Args:
        athena_binary: The Athena++ executable.
        work_directory: The directory for Athena++'s files.
    """
    REGRESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for case in REGRESSION_CASES:
        _, params, _, initial_state = astronomix_setup(case, discrete_laplacian=True)
        initial = np.asarray(initial_state)
        final, cycles, _ = run_athena(athena_binary, case, initial, float(params.t_end), work_directory)
        data_file = REGRESSION_DATA_DIR / f"{case.name}.npz"
        np.savez(
            data_file,
            initial_state=initial,
            final_state=final,
            cycles=cycles,
            t_end=float(params.t_end),
            box_size=np.array(case.box_size, dtype=float),
            gravitational_constant=case.gravitational_constant,
            courant_number=COURANT_NUMBER,
            gamma=GAMMA,
        )
        print(f"{case.name}: {cycles} cycles -> {data_file}")


def main():
    """Run all cases and write the comparison (or the regression data)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--athena", required=True, help="the Athena++ executable")
    parser.add_argument("--work-dir", default="/e/scratch/astronomix/revision/athenapp_self_gravity")
    parser.add_argument(
        "--regression-data",
        action="store_true",
        help="store the small regression cases for the pytest instead",
    )
    args = parser.parse_args()

    if args.regression_data:
        write_regression_data(Path(args.athena), args.work_dir)
        return

    all_metrics = []
    for case in CASES:
        metrics = compare_case(Path(args.athena), case, args.work_dir)
        all_metrics.append(metrics)
        print(f"{case.name}: cycles {metrics['cycles']}")
        for key in (
            "discrete_kernel_vs_athena",
            "spectral_kernel_vs_athena",
            "discrete_kernel_truncation_error",
            "athena_truncation_error",
        ):
            values = ", ".join(f"{name} {value:.2e}" for name, value in metrics[key].items())
            print(f"  {key:34s} {values}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = RESULTS_DIR / f"comparison_{jax.devices()[0].platform}.json"
    results_file.write_text(json.dumps(dict(device=str(jax.devices()[0]), cases=all_metrics), indent=2))
    print(f"Saved -> {results_file}")


if __name__ == "__main__":
    main()
