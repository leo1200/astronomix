"""
Isothermal shock tubes of the finite-volume framework against the exact solution.

An isothermal gas (``p = c_s^2 rho``, no energy equation) has two waves, each a shock or a rarefaction, around one
middle state. The exact solution takes one scalar root for the middle density (``_exact_solution``). Two cases:
two shocks from colliding streams, and a rarefaction with a shock from a density jump. Each integrator (``VL2``,
``RK2_SSP``, ``MUSCL_HANCOCK``) with the HLL solver is run at 200 and 400 cells: the L1 errors of density and
velocity fall at first order (the discontinuities set the error) and the density stays positive. The local
Lax-Friedrichs solver runs the same tubes with a larger error.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_isothermal_shock_tube.py
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# astronomix constants
from astronomix import FINITE_VOLUME, OPEN_BOUNDARY
from astronomix.option_classes.simulation_config import (
    DONOR_CELL,
    HLL,
    ISOTHERMAL,
    LAX_FRIEDRICHS,
    MUSCL_HANCOCK,
    PIECEWISE_LINEAR,
    RK2_SSP,
    VL2,
)

# astronomix containers
from astronomix import (
    BoundarySettings1D,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)

SOUND_SPEED = 0.8
T_END = 0.15
INTEGRATORS = (("VL2", VL2), ("RK2_SSP", RK2_SSP), ("MUSCL_HANCOCK", MUSCL_HANCOCK))
# name, (density, velocity) on the left, (density, velocity) on the right
CASES = (
    ("two shocks", (1.0, 1.0), (1.0, -1.0)),
    ("rarefaction and shock", (1.0, 0.0), (0.25, 0.0)),
)


# -------------------------------------------------------------
# ================ ↓ Exact isothermal solution ↓ ==============
# -------------------------------------------------------------


def _velocity_jump(density, side_density):
    """Velocity change across one wave to the middle density: a shock above the side density, else a rarefaction."""
    if density > side_density:
        return SOUND_SPEED * (density - side_density) / np.sqrt(density * side_density)
    return SOUND_SPEED * np.log(density / side_density)


def _middle_state(left, right):
    """Middle density and velocity: the root of ``f_L(rho) + f_R(rho) + u_R - u_L`` by bisection in ``log rho``."""
    (left_density, left_velocity), (right_density, right_velocity) = left, right

    def mismatch(log_density):
        density = np.exp(log_density)
        jumps = _velocity_jump(density, left_density) + _velocity_jump(density, right_density)
        return jumps + right_velocity - left_velocity

    lower, upper = np.log(1e-12), np.log(1e12)
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        lower, upper = (middle, upper) if mismatch(middle) < 0.0 else (lower, middle)
    density = np.exp(0.5 * (lower + upper))
    return density, left_velocity - _velocity_jump(density, left_density)


def _exact_solution(left, right, speed):
    """Density and velocity of the isothermal Riemann problem at the similarity coordinate ``speed = x / t``."""
    (left_density, left_velocity), (right_density, right_velocity) = left, right
    middle_density, middle_velocity = _middle_state(left, right)
    density = np.full_like(speed, middle_density)
    velocity = np.full_like(speed, middle_velocity)

    if middle_density > left_density:
        ahead = speed < left_velocity - SOUND_SPEED * np.sqrt(middle_density / left_density)
    else:
        ahead = speed < left_velocity - SOUND_SPEED
        fan = ~ahead & (speed < middle_velocity - SOUND_SPEED)
        density = np.where(fan, left_density * np.exp((left_velocity - SOUND_SPEED - speed) / SOUND_SPEED), density)
        velocity = np.where(fan, speed + SOUND_SPEED, velocity)
    density, velocity = np.where(ahead, left_density, density), np.where(ahead, left_velocity, velocity)

    if middle_density > right_density:
        ahead = speed > right_velocity + SOUND_SPEED * np.sqrt(middle_density / right_density)
    else:
        ahead = speed > right_velocity + SOUND_SPEED
        fan = ~ahead & (speed > middle_velocity + SOUND_SPEED)
        density = np.where(fan, right_density * np.exp((speed - SOUND_SPEED - right_velocity) / SOUND_SPEED), density)
        velocity = np.where(fan, speed - SOUND_SPEED, velocity)
    return np.where(ahead, right_density, density), np.where(ahead, right_velocity, velocity)


# -------------------------------------------------------------
# ================ ↑ Exact isothermal solution ↑ ==============
# -------------------------------------------------------------


def _l1_errors(left, right, cells, time_integrator, riemann_solver=HLL, reconstruction=PIECEWISE_LINEAR):
    """L1 errors of density and velocity of one run, and its smallest density."""
    config = SimulationConfig(
        dimensionality=1,
        box_size=1.0,
        boundary_settings=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        solver_mode=FINITE_VOLUME,
        equation_of_state=ISOTHERMAL,
        solver_config=FVSolverConfig(
            time_integrator=time_integrator, riemann_solver=riemann_solver, reconstruction=reconstruction
        ),
        mhd=False,
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    registered_variables = get_registered_variables(config)
    x = (jnp.arange(cells) + 0.5) / cells
    initial_state = construct_primitive_state(
        config,
        registered_variables,
        density=jnp.where(x < 0.5, left[0], right[0]),
        velocity_x=jnp.where(x < 0.5, left[1], right[1]),
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=T_END, C_cfl=0.4, isothermal_sound_speed=SOUND_SPEED)
    final_state = np.asarray(time_integration(initial_state, config, params, registered_variables).final_state)
    assert final_state.shape == (2, cells) and np.all(np.isfinite(final_state))

    density, velocity = _exact_solution(left, right, (np.asarray(x) - 0.5) / T_END)
    return (
        float(np.abs(final_state[registered_variables.density_index] - density).mean()),
        float(np.abs(final_state[registered_variables.velocity_index] - velocity).mean()),
        float(final_state[registered_variables.density_index].min()),
    )


def test_exact_solution_of_the_two_shock_case():
    """Colliding streams of Mach number M: the middle density solves ``sqrt(r) - 1 / sqrt(r) = M``."""
    density, velocity = _middle_state((1.0, 1.0), (1.0, -1.0))
    mach = 1.0 / SOUND_SPEED
    assert abs(density - (0.5 * (mach + np.sqrt(mach**2 + 4.0))) ** 2) < 1e-12
    assert abs(velocity) < 1e-12


def test_isothermal_shock_tubes_converge_to_the_exact_solution():
    """HLL with each integrator: first-order L1 convergence of density and velocity, positive density."""
    for case, left, right in CASES:
        for name, time_integrator in INTEGRATORS:
            coarse = _l1_errors(left, right, 200, time_integrator)
            fine = _l1_errors(left, right, 400, time_integrator)
            density_order, velocity_order = np.log2(coarse[0] / fine[0]), np.log2(coarse[1] / fine[1])
            print(
                f"{case}, {name}: L1 density {coarse[0]:.3e} (200), {fine[0]:.3e} (400), order {density_order:.2f}; "
                f"L1 velocity {coarse[1]:.3e}, {fine[1]:.3e}, order {velocity_order:.2f}; "
                f"smallest density {min(coarse[2], fine[2]):.4f}"
            )
            assert min(coarse[2], fine[2]) > 0.0
            assert fine[0] < 1e-2 and fine[1] < 4e-3
            assert density_order > 0.7 and velocity_order > 0.7


def test_lax_friedrichs_runs_the_isothermal_shock_tubes():
    """The local Lax-Friedrichs solver (donor cell under VL2, as that scheme asks) stays close to the solution."""
    for case, left, right in CASES:
        for name, time_integrator in INTEGRATORS:
            reconstruction = DONOR_CELL if time_integrator == VL2 else PIECEWISE_LINEAR
            density_error, velocity_error, smallest_density = _l1_errors(
                left, right, 200, time_integrator, LAX_FRIEDRICHS, reconstruction
            )
            print(f"{case}, {name}, LAX_FRIEDRICHS: L1 density {density_error:.3e}, velocity {velocity_error:.3e}")
            assert smallest_density > 0.0
            assert density_error < 0.1 and velocity_error < 0.1


if __name__ == "__main__":
    test_exact_solution_of_the_two_shock_case()
    test_isothermal_shock_tubes_converge_to_the_exact_solution()
    test_lax_friedrichs_runs_the_isothermal_shock_tubes()
