"""
Finite-difference baseline for AD gradients through shocks.

The smooth test in ``sensitivity.py`` validates AD against the exact gradient
of the linearized Euler equations; it cannot reach the shock regime, where AD
must still produce a correct (sub-)gradient of the *discrete* cost J(theta).
This script is the complementary check. The Sod shock tube with open
boundaries is parameterized by six scalars

    theta = (rho_L, p_L, u_L, rho_R, p_R, u_R),

and the cost J(theta) = 1/2 int (rho^2 + v^2 + p^2) dx of the final state is
differentiated through the solvers by AD and compared with finite differences
of the same J(theta):

* The initial condition ``jnp.where(x < x_split, left, right)`` is exactly
  linear in theta, so the initial-condition derivative is clean.
* A fixed time step removes the adaptive-CFL kinks: otherwise the step count
  is piecewise constant in theta and puts O(dt) jumps into J that AD and
  finite differences resolve differently. What remains are the limiter /
  Riemann-solver switches.
* At such a switch AD returns the derivative of one branch, while central
  differences average both. The per-parameter certificate is therefore the
  minimum over the two one-sided differences,
  min(|g_AD - g_F|, |g_AD - g_B|), which falls at the one-sided truncation
  rate O(h) when AD matches one of the two slopes (the step-sweep figure).
* A random direction generically misses the codimension-one kink surfaces, so
  along it J is smooth and central differences resolve v . grad J to
  O(h^2) down to round-off, which arbitrates AD correctness in the shock
  regime (printed). This needs theta0 itself off the kink surfaces. For the
  finite-difference solver it is not: the Lax-Friedrichs splitting speed
  max(|u| + c) has a kink at u = 0, and Sod starts at rest (u_L = u_R = 0),
  so every direction crosses it and the directional residual stays at the
  size of that kink (~1e-4) instead of reaching round-off; the
  min-one-sided certificate is unaffected.

Schemes: the finite-difference WENO-Z solver and the finite-volume VL2 scheme (AthenaPK's van Leer integrator, PLM, HLLC),
each on the native-JAX and the Pallas backend (Pallas primal, native tangent).
Figures: ``figures/shock_tube_sensitivity_step_sweep.svg`` (paper) and
``figures/shock_tube_sensitivity.svg``; numbers in
``data/shock_tube_sensitivity.npz``, from which ``--plot-only`` redraws the
figures.
"""

# ==== GPU selection ====
import sys as _sys
if "--plot-only" not in _sys.argv:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# typing
from typing import NamedTuple

# jax
import jax
import jax.numpy as jnp

# Finite differences down to h = 1e-6 (one-sided) and 1e-8 (directional)
# need double precision.
jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    BACKWARDS,
    CARTESIAN,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLLC,
    NATIVE_JAX,
    OPEN_BOUNDARY,
    PALLAS,
    VL2,
)

from astronomix.plotting_helpers.paper_style import (
    FD_COLOR,
    FV_COLOR,
    GUIDE_COLOR,
    SINGLE_PANEL_WIDTH,
    TEXT_WIDTH,
    layout_path,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings1D,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.plotting_helpers.paper_style import (
    use_paper_style,
    variant_style,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
FIGURES_DIRECTORY = SCRIPT_DIRECTORY / "figures"
DATA_DIRECTORY = SCRIPT_DIRECTORY / "data"

PARAMETER_NAMES = ["rho_L", "p_L", "u_L", "rho_R", "p_R", "u_R"]
SOD_PARAMETERS = (1.0, 1.0, 0.0, 0.125, 0.1, 0.0)

NUM_CELLS = 200
BOX_LENGTH = 1.0
SPLIT_POSITION = 0.5
T_END = 0.2
# Fixed dt = 8e-4 (a Courant number of about 0.3 at the largest Sod signal
# speed), small enough that every perturbed state of the sweep stays stable.
NUM_TIMESTEPS = 250

STEP_SIZES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
# The random slices are smooth, so the directional test runs further, down to
# the round-off floor of central differences (~eps^(2/3) ~ 1e-11).
DIRECTIONAL_STEP_SIZES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
NUM_RANDOM_DIRECTIONS = 4


class Scheme(NamedTuple):
    """
    One solver configuration of the test.

    Attributes:
        label: The legend label.
        solver_mode: FINITE_DIFFERENCE or FINITE_VOLUME.
        backend: NATIVE_JAX or PALLAS.
        style: Matplotlib keyword arguments of the scheme's step-sweep curve.
    """

    label: str
    solver_mode: int
    backend: int
    style: dict


# The colours of the schemes in all figures of the paper; the Pallas runs are
# drawn as filled markers on solid lines on top of the native-JAX runs, which
# are the dashed, hollow variant, so that both stay visible where they
# coincide.
SCHEMES = (
    Scheme(
        "FD WENO-Z (JAX)",
        FINITE_DIFFERENCE,
        NATIVE_JAX,
        dict(marker="o", markersize=7, zorder=2, **variant_style(FD_COLOR)),
    ),
    Scheme(
        "FD WENO-Z (Pallas)",
        FINITE_DIFFERENCE,
        PALLAS,
        dict(color=FD_COLOR, linestyle="-", marker="o", zorder=3),
    ),
    Scheme(
        "FV VL2 (JAX)",
        FINITE_VOLUME,
        NATIVE_JAX,
        dict(marker="s", markersize=7, zorder=2, **variant_style(FV_COLOR)),
    ),
    Scheme(
        "FV VL2 (Pallas)",
        FINITE_VOLUME,
        PALLAS,
        dict(color=FV_COLOR, linestyle="-", marker="s", zorder=3),
    ),
)


# -------------------------------------------------------------
# ============== ↓ Configuration and parameterized IC ↓ =======
# -------------------------------------------------------------


def make_config_and_params(scheme):
    """
    The 1D open-boundary shock tube with a fixed time step.

    Under an adaptive CFL step the step count n(theta) = ceil(t_end /
    dt(theta)) is piecewise constant in theta, and crossing a threshold puts an
    O(dt) jump into J(theta). The AD gradient is unaffected (it differentiates
    the smooth branch), but the finite-difference baseline cannot tell such a
    jump from a derivative. A fixed dt isolates the limiter / Riemann-solver
    switches, the only non-smoothness this test is meant to probe.

    The Pallas block (4, 1, 1) divides both the padded finite-difference grid
    (200 + 2 x 4 ghost cells) and the VL2 interior (200); the default block
    of 128 cells would not, and the run would fall back to the native kernels.

    Args:
        scheme: The Scheme (solver and backend).

    Returns:
        The tuple ``(config, params)``.
    """
    if scheme.solver_mode == FINITE_VOLUME:
        scheme_options = dict(time_integrator=VL2, riemann_solver=HLLC)
    else:
        scheme_options = dict(weno_z=True)

    config = SimulationConfig(
        solver_mode=scheme.solver_mode,
        geometry=CARTESIAN,
        progress_bar=False,
        differentiation_mode=BACKWARDS,
        backend_config=BackendConfig(
            backend=scheme.backend,
            pallas_block_shape=(4, 1, 1),
        ),
        mhd=False,
        dimensionality=1,
        box_size=BOX_LENGTH,
        num_cells=NUM_CELLS,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY,
            right_boundary=OPEN_BOUNDARY,
        ),
        return_snapshots=False,
        fixed_timestep=True,
        num_timesteps=NUM_TIMESTEPS,
        **scheme_options,
    )
    params = SimulationParams(
        C_cfl=0.4,
        t_end=T_END,
        gamma=5.0 / 3.0,
        gravitational_constant=0.0,
    )
    return config, params


def build_initial_state(theta, config, registered_variables, helper_data):
    """
    The Sod-type initial state of the parameters theta = (rho_L, p_L, u_L,
    rho_R, p_R, u_R): cells left of the split take the left state, the rest
    the right state.

    Args:
        theta: The six parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        helper_data: The helper data (cell centres).

    Returns:
        The primitive initial state.
    """
    left_density, left_pressure, left_velocity, right_density, right_pressure, right_velocity = theta
    cell_centers = jnp.squeeze(helper_data.geometric_centers)
    is_left = cell_centers < SPLIT_POSITION
    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.where(is_left, left_density, right_density),
        velocity_x=jnp.where(is_left, left_velocity, right_velocity),
        gas_pressure=jnp.where(is_left, left_pressure, right_pressure),
    )


def cost(theta, config, params, registered_variables, helper_data):
    """
    J(theta) = 1/2 sum_x (rho^2 + v^2 + p^2) dx of the final state.

    Args:
        theta: The six parameters.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        helper_data: The helper data.

    Returns:
        The scalar cost.
    """
    initial_state = build_initial_state(theta, config, registered_variables, helper_data)
    config_final = finalize_config(config, initial_state.shape)
    final_state = time_integration(initial_state, config_final, params, registered_variables)
    density = final_state[registered_variables.density_index]
    velocity = final_state[registered_variables.velocity_index]
    pressure = final_state[registered_variables.pressure_index]
    grid_spacing = BOX_LENGTH / NUM_CELLS
    return 0.5 * jnp.sum(density**2 + velocity**2 + pressure**2) * grid_spacing


# -------------------------------------------------------------
# ============== ↑ Configuration and parameterized IC ↑ =======
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================ ↓ Finite-difference baselines ↓ ============
# -------------------------------------------------------------


def axis_finite_differences(theta, step_size, jitted_cost, cost_at_theta):
    """
    Central, forward and backward differences of J along every parameter axis
    for one step size (two cost evaluations per parameter plus the shared
    J(theta)).

    Forward differences approximate the right derivative, backward ones the
    left derivative. At a kink AD has picked one side; central differences
    average both and cannot match AD there, the closer one-sided difference
    can.

    Args:
        theta: The parameters.
        step_size: The finite-difference step h.
        jitted_cost: The compiled cost function.
        cost_at_theta: J(theta).

    Returns:
        The tuple ``(central, forward, backward)`` of gradient arrays.
    """
    central = np.zeros(theta.size)
    forward = np.zeros(theta.size)
    backward = np.zeros(theta.size)
    for parameter_index in range(theta.size):
        unit_vector = jnp.zeros_like(theta).at[parameter_index].set(1.0)
        cost_plus = float(jitted_cost(theta + step_size * unit_vector))
        cost_minus = float(jitted_cost(theta - step_size * unit_vector))
        central[parameter_index] = (cost_plus - cost_minus) / (2.0 * step_size)
        forward[parameter_index] = (cost_plus - cost_at_theta) / step_size
        backward[parameter_index] = (cost_at_theta - cost_minus) / step_size
    return central, forward, backward


def directional_differences(theta, ad_gradient, jitted_cost, step_sizes, num_directions, seed):
    """
    Relative difference between v . grad J (AD) and the central difference of
    J along random unit vectors v.

    The kinks of J(theta) lie on codimension-one surfaces, so a random v
    generically misses them; along v, J is smooth, and central differences
    resolve the directional derivative with O(h^2) truncation down to the
    round-off floor.

    Args:
        theta: The parameters.
        ad_gradient: The AD gradient at theta.
        jitted_cost: The compiled cost function.
        step_sizes: The finite-difference steps h.
        num_directions: The number of random directions.
        seed: The seed of the directions.

    Returns:
        An array (num_directions, len(step_sizes)) of relative differences.
    """
    random_generator = np.random.default_rng(seed)
    relative_differences = np.zeros((num_directions, len(step_sizes)))
    for direction_index in range(num_directions):
        direction = random_generator.normal(size=theta.size)
        direction = direction / np.linalg.norm(direction)
        ad_directional_derivative = float(np.dot(ad_gradient, direction))
        for step_index, step_size in enumerate(step_sizes):
            cost_plus = float(jitted_cost(theta + step_size * jnp.asarray(direction)))
            cost_minus = float(jitted_cost(theta - step_size * jnp.asarray(direction)))
            central_difference = (cost_plus - cost_minus) / (2.0 * step_size)
            relative_differences[direction_index, step_index] = abs(
                central_difference - ad_directional_derivative
            ) / (abs(ad_directional_derivative) + 1e-30)
    return relative_differences


def evaluate_scheme(scheme, theta):
    """
    The AD gradient, the axis-aligned finite differences for every step size
    and the random-direction check of one scheme.

    Args:
        scheme: The Scheme.
        theta: The parameters.

    Returns:
        A dict with the AD gradient, the central / forward / backward
        differences per step size, the relative central and min-one-sided
        residuals per step size and the directional residuals.
    """
    config, params = make_config_and_params(scheme)
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    def scheme_cost(parameters):
        return cost(parameters, config, params, registered_variables, helper_data)

    jitted_cost = jax.jit(scheme_cost)
    jitted_gradient = jax.jit(jax.grad(scheme_cost))

    cost_at_theta = float(jitted_cost(theta))
    ad_gradient = np.asarray(jitted_gradient(theta))
    print(f"  J(theta0) = {cost_at_theta:.10e}")
    for name, gradient_component in zip(PARAMETER_NAMES, ad_gradient):
        print(f"    dJ/d{name:<6s} = {gradient_component: .10e}")

    central, forward, backward = [], [], []
    central_residual, min_one_sided_residual = [], []
    ad_norm = np.linalg.norm(ad_gradient) + 1e-30
    for step_size in STEP_SIZES:
        central_h, forward_h, backward_h = axis_finite_differences(theta, step_size, jitted_cost, cost_at_theta)
        central.append(central_h)
        forward.append(forward_h)
        backward.append(backward_h)
        central_residual.append(np.linalg.norm(ad_gradient - central_h) / ad_norm)
        closer_one_sided = np.minimum(np.abs(ad_gradient - forward_h), np.abs(ad_gradient - backward_h))
        min_one_sided_residual.append(np.linalg.norm(closer_one_sided) / ad_norm)
        print(
            f"  h = {step_size:.0e}: central {central_residual[-1]:.3e}, "
            f"min one-sided {min_one_sided_residual[-1]:.3e}"
        )

    directional = directional_differences(
        theta,
        ad_gradient,
        jitted_cost,
        DIRECTIONAL_STEP_SIZES,
        NUM_RANDOM_DIRECTIONS,
        seed=0,
    )
    best_per_direction = directional.min(axis=1)
    print(
        f"  random directions, best-of-h |FD - AD| / |AD|: min {best_per_direction.min():.3e}, "
        f"max {best_per_direction.max():.3e}"
    )

    return dict(
        ad_gradient=ad_gradient,
        central=np.array(central),
        forward=np.array(forward),
        backward=np.array(backward),
        central_residual=np.array(central_residual),
        min_one_sided_residual=np.array(min_one_sided_residual),
        directional_residual=directional,
    )


# -------------------------------------------------------------
# ================ ↑ Finite-difference baselines ↑ ============
# -------------------------------------------------------------


# -------------------------------------------------------------
# ========================= ↓ Figures ↓ =======================
# -------------------------------------------------------------


def plot_per_parameter(results):
    """
    Per scheme: the AD gradient with the closer one-sided difference per
    parameter (top) and the per-parameter relative residual (bottom), for
    h = 1e-4 and 1e-6. Writes ``figures/shock_tube_sensitivity.svg``.

    Args:
        results: The per-scheme dicts of :func:`evaluate_scheme`.
    """
    figure, axes = plt.subplots(2, len(SCHEMES), figsize=(7 * len(SCHEMES), 9), sharex="col")
    parameter_positions = np.arange(len(PARAMETER_NAMES))
    plotted_step_sizes = [1e-4, 1e-6]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(plotted_step_sizes)))

    for column, scheme in enumerate(SCHEMES):
        result = results[scheme.label]
        ad_gradient = result["ad_gradient"]
        top_axis, bottom_axis = axes[0, column], axes[1, column]
        for color, step_size in zip(colors, plotted_step_sizes):
            step_index = STEP_SIZES.index(step_size)
            forward = result["forward"][step_index]
            backward = result["backward"][step_index]
            forward_is_closer = np.abs(ad_gradient - forward) <= np.abs(ad_gradient - backward)
            closer_one_sided = np.where(forward_is_closer, forward, backward)
            relative_residual = np.abs(ad_gradient - closer_one_sided) / (np.abs(ad_gradient) + 1e-30)
            marker_style = dict(
                color=color,
                s=80,
                alpha=0.85,
                marker="^",
                edgecolors="black",
                linewidths=0.6,
                label=f"closer one-sided FD, h = {step_size:.0e}",
            )
            top_axis.scatter(parameter_positions, closer_one_sided, **marker_style)
            bottom_axis.scatter(parameter_positions, relative_residual, **marker_style)
        top_axis.scatter(parameter_positions, ad_gradient, marker="x", s=180, color="black", linewidths=2.5, label="AD")
        top_axis.set_title(f"{scheme.label}: AD vs closer one-sided FD")
        top_axis.set_ylabel(r"$\partial J / \partial \theta_i$")
        top_axis.axhline(0, color="gray", linewidth=0.7)
        top_axis.grid(True, alpha=0.4)
        top_axis.legend(fontsize=8, loc="best")

        bottom_axis.set_yscale("log")
        bottom_axis.set_xticks(parameter_positions)
        bottom_axis.set_xticklabels(PARAMETER_NAMES, rotation=30)
        bottom_axis.set_ylabel(r"$|g_{\mathrm{AD},i} - g_{\mathrm{FD},i}| \, / \, |g_{\mathrm{AD},i}|$")
        bottom_axis.set_title(f"{scheme.label}: per-parameter relative residual")
        bottom_axis.grid(True, which="both", alpha=0.4)
        bottom_axis.legend(fontsize=8, loc="best")

    figure.tight_layout()
    figure_path = FIGURES_DIRECTORY / "shock_tube_sensitivity.svg"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {figure_path}")


def plot_step_sweep(results):
    """
    The min-one-sided residual over the step size h for every scheme, with
    the O(h) truncation of one-sided differences as reference. Writes
    ``figures/shock_tube_sensitivity_step_sweep.svg`` (paper figure).

    Args:
        results: The per-scheme dicts of :func:`evaluate_scheme`.
    """
    use_paper_style()
    figure, axis = plt.subplots(1, 1, figsize=(SINGLE_PANEL_WIDTH, 3.1), layout="constrained")
    step_size_array = np.array(STEP_SIZES)
    for scheme in SCHEMES:
        axis.loglog(
            step_size_array,
            results[scheme.label]["min_one_sided_residual"],
            label=scheme.label,
            **scheme.style,
        )

    # The O(h) reference, anchored at the smallest residual at the smallest h
    # so that it lies just under the data. Deviating from it would indicate
    # kink contamination or an AD error.
    smallest_residual = min(results[scheme.label]["min_one_sided_residual"][-1] for scheme in SCHEMES)
    axis.loglog(
        step_size_array,
        smallest_residual * step_size_array / step_size_array[-1],
        linestyle=":",
        color=GUIDE_COLOR,
        linewidth=1.0,
        label=r"$\propto h$",
    )
    axis.set_xlabel("finite difference step size $h$")
    axis.set_ylabel(r"$\| \min_{F,B} | g_{\mathrm{AD}} - g_{\mathrm{FD}} | \|_2 \, / \, \| g_{\mathrm{AD}} \|_2$")
    axis.legend(loc="upper left")
    figure_path = layout_path(FIGURES_DIRECTORY / "shock_tube_sensitivity_step_sweep.svg")
    figure.savefig(figure_path)
    plt.close(figure)
    print(f"saved {figure_path}")


def load_results():
    """
    The per-scheme results of :func:`evaluate_scheme` as stored in
    ``data/shock_tube_sensitivity.npz``.

    Returns:
        The per-scheme dicts, keyed by scheme label.
    """
    cached = np.load(DATA_DIRECTORY / "shock_tube_sensitivity.npz")
    results = {scheme.label: {} for scheme in SCHEMES}
    for key in cached.files:
        for scheme in SCHEMES:
            prefix = f"{scheme.label}_"
            if key.startswith(prefix):
                results[scheme.label][key[len(prefix):]] = cached[key]
    return results


# -------------------------------------------------------------
# ========================= ↑ Figures ↑ =======================
# -------------------------------------------------------------


def run_shock_tube_sensitivity_test():
    """Run every scheme, print the summary, write the figures and the data."""
    theta = jnp.array(SOD_PARAMETERS)
    results = {}
    for scheme in SCHEMES:
        print(f"\n=== {scheme.label} ===")
        results[scheme.label] = evaluate_scheme(scheme, theta)

    print("\n=== summary ===")
    for scheme in SCHEMES:
        result = results[scheme.label]
        best_central = int(np.argmin(result["central_residual"]))
        best_one_sided = int(np.argmin(result["min_one_sided_residual"]))
        print(
            f"  {scheme.label:24s}: central best {result['central_residual'][best_central]:.3e} "
            f"(h = {STEP_SIZES[best_central]:.0e}); min one-sided best "
            f"{result['min_one_sided_residual'][best_one_sided]:.3e} (h = {STEP_SIZES[best_one_sided]:.0e}); "
            f"random directions best-of-h max {result['directional_residual'].min(axis=1).max():.3e}"
        )

    # The JAX and Pallas gradients of a solver differ only through the
    # round-off of their primal states (the Pallas tangent is the native one).
    for jax_label, pallas_label in ((SCHEMES[0].label, SCHEMES[1].label), (SCHEMES[2].label, SCHEMES[3].label)):
        jax_gradient = results[jax_label]["ad_gradient"]
        pallas_gradient = results[pallas_label]["ad_gradient"]
        difference = np.linalg.norm(jax_gradient - pallas_gradient) / np.linalg.norm(jax_gradient)
        print(f"  |g({pallas_label}) - g({jax_label})| / |g| = {difference:.3e}")

    DATA_DIRECTORY.mkdir(exist_ok=True)
    np.savez(
        DATA_DIRECTORY / "shock_tube_sensitivity.npz",
        step_sizes=np.array(STEP_SIZES),
        directional_step_sizes=np.array(DIRECTIONAL_STEP_SIZES),
        **{
            f"{scheme.label}_{key}": value
            for scheme in SCHEMES
            for key, value in results[scheme.label].items()
        },
    )

    FIGURES_DIRECTORY.mkdir(exist_ok=True)
    plot_per_parameter(results)
    plot_step_sweep(results)


if __name__ == "__main__" and "--plot-only" in _sys.argv:
    plot_step_sweep(load_results())
elif __name__ == "__main__":
    run_shock_tube_sensitivity_test()
