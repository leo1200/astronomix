"""
Reverse-mode gradients against the exact functional derivative of the
linearized Euler equations.

A small density perturbation (amplitude 1e-6) on a uniform background evolves
as linear acoustics, whose time evolution operator S_t is known in Fourier
space. The functional derivative of J = 1/2 <S_t U_0, S_t U_0> with respect to
the full initial state U_0 = (rho_P, v_P) is therefore exact, and the
reverse-mode AD gradient through the solver must converge to it at the spatial
order of the scheme. Two paper figures come out of this script:

* ``figures/gradient_3d_gaussian.svg``: the full-field 3D Gaussian case (N = 64),
  a central slice of the AD gradient of every scheme against the exact one.
* ``figures/gradient_convergence_test.svg``: the L1 error of the gradient of the
  1D sinusoidal case over N = 16 ... 256.

The schemes are the finite-difference WENO-Z solver (native JAX and Pallas)
and the finite-volume VL2 scheme (AthenaPK's van Leer integrator, PLM, HLLC; native
JAX and Pallas). Every backend is set explicitly; the Pallas derivatives go
through the native-JAX tangent (``diffable_pallas_call``). The reduced numbers
are written to ``data/``; ``--plot-only`` redraws both figures from
``data/gradient_3d_gaussian.npz`` and ``data/gradient_convergence_test.npz``.
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

# Exact AD of a 1e-6 perturbation needs double precision: in single precision
# the perturbation is at the round-off level of the background state.
jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# astronomix constants
from astronomix import (
    BACKWARDS,
    CARTESIAN,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLLC,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    VL2,
)
from astronomix.option_classes.simulation_config import PERIODIC_ROLL
from astronomix.plotting_helpers.paper_style import (
    FD_COLOR,
    FV_COLOR,
    GUIDE_COLOR,
    REFERENCE_COLOR,
    COLUMN_WIDTH,
    SINGLE_PANEL_WIDTH,
    TEXT_WIDTH,
    TWO_COLUMN,
    layout_path,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
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
from astronomix.plotting_helpers.paper_style import (
    figure_legend,
    use_paper_style,
    variant_style,
)
from astronomix.plotting_helpers.power_law_indicators import add_power_law_indicators


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
FIGURES_DIRECTORY = SCRIPT_DIRECTORY / "figures"
DATA_DIRECTORY = SCRIPT_DIRECTORY / "data"

# The linear acoustic problem: background density, sound speed and adiabatic
# index; the background pressure follows from c_s^2 = gamma P_B / rho_B.
BACKGROUND_DENSITY = 1.0
SOUND_SPEED = 2.0
GAMMA = 5.0 / 3.0
BACKGROUND_PRESSURE = SOUND_SPEED**2 * BACKGROUND_DENSITY / GAMMA
PERTURBATION_AMPLITUDE = 1e-6


class Scheme(NamedTuple):
    """
    One solver configuration whose gradient is compared with the exact one.

    Attributes:
        label: The legend label.
        solver_mode: FINITE_DIFFERENCE or FINITE_VOLUME.
        backend: NATIVE_JAX or PALLAS.
        positivity_preserving: Positivity-preserving WENO (finite difference).
        style: Matplotlib keyword arguments of the scheme's curve or markers.
    """

    label: str
    solver_mode: int
    backend: int
    positivity_preserving: bool
    style: dict


# The colours of the schemes in all figures of the paper; the Pallas runs are
# drawn as filled markers on solid lines on top of the native-JAX runs, which
# are the dashed, hollow variant, so that both stay visible where they
# coincide.
CONVERGENCE_SCHEMES = (
    Scheme(
        "FD WENO-Z (JAX)",
        FINITE_DIFFERENCE,
        NATIVE_JAX,
        False,
        dict(marker="o", markersize=7, zorder=2, **variant_style(FD_COLOR)),
    ),
    Scheme(
        "FD WENO-Z (Pallas)",
        FINITE_DIFFERENCE,
        PALLAS,
        False,
        dict(color=FD_COLOR, marker="o", linestyle="-", zorder=3),
    ),
    Scheme(
        "FV VL2 (JAX)",
        FINITE_VOLUME,
        NATIVE_JAX,
        False,
        dict(marker="s", markersize=7, zorder=2, **variant_style(FV_COLOR)),
    ),
    Scheme(
        "FV VL2 (Pallas)",
        FINITE_VOLUME,
        PALLAS,
        False,
        dict(color=FV_COLOR, marker="s", linestyle="-", zorder=3),
    ),
)

GAUSSIAN_SCHEMES = (
    Scheme(
        "FD WENO-Z (JAX)",
        FINITE_DIFFERENCE,
        NATIVE_JAX,
        False,
        dict(edgecolors=FD_COLOR, facecolors="none", marker="o", s=24, linewidths=1.0, zorder=2),
    ),
    Scheme(
        "FD WENO-Z (Pallas)",
        FINITE_DIFFERENCE,
        PALLAS,
        False,
        dict(color=FD_COLOR, marker="x", s=14, linewidths=1.0, zorder=3),
    ),
    Scheme(
        "FV VL2 (JAX)",
        FINITE_VOLUME,
        NATIVE_JAX,
        False,
        dict(edgecolors=FV_COLOR, facecolors="none", marker="s", s=24, linewidths=1.0, zorder=2),
    ),
    Scheme(
        "FV VL2 (Pallas)",
        FINITE_VOLUME,
        PALLAS,
        False,
        dict(color=FV_COLOR, marker="x", s=14, linewidths=1.0, zorder=3),
    ),
)


# -------------------------------------------------------------
# =========== ↓ Exact Fourier-space functional derivative ↓ ===
# -------------------------------------------------------------


def compute_analytic_gradients_fourier(
    initial_density_perturbation,
    initial_velocity_perturbations,
    box_length,
    sound_speed,
    background_density,
    t_end,
    mass_matrix="first_order",
):
    """
    Exact functional derivative of J = 1/2 <S_t U_0, S_t U_0> for linear
    acoustics in N dimensions, evaluated with the Fourier-space operator S*S
    (derivation in the paper appendix).

    The ``mass_matrix`` argument selects the inner product of the discrete
    objective:

    - ``'first_order'``: J = 1/2 sum |U_bar|^2 dV (midpoint quadrature on the
      cell averages). The gradient is the Fourier multiplier applied to the DFT
      of the cell averages.
    - ``'fourier'``: J = 1/2 int u^2 dV with the spectrally accurate Parseval
      quadrature of the reconstructed point values. The same multiplier is then
      divided by prod_d sinc(k_d h_d / 2)^2, the Fourier-space Jacobian of the
      cell-average to point-value map.

    Args:
        initial_density_perturbation: rho_P at t = 0 (cell averages).
        initial_velocity_perturbations: The tuple of the velocity components
            v_P at t = 0.
        box_length: The (cubic) box length L.
        sound_speed: The background sound speed c_s.
        background_density: The background density rho_B.
        t_end: The evolution time t.
        mass_matrix: ``'first_order'`` or ``'fourier'`` (see above).

    Returns:
        The tuple ``(density_gradient, velocity_gradients)`` in real space.
    """
    dimensionality = initial_density_perturbation.ndim
    num_cells_per_axis = initial_density_perturbation.shape

    wavenumber_axes = [
        jnp.fft.fftfreq(num_cells_per_axis[axis], d=box_length / num_cells_per_axis[axis]) * 2 * jnp.pi
        for axis in range(dimensionality)
    ]
    wavevector = jnp.meshgrid(*wavenumber_axes, indexing="ij")
    wavenumber = jnp.sqrt(sum(component**2 for component in wavevector))
    angular_frequency = sound_speed * wavenumber

    # The unit wavevector; the k = 0 mode carries no acoustic coupling.
    unit_wavevector = [
        jnp.where(wavenumber > 0, component / wavenumber, 0.0) for component in wavevector
    ]

    density_hat = jnp.fft.fftn(initial_density_perturbation)
    velocity_hat = [jnp.fft.fftn(component) for component in initial_velocity_perturbations]
    longitudinal_velocity_hat = sum(
        unit_wavevector[axis] * velocity_hat[axis] for axis in range(dimensionality)
    )

    # The multipliers of S*S.
    cosine = jnp.cos(angular_frequency * t_end)
    sine = jnp.sin(angular_frequency * t_end)
    density_factor = cosine**2 + (sound_speed / background_density) ** 2 * sine**2
    cross_factor = 1j * sine * cosine * (
        background_density / sound_speed - sound_speed / background_density
    )
    longitudinal_factor = ((background_density / sound_speed) ** 2 - 1.0) * sine**2

    density_gradient_hat = density_factor * density_hat - cross_factor * longitudinal_velocity_hat
    velocity_gradient_hat = [
        cross_factor * density_hat * unit_wavevector[axis]
        + velocity_hat[axis]
        + longitudinal_factor * longitudinal_velocity_hat * unit_wavevector[axis]
        for axis in range(dimensionality)
    ]

    if mass_matrix == "fourier":
        # With the high-order mass matrix both the forward and the adjoint map
        # go through the cell-average <-> point-value transform sinc(k h / 2),
        # so the per-cell-average gradient is (S*S U_bar) / sinc^2.
        sinc_product = _sinc_cell_average_factor(num_cells_per_axis, box_length)
        density_gradient_hat = density_gradient_hat / sinc_product**2
        velocity_gradient_hat = [component / sinc_product**2 for component in velocity_gradient_hat]
    elif mass_matrix != "first_order":
        raise ValueError(f"Unknown mass_matrix option: {mass_matrix!r}")

    density_gradient = jnp.fft.ifftn(density_gradient_hat).real
    velocity_gradients = [jnp.fft.ifftn(component).real for component in velocity_gradient_hat]
    return density_gradient, velocity_gradients


def _sinc_cell_average_factor(num_cells_per_axis, box_length):
    """
    Return prod_d sinc(k_d h_d / 2), the Fourier multiplier that maps the point
    values of a smooth periodic function to its cell averages.
    """
    dimensionality = len(num_cells_per_axis)
    sinc_product = None
    for axis in range(dimensionality):
        grid_spacing = box_length / num_cells_per_axis[axis]
        frequency = jnp.fft.fftfreq(num_cells_per_axis[axis], d=grid_spacing)
        # jnp.sinc(x) = sin(pi x) / (pi x), and k h / 2 = pi * frequency * h.
        axis_factor = jnp.sinc(frequency * grid_spacing)
        broadcast_shape = [1] * dimensionality
        broadcast_shape[axis] = num_cells_per_axis[axis]
        axis_factor = axis_factor.reshape(broadcast_shape)
        sinc_product = axis_factor if sinc_product is None else sinc_product * axis_factor
    return sinc_product


def _spectral_squared_integral(field, box_length):
    """
    Spectrally accurate integral of u(x)^2 over the periodic box, given u as
    cell averages of a smooth function.

    The cell averages and the Fourier coefficients of u are related by
    U_bar_k = u_hat_k * prod_d sinc(k_d h_d / 2); dividing by that factor
    recovers the coefficients of the point-valued field, and Parseval gives the
    integral of its square. For WENO on smooth periodic data this is the
    matching high-order mass matrix.

    Args:
        field: The cell averages of u.
        box_length: The (cubic) box length.

    Returns:
        The integral of u^2 over the box.
    """
    num_cells_total = field.size
    box_volume = box_length**field.ndim
    point_value_hat = jnp.fft.fftn(field) / _sinc_cell_average_factor(field.shape, box_length)
    return (box_volume / num_cells_total**2) * jnp.sum(jnp.abs(point_value_hat) ** 2)


# -------------------------------------------------------------
# =========== ↑ Exact Fourier-space functional derivative ↑ ===
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================= ↓ Simulation setup and AD ↓ ===============
# -------------------------------------------------------------


def simulation_config_and_params(dimensionality, num_cells, box_length, t_end, scheme):
    """
    The fully periodic, reverse-mode configuration of one scheme.

    Args:
        dimensionality: 1, 2 or 3.
        num_cells: The number of cells per axis.
        box_length: The (cubic) box length.
        t_end: The evolution time.
        scheme: The Scheme (solver, backend, positivity preservation).

    Returns:
        The tuple ``(config, params)``.
    """
    periodic = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
    if dimensionality == 1:
        box_size = box_length
        num_cells_vector = num_cells
        boundary_settings = periodic
    else:
        box_size = StaticFloatVector(*([box_length] * dimensionality))
        num_cells_vector = StaticIntVector(*([num_cells] * dimensionality))
        boundary_settings = BoundarySettings(*([periodic] * dimensionality))

    if scheme.solver_mode == FINITE_VOLUME:
        solver_config = FVSolverConfig(time_integrator=VL2, riemann_solver=HLLC)
    else:
        solver_config = FDSolverConfig(
            weno_z=True,
            weno_positivity_preserving=scheme.positivity_preserving,
        )

    # Fully periodic: the arrays are rolled instead of padded with ghost cells
    # (finalize_config selects the same layout for FD and VL2).
    config = SimulationConfig(
        solver_mode=scheme.solver_mode,
        backend_config=BackendConfig(backend=scheme.backend),
        geometry=CARTESIAN,
        progress_bar=False,
        boundary_handling=PERIODIC_ROLL,
        differentiation_mode=BACKWARDS,
        num_ghost_cells=0,
        mhd=False,
        dimensionality=dimensionality,
        box_size=box_size,
        num_cells=num_cells_vector,
        boundary_settings=boundary_settings,
        return_snapshots=False,
        solver_config=solver_config,
    )

    # VL2 takes AthenaPK's CFL number in 3D (0.3; its time step uses the
    # largest single-axis signal speed, not their sum); everything else keeps
    # the 0.4 of the finite-difference runs.
    if scheme.solver_mode == FINITE_VOLUME and dimensionality == 3:
        cfl_number = 0.3
    else:
        cfl_number = 0.4

    params = SimulationParams(
        C_cfl=cfl_number,
        t_end=t_end,
        gamma=GAMMA,
        gravitational_constant=0.0,
    )
    return config, params


def initial_perturbation(helper_data, dimensionality, ic_type, box_length):
    """
    The initial perturbation: an axis-angled sine wave or a centred Gaussian
    density perturbation, at rest.

    Args:
        helper_data: The helper data (cell centres).
        dimensionality: 1, 2 or 3.
        ic_type: ``'wave'`` (two wavelengths along each axis, k_d = 4 pi / L)
            or ``'gaussian'`` (sigma = L / 10).
        box_length: The (cubic) box length.

    Returns:
        The tuple ``(coordinates, density_perturbation, velocity_perturbations)``.
    """
    if dimensionality == 1:
        coordinates = [jnp.squeeze(helper_data.geometric_centers)]
    else:
        centers = jnp.moveaxis(helper_data.geometric_centers, -1, 0)
        coordinates = [centers[axis] for axis in range(dimensionality)]

    if ic_type == "wave":
        wavenumber = 2 * jnp.pi * 2 / box_length
        phase = sum(wavenumber * coordinates[axis] for axis in range(dimensionality))
        density_perturbation = PERTURBATION_AMPLITUDE * jnp.sin(phase)
    elif ic_type == "gaussian":
        squared_radius = sum((coordinates[axis] - box_length / 2) ** 2 for axis in range(dimensionality))
        width = 0.1 * box_length
        density_perturbation = PERTURBATION_AMPLITUDE * jnp.exp(-squared_radius / (2 * width**2))
    else:
        raise ValueError(f"Unknown ic_type: {ic_type!r}")

    velocity_perturbations = tuple(jnp.zeros_like(density_perturbation) for _ in range(dimensionality))
    return coordinates, density_perturbation, velocity_perturbations


def final_state_cost(
    density_perturbation,
    velocity_perturbations,
    config,
    params,
    box_length,
    mass_matrix="first_order",
):
    """
    Evolve the perturbed background and return J = 1/2 int (rho_P^2 + v_P^2) dV
    of the final state.

    Args:
        density_perturbation: The initial rho_P.
        velocity_perturbations: The tuple of initial velocity components.
        config: The simulation configuration.
        params: The simulation parameters.
        box_length: The (cubic) box length.
        mass_matrix: ``'first_order'`` (midpoint sum over the cell averages)
            or ``'fourier'`` (sinc-corrected spectral quadrature).

    Returns:
        The scalar cost J.
    """
    registered_variables = get_registered_variables(config)
    dimensionality = config.dimensionality
    cell_volume = (box_length / density_perturbation.shape[0]) ** dimensionality

    # The perturbation is an acoustic one: p_P = c_s^2 rho_P.
    velocity_names = ["velocity_x", "velocity_y", "velocity_z"][:dimensionality]
    initial_state = construct_primitive_state(
        config,
        registered_variables,
        density=BACKGROUND_DENSITY + density_perturbation,
        gas_pressure=BACKGROUND_PRESSURE + SOUND_SPEED**2 * density_perturbation,
        **dict(zip(velocity_names, velocity_perturbations)),
    )
    config_final = finalize_config(config, initial_state.shape)
    final_state = time_integration(initial_state, config_final, params, registered_variables)

    final_density_perturbation = final_state[registered_variables.density_index] - BACKGROUND_DENSITY
    if dimensionality == 1:
        final_velocities = [final_state[registered_variables.velocity_index]]
    else:
        velocity_index = registered_variables.velocity_index
        velocity_indices = (velocity_index.x, velocity_index.y, velocity_index.z)[:dimensionality]
        final_velocities = [final_state[index] for index in velocity_indices]

    if mass_matrix == "first_order":
        squared_norm = final_density_perturbation**2 + sum(velocity**2 for velocity in final_velocities)
        return 0.5 * jnp.sum(squared_norm) * cell_volume
    if mass_matrix == "fourier":
        return 0.5 * sum(
            _spectral_squared_integral(field, box_length)
            for field in [final_density_perturbation] + final_velocities
        )
    raise ValueError(f"Unknown mass_matrix option: {mass_matrix!r}")


def ad_gradients(density_perturbation, velocity_perturbations, config, params, box_length):
    """
    Reverse-mode AD gradient of the cost with respect to the initial state,
    divided by the cell volume so that it approximates the functional
    derivative (the gradient of a midpoint quadrature carries a factor dV).

    Args:
        density_perturbation: The initial rho_P.
        velocity_perturbations: The tuple of initial velocity components.
        config: The simulation configuration.
        params: The simulation parameters.
        box_length: The (cubic) box length.

    Returns:
        The tuple ``(density_gradient, velocity_gradients)``.
    """
    cell_volume = (box_length / density_perturbation.shape[0]) ** config.dimensionality

    def cost(density, velocities):
        return final_state_cost(density, velocities, config, params, box_length)

    density_gradient, velocity_gradients = jax.jit(jax.grad(cost, argnums=(0, 1)))(
        density_perturbation,
        velocity_perturbations,
    )
    return density_gradient / cell_volume, [gradient / cell_volume for gradient in velocity_gradients]


# -------------------------------------------------------------
# ================= ↑ Simulation setup and AD ↑ ===============
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============ ↓ Full-field gradient (3D Gaussian) ↓ ==========
# -------------------------------------------------------------


def run_gaussian_gradient_test(dimensionality=3, num_cells=64):
    """
    The full-field Gaussian case (L = 10, t = 1.5, by which time the blob has
    split into outgoing acoustic waves): the AD gradient of every scheme
    against the exact one, as a central slice along x. Writes
    ``figures/gradient_{dimensionality}d_gaussian.svg`` and the slices to
    ``data/gradient_{dimensionality}d_gaussian.npz``.

    Args:
        dimensionality: 1, 2 or 3.
        num_cells: The number of cells per axis.
    """
    box_length, t_end = 10.0, 1.5
    print(f"\n=== {dimensionality}D Gaussian: AD vs exact gradient (N = {num_cells}) ===")

    reference_config, _ = simulation_config_and_params(
        dimensionality,
        num_cells,
        box_length,
        t_end,
        GAUSSIAN_SCHEMES[0],
    )
    coordinates, density_perturbation, velocity_perturbations = initial_perturbation(
        get_helper_data(reference_config),
        dimensionality,
        "gaussian",
        box_length,
    )
    exact_density_gradient, exact_velocity_gradients = compute_analytic_gradients_fourier(
        density_perturbation,
        velocity_perturbations,
        box_length,
        SOUND_SPEED,
        BACKGROUND_DENSITY,
        t_end,
    )
    exact_norm = jnp.sqrt(
        jnp.sum(exact_density_gradient**2)
        + sum(jnp.sum(gradient**2) for gradient in exact_velocity_gradients)
    )

    # The central slice along x through the box centre, normalised by the
    # perturbation amplitude.
    central_slice = (slice(None),) + (num_cells // 2,) * (dimensionality - 1)
    x_axis = np.asarray(coordinates[0][(slice(None),) + (0,) * (dimensionality - 1)])
    slices = {
        "x": x_axis,
        "exact_density": np.asarray(exact_density_gradient[central_slice]) / PERTURBATION_AMPLITUDE,
        "exact_velocity_x": np.asarray(exact_velocity_gradients[0][central_slice]) / PERTURBATION_AMPLITUDE,
    }

    for scheme in GAUSSIAN_SCHEMES:
        config, params = simulation_config_and_params(dimensionality, num_cells, box_length, t_end, scheme)
        density_gradient, velocity_gradients = ad_gradients(
            density_perturbation,
            velocity_perturbations,
            config,
            params,
            box_length,
        )
        error_norm = jnp.sqrt(
            jnp.sum((density_gradient - exact_density_gradient) ** 2)
            + sum(
                jnp.sum((ad_gradient - exact_gradient) ** 2)
                for ad_gradient, exact_gradient in zip(velocity_gradients, exact_velocity_gradients)
            )
        )
        relative_error = float(error_norm / exact_norm)
        print(f"  {scheme.label:22s}: ||g_AD - g_exact||_2 / ||g_exact||_2 = {relative_error:.3e}")
        slices[f"{scheme.label}_density"] = np.asarray(density_gradient[central_slice]) / PERTURBATION_AMPLITUDE
        slices[f"{scheme.label}_velocity_x"] = (
            np.asarray(velocity_gradients[0][central_slice]) / PERTURBATION_AMPLITUDE
        )
        slices[f"{scheme.label}_relative_l2_error"] = relative_error

    # The JAX and Pallas gradients differ only through the round-off of their
    # primal states, since the Pallas tangent is the native one.
    for solver_prefix in ("FD WENO-Z", "FV VL2"):
        jax_slice = slices[f"{solver_prefix} (JAX)_density"]
        pallas_slice = slices[f"{solver_prefix} (Pallas)_density"]
        difference = np.max(np.abs(jax_slice - pallas_slice)) / np.max(np.abs(jax_slice))
        print(f"  {solver_prefix}: max |JAX - Pallas| / max |JAX| on the slice = {difference:.3e}")

    DATA_DIRECTORY.mkdir(exist_ok=True)
    np.savez(DATA_DIRECTORY / f"gradient_{dimensionality}d_gaussian.npz", **slices)

    plot_gaussian_gradient(slices, dimensionality)


def plot_gaussian_gradient(slices, dimensionality):
    """
    Draw ``figures/gradient_{dimensionality}d_gaussian.svg``: the central slice
    of the AD gradient of every scheme against the exact one, for the initial
    density (left) and velocity (right).

    Args:
        slices: The slices written by ``run_gaussian_gradient_test`` (or
            loaded from its ``.npz``).
        dimensionality: 1, 2 or 3.
    """
    use_paper_style()
    # Side by side in the single-column layout, stacked in one column in the
    # two-column layout.
    if TWO_COLUMN:
        figure, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 5.0), sharex=True, layout="constrained")
    else:
        figure, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9), sharex=True, layout="constrained")
    panels = (
        ("density", "exact_density", r"$\partial J / \partial \rho_0$"),
        ("velocity_x", "exact_velocity_x", r"$\partial J / \partial v_{x,0}$"),
    )
    x_axis = slices["x"]
    for axis, (field, exact_key, title) in zip(axes, panels):
        axis.plot(x_axis, slices[exact_key], "-", color=REFERENCE_COLOR, linewidth=1.0, label="exact gradient")
        for scheme in GAUSSIAN_SCHEMES:
            axis.scatter(
                x_axis,
                slices[f"{scheme.label}_{field}"],
                label=f"AD: {scheme.label}",
                **scheme.style,
            )
        axis.set_title(title)
        if TWO_COLUMN:
            axis.set_ylabel("gradient")
        else:
            axis.set_xlabel("$x$ (central slice)")
    axes[-1].set_xlabel("$x$ (central slice)")
    axes[0].set_ylabel("gradient")
    figure_legend(figure, ncols=2 if TWO_COLUMN else 3)
    FIGURES_DIRECTORY.mkdir(exist_ok=True)
    figure_path = layout_path(FIGURES_DIRECTORY / f"gradient_{dimensionality}d_gaussian.svg")
    figure.savefig(figure_path)
    plt.close(figure)
    print(f"  saved {figure_path}")


# -------------------------------------------------------------
# ============ ↑ Full-field gradient (3D Gaussian) ↑ ==========
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============= ↓ Convergence of the AD gradient ↓ ============
# -------------------------------------------------------------


def run_gradient_convergence_test(num_cells_values=(16, 32, 64, 128, 256)):
    """
    Mean L1 error of the AD gradient of the 1D sinusoidal case (L = 1,
    t = 0.15) against the exact gradient over the resolution, for every
    scheme. Writes ``figures/gradient_convergence_test.svg`` and
    ``data/gradient_convergence_test.npz``.

    Args:
        num_cells_values: The resolutions N.
    """
    print("\n=== 1D sine wave: convergence of the AD gradient ===")
    box_length, t_end = 1.0, 0.15
    dimensionality = 1

    errors = {scheme.label: [] for scheme in CONVERGENCE_SCHEMES}
    for scheme in CONVERGENCE_SCHEMES:
        for num_cells in num_cells_values:
            config, params = simulation_config_and_params(dimensionality, num_cells, box_length, t_end, scheme)
            _, density_perturbation, velocity_perturbations = initial_perturbation(
                get_helper_data(config),
                dimensionality,
                "wave",
                box_length,
            )
            density_gradient, velocity_gradients = ad_gradients(
                density_perturbation,
                velocity_perturbations,
                config,
                params,
                box_length,
            )
            exact_density_gradient, exact_velocity_gradients = compute_analytic_gradients_fourier(
                density_perturbation,
                velocity_perturbations,
                box_length,
                SOUND_SPEED,
                BACKGROUND_DENSITY,
                t_end,
            )
            # The mean L1 error of the density and velocity gradients, relative
            # to the perturbation amplitude (the exact gradient is O(1) then).
            density_error = jnp.mean(jnp.abs(density_gradient - exact_density_gradient))
            velocity_error = jnp.mean(jnp.abs(velocity_gradients[0] - exact_velocity_gradients[0]))
            l1_error = float(0.5 * (density_error + velocity_error) / PERTURBATION_AMPLITUDE)
            errors[scheme.label].append(l1_error)
            print(f"  {scheme.label:22s} N = {num_cells:4d}: mean L1 error = {l1_error:.3e}")

    for label, error_list in errors.items():
        orders = np.log2(np.array(error_list[:-1]) / np.array(error_list[1:]))
        print(f"  {label:22s} observed orders: " + ", ".join(f"{order:.2f}" for order in orders))

    DATA_DIRECTORY.mkdir(exist_ok=True)
    np.savez(
        DATA_DIRECTORY / "gradient_convergence_test.npz",
        num_cells=np.array(num_cells_values),
        **{label: np.array(error_list) for label, error_list in errors.items()},
    )
    plot_gradient_convergence(num_cells_values, errors)


def plot_gradient_convergence(num_cells_values, errors):
    """
    Draw ``figures/gradient_convergence_test.svg``.

    Args:
        num_cells_values: The resolutions N.
        errors: The mean L1 gradient errors per scheme label.
    """
    use_paper_style()
    figure, axis = plt.subplots(1, 1, figsize=(SINGLE_PANEL_WIDTH, 3.1), layout="constrained")
    num_cells_array = np.array(num_cells_values)
    for scheme in CONVERGENCE_SCHEMES:
        axis.loglog(num_cells_array, errors[scheme.label], label=scheme.label, **scheme.style)

    # Reference slopes next to the curves they describe, one octave long from
    # N = 48: -2 just above the finite-volume rows, -5 just below the
    # finite-difference rows (anchored on their N = 32 and N = 64 errors).
    anchor_resolution = 48.0
    finite_volume_scale = 2.0 * np.sqrt(errors["FV VL2 (JAX)"][1] * errors["FV VL2 (JAX)"][2])
    finite_difference_scale = 0.3 * np.sqrt(errors["FD WENO-Z (JAX)"][1] * errors["FD WENO-Z (JAX)"][2])
    add_power_law_indicators(
        axis,
        anchor=(anchor_resolution, 1.0),
        exponents=[-2, -5],
        scales=[finite_volume_scale, finite_difference_scale],
        x_span=2.0,
        x_label="N",
        line_kwargs=dict(color=GUIDE_COLOR, linestyle="--", linewidth=1.0),
        text_kwargs=dict(fontsize=8, ha="left", va="center", color=GUIDE_COLOR),
    )

    axis.set_xlabel("$N$")
    axis.set_ylabel(r"mean $L_1$ error of $\partial J/\partial U_0$")
    # Integer N ticks only; the default log minor-tick labels (2x10^1, ...)
    # would overlap them.
    axis.set_xticks(list(num_cells_values))
    axis.set_xticklabels([str(num_cells) for num_cells in num_cells_values])
    axis.xaxis.set_minor_formatter(mticker.NullFormatter())
    axis.legend(loc="lower left")
    figure_path = layout_path(FIGURES_DIRECTORY / "gradient_convergence_test.svg")
    figure.savefig(figure_path)
    plt.close(figure)
    print(f"  saved {figure_path}")


# -------------------------------------------------------------
# ============= ↑ Convergence of the AD gradient ↑ ============
# -------------------------------------------------------------


if __name__ == "__main__" and "--plot-only" in _sys.argv:
    plot_gaussian_gradient(dict(np.load(DATA_DIRECTORY / "gradient_3d_gaussian.npz")), dimensionality=3)
    cached = np.load(DATA_DIRECTORY / "gradient_convergence_test.npz")
    plot_gradient_convergence(
        tuple(int(num_cells) for num_cells in cached["num_cells"]),
        {scheme.label: list(cached[scheme.label]) for scheme in CONVERGENCE_SCHEMES},
    )
elif __name__ == "__main__":
    FIGURES_DIRECTORY.mkdir(exist_ok=True)

    # Paper figure gradient_3d_gaussian.svg.
    run_gaussian_gradient_test(dimensionality=3, num_cells=64)

    # Paper figure gradient_convergence_test.svg.
    run_gradient_convergence_test()
