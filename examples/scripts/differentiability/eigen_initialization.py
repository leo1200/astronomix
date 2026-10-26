"""
Eigen-initialization of the Kelvin-Helmholtz instability with the AD Jacobian.

The semi-discrete finite-difference right-hand side dq/dt = R(q) is linearized
by forward-mode AD about a base state that depends on y only. Since the x axis
is periodic and the base state streamwise homogeneous, a single x Fourier mode
q_hat(y) exp(i k_x x) decouples, and the reduced Jacobian of that mode
(n_var N_y x n_var N_y) is small enough to be diagonalized densely
(``single_xmode_rhs_jacobian2D``). Its eigenpairs are exact eigenpairs of the
discrete operator: growth rate Re(lambda), frequency Im(lambda).

Setup (paper Sec. "Sensitivity analysis and eigen-initialization"): pure
hydrodynamics on [0, 1]^2 at 300^2 cells, periodic in x and open in y, in single
precision, WENO-Z. A single tanh transition across y_c = 1/2 in density (chi = 10
below, 1 above) and streamwise velocity (-/+ v_shear / 2, v_shear = M_b c_b with
M_b = 0.5, c_b = sqrt(gamma P / rho_b)), uniform pressure P = 1, gamma = 5/3,
smoothing length sigma = lambda_x / 10 with lambda_x = L_x / 2; the sharp
reference interface is the sigma -> 0 limit. The KH time is
tau_KH = sqrt(Delta) lambda_x / v_shear with Delta = (rho_1 + rho_2)^2 /
(rho_1 rho_2).

The script

1. computes the eigenspectra of the dilute upper and the dense lower state in
   isolation and of the sharp and the smoothed interface, coloured by the
   effective transverse wavenumber, and selects the KH eigenmode (growing,
   interface-localized, transversely smooth): ``figures/eigenvalue_spectrum.png``
   (paper: eigenvalue_spectrum_final.png);
2. evolves the eigenmode and the usual ad-hoc velocity perturbation
   v_y = A sin(k_x x) exp(-((y - y_c) / (0.2 lambda_x))^2) of the same peak
   amplitude A = M_b c_b / 20 to t = 1.5 tau_KH and compares the growth of the
   k_x Fourier amplitude of v_y, ``figures/eigenmode_transient_test.png``, and the
   final states, ``figures/eigenmode_final_states.png``.

The spectra, the selected eigenvalue, the amplitude curves and the final
states are written to ``data/eigen_initialization.npz``, from which
``--plot-only`` redraws the three figures.

The slab geometry (a dense stream of half-width R_s, two interfaces, Mandelker
et al. 2016) and a pressure perturbation are available through
``interface_mode = SLAB`` and ``PRESSURE_PERTURBATION``; the paper uses the
single interface.
"""

# ==== GPU selection ====
import sys as _sys
if "--plot-only" not in _sys.argv:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from dataclasses import dataclass
from pathlib import Path

# typing
from typing import NamedTuple

# jax
import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# astronomix constants
from astronomix import (
    FINITE_DIFFERENCE,
    OPEN_BOUNDARY,
    PALLAS,
    PERIODIC_BOUNDARY,
)
from astronomix.option_classes.simulation_config import DYNAMIC_VISCOSITY
from astronomix.plotting_helpers.paper_style import (
    DIVERGING_COLORMAP,
    FD_COLOR,
    REFERENCE_COLOR,
    SEQUENTIAL_COLORMAP,
    SINGLE_PANEL_WIDTH,
    TEXT_WIDTH,
    layout_path,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix._fluid_equations._equations import (
    conserved_state_from_primitive,
    primitive_state_from_conserved,
)
from astronomix.analysis_helpers.jacobians import single_xmode_rhs_jacobian2D
from astronomix.plotting_helpers.inset_box import add_inset_box
from astronomix.plotting_helpers.paper_style import (
    attach_colorbar,
    figure_legend,
    reserve_colorbar_space,
    use_paper_style,
    variant_style,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
FIGURES_DIRECTORY = SCRIPT_DIRECTORY / "figures"
DATA_DIRECTORY = SCRIPT_DIRECTORY / "data"

# Geometry of the base state.
SINGLE_INTERFACE = 0
SLAB = 1
interface_mode = SINGLE_INTERFACE

# Perturbation types.
VELOCITY_PERTURBATION = 0
PRESSURE_PERTURBATION = 1
EIGENMODE_PERTURBATION = 2

# Global parameters of the base state.
y_center = 0.5
background_density = 1.0
pressure = 1.0
background_velocity = 0.0
box_size = 1.0
gamma = 5 / 3

# The smooth KH eigenmode (k_y^eff / k_Nyq ~ 0.05) separates cleanly from the
# grid-scale modes only once the smoothed interface (sigma = lambda_x / 10) is
# well resolved; at much lower resolution the selection fails.
num_cells = 300


@dataclass
class PerturbationSetup:
    """Amplitude and wavelength of the perturbation."""

    #: The amplitude of the perturbation (peak transverse velocity).
    amplitude: float

    #: The wavelength of the perturbation.
    wavelength: float


@dataclass
class PressurePerturbationSetup(PerturbationSetup):
    """A perturbation setup with the Gaussian width of a pressure perturbation."""

    #: The width of the Gaussian envelope of a pressure perturbation.
    gaussian_width: float


class KHISetup(NamedTuple):
    """The Kelvin-Helmholtz setup."""

    #: The number of cells in the x and y directions.
    num_cells: int

    #: Simulation duration.
    simulation_time: float

    #: The perturbation type.
    perturbation_type: int

    #: The perturbation setup.
    perturbation_setup: PerturbationSetup

    #: Switch for whether to include diffusion (viscosity) or not.
    diffusion: bool

    #: The viscosity of the fluid, if diffusion is included.
    viscosity: float

    #: The Mach number of the shear flow, M_b = v_shear / c_b.
    mach_number: float

    #: The density contrast chi between the dense and the dilute fluid.
    density_contrast: float

    #: The radius (half-width) of the slab (slab mode only).
    slab_radius: float

    #: The smoothing length sigma of the tanh profiles.
    smoothing_length: float

    #: Smooth tanh profiles (True) or sharp jumps (False).
    smooth_profiles: bool = True


# -------------------------------------------------------------
# ================ ↓ Base state and perturbations ↓ ===========
# -------------------------------------------------------------


def slab_profile(f_b, f_s, Y, y_center, slab_radius, smoothing_length):
    """
    Tanh transition from f_b (background) to f_s (stream) on both sides of a
    slab: f = f_b + (f_s - f_b) / 4 (1 + tanh((R_s - (y - y_c)) / sigma))
    (1 + tanh((R_s + (y - y_c)) / sigma)).
    """
    return f_b + 0.25 * (f_s - f_b) * (
        (1 + jnp.tanh((slab_radius - (Y - y_center)) / smoothing_length))
        * (1 + jnp.tanh((slab_radius + (Y - y_center)) / smoothing_length))
    )


def single_interface(f_l, f_u, Y, y_center, smoothing_length):
    """Tanh transition from f_l (below y_c) to f_u (above y_c)."""
    transition = jnp.tanh((Y - y_center) / smoothing_length)
    return 0.5 * (f_l * (1 - transition) + f_u * (1 + transition))


def velocity_perturbation(cell_centers, slab_radius, perturbation_setup):
    """
    The transverse-velocity perturbation of a slab (Roediger et al. 2013, Sec.
    3.1): v_y = A sin(k x) [exp(-((y - y_c) - R_s)^2 / sigma_y^2) +
    exp(-((y - y_c) + R_s)^2 / sigma_y^2)] with sigma_y = 0.3 lambda.
    """
    wavenumber = 2 * jnp.pi / perturbation_setup.wavelength
    X = cell_centers[:, :, 0]
    Y = cell_centers[:, :, 1]
    width = 0.3 * perturbation_setup.wavelength
    return perturbation_setup.amplitude * jnp.sin(wavenumber * X) * (
        jnp.exp(-(((Y - y_center) - slab_radius) ** 2) / width**2)
        + jnp.exp(-(((Y - y_center) + slab_radius) ** 2) / width**2)
    )


def single_interface_velocity_perturbation(cell_centers, perturbation_setup):
    """
    The ad-hoc transverse-velocity perturbation of the single interface:
    v_y = A sin(k x) exp(-((y - y_c) / sigma_v)^2) with sigma_v = 0.2 lambda.
    """
    wavenumber = 2 * jnp.pi / perturbation_setup.wavelength
    X = cell_centers[:, :, 0]
    Y = cell_centers[:, :, 1]
    width = 0.2 * perturbation_setup.wavelength
    return perturbation_setup.amplitude * jnp.sin(wavenumber * X) * jnp.exp(-(((Y - y_center) / width) ** 2))


def pressure_perturbation(cell_centers, slab_radius, y_center, perturbation_setup):
    """
    The pressure perturbation of a slab (Mandelker et al. 2016, Sec. 3.3):
    P_1 = A cos(k x) [exp(-((y - y_c) - R_s)^2 / (2 Sigma^2)) +
    exp(-((y - y_c) + R_s)^2 / (2 Sigma^2))].
    """
    wavenumber = 2 * jnp.pi / perturbation_setup.wavelength
    X = cell_centers[:, :, 0]
    Y = cell_centers[:, :, 1]
    width = perturbation_setup.gaussian_width
    return perturbation_setup.amplitude * jnp.cos(wavenumber * X) * (
        jnp.exp(-(((Y - y_center) - slab_radius) ** 2) / (2 * width**2))
        + jnp.exp(-(((Y - y_center) + slab_radius) ** 2) / (2 * width**2))
    )


def setup_khi_unperturbed_base(setup: KHISetup, return_snapshots=False):
    """
    The configuration, parameters and the unperturbed (y-dependent) base state.

    Args:
        setup: The KH setup.
        return_snapshots: Return 300 snapshots of the states.

    Returns:
        The tuple ``(base_state, config, params, registered_variables,
        helper_data)``.
    """
    # The padded grid is 300 + 2 x 4 = 308 cells per axis; a Pallas block has
    # to divide it and be a power of two, so it is (4, 4).
    config = SimulationConfig(
        backend_config=BackendConfig(backend=PALLAS, pallas_block_shape=(4, 4, 1)),
        weno_z=True,
        progress_bar=True,
        dimensionality=2,
        box_size=box_size,
        num_cells=setup.num_cells,
        diffusion=setup.diffusion,
        viscosity_type=DYNAMIC_VISCOSITY,
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
        return_snapshots=return_snapshots,
        num_snapshots=300,
        snapshot_settings=SnapshotSettings(return_states=True),
    )

    # The finite-difference CFL number counts the sum of the axis signal
    # speeds; 1.5 is close to the SSP limit of the five-stage SSPRK4 and
    # stable for this smooth, subsonic flow.
    params = SimulationParams(
        viscosity=setup.viscosity,
        t_end=setup.simulation_time,
        C_cfl=1.5 if config.solver_mode == FINITE_DIFFERENCE else 0.4,
        gamma=gamma,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    Y = helper_data.geometric_centers[:, :, 1]
    stream_density = setup.density_contrast * background_density
    background_sound_speed = jnp.sqrt(params.gamma * pressure / background_density)
    shear_velocity = setup.mach_number * background_sound_speed

    if interface_mode == SINGLE_INTERFACE:
        if setup.smooth_profiles:
            density = single_interface(stream_density, background_density, Y, y_center, setup.smoothing_length)
            velocity_x = single_interface(-shear_velocity / 2, shear_velocity / 2, Y, y_center, setup.smoothing_length)
        else:
            density = jnp.where(Y < y_center, stream_density, background_density)
            velocity_x = jnp.where(Y < y_center, -shear_velocity / 2, shear_velocity / 2)
    elif interface_mode == SLAB:
        if setup.smooth_profiles:
            density = slab_profile(
                background_density,
                stream_density,
                Y,
                y_center,
                setup.slab_radius,
                setup.smoothing_length,
            )
            velocity_x = slab_profile(
                background_velocity,
                shear_velocity,
                Y,
                y_center,
                setup.slab_radius,
                setup.smoothing_length,
            )
        else:
            inside_slab = jnp.abs(Y - y_center) <= setup.slab_radius
            density = jnp.where(inside_slab, stream_density, background_density)
            velocity_x = jnp.where(inside_slab, shear_velocity, background_velocity)
    else:
        raise ValueError(f"Unknown interface_mode {interface_mode}.")

    base_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity_x,
        velocity_y=jnp.zeros_like(velocity_x),
        gas_pressure=jnp.full_like(velocity_x, pressure),
    )
    config = finalize_config(config, base_state.shape)
    return base_state, config, params, registered_variables, helper_data


def perturb_khi(base_state, setup, helper_data, registered_variables, eigenmode_state=None):
    """
    Add the configured perturbation to the base state.

    Args:
        base_state: The unperturbed primitive state.
        setup: The KH setup (its ``perturbation_type`` selects the perturbation).
        helper_data: The helper data (cell centres).
        registered_variables: The registered variables.
        eigenmode_state: For ``EIGENMODE_PERTURBATION`` the eigenmode-perturbed
            state from :func:`find_numerical_eigenmode`.

    Returns:
        The perturbed primitive state.
    """
    cell_centers = helper_data.geometric_centers
    if setup.perturbation_type == VELOCITY_PERTURBATION:
        if interface_mode == SLAB:
            transverse_velocity = velocity_perturbation(cell_centers, setup.slab_radius, setup.perturbation_setup)
        else:
            transverse_velocity = single_interface_velocity_perturbation(cell_centers, setup.perturbation_setup)
        return base_state.at[registered_variables.velocity_index.y].add(transverse_velocity)

    if setup.perturbation_type == PRESSURE_PERTURBATION:
        if interface_mode != SLAB:
            raise NotImplementedError("The pressure perturbation is implemented for the slab only.")
        pressure_change = pressure_perturbation(cell_centers, setup.slab_radius, y_center, setup.perturbation_setup)
        return base_state.at[registered_variables.pressure_index].add(pressure_change)

    if setup.perturbation_type == EIGENMODE_PERTURBATION:
        if eigenmode_state is None:
            raise ValueError("EIGENMODE_PERTURBATION needs the eigenmode state of find_numerical_eigenmode.")
        return eigenmode_state

    raise NotImplementedError(f"Unknown perturbation type {setup.perturbation_type}.")


# -------------------------------------------------------------
# ================ ↑ Base state and perturbations ↑ ===========
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================ ↓ Eigenmode initialization ↓ ===============
# -------------------------------------------------------------


def effective_transverse_wavenumber(modes, transverse_spacing):
    """
    The effective transverse wavenumber k_y^eff / k_y^Nyq of each eigenvector:
    the spectral centroid sqrt(sum k_y^2 |q_hat(k_y)|^2 / sum |q_hat(k_y)|^2)
    of its y Fourier spectrum (all conserved variables), normalized to the
    Nyquist wavenumber. Unlike a centred-gradient estimate it detects
    Nyquist (checkerboard) modes correctly.

    Args:
        modes: The eigenvectors as an array (num_modes, n_var, N_y).
        transverse_spacing: The cell size in y.

    Returns:
        The array of k_y^eff / k_y^Nyq.
    """
    num_transverse_cells = modes.shape[-1]
    spectrum = jnp.fft.fft(modes, axis=-1)
    transverse_wavenumbers = 2.0 * jnp.pi * jnp.fft.fftfreq(num_transverse_cells, d=transverse_spacing)
    power = jnp.sum(jnp.abs(spectrum) ** 2, axis=1)
    effective_wavenumber = jnp.sqrt(
        jnp.sum(power * (transverse_wavenumbers**2)[None, :], axis=1) / (jnp.sum(power, axis=1) + 1e-30)
    )
    return effective_wavenumber / (jnp.pi / transverse_spacing)


def find_numerical_eigenmode(
    base_state,
    config,
    params,
    registered_variables,
    helper_data,
    setup: KHISetup,
    assembly_batch_size=4,
    require_growth=True,
    localization_threshold=0.20,
    wavenumber_threshold=0.25,
    num_printed_candidates=12,
):
    """
    Select the KH eigenmode of the AD Fourier-block Jacobian and return the
    eigenmode-perturbed state.

    Among the growing modes (Re(lambda) > 0) those localized at the interface
    (L > ``localization_threshold``) and smooth in y (k_y^eff / k_y^Nyq <
    ``wavenumber_threshold``) are accepted, and the fastest growing one is
    taken. The spurious fast-growing modes of the discretization are
    grid-scale in y (k_y^eff / k_y^Nyq ~ 0.6), the physical KH mode is smooth
    (~ 0.05), so one cutoff separates them. The mode is phase-aligned and
    scaled to the peak transverse velocity ``setup.perturbation_setup.amplitude``.

    Args:
        base_state: The unperturbed primitive state.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        helper_data: The helper data (cell centres).
        setup: The KH setup.
        assembly_batch_size: Jacobian columns per JVP batch.
        require_growth: Accept growing modes only.
        localization_threshold: The minimum interface localization L.
        wavenumber_threshold: The maximum k_y^eff / k_y^Nyq.
        num_printed_candidates: The number of best candidates printed.

    Returns:
        The tuple ``(perturbed_state, eigenvalue)``.
    """
    print(f"Extracting the AD KH eigenmode at lambda_x = {setup.perturbation_setup.wavelength} ...")

    config = finalize_config(config, base_state.shape)
    density_index = registered_variables.density_index
    momentum_y_index = registered_variables.velocity_index.y

    conserved_base_state = conserved_state_from_primitive(base_state, params.gamma, config, registered_variables)
    num_variables, _, num_transverse_cells = conserved_base_state.shape

    box_length_x = float(config.box_size.x)
    wavelength_ratio = box_length_x / float(setup.perturbation_setup.wavelength)
    mode_number = int(round(wavelength_ratio))
    if abs(wavelength_ratio - mode_number) > 1e-12:
        raise ValueError(
            f"wavelength={setup.perturbation_setup.wavelength} is not commensurate with "
            f"L_x={box_length_x} (L_x / wavelength = {wavelength_ratio})."
        )
    streamwise_wavenumber = 2.0 * jnp.pi * mode_number / box_length_x

    X = helper_data.geometric_centers[:, :, 0]
    y = helper_data.geometric_centers[0, :, 1]
    transverse_spacing = y[1] - y[0]

    base_density = base_state[density_index, 0, :]
    base_velocity_y = base_state[momentum_y_index, 0, :]

    # -------------------------------------------------------------
    # ========== ↓ Fourier-block Jacobian and eigensystem ↓ =======
    # -------------------------------------------------------------

    print("  Building the AD Fourier-block Jacobian ...")
    jacobian = single_xmode_rhs_jacobian2D(
        base_state,
        config,
        params,
        registered_variables,
        helper_data,
        wavelength=setup.perturbation_setup.wavelength,
        assembly_batch_size=assembly_batch_size,
    )
    print("  Solving the dense eigenproblem ...")
    eigenvalues, eigenvectors = jnp.linalg.eig(jacobian)

    # eigenvectors[:, i] is eigenvector i; reshape to modes[i, variable, y].
    modes = eigenvectors.T.reshape((-1, num_variables, num_transverse_cells))

    # The linearized conserved-to-primitive map gives the transverse-velocity
    # perturbation of every mode.
    density_perturbation = modes[:, density_index, :]
    transverse_velocity_perturbation = (
        modes[:, momentum_y_index, :] / base_density[None, :]
        - base_velocity_y[None, :] * density_perturbation / base_density[None, :]
    )

    # -------------------------------------------------------------
    # ========== ↑ Fourier-block Jacobian and eigensystem ↑ =======
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ======================= ↓ Mode selection ↓ ==================
    # -------------------------------------------------------------

    if interface_mode == SLAB:
        envelope = jnp.exp(-streamwise_wavenumber * jnp.abs(y - (y_center - setup.slab_radius))) + jnp.exp(
            -streamwise_wavenumber * jnp.abs(y - (y_center + setup.slab_radius))
        )
    else:
        envelope = jnp.exp(-streamwise_wavenumber * jnp.abs(y - y_center))
    envelope = envelope / jnp.max(envelope)

    transverse_velocity_power = jnp.sum(jnp.abs(transverse_velocity_perturbation) ** 2, axis=1) + 1e-30
    localization = (
        jnp.sum(jnp.abs(transverse_velocity_perturbation) ** 2 * envelope[None, :], axis=1)
        / transverse_velocity_power
    )
    wavenumber_ratio = effective_transverse_wavenumber(modes, transverse_spacing)

    finite = (
        jnp.isfinite(jnp.real(eigenvalues))
        & jnp.isfinite(jnp.imag(eigenvalues))
        & jnp.isfinite(localization)
        & jnp.isfinite(wavenumber_ratio)
    )
    growing = jnp.real(eigenvalues) > 0.0 if require_growth else jnp.ones_like(finite)
    is_kh_like = (
        finite
        & growing
        & (localization > localization_threshold)
        & (wavenumber_ratio < wavenumber_threshold)
    )

    # Among the accepted modes, the fastest growing one.
    score = jnp.where(is_kh_like, jnp.real(eigenvalues), -1e30)
    if not bool(jnp.any(is_kh_like)):
        print(
            "  Warning: no mode passed the KH filters; falling back to the growing "
            "mode that is most localized and smoothest in y."
        )
        score = jnp.where(finite & growing, localization - wavenumber_ratio, -1e30)

    selected_index = int(jnp.argmax(score))
    eigenvalue = eigenvalues[selected_index]
    selected_mode = modes[selected_index]

    print("  Best eigenmode candidates:")
    for rank, mode_index in enumerate(np.asarray(jnp.argsort(score)[::-1][:num_printed_candidates])):
        marker = " <--- selected" if mode_index == selected_index else ""
        print(
            f"    {rank:02d}: index {int(mode_index):04d}, "
            f"lambda = {float(jnp.real(eigenvalues[mode_index])):+.6e} "
            f"{float(jnp.imag(eigenvalues[mode_index])):+.6e} i, "
            f"L = {float(localization[mode_index]):.3f}, "
            f"k_y^eff/k_Nyq = {float(wavenumber_ratio[mode_index]):.3f}{marker}"
        )

    # -------------------------------------------------------------
    # ======================= ↑ Mode selection ↑ ==================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =============== ↓ Phase alignment and scaling ↓ =============
    # -------------------------------------------------------------

    def transverse_velocity_profile(mode):
        """The transverse-velocity perturbation profile of one mode."""
        return mode[momentum_y_index] / base_density - base_velocity_y * mode[density_index] / base_density

    # Rotate the phase so that v_y is real at its interface peak.
    anchor = int(jnp.argmax(jnp.abs(transverse_velocity_profile(selected_mode)) * envelope))
    selected_mode = selected_mode * jnp.exp(-1j * jnp.angle(transverse_velocity_profile(selected_mode)[anchor]))

    phase = jnp.exp(1j * streamwise_wavenumber * X).astype(selected_mode.dtype)
    transverse_velocity_field = jnp.real(transverse_velocity_profile(selected_mode)[None, :] * phase)
    peak_transverse_velocity = jnp.max(jnp.abs(transverse_velocity_field))
    if float(peak_transverse_velocity) <= 0.0 or not bool(jnp.isfinite(peak_transverse_velocity)):
        raise RuntimeError("The selected eigenmode has an invalid transverse-velocity amplitude.")
    scale = setup.perturbation_setup.amplitude / peak_transverse_velocity

    conserved_perturbation = scale * jnp.real(selected_mode[:, None, :] * phase[None, :, :])
    perturbed_state = primitive_state_from_conserved(
        conserved_base_state + conserved_perturbation.astype(conserved_base_state.dtype),
        params.gamma,
        config,
        registered_variables,
    )
    minimum_density = float(jnp.min(perturbed_state[density_index]))
    minimum_pressure = float(jnp.min(perturbed_state[registered_variables.pressure_index]))
    if minimum_density <= 0.0 or minimum_pressure <= 0.0:
        raise RuntimeError(
            f"Eigenmode amplitude too large: min rho = {minimum_density:.3e}, "
            f"min p = {minimum_pressure:.3e}. Reduce perturbation_setup.amplitude."
        )

    # -------------------------------------------------------------
    # =============== ↑ Phase alignment and scaling ↑ =============
    # -------------------------------------------------------------

    mode_vector = selected_mode.reshape(-1)
    jacobian_action = jacobian @ mode_vector
    residual = jnp.linalg.norm(jacobian_action - eigenvalue * mode_vector) / (
        jnp.linalg.norm(jacobian_action) + jnp.abs(eigenvalue) * jnp.linalg.norm(mode_vector) + 1e-30
    )
    print(
        f"Selected KH eigenmode: lambda = {float(jnp.real(eigenvalue)):.6e} + "
        f"{float(jnp.imag(eigenvalue)):.6e} i, growth time {1.0 / float(jnp.real(eigenvalue)):.6e}, "
        f"eigen residual {float(residual):.3e}"
    )
    return perturbed_state, eigenvalue


# -------------------------------------------------------------
# ================ ↑ Eigenmode initialization ↑ ===============
# -------------------------------------------------------------


# -------------------------------------------------------------
# ===================== ↓ Paper figures ↓ =====================
# -------------------------------------------------------------


def paper_setup():
    """
    The setup of the paper: chi = 10, M_b = 0.5, lambda_x = L_x / 2,
    sigma = lambda_x / 10, A = M_b c_b / 20, no viscosity, evolved to
    1.5 tau_KH.

    Returns:
        The tuple ``(setup, kh_time)``.
    """
    density_contrast = 10.0
    mach_number = 0.5
    background_sound_speed = float(jnp.sqrt(gamma * pressure / background_density))
    wavelength = box_size / 2
    smoothing_length = wavelength / 10

    dense_density = density_contrast * background_density
    density_factor = (dense_density + background_density) ** 2 / (dense_density * background_density)
    shear_velocity = mach_number * background_sound_speed
    kh_time = float(np.sqrt(density_factor) * wavelength / shear_velocity)

    setup = KHISetup(
        num_cells=num_cells,
        simulation_time=1.5 * kh_time,
        perturbation_type=VELOCITY_PERTURBATION,
        perturbation_setup=PressurePerturbationSetup(
            amplitude=shear_velocity / 20,
            wavelength=wavelength,
            gaussian_width=5 * smoothing_length,
        ),
        diffusion=False,
        viscosity=0.0,
        mach_number=mach_number,
        density_contrast=density_contrast,
        slab_radius=0.1,
        smoothing_length=smoothing_length,
    )
    return setup, kh_time


def eigenvalue_plot(setup: KHISetup):
    """
    The eigenspectra of the upper and the lower state, the sharp and the
    smoothed interface, and the selected KH eigenmode (Fig. kh-spectrum).
    Writes ``figures/eigenvalue_spectrum.png``.

    Args:
        setup: The KH setup (smooth profiles).

    Returns:
        The tuple ``(eigenmode_state, eigenvalue, spectra)`` with the
        eigenmode-perturbed smoothed state, the selected eigenvalue and a dict
        of the spectra and their k_y^eff / k_Nyq.
    """
    sharp_state, config, params, registered_variables, helper_data = setup_khi_unperturbed_base(
        setup._replace(smooth_profiles=False)
    )
    smoothed_state, _, _, _, _ = setup_khi_unperturbed_base(setup._replace(smooth_profiles=True))

    # The upper and the lower fluid in isolation (single-interface mode).
    num_transverse_cells = sharp_state.shape[2]
    upper_state = jnp.repeat(sharp_state[:, :, -1:], num_transverse_cells, axis=2)
    lower_state = jnp.repeat(sharp_state[:, :, :1], num_transverse_cells, axis=2)
    transverse_spacing = float(helper_data.geometric_centers[0, 1, 1] - helper_data.geometric_centers[0, 0, 1])

    spectra = {}
    states = (
        ("upper state", upper_state),
        ("lower state", lower_state),
        ("khi unperturbed", sharp_state),
        ("khi unperturbed smoothed", smoothed_state),
    )
    for label, state in states:
        jacobian = single_xmode_rhs_jacobian2D(
            state,
            config,
            params,
            registered_variables,
            helper_data,
            wavelength=setup.perturbation_setup.wavelength,
        )
        eigenvalues, eigenvectors = jnp.linalg.eig(jacobian)
        modes = eigenvectors.T.reshape((-1, state.shape[0], num_transverse_cells))
        wavenumber_ratio = effective_transverse_wavenumber(modes, transverse_spacing)
        spectra[label] = (np.asarray(eigenvalues), np.asarray(wavenumber_ratio))
        print(
            f"{label}: max Re(lambda) = {float(jnp.max(jnp.real(eigenvalues))):.4e}, "
            f"min Re(lambda) = {float(jnp.min(jnp.real(eigenvalues))):.4e}, "
            f"max |Im(lambda)| = {float(jnp.max(jnp.abs(jnp.imag(eigenvalues)))):.4e}"
        )

    # Select the KH eigenmode of the smoothed interface.
    eigenmode_state, eigenvalue = find_numerical_eigenmode(
        smoothed_state,
        config,
        params,
        registered_variables,
        helper_data,
        setup,
    )
    plot_eigenvalue_spectrum(spectra, np.asarray(eigenvalue))
    return eigenmode_state, eigenvalue, spectra


#: Panel titles of the eigenvalue spectra (the spectrum labels are also the cache keys).
SPECTRUM_TITLES = {
    "upper state": "upper state",
    "lower state": "lower state",
    "khi unperturbed": "sharp interface",
    "khi unperturbed smoothed": "smoothed interface",
}


def plot_eigenvalue_spectrum(spectra, eigenvalue):
    """
    Draw ``figures/eigenvalue_spectrum.png``: the four eigenspectra coloured by
    the effective transverse wavenumber, with the selected KH eigenmode marked
    and magnified in the spectrum of the smoothed interface.

    Args:
        spectra: Per panel label, the eigenvalues and their k_y^eff / k_Nyq.
        eigenvalue: The selected eigenvalue.
    """
    use_paper_style()
    figure, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 5.8), sharex=True, sharey=True)
    imaginary_range = (-650, 650)
    real_range = (-450, 50)
    last_scatter = None
    for panel_index, (label, (eigenvalues, wavenumber_ratio)) in enumerate(spectra.items()):
        axis = axes[panel_index // 2, panel_index % 2]
        last_scatter = axis.scatter(
            np.real(eigenvalues),
            np.imag(eigenvalues),
            c=wavenumber_ratio,
            s=4,
            cmap=SEQUENTIAL_COLORMAP,
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(SPECTRUM_TITLES.get(label, label))
        axis.set_xlim(real_range)
        axis.set_ylim(imaginary_range)
    for axis in axes[1, :]:
        axis.set_xlabel(r"$\operatorname{Re}(\lambda)$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\operatorname{Im}(\lambda)$")

    axis = axes[1, 1]
    axis.scatter(
        np.real(eigenvalue),
        np.imag(eigenvalue),
        marker="*",
        s=80,
        facecolors="none",
        edgecolors=REFERENCE_COLOR,
        linewidths=1.0,
        label=f"selected KH eigenmode, $\\lambda = {float(np.real(eigenvalue)):.2f} + {float(np.imag(eigenvalue)):.2f}i$",
        zorder=10,
    )
    add_inset_box(axis, x1=-2.0, x2=2.0, y1=0.0, y2=3.0, loc="lower left", connect_loc1=2, connect_loc2=4)

    # One colorbar per row, of the height of the panels (all panels share the
    # range); the left panels keep the same size.
    for row in range(2):
        reserve_colorbar_space(axes[row, 0])
        attach_colorbar(figure, last_scatter, axes[row, 1], label=r"$k_y^\mathrm{eff}/k_{y,\mathrm{Nyq}}$")
    figure_legend(figure, ncols=1)
    figure_path = layout_path(FIGURES_DIRECTORY / "eigenvalue_spectrum.png")
    figure.savefig(figure_path)
    plt.close(figure)
    print(f"saved {figure_path}")


def mode_amplitude(states, velocity_y_index, wavenumber, X):
    """
    A(t) = max_y sqrt(a_s^2 + a_c^2), the transverse peak of the modulus of
    the k_x Fourier pair of v_y, a_s = 2/N_x sum_x v_y sin(k_x x) and
    a_c = 2/N_x sum_x v_y cos(k_x x); insensitive to the x phase of the
    travelling wave.

    Args:
        states: The snapshot states (num_snapshots, n_var, N_x, N_y).
        velocity_y_index: The index of v_y.
        wavenumber: The streamwise wavenumber k_x.
        X: The x coordinates of the cell centres.

    Returns:
        The array of A(t).
    """
    sine = jnp.sin(wavenumber * X)
    cosine = jnp.cos(wavenumber * X)
    amplitudes = []
    for state in states:
        transverse_velocity = state[velocity_y_index]
        sine_coefficient = 2 * jnp.mean(transverse_velocity * sine, axis=0)
        cosine_coefficient = 2 * jnp.mean(transverse_velocity * cosine, axis=0)
        amplitudes.append(float(jnp.max(jnp.sqrt(sine_coefficient**2 + cosine_coefficient**2))))
    return np.array(amplitudes)


def eigenmode_growth_test(setup: KHISetup, kh_time, eigenmode_state, eigenvalue, spectra):
    """
    Evolve the ad-hoc velocity perturbation and the eigenmode to 1.5 tau_KH,
    compare the growth of the mode amplitude (Fig. kh-transient) and the final
    states (Fig. kh-final). Writes ``figures/eigenmode_transient_test.png``,
    ``figures/eigenmode_final_states.png`` and ``data/eigen_initialization.npz``.

    Args:
        setup: The KH setup (smooth profiles, velocity perturbation).
        kh_time: tau_KH.
        eigenmode_state: The eigenmode-perturbed state.
        eigenvalue: The selected eigenvalue.
        spectra: The spectra of :func:`eigenvalue_plot`, stored with the data.
    """
    print("\nEvolving the velocity perturbation and the eigenmode ...")
    base_state, config, params, registered_variables, helper_data = setup_khi_unperturbed_base(
        setup,
        return_snapshots=True,
    )
    velocity_state = perturb_khi(base_state, setup, helper_data, registered_variables)
    result_velocity = time_integration(velocity_state, config, params, registered_variables)
    result_eigenmode = time_integration(eigenmode_state, config, params, registered_variables)

    growth_rate = float(jnp.real(eigenvalue))
    X = helper_data.geometric_centers[:, :, 0]
    wavenumber = 2 * jnp.pi / setup.perturbation_setup.wavelength
    velocity_y_index = registered_variables.velocity_index.y
    amplitude_velocity = mode_amplitude(result_velocity.states, velocity_y_index, wavenumber, X)
    amplitude_eigenmode = mode_amplitude(result_eigenmode.states, velocity_y_index, wavenumber, X)
    time_velocity = np.asarray(result_velocity.time_points)
    time_eigenmode = np.asarray(result_eigenmode.time_points)
    log_growth_velocity = np.log(amplitude_velocity / amplitude_velocity[0])
    log_growth_eigenmode = np.log(amplitude_eigenmode / amplitude_eigenmode[0])

    # The measured growth rate of the eigenmode run over its first 0.6 tau_KH
    # (linear fit of ln A), to compare with Re(lambda).
    early = time_eigenmode <= 0.6 * kh_time
    measured_rate = float(np.polyfit(time_eigenmode[early], log_growth_eigenmode[early], 1)[0])
    print(f"Re(lambda) = {growth_rate:.4f}, measured eigenmode growth rate over t < 0.6 tau_KH = {measured_rate:.4f}")
    minimum_index = int(np.argmin(log_growth_velocity))
    print(
        f"velocity perturbation: minimum ln(A/A0) = {log_growth_velocity[minimum_index]:.3f} "
        f"at t / tau_KH = {time_velocity[minimum_index] / kh_time:.3f}; final ln(A/A0): velocity "
        f"{log_growth_velocity[-1]:.3f}, eigenmode {log_growth_eigenmode[-1]:.3f}"
    )

    # The final density, transverse velocity and pressure of both runs, for
    # the final-state figure.
    final_fields = {}
    for run_name, result in (("velocity", result_velocity), ("eigenmode", result_eigenmode)):
        final_state = result.states[-1]
        final_fields[f"final_density_{run_name}"] = np.asarray(final_state[registered_variables.density_index])
        final_fields[f"final_velocity_y_{run_name}"] = np.asarray(final_state[registered_variables.velocity_index.y])
        final_fields[f"final_pressure_{run_name}"] = np.asarray(final_state[registered_variables.pressure_index])

    DATA_DIRECTORY.mkdir(exist_ok=True)
    np.savez(
        DATA_DIRECTORY / "eigen_initialization.npz",
        eigenvalue=np.asarray(eigenvalue),
        kh_time=kh_time,
        simulation_time=setup.simulation_time,
        time_velocity=time_velocity,
        amplitude_velocity=amplitude_velocity,
        time_eigenmode=time_eigenmode,
        amplitude_eigenmode=amplitude_eigenmode,
        **final_fields,
        **{f"spectrum_{label.replace(' ', '_')}": values[0] for label, values in spectra.items()},
        **{f"wavenumber_ratio_{label.replace(' ', '_')}": values[1] for label, values in spectra.items()},
    )
    plot_growth_transient(
        time_velocity / kh_time,
        log_growth_velocity,
        time_eigenmode / kh_time,
        log_growth_eigenmode,
        growth_rate * kh_time,
        setup.simulation_time / kh_time,
    )
    plot_final_states(final_fields)


def plot_growth_transient(
    time_velocity,
    log_growth_velocity,
    time_eigenmode,
    log_growth_eigenmode,
    growth_rate_per_kh_time,
    end_time,
):
    """
    Draw ``figures/eigenmode_transient_test.png``: the growth of the mode
    amplitude of the ad-hoc velocity perturbation and of the eigenmode, with
    the growth expected from the eigenvalue.

    Args:
        time_velocity: The snapshot times of the velocity-perturbation run (tau_KH).
        log_growth_velocity: Its ln(A / A_0).
        time_eigenmode: The snapshot times of the eigenmode run (tau_KH).
        log_growth_eigenmode: Its ln(A / A_0).
        growth_rate_per_kh_time: Re(lambda) tau_KH.
        end_time: The end time of the runs (tau_KH).
    """
    use_paper_style()
    figure, axis = plt.subplots(figsize=(SINGLE_PANEL_WIDTH, 2.9), layout="constrained")
    axis.plot(
        time_velocity,
        log_growth_velocity,
        label="transverse-velocity perturbation",
        **variant_style(FD_COLOR),
    )
    axis.plot(
        time_eigenmode,
        log_growth_eigenmode,
        label="selected eigenmode",
        color=FD_COLOR,
    )
    reference_time = np.linspace(0, end_time, 100)
    axis.plot(
        reference_time,
        growth_rate_per_kh_time * reference_time,
        label=r"$\operatorname{Re}[\lambda]\, t$ (expected growth)",
        linestyle=":",
        color=REFERENCE_COLOR,
        linewidth=1.0,
    )
    axis.set_ylim(top=2.0)
    axis.set_xlabel(r"$t / \tau_\mathrm{KH}$")
    axis.set_ylabel(r"$\ln(A(t) / A_0)$")
    axis.legend(loc="lower right")
    figure_path = layout_path(FIGURES_DIRECTORY / "eigenmode_transient_test.png")
    figure.savefig(figure_path)
    plt.close(figure)
    print(f"saved {figure_path}")


def plot_final_states(final_fields):
    """
    Draw ``figures/eigenmode_final_states.png``: density (logarithmic),
    transverse velocity and pressure at the final time of the velocity
    perturbation (top) and the eigenmode run (bottom).

    Args:
        final_fields: The final fields as stored in the ``.npz``.
    """
    use_paper_style()
    figure, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.8))
    extent = [0, box_size, 0, box_size]
    panels = (
        ("final_density", SEQUENTIAL_COLORMAP, LogNorm, "density"),
        ("final_velocity_y", DIVERGING_COLORMAP, None, "$v_y$"),
        ("final_pressure", SEQUENTIAL_COLORMAP, None, "pressure"),
    )
    run_names = ("velocity", "eigenmode")
    for row, run_name in enumerate(run_names):
        for column, (field, colormap, norm_class, title) in enumerate(panels):
            values = final_fields[f"{field}_{run_name}"]
            # One colour scale per column, shared by both runs.
            both_runs = [np.asarray(final_fields[f"{field}_{name}"]) for name in run_names]
            smallest = float(min(np.min(run) for run in both_runs))
            largest = float(max(np.max(run) for run in both_runs))
            if norm_class is not None:
                image_options = dict(norm=norm_class(vmin=smallest, vmax=largest))
            elif colormap == DIVERGING_COLORMAP:
                # A diverging map centred on zero.
                largest = float(max(np.max(np.abs(run)) for run in both_runs))
                image_options = dict(vmin=-largest, vmax=largest)
            else:
                image_options = dict(vmin=smallest, vmax=largest)
            image_options.update(cmap=colormap, origin="lower", extent=extent)
            axis = axes[row, column]
            image = axis.imshow(values.T, **image_options)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            attach_colorbar(figure, image, axis)
            if row == 0:
                axis.set_title(title)
    axes[0, 0].set_ylabel("velocity perturbation")
    axes[1, 0].set_ylabel("eigenmode")
    figure.tight_layout()
    figure_path = layout_path(FIGURES_DIRECTORY / "eigenmode_final_states.png")
    figure.savefig(figure_path)
    plt.close(figure)
    print(f"saved {figure_path}")


# -------------------------------------------------------------
# ===================== ↑ Paper figures ↑ =====================
# -------------------------------------------------------------


def plot_from_cache():
    """Redraw the three figures from ``data/eigen_initialization.npz``."""
    cached = dict(np.load(DATA_DIRECTORY / "eigen_initialization.npz"))
    labels = ("upper state", "lower state", "khi unperturbed", "khi unperturbed smoothed")
    spectra = {
        label: (
            cached[f"spectrum_{label.replace(' ', '_')}"],
            cached[f"wavenumber_ratio_{label.replace(' ', '_')}"],
        )
        for label in labels
    }
    plot_eigenvalue_spectrum(spectra, cached["eigenvalue"])
    kh_time = float(cached["kh_time"])
    plot_growth_transient(
        cached["time_velocity"] / kh_time,
        np.log(cached["amplitude_velocity"] / cached["amplitude_velocity"][0]),
        cached["time_eigenmode"] / kh_time,
        np.log(cached["amplitude_eigenmode"] / cached["amplitude_eigenmode"][0]),
        float(np.real(cached["eigenvalue"])) * kh_time,
        float(cached["simulation_time"]) / kh_time,
    )
    plot_final_states(cached)


if __name__ == "__main__" and "--plot-only" in _sys.argv:
    plot_from_cache()
elif __name__ == "__main__":
    FIGURES_DIRECTORY.mkdir(exist_ok=True)
    paper_khi_setup, paper_kh_time = paper_setup()

    # Fig. kh-spectrum: the eigenspectra and the selected KH eigenmode.
    paper_eigenmode_state, paper_eigenvalue, paper_spectra = eigenvalue_plot(paper_khi_setup)

    # Figs. kh-transient and kh-final: growth transient and final states.
    eigenmode_growth_test(
        paper_khi_setup,
        paper_kh_time,
        paper_eigenmode_state,
        paper_eigenvalue,
        paper_spectra,
    )
