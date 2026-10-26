"""
Origin of the growing grid-scale modes of the linearized KH operator.

The AD-linearized operator of ``eigen_initialization.py`` has growing eigenmodes
of grid-scale transverse structure besides the physical Kelvin-Helmholtz mode.
This diagnostic computes the single-mode spectra of the sharp and the smoothed
interface twice: with the full linearization of the WENO reconstruction (the
paper figure) and with its nonlinear weights, characteristic eigenvectors and
splitting speed frozen (``weno_ad_frozen_weights``), i.e. the linearization of
the linear scheme the primal step actually uses. It prints, per spectrum, the
growing modes sorted by growth rate with their effective transverse
wavenumber, and writes them to ``data/eigen_spectrum_frozen_weights.npz``.

    sbatch --export=ALL,RUN_DIR=examples/scripts/differentiability \\
        pytests/runners/gpu1.sh python eigen_spectrum_frozen_weights.py
"""

# general
from pathlib import Path

# numerics
import numpy as np

# jax
import jax.numpy as jnp

# astronomix functions
from astronomix.analysis_helpers.jacobians import single_xmode_rhs_jacobian2D

# the KH setup of the paper figure (importing it selects the GPU)
from eigen_initialization import (
    effective_transverse_wavenumber,
    paper_setup,
    setup_khi_unperturbed_base,
)

DATA_FILE = Path(__file__).resolve().parent / "data" / "eigen_spectrum_frozen_weights.npz"

#: Growth rates below this are taken as neutral (float32 round-off of the
#: damped and the neutral modes of the homogeneous states).
NEUTRAL_GROWTH_RATE = 1e-3

#: Effective transverse wavenumber (in units of the Nyquist wavenumber) above
#: which a mode counts as grid-scale, the cutoff of the mode selection.
GRID_SCALE_WAVENUMBER = 0.25


def spectrum(setup, smooth_profiles, frozen_weights):
    """
    The eigenvalues and effective transverse wavenumbers of the single-mode
    Jacobian of one KH base state.

    Args:
        setup: The KH setup of the paper.
        smooth_profiles: The smoothed (True) or the sharp (False) interface.
        frozen_weights: Linearize with the WENO weights, eigenvectors and
            splitting speed frozen.

    Returns:
        The tuple ``(eigenvalues, wavenumber_ratio)``.
    """
    state, config, params, registered_variables, helper_data = setup_khi_unperturbed_base(
        setup._replace(smooth_profiles=smooth_profiles)
    )
    config = config._replace(weno_ad_frozen_weights=frozen_weights)
    jacobian = single_xmode_rhs_jacobian2D(
        state,
        config,
        params,
        registered_variables,
        helper_data,
        wavelength=setup.perturbation_setup.wavelength,
    )
    eigenvalues, eigenvectors = jnp.linalg.eig(jacobian)
    num_transverse_cells = state.shape[2]
    modes = eigenvectors.T.reshape((-1, state.shape[0], num_transverse_cells))
    transverse_spacing = float(helper_data.geometric_centers[0, 1, 1] - helper_data.geometric_centers[0, 0, 1])
    wavenumber_ratio = effective_transverse_wavenumber(modes, transverse_spacing)
    return np.asarray(eigenvalues), np.asarray(wavenumber_ratio)


def main():
    """Compute, print and store the four spectra."""
    setup, _ = paper_setup()
    results = {}
    for smooth_profiles, interface in ((False, "sharp"), (True, "smoothed")):
        for frozen_weights, linearization in ((False, "full"), (True, "frozen")):
            eigenvalues, wavenumber_ratio = spectrum(setup, smooth_profiles, frozen_weights)
            key = f"{interface}_{linearization}"
            results[f"eigenvalues_{key}"] = eigenvalues
            results[f"wavenumber_ratio_{key}"] = wavenumber_ratio
            growing = eigenvalues.real > NEUTRAL_GROWTH_RATE
            grid_scale = growing & (wavenumber_ratio > GRID_SCALE_WAVENUMBER)
            order = np.argsort(-eigenvalues.real[growing])
            listed = ", ".join(
                f"{eigenvalues.real[growing][index]:.3f} (k {wavenumber_ratio[growing][index]:.2f})"
                for index in order[:8]
            )
            fastest_grid_scale = eigenvalues.real[grid_scale].max() if grid_scale.any() else 0.0
            print(
                f"{interface:8s} {linearization:6s}: {growing.sum():3d} growing modes, "
                f"{grid_scale.sum():3d} grid-scale, fastest grid-scale Re(lambda) = {fastest_grid_scale:.3f}; "
                f"largest: {listed}",
                flush=True,
            )
    DATA_FILE.parent.mkdir(exist_ok=True)
    np.savez(DATA_FILE, **results)
    print(f"wrote {DATA_FILE}")


if __name__ == "__main__":
    main()
