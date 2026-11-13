"""
The scheme families of the cross-scheme module tests (``test_*_across_schemes.py``) and a small runner.

Every module test runs one problem with an analytic (or linear-theory) answer in each scheme family that carries
the module, and checks all of them against the same tolerance.
"""

# numerics
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# astronomix constants
from astronomix import FINITE_DIFFERENCE, FINITE_VOLUME, PERIODIC_BOUNDARY  # noqa: E402
from astronomix.option_classes.simulation_config import (  # noqa: E402
    CONSTRAINED_TRANSPORT,
    HLLC,
    MUSCL_HANCOCK,
    RK2_SSP,
    VAN_LEER,
    VL2,
)

# astronomix containers
from astronomix import (  # noqa: E402
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)
from astronomix.option_classes.simulation_config import StaticFloatVector  # noqa: E402
from astronomix.option_classes.presets import (  # noqa: E402
    BEATTIE_REFERENCE,
    FD_RIEMANN_IDEAL_GAS,
)

# astronomix functions
from astronomix import (  # noqa: E402
    finalize_config,
    get_registered_variables,
    time_integration,
)

#: The hydrodynamic scheme families: name -> (solver_mode, solver_config).
HYDRO_SCHEMES = {
    "FD WENO_FLUX_SPLITTING": (FINITE_DIFFERENCE, FDSolverConfig()),
    "FD RIEMANN_FLUX": (FINITE_DIFFERENCE, FD_RIEMANN_IDEAL_GAS),
    "FV VL2": (FINITE_VOLUME, FVSolverConfig(time_integrator=VL2, riemann_solver=HLLC)),
    "FV RK2_SSP": (FINITE_VOLUME, FVSolverConfig(time_integrator=RK2_SSP, riemann_solver=HLLC, limiter=VAN_LEER)),
    "FV MUSCL_HANCOCK": (
        FINITE_VOLUME,
        FVSolverConfig(time_integrator=MUSCL_HANCOCK, riemann_solver=HLLC, limiter=VAN_LEER),
    ),
}

#: The MHD scheme families: name -> (solver_mode, solver_config, smallest dimensionality).
MHD_SCHEMES = {
    "FD WENO_FLUX_SPLITTING": (FINITE_DIFFERENCE, FDSolverConfig(), 1),
    "FD RIEMANN_FLUX": (FINITE_DIFFERENCE, FD_RIEMANN_IDEAL_GAS, 3),
    "FV VL2 GLM": (FINITE_VOLUME, FVSolverConfig(time_integrator=VL2), 1),
    "FV VL2 CT": (FINITE_VOLUME, FVSolverConfig(time_integrator=VL2, divergence_control=CONSTRAINED_TRANSPORT), 2),
    "FV MUSCL_HANCOCK (BEATTIE_REFERENCE)": (FINITE_VOLUME, BEATTIE_REFERENCE, 1),
}


def periodic_boundaries(dimensionality):
    """Periodic boundaries on every axis."""
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    if dimensionality == 1:
        return periodic
    return BoundarySettings(*([periodic] * dimensionality + [BoundarySettings1D()] * (3 - dimensionality)))


def make_config(solver_mode, solver_config, shape, boundary_settings=None, **settings):
    """The (unfinalized) configuration of a run on a box of length 1 along x with cubic cells."""
    dimensionality = len(shape)
    box_lengths = [num_cells / shape[0] for num_cells in shape] + [1.0] * (3 - dimensionality)
    return SimulationConfig(
        dimensionality=dimensionality,
        box_size=StaticFloatVector(*box_lengths),
        solver_mode=solver_mode,
        solver_config=solver_config,
        boundary_settings=boundary_settings or periodic_boundaries(dimensionality),
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
        **settings,
    )


def run(config, initial_state, params: SimulationParams):
    """Finalize the configuration for the state and integrate; returns the final state and the configuration."""
    registered_variables = get_registered_variables(config)
    config = finalize_config(config, initial_state.shape)
    result = time_integration(jnp.asarray(initial_state), config, params, registered_variables)
    return np.asarray(result.final_state), config


def cell_centres(num_cells, box_length):
    """The cell centres of a uniform axis."""
    return (np.arange(num_cells) + 0.5) * box_length / num_cells


def mode_amplitude(field, coordinate, wavenumber, phase):
    """The amplitude of the mode ``sin(k x + phase)`` of a field that varies along the first axis."""
    profile = field.reshape(field.shape[0], -1).mean(axis=1)
    return 2.0 * np.mean(profile * np.sin(wavenumber * coordinate + phase))
