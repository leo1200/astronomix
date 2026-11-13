"""
The field-level inference problem of the paper (Sec. "Inverse modeling 3D
example"): code units, the MHD configuration, the driven turbulent background,
the logo target and the projected-density loss.

Shared by the optimisation driver, the gradient-cost probe and the panel
renderer, so that all of them integrate exactly the same equations. The solver
is the baseline finite-difference scheme (WENO-Z weights, positivity-preserving
WENO at C_cfl = 0.75 with SSPRK4) on the Pallas backend; reverse-mode
derivatives go through the native-JAX tangent of the Pallas kernels or, with
``pallas_adjoint``, through the Pallas adjoint of the ideal-MHD WENO flux.
"""

# general
import os
from timeit import default_timer as timer

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# images
from PIL import Image

# units and constants
from astropy import units as u
import astropy.constants as c

# astronomix constants
from astronomix import (
    BACKWARDS,
    FORWARDS,
    PALLAS,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    CodeUnits,
    FDSolverConfig,
    SimulationConfig,
    SimulationParams,
)
from astronomix import TurbulenceDrivingConfig, TurbulenceDrivingParams

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    initialize_interface_fields,
    time_integration,
)

# Code units of the paper: 3 pc, 100 M_sun, 100 km/s.
CODE_UNITS = CodeUnits(3 * u.parsec, 100 * u.M_sun, 100 * u.km / u.s)

# The ISM-like medium: n_H = 2 cm^-3, p / k_B = 3e4 K cm^-3, B = 13.5 muG.
DENSITY_0 = (2 * c.m_p / u.cm**3).to(CODE_UNITS.code_density).value
PRESSURE_0 = (3e4 * u.K / u.cm**3 * c.k_B).to(CODE_UNITS.code_pressure).value
MAGNETIC_FIELD_0 = (13.5 * u.microgauss / c.mu0**0.5).to(CODE_UNITS.code_magnetic_field).value
# Target velocity dispersion of the TurbGen driving: the v_rms ~ 0.42 v_0 that the
# paper's background reached with the earlier forcing module (tag pre-streamline;
# a cached background of that module stays a valid input, a new one is driven by TurbGen).
TURBULENT_VELOCITY_DISPERSION = 0.42

# Duration of the driven phase (~3.5 crossing times at v_rms ~ 0.42 v_0).
DRIVING_TIME = (24 * 1e4 * u.yr).to(CODE_UNITS.code_time).value

# Positivity-preserving WENO is provably positive up to C_cfl = 0.75 with SSPRK4.
COURANT_NUMBER = 0.75

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")


def make_config(
    num_cells,
    turbulence_driving=False,
    backward=False,
    num_checkpoints=16,
    rematerialize_stages=True,
    return_snapshots=False,
    backend=PALLAS,
    pallas_adjoint=False,
):
    """
    The MHD simulation configuration of the problem (not yet finalized).

    Args:
        num_cells: Cells per dimension of the periodic unit box.
        turbulence_driving: Drive the turbulence (the background generation).
        backward: Reverse-mode differentiable time loop (checkpointed).
        num_checkpoints: Checkpoints of the reverse-mode time loop.
        rematerialize_stages: Rematerialise the Runge-Kutta stages in the backward.
        return_snapshots: Return snapshot data (used to count time steps).
        backend: The backend of the primal (PALLAS or NATIVE_JAX).
        pallas_adjoint: Differentiate the WENO flux of the reverse-mode time
            loop with its Pallas adjoint (``BackendConfig.pallas_adjoint``)
            instead of the transposed native tangent.

    Returns:
        The SimulationConfig.
    """
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    return SimulationConfig(
        mhd=True,
        solver_config=FDSolverConfig(weno_z=True, weno_positivity_preserving=True),
        backend_config=BackendConfig(backend=backend, pallas_adjoint=pallas_adjoint),
        progress_bar=False,
        donate_state=False,
        dimensionality=3,
        box_size=1.0,
        num_cells=num_cells,
        differentiation_mode=BACKWARDS if backward else FORWARDS,
        num_checkpoints=num_checkpoints,
        rematerialize_stages=rematerialize_stages,
        return_snapshots=return_snapshots,
        num_snapshots=2,
        turbulence_driving_config=TurbulenceDrivingConfig(turbulence_driving=turbulence_driving),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
    )


def make_params(t_end):
    """
    The simulation parameters of the problem.

    Args:
        t_end: End time of the integration in code units.

    Returns:
        The SimulationParams (the driving parameters only act with the driving on).
    """
    return SimulationParams(
        C_cfl=COURANT_NUMBER,
        dt_max=0.1,
        gamma=5 / 3,
        t_end=t_end,
        minimum_density=1e-2 * DENSITY_0,
        minimum_pressure=1e-2 * PRESSURE_0,
        turbulence_driving_params=TurbulenceDrivingParams(velocity=TURBULENT_VELOCITY_DISPERSION),
    )


def finalized(config):
    """
    Finalize a configuration for the MHD state shape of its grid.

    Args:
        config: The configuration from ``make_config``.

    Returns:
        The registered variables and the finalized configuration.
    """
    registered_variables = get_registered_variables(config)
    shape = (registered_variables.num_vars,) + (config.num_cells,) * 3
    return registered_variables, finalize_config(config, shape)


def build_state(config, registered_variables, density, velocity, pressure, magnetic_field):
    """
    Assemble the MHD state, with interface fields from the cell-centred field.

    Args:
        config: The simulation configuration.
        registered_variables: The registered variables.
        density: Density, shape (N, N, N).
        velocity: Velocity, shape (3, N, N, N).
        pressure: Gas pressure, shape (N, N, N).
        magnetic_field: Magnetic field, shape (3, N, N, N).

    Returns:
        The primitive state array.
    """
    interface_x, interface_y, interface_z = initialize_interface_fields(
        magnetic_field[0],
        magnetic_field[1],
        magnetic_field[2],
    )
    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        gas_pressure=pressure,
        velocity_x=velocity[0],
        velocity_y=velocity[1],
        velocity_z=velocity[2],
        magnetic_field_x=magnetic_field[0],
        magnetic_field_y=magnetic_field[1],
        magnetic_field_z=magnetic_field[2],
        interface_magnetic_field_x=interface_x,
        interface_magnetic_field_y=interface_y,
        interface_magnetic_field_z=interface_z,
    )


def driven_turbulence(num_cells, cache_path):
    """
    The saturated turbulent state after the driven phase, generated once and
    cached (the stochastic driving is deterministic, so the cache is exact).

    Args:
        num_cells: Cells per dimension.
        cache_path: The .npy file of the cached state.

    Returns:
        The turbulent primitive state and the wall time of its generation [s]
        (0 when it was read from the cache).
    """
    if os.path.exists(cache_path):
        return jnp.asarray(np.load(cache_path)), 0.0
    registered_variables, config = finalized(make_config(num_cells, turbulence_driving=True))
    shape = (num_cells,) * 3
    zero = jnp.zeros(shape)
    uniform_state = build_state(
        config,
        registered_variables,
        jnp.full(shape, DENSITY_0),
        (zero, zero, zero),
        jnp.full(shape, PRESSURE_0),
        (jnp.full(shape, MAGNETIC_FIELD_0), zero, zero),
    )
    start = timer()
    turbulence = jax.block_until_ready(
        time_integration(uniform_state, config, make_params(DRIVING_TIME), registered_variables)
    )
    generation_time = timer() - start
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.save(cache_path, np.asarray(turbulence))
    return turbulence, generation_time


def rms_velocity(state, registered_variables):
    """
    The rms velocity of a state.

    Args:
        state: The primitive state.
        registered_variables: The registered variables.

    Returns:
        sqrt(<v^2>) in code units.
    """
    velocity_index = registered_variables.velocity_index
    return float(jnp.sqrt(jnp.mean(
        state[velocity_index.x] ** 2 + state[velocity_index.y] ** 2 + state[velocity_index.z] ** 2
    )))


def logo_target(num_cells, total_mass):
    """
    The logo as a column-density target: dark pixels map to a 1 % excess over
    a flat background, normalised to the mass of the column-density map.

    Args:
        num_cells: Cells per dimension (the target is num_cells x num_cells).
        total_mass: Sum of the density cube the projection is compared with.

    Returns:
        The target projection, shape (N, N).
    """
    image = jnp.array(Image.open(LOGO_PATH).convert("L"))
    image = 1.0 - image / 255.0
    height, width = image.shape
    pad_height, pad_width = (-height) % num_cells, (-width) % num_cells
    image = jnp.pad(
        image,
        ((pad_height // 2, pad_height - pad_height // 2), (pad_width // 2, pad_width - pad_width // 2)),
    )
    padded_height, padded_width = image.shape
    pixels = image.reshape(
        num_cells, padded_height // num_cells, num_cells, padded_width // num_cells
    ).mean(axis=(1, 3))
    target = 1.0 + (pixels - 0.01) / 0.99 * 0.01
    return target / jnp.sum(target) * total_mass


def make_loss(turbulence, config, registered_variables, t_end, target):
    """
    The objective: MSE between the z-projected density after free evolution
    for t_end and the target, as a function of the initial velocity field
    (density, pressure and magnetic field of the turbulent state fixed).

    Args:
        turbulence: The driven turbulent state (background and initial guess).
        config: The finalized (BACKWARDS for gradients) configuration.
        registered_variables: The registered variables.
        t_end: Duration of the free evolution.
        target: The target projection.

    Returns:
        loss(velocity) with velocity of shape (3, N, N, N).
    """
    density_index = registered_variables.density_index
    magnetic_index = registered_variables.magnetic_index
    density = turbulence[density_index]
    pressure = turbulence[registered_variables.pressure_index]
    magnetic_field = jnp.stack([
        turbulence[magnetic_index.x],
        turbulence[magnetic_index.y],
        turbulence[magnetic_index.z],
    ])
    params = make_params(t_end)

    def loss(velocity):
        initial_state = build_state(
            config,
            registered_variables,
            density,
            velocity,
            pressure,
            magnetic_field,
        )
        final_state = time_integration(initial_state, config, params, registered_variables)
        projection = jnp.sum(final_state[density_index], axis=2)
        return jnp.mean((projection - target) ** 2)

    return loss


def initial_velocity(state, registered_variables):
    """
    The velocity of a state as a (3, N, N, N) control array.

    Args:
        state: The primitive state.
        registered_variables: The registered variables.

    Returns:
        The stacked velocity components.
    """
    velocity_index = registered_variables.velocity_index
    return jnp.stack([
        state[velocity_index.x],
        state[velocity_index.y],
        state[velocity_index.z],
    ])
