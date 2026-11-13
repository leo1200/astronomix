"""
Turbulence driving with an Ornstein-Uhlenbeck acceleration field.

A JAX port of TurbGen by Christoph Federrath (Federrath et al. 2010, A&A 512,
A81; Federrath et al. 2022, ascl:2204.001), from TurbGen.h of
github.com/chfeder/turbulence_generator, commit d9a7d74 (MIT licence,
Copyright (c) 2010-2025 Christoph Federrath; the notice is the file
``LICENSE_TurbGen`` in this directory). The line numbers in the docstrings
refer to that file.

The acceleration is a sum over a fixed list of Fourier modes,

    a_d(x, t) = Re sum_m C_md(t) exp(i k_m . x),

whose vectors follow independent Ornstein-Uhlenbeck (OU) processes with the
turnover time as correlation time, projected to the wanted mixture of
solenoidal and compressive modes. The sum is evaluated directly (no FFT) in
separable form, and every cell needs only its own coordinates.

Differences to TurbGen: the Gaussian deviates come from ``jax.random``, not
from TurbGen's own generator (same statistics, other numbers), and those of
pattern n depend on the key and n only, so the driving sequence does not
depend on the time steps of the run.
"""

# typing
from typing import Sequence, Union

# jax
import jax
import jax.numpy as jnp

# astronomix
from astronomix._modules._turbulence_driving._modes import (
    build_driving_modes,
    separable_lattice,
)
from astronomix._modules._turbulence_driving._turbulence_driving_options import (
    TurbulenceDrivingConfig,
    TurbulenceDrivingParams,
    TurbulenceDrivingState,
)

# -------------------------------------------------------------
# ================ ↓ scales of the driving ↓ ==================
# -------------------------------------------------------------


def turnover_time(config: TurbulenceDrivingConfig, params: TurbulenceDrivingParams):
    """Correlation time of the driving, T = L_x / (k_driving velocity) (TurbGen.h, 334)."""
    return config.box_size[0] / params.k_driving / params.velocity


def update_interval(config: TurbulenceDrivingConfig, params: TurbulenceDrivingParams):
    """Time between two patterns, T / num_steps_per_turnover (TurbGen.h, 336)."""
    return turnover_time(config, params) / config.num_steps_per_turnover


def ou_standard_deviation(config: TurbulenceDrivingConfig, params: TurbulenceDrivingParams):
    """
    Stationary standard deviation of the OU process (TurbGen.h, 340-346),

        sigma = sqrt(E / T),   E = (0.15 velocity)^3 / L_x,

    with the energy injection rate E; sigma scales as velocity^2.
    """
    energy_injection_rate = (0.15 * params.velocity) ** 3 / config.box_size[0]
    return jnp.sqrt(energy_injection_rate / turnover_time(config, params))


def solenoidal_weight_normalisation(solenoidal_weight, num_components: int):
    """
    sqrt(3 / n) sqrt(3) / sqrt(1 - 2 w + n w^2) for the solenoidal weight w
    and n components (TurbGen.h, 163-166; eq. 9 of Federrath et al. 2010):
    makes the rms of the field independent of w.
    """
    weight_sum = 1.0 - 2.0 * solenoidal_weight + num_components * solenoidal_weight**2
    return jnp.sqrt(3.0 / num_components) * jnp.sqrt(3.0) / jnp.sqrt(weight_sum)


# -------------------------------------------------------------
# ================ ↑ scales of the driving ↑ ==================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ============ ↓ Ornstein-Uhlenbeck mode state ↓ ==============
# -------------------------------------------------------------


def init_turbulence_driving(config: TurbulenceDrivingConfig, dtype=None) -> TurbulenceDrivingState:
    """
    The state before the first pattern (TurbGen.h, 337 and 934-946): step -1
    and every mode drawn from the stationary distribution of the OU process.
    """
    dtype = jnp.result_type(float) if dtype is None else dtype
    num_modes = build_driving_modes(config).amplitudes.size
    initial_key, key = jax.random.split(jax.random.key(config.random_seed))
    ou_phases = jax.random.normal(initial_key, (num_modes, config.dimensionality, 2), dtype)
    return TurbulenceDrivingState(ou_phases, jnp.asarray(-1, jnp.int32), key, jnp.ones(config.dimensionality, dtype))


def ornstein_uhlenbeck_update(ou_phases, deviates, damping_factor):
    """
    One step of the OU sequence in units of its standard deviation
    (TurbGen.h, 950-988; eq. 4 of Federrath et al. 2010):

        x <- f x + sqrt(1 - f^2) z,   f = exp(-interval / T),   z ~ N(0, 1).
    """
    return damping_factor * ou_phases + jnp.sqrt(1.0 - damping_factor**2) * deviates


def adjust_amplitude(amplitude_adjustment, velocity_dispersion, config, params):
    """
    One step of the automatic amplitude adjustment (TurbGen.h, 408-433,
    marked experimental there). With the measured velocity dispersion s_d of
    component d the amplitude that would meet the target is

        A_new = A r,   r = (velocity / (sqrt(n) s_d))^1.5,

    and A relaxes towards it on the time sqrt(r) T:

        A <- g A + (1 - g) A_new,   g = exp(-interval / (sqrt(r) T)).
    """
    ratio = (params.velocity / (velocity_dispersion * config.dimensionality**0.5)) ** 1.5
    relaxation = jnp.exp(-1.0 / (config.num_steps_per_turnover * jnp.sqrt(ratio)))
    return amplitude_adjustment * (relaxation + (1.0 - relaxation) * ratio)


def advance_turbulence_driving(
    driving_state: TurbulenceDrivingState,
    time,
    config: TurbulenceDrivingConfig,
    params: TurbulenceDrivingParams,
    velocity_dispersion=None,
) -> TurbulenceDrivingState:
    """
    Moves the driving to the pattern of the given time (TurbGen.h, 381-446).
    Pattern n = floor(time / interval) holds for n <= time / interval < n + 1;
    the first call, coming from step -1, therefore always draws a pattern.
    Between two patterns the state is returned unchanged.

    If n > 1 patterns are due, TurbGen applies n single updates; here they
    are applied as one update with f^n, which has the same distribution.

    With ``config.amplitude_auto_adjust``, ``velocity_dispersion`` is the
    measured velocity dispersion per component, shape (num_components,); the
    amplitude is adjusted once per call that updates the pattern, after 0.1
    turnover times, as in TurbGen.
    """
    dtype = driving_state.ou_phases.dtype
    target_step = jnp.floor(time / update_interval(config, params)).astype(jnp.int32)
    num_due = jnp.maximum(target_step - driving_state.step, 0)
    damping_factor = jnp.exp(-num_due.astype(dtype) / config.num_steps_per_turnover)
    pattern_key = jax.random.fold_in(driving_state.key, target_step)
    deviates = jax.random.normal(pattern_key, driving_state.ou_phases.shape, dtype)
    ou_phases = ornstein_uhlenbeck_update(driving_state.ou_phases, deviates, damping_factor)

    amplitude_adjustment = driving_state.amplitude_adjustment
    if config.amplitude_auto_adjust and velocity_dispersion is not None:
        velocity_dispersion = jnp.asarray(velocity_dispersion, dtype)
        adjusted = adjust_amplitude(amplitude_adjustment, velocity_dispersion, config, params)
        adjust = (num_due > 0) & (time > 0.1 * turnover_time(config, params)) & (velocity_dispersion[0] > 0)
        amplitude_adjustment = jnp.where(adjust, adjusted.astype(dtype), amplitude_adjustment)

    return TurbulenceDrivingState(ou_phases, driving_state.step + num_due, driving_state.key, amplitude_adjustment)


# -------------------------------------------------------------
# ============ ↑ Ornstein-Uhlenbeck mode state ↑ ==============
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ acceleration field ↓ =====================
# -------------------------------------------------------------


def project_modes(ou_phases, wave_vectors, solenoidal_weight):
    """
    Mixture of the solenoidal and the compressive part of the mode vectors
    (TurbGen.h, 992-1038; eq. 6 of Federrath et al. 2010). With the complex
    vector p = OU real part + i OU imaginary part of a mode and its
    compressive part p_c = k (k . p) / k^2,

        c = w (p - p_c) + (1 - w) p_c.

    Returns c with shape (num_modes, num_components).
    """
    phases = jax.lax.complex(ou_phases[..., 0], ou_phases[..., 1])
    k_squared = jnp.sum(wave_vectors**2, axis=1, keepdims=True)
    compressive = wave_vectors * jnp.sum(wave_vectors * phases, axis=1, keepdims=True) / k_squared
    return solenoidal_weight * (phases - compressive) + (1.0 - solenoidal_weight) * compressive


def mode_coefficients(driving_state: TurbulenceDrivingState, config, params):
    """
    The complex coefficients of a_d(x) = Re sum_m C_md exp(i k_m . x)
    (TurbGen.h, 603 and 632-634):

        C_md = 2 N(w) A_m sigma f_d^1.5 g_d c_md

    with the normalisation N(w) of the solenoidal weight, the amplitude A_m
    of the mode, the OU standard deviation sigma, the amplitude factor f_d
    (to the power 1.5: TurbGen.h, 350-352), the automatic adjustment g_d and
    the projected mode vectors c. Shape (num_modes, num_components).
    """
    dtype = driving_state.ou_phases.dtype
    modes = build_driving_modes(config)
    wave_vectors = jnp.asarray(modes.wave_vectors, dtype)
    projected = project_modes(driving_state.ou_phases, wave_vectors, params.solenoidal_weight)
    weight_normalisation = solenoidal_weight_normalisation(params.solenoidal_weight, config.dimensionality)
    scale = 2.0 * weight_normalisation * ou_standard_deviation(config, params)
    component_factor = jnp.asarray(params.amplitude_factor, dtype) ** 1.5 * driving_state.amplitude_adjustment
    coefficients = scale * jnp.asarray(modes.amplitudes, dtype)[:, None] * component_factor * projected
    return coefficients.astype(projected.dtype)


def cell_center_coordinates(num_cells: int, cell_width, index_offset: int = 0, dtype=None):
    """
    Cell centres of one axis, x_i = (index_offset + i + 1/2) cell_width for
    i = 0, ..., num_cells - 1 with the lower box edge at x = 0. The offset is
    the global index of the first cell: that of a shard, or minus the number
    of ghost cells for a padded array.
    """
    return (index_offset + jnp.arange(num_cells, dtype=dtype) + 0.5) * cell_width


def driving_acceleration(
    driving_state: TurbulenceDrivingState,
    config: TurbulenceDrivingConfig,
    params: TurbulenceDrivingParams,
    cell_centers: Sequence[jnp.ndarray],
) -> jnp.ndarray:
    """
    The acceleration a_d(x) = Re sum_m C_md exp(i k_m . x) on the tensor grid
    of the given cell centres (TurbGen.h, 590-637).

    Args:
        cell_centers: One 1D coordinate array per axis, of the whole grid or
            of any part of it (a shard, with or without ghost cells); see
            ``cell_center_coordinates``. Nothing else of the grid enters.

    Returns:
        The acceleration, shape (num_components, nx[, ny[, nz]]).

    The sum is separable: with the modes on the lattice of the distinct wave
    numbers per axis (k_z >= 0 after mirroring), the sums over k_x and k_y
    are formed first, on planes, and the sum over k_z last,

        a_d(x, y, z) = sum_c [cos(k_c z) Re P_dc(x, y) - sin(k_c z) Im P_dc(x, y)],
        P_dc(x, y) = sum_ab C_dabc exp(i k_a x) exp(i k_b y),

    as one matrix product, which writes the field in memory order. The cost
    per cell and component is two multiply-adds per distinct k_z (8 for the
    modes of Beattie et al.) instead of one sine and cosine per mode (122).
    In 2D the last axis is y, in 1D only the matrix product remains. The
    contractions run at full precision: the default single-precision matrix
    product of a GPU (TF32) would leave relative errors of 1e-4 to 1e-3.
    """
    highest = jax.lax.Precision.HIGHEST
    lattice = separable_lattice(config)
    coefficients = mode_coefficients(driving_state, config, params)
    coefficients = jnp.where(lattice.mirrored[:, None], coefficients.conj(), coefficients)
    lattice_shape = tuple(k.size for k in lattice.wave_numbers)
    planes = jnp.zeros((config.dimensionality, *lattice_shape), coefficients.dtype)
    planes = planes.at[(slice(None), *lattice.index)].add(coefficients.T)

    phase = [jnp.exp(1j * jnp.outer(jnp.asarray(k, x.dtype), x)) for k, x in zip(lattice.wave_numbers, cell_centers)]
    if config.dimensionality >= 2:
        planes = jnp.einsum("da...,ax->dx...", planes, phase[0], precision=highest)
    if config.dimensionality == 3:
        planes = jnp.einsum("dxbc,by->dxyc", planes, phase[1], precision=highest)
    tables = jnp.concatenate([phase[-1].real, -phase[-1].imag])
    planes = jnp.concatenate([planes.real, planes.imag], axis=-1)
    return jnp.matmul(planes, tables, precision=highest)


def driven_velocity_indices(velocity_index: Union[int, Sequence[int]], num_components: int):
    """The state rows of the first ``num_components`` velocity components."""
    return (velocity_index,) if isinstance(velocity_index, int) else tuple(velocity_index)[:num_components]


def add_velocity_kick(primitive_state, kick, velocity_index: Union[int, Sequence[int]], cells=()):
    """
    v_d <- v_d + kick_d on the velocity components of a primitive state, in
    the cells selected by the slices ``cells`` (all cells if empty).
    Adjacent velocity components (the layout of the variable registry) are
    updated as one block: on a GPU every separate update of a component
    costs a pass over the whole state.
    """
    num_components = kick.shape[0]
    indices = driven_velocity_indices(velocity_index, num_components)
    kick = kick.astype(primitive_state.dtype)
    if indices == tuple(range(indices[0], indices[0] + num_components)):
        return primitive_state.at[(slice(indices[0], indices[0] + num_components), *cells)].add(kick)
    for component, index in enumerate(indices):
        primitive_state = primitive_state.at[(index, *cells)].add(kick[component])
    return primitive_state


# -------------------------------------------------------------
# ================ ↑ acceleration field ↑ =====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ============== ↓ driving of a simulation state ↓ ============
# -------------------------------------------------------------

# The functions below take the simulation configuration, the simulation
# parameters and the variable registry of astronomix (duck-typed, so that the
# configuration module can import the options above without a cycle).


def physical_cells(primitive_state, config):
    """
    Slices of the physical cells of a primitive state: a padded state carries
    the same number of ghost layers on every axis; an empty tuple if it
    carries none.
    """
    num_cells = tuple(config.num_cells)[: config.dimensionality]
    ghost_layers = (primitive_state.shape[1] - num_cells[0]) // 2
    if ghost_layers == 0:
        return ()
    return tuple(slice(ghost_layers, ghost_layers + n) for n in num_cells)


def physical_cell_centers(config, dtype):
    """The cell centres of the physical grid per axis, lower box edge at 0."""
    driving_config = config.turbulence_driving_config
    num_cells = tuple(config.num_cells)[: config.dimensionality]
    return [cell_center_coordinates(n, length / n, 0, dtype) for n, length in zip(num_cells, driving_config.box_size)]


def velocity_dispersion(primitive_state, config, registered_variables):
    """
    The velocity dispersion per driven component as the FLASH plugin of
    TurbGen measures it for the automatic amplitude adjustment
    (plugins/flash/StirMain/Stir.F90, the sums before
    ``st_stir_check_for_update_of_turb_pattern_c``):

        s_d = sqrt(<v_d^2>_V - (P_d / M)^2)

    with the volume average <.>_V, the momentum P_d and the mass M of the
    physical cells. Shape (num_components,).
    """
    cells = physical_cells(primitive_state, config)
    density = primitive_state[(registered_variables.density_index, *cells)]
    indices = driven_velocity_indices(registered_variables.velocity_index, config.dimensionality)
    mass = jnp.sum(density)
    dispersion = []
    for index in indices:
        velocity = primitive_state[(index, *cells)]
        mean_velocity = jnp.sum(density * velocity) / mass
        variance = jnp.maximum(jnp.mean(velocity**2) - mean_velocity**2, 0.0)
        dispersion.append(jnp.sqrt(variance + jnp.finfo(primitive_state.dtype).tiny))
    return jnp.stack(dispersion)


def apply_turbulence_driving(
    primitive_state, driving_state: TurbulenceDrivingState, dt, config, params, registered_variables
):
    """
    The kick of the turbulence driving at the end of a time step, as the
    FLASH plugin of TurbGen applies it (plugins/flash/StirMain/Stir.F90 with
    CORRECT_BULK_MOTION, the only setting there):

        v <- v + dt a - (dt F + P) / M,   F = sum rho a dV,

    with the acceleration a of the current pattern and the mass M and
    momentum P of the physical cells before the kick, so that the net
    momentum after it is zero. Density, pressure and magnetic field are
    unchanged: for the ideal gas the internal energy stays and the total
    energy grows by the change of the kinetic energy (Stir.F90: ENER, the
    specific total energy, gets d_ekin = e_kin_new - e_kin_old).

    The acceleration is recomputed from the mode state on every call; nothing
    of grid size is stored. Only the physical cells are kicked; the ghost
    cells are left to the boundary fill of the scheme.
    """
    driving_config = config.turbulence_driving_config
    cells = physical_cells(primitive_state, config)
    cell_centers = physical_cell_centers(config, primitive_state.dtype)
    acceleration = driving_acceleration(driving_state, driving_config, params.turbulence_driving_params, cell_centers)
    acceleration = acceleration.astype(primitive_state.dtype)

    density = primitive_state[(registered_variables.density_index, *cells)]
    indices = driven_velocity_indices(registered_variables.velocity_index, driving_config.dimensionality)
    spatial_axes = tuple(range(1, acceleration.ndim))
    mass = jnp.sum(density)
    momentum = jnp.stack([jnp.sum(density * primitive_state[(index, *cells)]) for index in indices])
    force = jnp.sum(density * acceleration, axis=spatial_axes)
    correction = (dt * force + momentum) / mass
    kick = dt * acceleration - jnp.expand_dims(correction, spatial_axes)
    return add_velocity_kick(primitive_state, kick, registered_variables.velocity_index, cells)


# -------------------------------------------------------------
# ============== ↑ driving of a simulation state ↑ ============
# -------------------------------------------------------------
