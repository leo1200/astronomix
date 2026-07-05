"""Per-snapshot diagnostics for the time integration.

A data-driven registry of the quantities that can be stored at each snapshot:
which ones a given configuration collects, how they are sized, and how each is
computed from the (unpadded) primitive state.  Adding a new diagnostic means
appending one ``SnapshotQuantity`` entry here and adding the matching field to
:class:`SnapshotData` — ``build_snapshot_store`` and ``record_snapshot`` below
are generic and need no changes.
"""

# typing
from typing import Callable, NamedTuple

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE

# astronomix containers
from astronomix.data_classes.simulation_snapshot_data import SnapshotData

# astronomix functions
from astronomix._finite_volume._magnetic_update._vector_maths import divergence3D
from astronomix._fluid_equations.total_quantities import (
    calculate_gravitational_energy,
    calculate_internal_energy,
    calculate_kinetic_energy,
    calculate_radial_momentum,
    calculate_total_energy,
    calculate_total_mass,
)
from astronomix._spatial_operators._differencing import _interface_field_divergence
from astronomix.analysis_helpers.energy_spectrum import (
    _wavenumber_bins,
    get_kinetic_energy_spectrum,
    get_magnetic_energy_spectrum,
    get_magnetic_helicity_spectrum,
)


class SnapshotQuantity(NamedTuple):
    """One diagnostic collected into the snapshot store.

    Attributes:
        field: The :class:`SnapshotData` field this quantity writes to.
        is_enabled: ``is_enabled(config) -> bool`` — collected for this config.
        per_snapshot_shape: ``per_snapshot_shape(config, state_shape) -> tuple``
            — the shape of one snapshot's entry (excluding the snapshot axis).
        compute: ``compute(primitive_state, helper_data, params, config,
            registered_variables) -> array`` — evaluated on the unpadded
            primitive state at each snapshot time.
    """

    field: str
    is_enabled: Callable
    per_snapshot_shape: Callable
    compute: Callable


# ---------------------------------------------------------------------------
# Spectra share a single wavenumber binning.  The bin count is only needed to
# size the spectral arrays; recomputing it here (an up-front, setup-time cost)
# keeps it out of the per-quantity shape signature.
# ---------------------------------------------------------------------------


def spectra_requested(config) -> bool:
    """Whether any spectral diagnostic (kinetic / magnetic / helicity) is on."""
    settings = config.snapshot_settings
    return (
        settings.return_kinetic_energy_spectrum
        or settings.return_magnetic_energy_spectrum
        or settings.return_helicity_spectrum
    )


def _spectral_bin_count(config) -> int:
    _, number_of_bins, _ = _wavenumber_bins(
        config.num_cells.x, config.num_cells.y, config.num_cells.z
    )
    return number_of_bins


def spectral_wavenumbers(config):
    """The shared wavenumber bin centers for the spectra, or ``None``."""
    if not spectra_requested(config):
        return None
    _, _, wavenumber_centers = _wavenumber_bins(
        config.num_cells.x, config.num_cells.y, config.num_cells.z
    )
    return wavenumber_centers


# ---------------------------------------------------------------------------
# Per-quantity ``compute`` functions.  All share the same signature so the
# recording loop can call them uniformly; each uses only the arguments it needs.
# ---------------------------------------------------------------------------


def _compute_states(primitive_state, helper_data, params, config, registered_variables):
    return primitive_state


def _compute_total_mass(primitive_state, helper_data, params, config, registered_variables):
    return calculate_total_mass(primitive_state, helper_data, config)


def _compute_total_energy(primitive_state, helper_data, params, config, registered_variables):
    return calculate_total_energy(
        primitive_state,
        helper_data,
        params.gamma,
        params.gravitational_constant,
        params,
        config,
        registered_variables,
    )


def _compute_internal_energy(primitive_state, helper_data, params, config, registered_variables):
    return calculate_internal_energy(
        primitive_state, helper_data, params.gamma, config, registered_variables
    )


def _compute_kinetic_energy(primitive_state, helper_data, params, config, registered_variables):
    return calculate_kinetic_energy(primitive_state, helper_data, config, registered_variables)


def _compute_radial_momentum(primitive_state, helper_data, params, config, registered_variables):
    return calculate_radial_momentum(primitive_state, helper_data, config, registered_variables)


def _compute_gravitational_energy(primitive_state, helper_data, params, config, registered_variables):
    return calculate_gravitational_energy(
        primitive_state,
        helper_data,
        params.gravitational_constant,
        params,
        config,
        registered_variables,
    )


def _compute_magnetic_divergence(primitive_state, helper_data, params, config, registered_variables):
    if config.solver_mode == FINITE_DIFFERENCE:
        interface_magnetic_field = registered_variables.interface_magnetic_field_index
        return jnp.max(jnp.abs(_interface_field_divergence(
            primitive_state[interface_magnetic_field.x],
            primitive_state[interface_magnetic_field.y],
            primitive_state[interface_magnetic_field.z],
            config.grid_spacing,
        )))
    magnetic_field = registered_variables.magnetic_index
    return jnp.max(jnp.abs(divergence3D(
        primitive_state[magnetic_field.x:magnetic_field.z + 1],
        config.grid_spacing,
    )))


def _compute_kinetic_energy_spectrum(primitive_state, helper_data, params, config, registered_variables):
    velocity = registered_variables.velocity_index
    _, spectrum = get_kinetic_energy_spectrum(
        primitive_state[velocity.x],
        primitive_state[velocity.y],
        primitive_state[velocity.z],
        primitive_state[registered_variables.density_index],
    )
    return spectrum


def _compute_magnetic_energy_spectrum(primitive_state, helper_data, params, config, registered_variables):
    magnetic_field = registered_variables.magnetic_index
    _, spectrum = get_magnetic_energy_spectrum(
        primitive_state[magnetic_field.x],
        primitive_state[magnetic_field.y],
        primitive_state[magnetic_field.z],
    )
    return spectrum


def _compute_helicity_spectrum(primitive_state, helper_data, params, config, registered_variables):
    magnetic_field = registered_variables.magnetic_index
    _, spectrum = get_magnetic_helicity_spectrum(
        primitive_state[magnetic_field.x],
        primitive_state[magnetic_field.y],
        primitive_state[magnetic_field.z],
    )
    return spectrum


def _compute_temperature_pdf(primitive_state, helper_data, params, config, registered_variables):
    # temperature from the ideal gas law, ASSUMING T = P / rho
    temperature = (
        primitive_state[registered_variables.pressure_index]
        / primitive_state[registered_variables.density_index]
    )
    log_temperature = jnp.log10(temperature)
    # temperature PDF (dV / dlogT)
    volume_per_log_temperature, _ = jnp.histogram(
        log_temperature.flatten(),
        range=(
            jnp.log10(config.snapshot_settings.temperature_pdf_min),
            jnp.log10(config.snapshot_settings.temperature_pdf_max),
        ),
        bins=config.snapshot_settings.num_temperature_bins,
    )
    return volume_per_log_temperature


SNAPSHOT_QUANTITIES = (
    SnapshotQuantity(
        field="states",
        is_enabled=lambda config: config.snapshot_settings.return_states,
        per_snapshot_shape=lambda config, state_shape: state_shape,
        compute=_compute_states,
    ),
    SnapshotQuantity(
        field="total_mass",
        is_enabled=lambda config: config.snapshot_settings.return_total_mass,
        per_snapshot_shape=lambda config, state_shape: (),
        compute=_compute_total_mass,
    ),
    SnapshotQuantity(
        field="total_energy",
        is_enabled=lambda config: config.snapshot_settings.return_total_energy,
        per_snapshot_shape=lambda config, state_shape: (),
        compute=_compute_total_energy,
    ),
    SnapshotQuantity(
        field="internal_energy",
        is_enabled=lambda config: config.snapshot_settings.return_internal_energy,
        per_snapshot_shape=lambda config, state_shape: (),
        compute=_compute_internal_energy,
    ),
    SnapshotQuantity(
        field="kinetic_energy",
        is_enabled=lambda config: config.snapshot_settings.return_kinetic_energy,
        per_snapshot_shape=lambda config, state_shape: (),
        compute=_compute_kinetic_energy,
    ),
    SnapshotQuantity(
        field="radial_momentum",
        is_enabled=lambda config: config.snapshot_settings.return_radial_momentum,
        per_snapshot_shape=lambda config, state_shape: (),
        compute=_compute_radial_momentum,
    ),
    SnapshotQuantity(
        field="gravitational_energy",
        is_enabled=lambda config: (
            config.snapshot_settings.return_gravitational_energy and config.gravity_config.gravity
        ),
        per_snapshot_shape=lambda config, state_shape: (),
        compute=_compute_gravitational_energy,
    ),
    SnapshotQuantity(
        field="magnetic_divergence",
        is_enabled=lambda config: (
            config.snapshot_settings.return_magnetic_divergence and config.mhd
        ),
        per_snapshot_shape=lambda config, state_shape: (),
        compute=_compute_magnetic_divergence,
    ),
    SnapshotQuantity(
        field="kinetic_energy_spectrum",
        is_enabled=lambda config: config.snapshot_settings.return_kinetic_energy_spectrum,
        per_snapshot_shape=lambda config, state_shape: (_spectral_bin_count(config),),
        compute=_compute_kinetic_energy_spectrum,
    ),
    SnapshotQuantity(
        field="magnetic_energy_spectrum",
        is_enabled=lambda config: (
            config.snapshot_settings.return_magnetic_energy_spectrum and config.mhd
        ),
        per_snapshot_shape=lambda config, state_shape: (_spectral_bin_count(config),),
        compute=_compute_magnetic_energy_spectrum,
    ),
    SnapshotQuantity(
        field="helicity_spectrum",
        is_enabled=lambda config: (
            config.snapshot_settings.return_helicity_spectrum and config.mhd
        ),
        per_snapshot_shape=lambda config, state_shape: (_spectral_bin_count(config),),
        compute=_compute_helicity_spectrum,
    ),
    SnapshotQuantity(
        field="temperature_pdf",
        is_enabled=lambda config: config.snapshot_settings.return_temperature_pdf,
        per_snapshot_shape=lambda config, state_shape: (
            config.snapshot_settings.num_temperature_bins,
        ),
        compute=_compute_temperature_pdf,
    ),
)


def enabled_snapshot_quantities(config):
    """The snapshot quantities collected for ``config``, in registry order."""
    return tuple(
        quantity for quantity in SNAPSHOT_QUANTITIES if quantity.is_enabled(config)
    )


def build_snapshot_store(config, number_of_snapshots, state_shape) -> SnapshotData:
    """Allocate the snapshot store with zeroed buffers for every enabled quantity."""
    quantity_buffers = {
        quantity.field: jnp.zeros(
            (number_of_snapshots, *quantity.per_snapshot_shape(config, state_shape))
        )
        for quantity in enabled_snapshot_quantities(config)
    }
    return SnapshotData(
        time_points=jnp.zeros(number_of_snapshots),
        k_spectra=spectral_wavenumbers(config),
        current_checkpoint=0,
        final_state=None,
        **quantity_buffers,
    )


def record_snapshot(
    store,
    snapshot_index,
    time,
    primitive_state,
    helper_data,
    params,
    config,
    registered_variables,
) -> SnapshotData:
    """Write the time and every enabled diagnostic into ``store`` at ``snapshot_index``."""
    store = store._replace(
        time_points=store.time_points.at[snapshot_index].set(time)
    )
    for quantity in enabled_snapshot_quantities(config):
        value = quantity.compute(
            primitive_state, helper_data, params, config, registered_variables
        )
        store = store._replace(
            **{quantity.field: getattr(store, quantity.field).at[snapshot_index].set(value)}
        )
    return store
