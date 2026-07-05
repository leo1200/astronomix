"""
Snapshot return container for the time integration.

Holds the per-snapshot diagnostics (states, energies, spectra, ...) plus
run-level metadata (runtime, iteration count, compiled-step memory usage)
returned by :func:`astronomix.time_stepping.time_integration.time_integration`
when ``config.return_snapshots`` is set.
"""

# typing
from typing import NamedTuple

# jax
import jax.numpy as jnp


class SnapshotData(NamedTuple):
    """Return format for the time integration, when snapshots are requested."""

    #: The times at which the snapshots were taken.
    time_points: jnp.ndarray = None

    #: The primitive states at the times the snapshots were taken.
    states: jnp.ndarray = None

    #: The final state of the simulation. This is especially useful
    #: when no snapshots are returned but only the statistics.
    final_state: jnp.ndarray = None

    #: The total mass at the times the snapshots were taken.
    total_mass: jnp.ndarray = None

    #: The total energy at the times the snapshots were taken.
    total_energy: jnp.ndarray = None

    #: internal energy
    internal_energy: jnp.ndarray = None

    #: kinetic energy
    kinetic_energy: jnp.ndarray = None

    #: gravitational energy
    gravitational_energy: jnp.ndarray = None

    #: Radial momentum
    radial_momentum: jnp.ndarray = None

    #: Mean absolute magnetic field divergence
    magnetic_divergence: jnp.ndarray = None

    #: The kinetic energy spectrum at the times the snapshots were taken.
    kinetic_energy_spectrum: jnp.ndarray = None

    #: The magnetic energy spectrum at the times the snapshots were taken.
    magnetic_energy_spectrum: jnp.ndarray = None

    #: The helicity spectrum at the times the snapshots were taken.
    helicity_spectrum: jnp.ndarray = None

    #: The k-vector corresponding to the spectra 
    #: (same for each time snapshot and each spectrum).
    k_spectra: jnp.ndarray = None

    #: The temperature PDF (dV/dlogT) at the times the snapshots were taken.
    temperature_pdf: jnp.ndarray = None

    #: The runtime of the simulation loop.
    runtime: float = 0.0

    #: Number of timesteps taken.
    num_iterations: int = 0

    #: Compiled-step temporary memory per device, in bytes
    #: (populated when config.memory_analysis is True).
    temporary_memory_bytes: int = 0

    #: Compiled-step argument memory per device, in bytes
    #: (populated when config.memory_analysis is True).
    argument_memory_bytes: int = 0

    #: Compiled-step total memory per device, in bytes
    #: (temp + argument + output - alias; populated when
    #: config.memory_analysis is True).
    total_memory_bytes: int = 0

    #: The current checkpoint, used internally.
    current_checkpoint: int = 0
