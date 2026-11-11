"""
Elementwise dual-energy switch (Bryan et al. 1995).

The internal energy recovered from the total energy, ``e_E = E - KE (- ME)``, is
a small difference of large numbers in kinetic- or magnetic-energy dominated
cells and is destroyed there by floating-point cancellation. The dual-energy
formalism carries a separately advected internal-energy density ``g`` and
selects, cell by cell,

    e_int = e_E   if  e_E > eta * E   (the total-energy value is accurate and
                                       captures shock heating)
          = g     otherwise           (the advected value avoids the
                                       cancellation)

This module holds the one switch formula shared by the pressure recoveries of
the finite-difference scheme (primitive recovery, eigenstructure):
``dual_energy_internal_energy`` switches an internal energy,
``dual_energy_switched_pressure`` a gas pressure recovered from the total
energy. Both are plain elementwise ``jax.numpy``, so they apply to whole arrays
and to the per-cell values inside a kernel alike.
"""

# jax
import jax.numpy as jnp


def dual_energy_internal_energy(
    internal_energy_from_total,
    total_energy,
    internal_energy_density,
    eta,
):
    """
    Select the internal energy density with the dual-energy switch.

    The total-energy value is kept where it is a non-negligible fraction of the
    total energy, ``e_E > eta * max(E, 1e-30)``, and is not NaN; everywhere
    else the advected ``g`` is used. The floor on ``E`` keeps the threshold
    meaningful for a (non-physical) non-positive total energy.

    NOTE: the NaN test is the self-comparison ``e_E == e_E``, deliberately not
    ``jnp.isfinite``: the two differ for ``e_E = +-inf``, and an infinite
    ``e_E`` that passes the threshold is kept.

    Args:
        internal_energy_from_total: The internal energy density recovered from
            the total energy, ``e_E = E - KE (- ME)``.
        total_energy: The total energy density ``E``.
        internal_energy_density: The separately advected internal energy
            density ``g``.
        eta: The switch threshold (``config.dual_energy_eta``).

    Returns:
        The switched internal energy density.
    """
    safe_total_energy = jnp.maximum(total_energy, 1e-30)
    total_energy_is_reliable = (
        internal_energy_from_total > eta * safe_total_energy
    ) & (internal_energy_from_total == internal_energy_from_total)
    return jnp.where(
        total_energy_is_reliable,
        internal_energy_from_total,
        internal_energy_density,
    )


def dual_energy_switched_pressure(
    gas_pressure,
    total_energy,
    gamma,
    internal_energy_density,
    eta,
):
    """
    Apply the dual-energy switch to a gas pressure recovered from the total energy.

    The internal energy is re-derived from the pressure as ``p / (gamma - 1)``,
    switched with ``dual_energy_internal_energy`` and converted back, so wave
    speeds computed from the result never see the cancellation-corrupted value.

    Args:
        gas_pressure: The gas pressure recovered from the total energy.
        total_energy: The total energy density ``E``.
        gamma: The adiabatic index.
        internal_energy_density: The separately advected internal energy
            density ``g``.
        eta: The switch threshold (``config.dual_energy_eta``).

    Returns:
        The switched gas pressure ``(gamma - 1) e_int``.
    """
    internal_energy_from_total = gas_pressure / (gamma - 1.0)
    internal_energy = dual_energy_internal_energy(
        internal_energy_from_total,
        total_energy,
        internal_energy_density,
        eta,
    )
    return (gamma - 1.0) * internal_energy
