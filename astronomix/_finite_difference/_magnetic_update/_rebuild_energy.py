"""
Total energy after the cell-centred magnetic field is rebuilt from the faces,
for the finite-difference scheme with a Riemann flux
(``FDSolverConfig.scheme = RIEMANN_FLUX``, ideal gas): the energy is held.

The scheme evolves the cell-centred field with the fluid's fluxes and the face
fields with the constrained transport; every stage and every step replace the
former by the face-to-centre interpolation of the latter. The two magnetic
energies differ by what the two discretisations dissipate differently.

With the pressure held (``E += (|B_new|^2 - |B_old|^2) / 2``, the rebuild of
the production scheme in ``_constrained_transport.py``) that difference leaves
or enters the total energy. Here it is exchanged with the thermal energy, so
the total energy is conserved where the thermal energy can carry the exchange.
At very low plasma beta the thermal energy is a small residual of the total
energy and the local mismatch can exceed it: the energy is held fully in cells
with beta >= ``ENERGY_BETA``, the pressure in cells with
beta <= ``ENERGY_BETA`` / 10, and log-linearly blended between (the usual
switch to the internal energy of low-beta MHD codes). Independently the rebuild
never lowers the thermal energy below ``PRESSURE_FRACTION`` of its value
before the rebuild (nor below the pressure floor); where that bound applies the
total energy changes by the least amount that keeps it.

Elementwise jnp only, so the array form and a kernel can share it.
"""

# jax
import jax.numpy as jnp

#: The plasma beta above which the energy is held fully.
ENERGY_BETA = 1e-4

#: The fraction of its thermal energy a rebuild may take from a cell at most.
PRESSURE_FRACTION = 0.5


def rebuilt_total_energy(
    energy,
    density,
    momentum_squared,
    magnetic_squared_old,
    magnetic_squared_new,
    gamma,
    minimum_density,
    minimum_pressure,
):
    """
    The total energy of a cell whose cell-centred field changes from
    ``|B_old|^2`` to ``|B_new|^2``.

    Args:
        energy: The total energy density before the rebuild.
        density: The mass density.
        momentum_squared: |rho v|^2.
        magnetic_squared_old: |B|^2 before the rebuild.
        magnetic_squared_new: |B|^2 after the rebuild.
        gamma: The adiabatic index.
        minimum_density: The density floor (guards the kinetic energy).
        minimum_pressure: The pressure floor.

    Returns:
        The total energy density after the rebuild.
    """
    magnetic_change = 0.5 * (magnetic_squared_new - magnetic_squared_old)
    # A zero of the array type, so that a kernel lowers no bare literal in another precision.
    zero = energy - energy
    kinetic = 0.5 * momentum_squared / jnp.maximum(density, minimum_density + zero)
    thermal_old = energy - kinetic - 0.5 * magnetic_squared_old
    # beta / ENERGY_BETA = 2 (gamma - 1) e / (ENERGY_BETA |B|^2); a field-free cell has weight 1, a cell without
    # thermal energy weight 0. Where the numerator is not below the denominator the rounded ratio is at least 1 and
    # the weight is 1 after the bounds; the denominator 1 there keeps the derivative of the quotient finite in
    # float32 at |B|^2 = 0 (JAX forms -x y^-2, and (ENERGY_BETA 1e-30)^-2 is infinite).
    numerator = 2.0 * (gamma - 1.0) * thermal_old
    denominator = ENERGY_BETA * (magnetic_squared_old + 1e-30)
    held = numerator >= denominator
    ratio = numerator / jnp.where(held, zero + 1.0, denominator)
    weight = 1.0 + jnp.log(jnp.maximum(ratio, zero + 1e-30)) / 2.302585092994046
    weight = jnp.where(held, zero + 1.0, weight)
    weight = jnp.minimum(jnp.maximum(weight, zero), zero + 1.0)
    thermal_target = thermal_old - weight * magnetic_change
    lowest = jnp.minimum(
        thermal_old,
        jnp.maximum(PRESSURE_FRACTION * thermal_old, minimum_pressure / (gamma - 1.0) + zero),
    )
    thermal_new = jnp.maximum(thermal_target, lowest)
    # E + dB^2/2 + (e_new - e_old), written so that a fully held cell away from
    # the bound keeps E bit for bit.
    return energy + (1.0 - weight) * magnetic_change + (thermal_new - thermal_target)
