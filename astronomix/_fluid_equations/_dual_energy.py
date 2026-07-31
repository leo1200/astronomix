"""
Dual-energy formalism (switch variant, Bryan et al. 1995) for adiabatic MHD.

In high-Mach or low-beta flows the internal energy recovered from the total
energy, ``e_int = E - KE - ME``, is a tiny difference of large numbers and is
destroyed by floating-point cancellation (see the M~50 adiabatic stress test).
The cure is to evolve a *separate* internal-energy density ``g = rho*e`` with its
own gas-energy equation

    d g / d t + div(g v) = - p div(v)                     (1)

and, when recovering the pressure, use a **switch**:

    e_int = e_E  if  e_E / E_total > eta   (internal energy non-negligible:
                                            total-energy value is accurate and
                                            captures shock heating)
          = g    otherwise                 (kinetic/magnetic-energy dominated:
                                            the separately-advected g avoids the
                                            cancellation)

with ``eta`` ~ 1e-3. Where the advected value is used, the total energy is reset
to ``E = e_int + KE + ME`` so the two energies stay consistent going forward.

This module provides the **pure, backend-agnostic** building blocks; the
operator-split wiring into the time loop (threading ``g`` through the carry and
calling these once per step) is layered on top. Everything here is plain JAX, so
it differentiates and runs on CPU/GPU and under either backend.
"""

from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from astronomix._stencil_operations._stencil_operations import _shift
from astronomix.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables


def _momentum_indices(config, rv):
    if config.dimensionality == 1:
        return [rv.momentum_index]
    if config.dimensionality == 2:
        return [rv.momentum_index.x, rv.momentum_index.y]
    return [rv.momentum_index.x, rv.momentum_index.y, rv.momentum_index.z]


def _velocity_components(conserved_state, config, rv):
    """v = momentum / rho, with rho floored to a tiny positive value."""
    rho = jnp.maximum(conserved_state[rv.density_index], 1e-30)
    return [conserved_state[i] / rho for i in _momentum_indices(config, rv)]


def kinetic_and_magnetic_energy(conserved_state, config, rv):
    """Kinetic + magnetic energy densities (KE, ME) for the ideal-gas MHD state."""
    rho = jnp.maximum(conserved_state[rv.density_index], 1e-30)
    mom = [conserved_state[i] for i in _momentum_indices(config, rv)]
    ke = 0.5 * sum(m * m for m in mom) / rho
    if config.mhd:
        bx = conserved_state[rv.magnetic_index.x]
        by = conserved_state[rv.magnetic_index.y]
        bz = conserved_state[rv.magnetic_index.z]
        me = 0.5 * (bx * bx + by * by + bz * bz)
    else:
        me = jnp.zeros_like(ke)
    return ke, me


def internal_energy_from_total(conserved_state, config, rv):
    """``e_E = E - KE - ME`` — internal energy density from the total energy
    (cancellation-prone in KE/ME-dominated cells)."""
    ke, me = kinetic_and_magnetic_energy(conserved_state, config, rv)
    return conserved_state[rv.energy_index] - ke - me


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def dual_energy_switch(
    conserved_state: STATE_TYPE,
    internal_energy_density: Float[Array, "..."],
    gamma: Union[float, Float[Array, ""]],
    eta: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Apply the dual-energy switch and recover a consistent (E, g, p).

    Args:
        conserved_state: ideal-gas MHD conserved state (rho, mom, E, B, ...).
        internal_energy_density: the separately-advected ``g = rho*e``.
        gamma, eta, minimum_pressure: EoS / switch / floor.

    Returns:
        ``(conserved_state, g, pressure)`` with ``E`` and ``g`` synchronised to
        the chosen internal energy and ``pressure = (gamma-1) e_int`` floored.
    """
    ke, me = kinetic_and_magnetic_energy(conserved_state, config, rv=registered_variables)
    E = conserved_state[registered_variables.energy_index]
    e_E = E - ke - me                          # total-energy internal energy

    g = internal_energy_density
    # The total-energy value is trustworthy when the internal energy is a
    # non-negligible fraction of the total energy (no catastrophic cancellation
    # and shock heating is captured). E is non-negative for a physical state.
    E_safe = jnp.maximum(E, 1e-30)
    reliable = (e_E > eta * E_safe) & jnp.isfinite(e_E)

    e_int = jnp.where(reliable, e_E, g)
    pressure = jnp.maximum((gamma - 1.0) * e_int, minimum_pressure)
    e_int = pressure / (gamma - 1.0)           # re-derive after the floor

    # Synchronise both energies to the chosen internal energy. In reliable cells
    # e_int == e_E so E is unchanged; in unreliable cells the (accurate) advected
    # g is promoted into the total energy, conserving it forward.
    E_new = e_int + ke + me
    conserved_state = conserved_state.at[registered_variables.energy_index].set(E_new)
    return conserved_state, e_int, pressure


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def advect_internal_energy(
    internal_energy_density: Float[Array, "..."],
    conserved_state: STATE_TYPE,
    pressure: Float[Array, "..."],
    dt: Union[float, Float[Array, ""]],
    grid_spacing: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """One operator-split update of ``g`` over ``dt`` solving
    ``d g/dt + div(g v) = -p div(v)`` on a periodic grid.

    First-order upwind for the conservative advection ``div(g v)`` (with the
    face velocity ``½(v_i+v_{i+1})``) plus a central-difference ``p div(v)``
    work term. First order is dissipative but stable and conservative; it is
    deliberately simple — the dual energy only *matters* where the total-energy
    internal energy is unusable, and there a robust low-order ``g`` beats a
    cancellation-destroyed high-order one.
    """
    ndim = int(config.dimensionality)
    vels = _velocity_components(conserved_state, config, registered_variables)
    g = internal_energy_density
    dtdx = dt / grid_spacing

    div_gv = jnp.zeros_like(g)
    div_v = jnp.zeros_like(g)
    for axis, v in enumerate(vels):
        sa = axis  # spatial axis of a single field (no leading var axis here)
        g_p = _shift(g, -1, axis=sa)            # g_{i+1}
        v_p = _shift(v, -1, axis=sa)            # v_{i+1}
        # face velocity at i+1/2 and first-order upwind value of g there
        vf = 0.5 * (v + v_p)
        g_face = jnp.where(vf >= 0.0, g, g_p)
        flux_p = vf * g_face                    # F_{i+1/2}
        flux_m = _shift(flux_p, 1, axis=sa)     # F_{i-1/2}
        div_gv = div_gv + (flux_p - flux_m)
        # central divergence of velocity for the pdV work
        div_v = div_v + 0.5 * (v_p - _shift(v, 1, axis=sa))

    g_new = g - dtdx * div_gv - dtdx * pressure * div_v
    return g_new
