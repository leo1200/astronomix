"""
Here we protect the density and pressure from going negative.

In my view this is a bit of a shady practice, hiding unphysical
updates under the rug. However, it is common practice.
"""

# general
import itertools
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    IDEAL_GAS,
    POSITIVITY_CONSERVATIVE,
    POSITIVITY_HARD_FLOOR,
    POSITIVITY_REDISTRIBUTE,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._pallas_helpers import diffable_pallas_call_n


def _enforce_positivity_native(
    conserved_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    return _enforce_positivity_native_impl(
        conserved_state, config, gamma,
        minimum_density, minimum_pressure, registered_variables,
    )


@partial(
    jax.jit, static_argnames=["registered_variables", "config"]
)
def _enforce_positivity(
    conserved_state: STATE_TYPE,
    config: SimulationConfig,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    if _enforce_positivity_pallas_supported(conserved_state, config):
        pallas = lambda s, g, mr, mp: _enforce_positivity_pallas(  # noqa: E731
            s, config, g, mr, mp, registered_variables,
        )
        native = lambda s, g, mr, mp: _enforce_positivity_native(  # noqa: E731
            s, g, mr, mp, config, registered_variables,
        )
        return diffable_pallas_call_n(
            (conserved_state, gamma, minimum_density, minimum_pressure),
            pallas_branch=pallas, native_branch=native,
        )
    return _enforce_positivity_native(
        conserved_state, gamma, minimum_density, minimum_pressure,
        config, registered_variables,
    )


def _enforce_positivity_native_impl(
    conserved_state: STATE_TYPE,
    config: SimulationConfig,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    # Optional NaN/inf backstop: jnp.maximum(NaN, floor) = NaN, so a non-finite
    # cell would otherwise survive the floor and propagate. Reset any non-finite
    # entry to zero first; the density/pressure floors below then turn it into a
    # valid floor state (rho -> minimum_density, momentum 0, pressure floored).
    if config.positivity_config.nan_safe:
        conserved_state = jnp.nan_to_num(
            conserved_state, nan=0.0, posinf=0.0, neginf=0.0
        )

    rho = conserved_state[registered_variables.density_index]

    # Vacuum-rest: cells below the density floor are effectively vacuum; zero
    # their momentum so the recovered velocity is 0 rather than
    # momentum / minimum_density (which spikes and triggers high-Mach blow-up).
    # Done before the floor + energy recovery so the kinetic energy is consistent.
    if config.positivity_config.vacuum_rest:
        floored = rho < minimum_density
        mom = registered_variables.momentum_index
        if config.dimensionality == 1:
            mom_indices = (mom,)
        elif config.dimensionality == 2:
            mom_indices = (mom.x, mom.y)
        else:
            mom_indices = (mom.x, mom.y, mom.z)
        for mi in mom_indices:
            conserved_state = conserved_state.at[mi].set(
                jnp.where(floored, 0.0, conserved_state[mi])
            )

    # enforce minimum density
    rho = jnp.maximum(rho, minimum_density)

    # the energy only needs to be updated in the ideal gas case
    if config.equation_of_state == IDEAL_GAS:

        if config.dimensionality == 1:
            v_x = conserved_state[registered_variables.momentum_index] / rho
        else:
            v_x = conserved_state[registered_variables.momentum_index.x] / rho

        if config.dimensionality == 2:
            v_y = conserved_state[registered_variables.momentum_index.y] / rho
            v_z = 0.0
        elif config.dimensionality == 3:
            v_y = conserved_state[registered_variables.momentum_index.y] / rho
            v_z = conserved_state[registered_variables.momentum_index.z] / rho

        energy = conserved_state[registered_variables.energy_index]

        if config.mhd:
            B_x = conserved_state[registered_variables.magnetic_index.x]
            B_y = conserved_state[registered_variables.magnetic_index.y]
            B_z = conserved_state[registered_variables.magnetic_index.z]

            b2 = B_x**2 + B_y**2 + B_z**2
        
        if config.dimensionality == 1:
            v2 = v_x**2
        elif config.dimensionality == 2:
            v2 = v_x**2 + v_y**2
        elif config.dimensionality == 3:
            v2 = v_x**2 + v_y**2 + v_z**2

        # calculate pressure
        if config.mhd:
            pressure = (gamma - 1.0) * (energy - 0.5 * rho * v2 - 0.5 * b2)
        else:
            pressure = (gamma - 1.0) * (energy - 0.5 * rho * v2)
        
        pressure = jnp.maximum(pressure, minimum_pressure)

        # redefine energy with new pressure
        if config.mhd:
            energy = pressure / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * b2
        else:
            energy = pressure / (gamma - 1.0) + 0.5 * rho * v2

        # reconstruct conserved state
        conserved_state = conserved_state.at[registered_variables.energy_index].set(energy)
    
    # for both the ideal gas and isothermal case, we need to update the density
    conserved_state = conserved_state.at[registered_variables.density_index].set(rho)

    return conserved_state

def _momentum_indices(config, registered_variables):
    """Conserved-state momentum component indices for the active dimensionality."""
    if config.dimensionality == 1:
        return [registered_variables.momentum_index]
    if config.dimensionality == 2:
        return [registered_variables.momentum_index.x,
                registered_variables.momentum_index.y]
    return [registered_variables.momentum_index.x,
            registered_variables.momentum_index.y,
            registered_variables.momentum_index.z]


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _redistribute_positivity(
    conserved_state: STATE_TYPE,
    threshold: Union[float, Float[Array, ""]],
    max_velocity: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Dispatch: Pallas neighbour-redistribution kernel when supported, else the
    native-JAX stencil. AD (jvp/vjp) routes through the native branch."""
    if _redistribute_positivity_pallas_supported(conserved_state, config):
        pallas = lambda s, thr, vmax, g, pmin: _redistribute_positivity_pallas(  # noqa: E731
            s, thr, vmax, g, pmin, config, registered_variables,
        )
        native = lambda s, thr, vmax, g, pmin: _redistribute_positivity_native(  # noqa: E731
            s, thr, vmax, g, pmin, config, registered_variables,
        )
        return diffable_pallas_call_n(
            (conserved_state, threshold, max_velocity, gamma, minimum_pressure),
            pallas_branch=pallas, native_branch=native,
        )
    return _redistribute_positivity_native(
        conserved_state, threshold, max_velocity, gamma, minimum_pressure,
        config, registered_variables,
    )


def _redistribute_positivity_native(
    conserved_state: STATE_TYPE,
    threshold: Union[float, Float[Array, ""]],
    max_velocity: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Neighbour redistribution of sub-threshold (vacuum) cells, a conserved-state
    generalisation of the HOW-MHD *isothermal* ``prot.f``.

    For every cell with ``rho <= threshold`` the density and momentum (and, for
    ideal gas, the total energy) are replaced by the average over the valid
    (``rho > threshold``) cells in its 3x3x3 neighbourhood; the patched velocity
    is the mass-weighted neighbour mean, clipped to ``max_velocity``. Isolated
    vacuum cells (no valid neighbour) fall back to ``rho = threshold`` keeping
    their momentum.

    vs HARD_FLOOR: a hard floor leaves a sharp ``threshold`` cell next to ~O(1)
    neighbours (a strong artificial gradient that the high-order WENO update then
    has to digest); this replaces it with a smooth, physically-plausible
    neighbour-averaged value, which is markedly gentler at strong shocks. NOTE: it
    is NOT strictly mass-conserving — like ``prot.f`` it copies the neighbour
    average without debiting the donors, so a tiny amount of mass is added at the
    (rare) patched cells; the win is smoothness/stability, not exact conservation.
    Native-only (3x3x3 neighbour stencil; no Pallas sibling).
    """
    ndim = config.dimensionality
    di = registered_variables.density_index
    rho = conserved_state[di]
    mom_idx = _momentum_indices(config, registered_variables)
    mom = [conserved_state[i] for i in mom_idx]

    is_invalid = rho <= threshold
    valid_f = (~is_invalid).astype(rho.dtype)

    axes = tuple(range(ndim))

    def sum_neighbors(arr):
        out = jnp.zeros_like(arr)
        for shift in itertools.product((-1, 0, 1), repeat=ndim):
            out = out + jnp.roll(arr, shift=shift, axis=axes)
        return out

    rho_sum = sum_neighbors(rho * valid_f)
    count_sum = sum_neighbors(valid_f)
    mom_sum = [sum_neighbors(m * valid_f) for m in mom]

    has = count_sum > 0
    count_safe = jnp.where(has, count_sum, 1.0)
    rho_sum_safe = jnp.where(has, rho_sum, 1.0)

    rho_patched = jnp.where(has, rho_sum / count_safe, threshold)
    mom_patched = []
    for m, ms in zip(mom, mom_sum):
        # mass-weighted neighbour mean velocity = sum(mom) / sum(rho); an
        # isolated cell (no valid neighbour) is a genuine deep-void / vacuum cell.
        # Without vacuum_rest it keeps v = mom/threshold, which RUNS AWAY in deep
        # voids: floored density pins rho at the threshold so any residual
        # momentum gives a large recovered velocity, that velocity is re-injected
        # every RK substage (the per-substage positivity runs before each flux),
        # and the high-Mach WENO reconstruction across the void then overshoots
        # until the step blows up. Resting the isolated cell (v=0) when
        # vacuum_rest is on removes the run-away while leaving the gentle
        # neighbour-fill for void *edges* (has=True) untouched. Mirrors the
        # `_enforce_positivity` vacuum_rest semantics (vacuum -> zero momentum).
        isolated_v = 0.0 if config.positivity_config.vacuum_rest else (m / threshold)
        v = jnp.where(has, ms / rho_sum_safe, isolated_v)
        v = jnp.clip(v, -max_velocity, max_velocity)
        mom_patched.append(rho_patched * v)

    out = conserved_state.at[di].set(jnp.where(is_invalid, rho_patched, rho))
    for i, m, mp in zip(mom_idx, mom, mom_patched):
        out = out.at[i].set(jnp.where(is_invalid, mp, m))

    if config.equation_of_state == IDEAL_GAS:
        # redistribute total energy the same way, then re-floor pressure
        ei = registered_variables.energy_index
        E = conserved_state[ei]
        E_patched = jnp.where(has, sum_neighbors(E * valid_f) / count_safe, E)
        out = out.at[ei].set(jnp.where(is_invalid, E_patched, E))
        out = _enforce_positivity_native_impl(
            out, config, gamma, threshold, minimum_pressure, registered_variables,
        )

    return out


def _internal_energy(conserved_state, rho, config, registered_variables):
    """Internal energy density E - KE - E_mag from a conserved state."""
    if config.dimensionality == 1:
        v_x = conserved_state[registered_variables.momentum_index] / rho
        v2 = v_x ** 2
    else:
        v_x = conserved_state[registered_variables.momentum_index.x] / rho
        v_y = conserved_state[registered_variables.momentum_index.y] / rho
        if config.dimensionality == 2:
            v2 = v_x ** 2 + v_y ** 2
        else:
            v_z = conserved_state[registered_variables.momentum_index.z] / rho
            v2 = v_x ** 2 + v_y ** 2 + v_z ** 2
    e_int = conserved_state[registered_variables.energy_index] - 0.5 * rho * v2
    if config.mhd:
        b2 = (conserved_state[registered_variables.magnetic_index.x] ** 2
              + conserved_state[registered_variables.magnetic_index.y] ** 2
              + conserved_state[registered_variables.magnetic_index.z] ** 2)
        e_int = e_int - 0.5 * b2
    return e_int


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _conservative_energy_positivity(
    conserved_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    positivity_max_velocity: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Conservative internal-energy positivity enforcement on the full state.

    Where the internal energy is (near) negative — exactly the cells the
    energy-conserving self-gravity scheme drives unstable on a violent collapse —
    pull internal energy in from hotter neighbours via an antisymmetric face flux
        f_{i+1/2} = coeff * A_{i+1/2} * (e_i - e_{i+1}),
    activated only at faces touching a near-floor cell. Because f is a face flux,
    total energy is conserved EXACTLY for any activation pattern. A density floor
    (+ optional vacuum-rest) handles voids, and a final minimal pressure floor
    guarantees positivity unconditionally — but the prior redistribution makes
    that residual injection negligible (vs the 100%+ a bare HARD_FLOOR causes on
    deep collapse). This sees the TRUE post-update internal energy (it runs in the
    SSPRK stage), unlike the source-level ``gravity_energy_pp_redistribute``.
    """
    if config.positivity_config.nan_safe:
        conserved_state = jnp.nan_to_num(
            conserved_state, nan=0.0, posinf=0.0, neginf=0.0
        )

    di = registered_variables.density_index
    rho = conserved_state[di]

    # vacuum-rest: zero momentum in below-floor (vacuum) cells before recovery
    if config.positivity_config.vacuum_rest:
        floored = rho < minimum_density
        for mi in _momentum_indices(config, registered_variables):
            conserved_state = conserved_state.at[mi].set(
                jnp.where(floored, 0.0, conserved_state[mi])
            )

    rho = jnp.maximum(rho, minimum_density)
    conserved_state = conserved_state.at[di].set(rho)

    if config.equation_of_state == IDEAL_GAS:
        e_int = _internal_energy(conserved_state, rho, config, registered_variables)
        e_floor = minimum_pressure / (gamma - 1.0)
        activate = config.positivity_config.cons_activate * e_floor
        w = config.positivity_config.cons_coeff

        corrected = e_int
        for _ in range(config.positivity_config.cons_passes):
            transfer = jnp.zeros_like(corrected)
            for ax in range(config.dimensionality):
                nbr = jnp.roll(corrected, shift=-1, axis=ax)   # cell i+1 at index i
                active = (jnp.minimum(corrected, nbr) < activate).astype(
                    corrected.dtype
                )
                f = w * active * (corrected - nbr)             # flux i -> i+1
                transfer = transfer + jnp.roll(f, shift=1, axis=ax) - f
            corrected = corrected + transfer

        # conservative correction: shift total energy by the redistribution
        conserved_state = conserved_state.at[registered_variables.energy_index].add(
            corrected - e_int
        )

    # velocity cap: clip each velocity component to +-positivity_max_velocity in
    # the rare runaway cells (a deep-collapse blow-up is a negative-pressure ->
    # velocity-spike cascade; capping |v| breaks it). Removes only the excess
    # kinetic energy of those pathological cells; the matching total-energy term
    # is updated so the recovered pressure is unchanged. A no-op when
    # positivity_max_velocity is inf (clip is identity, energy delta is 0).
    rho_c = conserved_state[registered_variables.density_index]
    for mi in _momentum_indices(config, registered_variables):
        v = conserved_state[mi] / rho_c
        v_cap = jnp.clip(v, -positivity_max_velocity, positivity_max_velocity)
        # adjust total energy by the kinetic-energy change so e_int is preserved
        conserved_state = conserved_state.at[registered_variables.energy_index].add(
            0.5 * rho_c * (v_cap ** 2 - v ** 2)
        )
        conserved_state = conserved_state.at[mi].set(rho_c * v_cap)

    # final density + (residual) pressure floor — the unconditional guarantee
    return _enforce_positivity_native_impl(
        conserved_state, config, gamma, minimum_density, minimum_pressure,
        registered_variables,
    )


def _apply_stage_positivity(
    conserved_state: STATE_TYPE,
    mode: int,
    config: SimulationConfig,
    gamma: Union[float, Float[Array, ""]],
    minimum_density: Union[float, Float[Array, ""]],
    minimum_pressure: Union[float, Float[Array, ""]],
    positivity_max_velocity: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Dispatch a positivity mode onto a conserved state (``mode`` is static)."""
    if mode == POSITIVITY_HARD_FLOOR:
        return _enforce_positivity(
            conserved_state, config, gamma,
            minimum_density, minimum_pressure, registered_variables,
        )
    if mode == POSITIVITY_REDISTRIBUTE:
        return _redistribute_positivity(
            conserved_state, minimum_density, positivity_max_velocity, gamma,
            minimum_pressure, config, registered_variables,
        )
    if mode == POSITIVITY_CONSERVATIVE:
        return _conservative_energy_positivity(
            conserved_state, gamma, minimum_density, minimum_pressure,
            positivity_max_velocity, config, registered_variables,
        )
    return conserved_state


# Bottom-of-file Pallas import (avoids circular import — see guide §2.4).
from astronomix._fluid_equations._enforce_positivity_pallas import (  # noqa: E402
    _enforce_positivity_pallas,
    _enforce_positivity_pallas_supported,
    _redistribute_positivity_pallas,
    _redistribute_positivity_pallas_supported,
)
