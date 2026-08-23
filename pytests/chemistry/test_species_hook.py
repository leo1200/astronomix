"""
Tests for the advected chemical-species block and the reaction source-term hook.

These exercise the mechanism only (no chemistry engine): that species register
as a contiguous state block, advect with the finite-volume flow, and that a
user-supplied ``source_term`` is invoked once per step and can modify them.
"""

# ==== GPU selection (repo convention) ====
from autocvd import autocvd

# least_used=True picks the least-loaded GPU immediately rather than blocking
# until one is fully idle, so the test does not hang on a shared machine.
autocvd(num_gpus=1, least_used=True)
# ruff: noqa: E402
# =========================================

import jax

# astronomix integrates in double precision; enable it up front so the initial
# state and the time-loop carry share a dtype.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    BoundarySettings1D,
    SimulationConfig,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix._modules._chemistry.chemistry_options import ChemistryConfig

NUM_CELLS = 64
ADVECTION_VELOCITY = 0.5
END_TIME = 0.2


def _run(source_term):
    """Run a 1-D periodic box that advects a single species bump at a uniform
    velocity, with an optional reaction source term. Returns (positions, initial
    species, final species)."""
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        backend=NATIVE_JAX,
        dimensionality=1,
        num_cells=NUM_CELLS,
        box_size=1.0,
        boundary_settings=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        progress_bar=False,
        chemistry_config=ChemistryConfig(
            chemistry=True,
            number_of_chemical_species=1,
            species_names=("tracer",),
            source_term=source_term,
        ),
    )
    params = SimulationParams(t_end=END_TIME, C_cfl=0.4)
    registered_variables = get_registered_variables(config)

    position = get_helper_data(config).geometric_centers
    density = jnp.ones_like(position)
    velocity = jnp.full_like(position, ADVECTION_VELOCITY)
    pressure = jnp.ones_like(position)
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity,
        gas_pressure=pressure,
    )

    # A smooth interior bump in the (single) species field.
    bump = jnp.exp(-(((position - 0.3) / 0.08) ** 2)) + 1e-6
    species_index = registered_variables.chemistry_species_index
    state = state.at[species_index].set(bump)

    config = finalize_config(config, state.shape)
    final_state = time_integration(state, config, params, registered_variables)
    return position, state[species_index], final_state[species_index]


def test_species_block_registration():
    """The species block is appended to the state only when chemistry is on."""
    base = SimulationConfig(solver_mode=FINITE_VOLUME, dimensionality=1, num_cells=16)
    registered_off = get_registered_variables(base)
    core_vars = registered_off.num_vars

    assert not registered_off.chemistry_species_active

    with_chemistry = base._replace(
        chemistry_config=ChemistryConfig(
            chemistry=True,
            number_of_chemical_species=4,
            species_names=("a", "b", "c", "d"),
        )
    )
    registered_on = get_registered_variables(with_chemistry)

    assert registered_on.chemistry_species_active
    assert registered_on.num_chemical_species == 4
    # the block is contiguous and appended after the core variables
    assert registered_on.chemistry_species_index == core_vars
    assert registered_on.num_vars == core_vars + 4


def test_species_advects_with_the_flow():
    """With no source term, the species bump is transported at the flow speed and
    its total mass is conserved."""
    position, initial_species, final_species = _run(source_term=None)

    def center_of_mass(field):
        return jnp.sum(position * field) / jnp.sum(field)

    shift = float(center_of_mass(final_species) - center_of_mass(initial_species))
    expected = ADVECTION_VELOCITY * END_TIME

    assert abs(shift - expected) < 0.03, f"bump advected by {shift}, expected {expected}"

    initial_mass = float(jnp.sum(initial_species))
    final_mass = float(jnp.sum(final_species))
    assert abs(final_mass - initial_mass) / initial_mass < 0.02

    assert bool(jnp.all(final_species >= 0.0))
    assert bool(jnp.all(jnp.isfinite(final_species)))


def test_source_term_hook_is_invoked():
    """A source term that scales the species down each step must leave the final
    mass well below the advection-only (no-source) case, proving the hook fires."""

    def scaling_source(primitive_state, registered_variables, chemistry_config, chemistry_params, dt):
        start = registered_variables.chemistry_species_index
        count = registered_variables.num_chemical_species
        return primitive_state.at[start : start + count].multiply(0.5)

    _, _, final_no_source = _run(source_term=None)
    _, _, final_with_source = _run(source_term=scaling_source)

    mass_no_source = float(jnp.sum(final_no_source))
    mass_with_source = float(jnp.sum(final_with_source))

    assert mass_with_source < 0.9 * mass_no_source, (
        f"source term had no effect: {mass_with_source} vs {mass_no_source}"
    )
    assert bool(jnp.all(jnp.isfinite(final_with_source)))
