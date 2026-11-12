"""
The 16^3 blast shared by the differentiability tests.

A dense, homologously expanding ejecta ball drives a Mach ~8 shock into a cold
ambient medium in a periodic box. It is evolved with the finite-difference WENO
solver in the configuration a supernova-remnant inference differentiates: the
positivity-preserving reconstruction, the dual-energy formalism, five bounded
composition passive scalars and frozen WENO weights in the tangent.

This is a helper module, not a test module; the tests import it as a sibling
module (pytest puts the test directory on ``sys.path``).
"""

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    CARTESIAN,
    FINITE_DIFFERENCE,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    PassiveScalarConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
)


#: The adiabatic index of the gas.
GAMMA = 5.0 / 3.0

#: Cells per axis of the cubic box.
NUM_CELLS = 16

#: The number of (composition) passive scalars.
NUM_PASSIVE_SCALARS = 5

#: Position of the ejecta fraction ``C_ej`` among the user's passive scalars. It
#: is exactly 1 in the ejecta and exactly 0 in the ambient medium, i.e. it sits on
#: one of its bounds almost everywhere.
EJECTA_FRACTION_SCALAR = 0

#: The passive-scalar settings of the blast (default sub-cycling); tests that
#: force sub-steps pass a ``_replace`` of it as ``passive_scalar_config``.
PASSIVE_SCALAR_CONFIG = PassiveScalarConfig(
    num_scalars=NUM_PASSIVE_SCALARS,
    bounds=tuple((0.0, 1.0) for _ in range(NUM_PASSIVE_SCALARS)),
)

#: End time of the adaptive runs: four CFL-limited steps of the blast.
T_END = 0.012

#: Number of steps of the fixed-step runs.
NUM_FIXED_STEPS = 3

#: Time step of the fixed-step runs. It lies well inside the CFL limit (the
#: first adaptive step is 0.008): the coarsely resolved ejecta edge only stays
#: benign there, while larger steps crush single cells and let the
#: linearisation grow by many orders of magnitude.
DT_FIXED = 0.0025


def blast_config(**overrides):
    """
    The solver configuration of the blast.

    Args:
        **overrides: SimulationConfig fields replacing the defaults below
            (differentiation mode, fixed steps, reverse-mode memory options,
            ``passive_scalar_config`` built from ``PASSIVE_SCALAR_CONFIG`` for
            the passive-scalar sub-cycling, ...).

    Returns:
        The simulation configuration, not yet finalized.
    """
    periodic_boundaries = BoundarySettings1D(
        PERIODIC_BOUNDARY,
        PERIODIC_BOUNDARY,
    )
    options = dict(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        geometry=CARTESIAN,
        solver_config=FDSolverConfig(
            weno_positivity_preserving=True,
            weno_ad_frozen_weights=True,
            dual_energy=True,
        ),
        box_size=1.0,
        num_cells=NUM_CELLS,
        boundary_settings=BoundarySettings(
            periodic_boundaries,
            periodic_boundaries,
            periodic_boundaries,
        ),
        passive_scalar_config=PASSIVE_SCALAR_CONFIG,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        progress_bar=False,
        num_checkpoints=8,
    )
    options.update(overrides)
    return SimulationConfig(**options)


def blast_params(t_end=T_END):
    """
    The simulation parameters of the blast.

    Args:
        t_end: The end time of the run.

    Returns:
        The simulation parameters.
    """
    return SimulationParams(
        gamma=GAMMA,
        C_cfl=0.3,
        t_end=t_end,
        minimum_density=1e-6,
        minimum_pressure=1e-8,
    )


def setup_blast(**overrides):
    """
    Set up the blast: a homologously expanding ejecta ball (``C_ej = 1``, exactly
    on its upper bound) driving a Mach ~8 shock into a cold ambient medium
    (``C_ej = 0``, exactly on its lower bound).

    Density and pressure carry weak smooth modulations, so that no direction of
    the box is special.

    Args:
        **overrides: SimulationConfig fields replacing the defaults of
            :func:`blast_config`.

    Returns:
        The finalized simulation configuration, the registered variables and the
        initial primitive state.
    """
    config = blast_config(**overrides)
    registered_variables = get_registered_variables(config)

    # -------------------------------------------------------------
    # ==================== ↓ Ejecta and ambient ↓ =================
    # -------------------------------------------------------------

    cell_centres = (jnp.arange(NUM_CELLS) + 0.5) / NUM_CELLS - 0.5
    x, y, z = jnp.meshgrid(
        cell_centres,
        cell_centres,
        cell_centres,
        indexing="ij",
    )
    radius = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)
    ejecta_radius = 0.2
    inside_ejecta = radius < ejecta_radius

    density = (
        jnp.where(inside_ejecta, 5.0, 1.0)
        + 0.05 * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    )
    # Homologous expansion: the radial velocity grows linearly with radius.
    radial_velocity = jnp.where(inside_ejecta, 0.8 * radius / ejecta_radius, 0.0)
    radius_away_from_zero = jnp.maximum(radius, 1e-12)
    pressure = jnp.where(inside_ejecta, 5e-2, 1e-2) * (1.0 + 0.1 * jnp.cos(2 * jnp.pi * z))

    # -------------------------------------------------------------
    # ==================== ↑ Ejecta and ambient ↑ =================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ====================== ↓ Composition ↓ ======================
    # -------------------------------------------------------------

    ejecta_fraction = jnp.where(inside_ejecta, 1.0, 0.0)
    fractional_ejecta_radius = jnp.clip(radius / ejecta_radius, 0.0, 1.0)
    passive_scalars = jnp.stack([
        # The ejecta fraction (EJECTA_FRACTION_SCALAR).
        ejecta_fraction,
        # An ejecta species concentrated towards the centre.
        ejecta_fraction * 0.4 * (1.0 - fractional_ejecta_radius) + 0.001,
        # An ejecta species concentrated towards the ejecta edge.
        ejecta_fraction * 0.3 * fractional_ejecta_radius + 0.0007,
        # A species varying smoothly across the whole box.
        0.25 + 0.2 * jnp.sin(2 * jnp.pi * x),
        # A species that is depleted in the ejecta.
        0.28 - 0.1 * ejecta_fraction,
    ])

    # -------------------------------------------------------------
    # ====================== ↑ Composition ↑ ======================
    # -------------------------------------------------------------

    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=radial_velocity * x / radius_away_from_zero,
        velocity_y=radial_velocity * y / radius_away_from_zero,
        velocity_z=radial_velocity * z / radius_away_from_zero,
        gas_pressure=pressure,
        passive_scalars=passive_scalars,
    )
    config = finalize_config(config, state.shape)
    return config, registered_variables, state


def fluid_only_registry(registered_variables):
    """
    The registry of the state with the passive-scalar block split off.

    The passive-scalar advection receives the fluid state without the scalar
    block, as in the solver's own update; this registry describes that state
    (the dual-energy row stays).

    Args:
        registered_variables: The registered variables of the full state.

    Returns:
        The registered variables of the fluid part of the state.
    """
    return registered_variables._replace(
        num_vars=registered_variables.passive_scalar_index,
        passive_scalar_index=-1,
        num_passive_scalars=0,
        passive_scalars_active=False,
    )


def get_fluid_state(state, registered_variables):
    """The rows of the state in front of the passive-scalar block (hydro and dual energy)."""
    return state[:registered_variables.passive_scalar_index]
