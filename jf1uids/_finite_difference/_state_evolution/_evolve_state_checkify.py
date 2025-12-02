# general imports
import jax.numpy as jnp
import jax
from functools import partial


# type checking imports
from jaxtyping import Array, Float
from beartype import beartype as typechecker
from typing import Union

# general jf1uids imports
from jf1uids._finite_difference._fluid_equations._equations import conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd
from jf1uids._finite_difference._time_integrators._ssprk import _ssprk4_with_ct
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.variable_registry.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import (
    STATE_TYPE,
    SimulationConfig,
)

from jf1uids.option_classes.simulation_params import SimulationParams
import jax.numpy as jnp
import jax
import jax.experimental.checkify as jax_checkify
from functools import partial

# @partial(jax.jit, static_argnames=["config", "registered_variables"])
def _evolve_state_fd(
    primitive_state,
    dt,
    gamma,
    gravitational_constant,
    config,
    params,
    helper_data,
    registered_variables,
):

    # -------------------------------------------------------
    # Basic physical validity checks on input primitive state
    # -------------------------------------------------------
    rho = primitive_state[registered_variables.density_index]
    pres = primitive_state[registered_variables.pressure_index]

    jax_checkify.check(jnp.all(rho > params.minimum_density * 0.1),
                       "Primitive input: density too close to minimum → unstable gradients")

    jax_checkify.check(jnp.all(pres > params.minimum_pressure * 0.1),
                       "Primitive input: pressure too close to minimum → unstable gradients")

    jax_checkify.check(jnp.all(jnp.isfinite(primitive_state)),
                       "Primitive input contains NaN/Inf")

    # -------------------------------------------------------
    # Convert primitive → conserved
    # -------------------------------------------------------
    conserved_state = conserved_state_from_primitive_mhd(
        primitive_state[:-3], gamma, registered_variables
    )

    jax_checkify.check(jnp.all(jnp.isfinite(conserved_state)),
                       "conserved_state_from_primitive_mhd produced NaN/Inf")

    jax_checkify.check(jnp.max(jnp.abs(conserved_state)) < 1e8,
                       "conserved_state magnitude too large → gradient explosion risk")

    # -------------------------------------------------------
    # Extract interface magnetic fields
    # -------------------------------------------------------
    bxb = primitive_state[registered_variables.interface_magnetic_field_index.x]
    byb = primitive_state[registered_variables.interface_magnetic_field_index.y]
    bzb = primitive_state[registered_variables.interface_magnetic_field_index.z]

    jax_checkify.check(jnp.all(jnp.isfinite(bxb)),
                       "bxb contains NaN/Inf")
    jax_checkify.check(jnp.all(jnp.isfinite(byb)),
                       "byb contains NaN/Inf")
    jax_checkify.check(jnp.all(jnp.isfinite(bzb)),
                       "bzb contains NaN/Inf")

    # -------------------------------------------------------
    # SSPRK4 time integration with constrained transport
    # -------------------------------------------------------
    conserved_state, bxb, byb, bzb = _ssprk4_with_ct(
        conserved_state,
        bxb,
        byb,
        bzb,
        gamma,
        config.grid_spacing,
        dt,
        params,
        helper_data,
        config,
        registered_variables,
    )

    # After RK step — inspect the updated arrays
    jax_checkify.check(jnp.all(jnp.isfinite(conserved_state)),
                       "SSPRK4: conserved_state contains NaN/Inf post-update")
    jax_checkify.check(jnp.max(jnp.abs(conserved_state)) < 1e8,
                       "SSPRK4: conserved_state magnitude too large → backward blowup")

    jax_checkify.check(jnp.all(jnp.isfinite(bxb)),
                       "SSPRK4: bxb contains NaN/Inf")
    jax_checkify.check(jnp.all(jnp.isfinite(byb)),
                       "SSPRK4: byb contains NaN/Inf")
    jax_checkify.check(jnp.all(jnp.isfinite(bzb)),
                       "SSPRK4: bzb contains NaN/Inf")

    # -------------------------------------------------------
    # Convert conserved → primitive
    # -------------------------------------------------------
    primitive_state = primitive_state_from_conserved_mhd(
        conserved_state, gamma, registered_variables
    )

    jax_checkify.check(jnp.all(jnp.isfinite(primitive_state)),
                       "primitive_state_from_conserved_mhd produced NaN/Inf")

    jax_checkify.check(
        jnp.all(primitive_state[registered_variables.density_index] > params.minimum_density * 0.1),
        "Post-update primitive density extremely small → unstable gradients"
    )

    jax_checkify.check(
        jnp.all(primitive_state[registered_variables.pressure_index] > params.minimum_pressure * 0.1),
        "Post-update primitive pressure extremely small → unstable gradients"
    )

    # -------------------------------------------------------
    # Append the updated interface magnetic fields
    # -------------------------------------------------------
    primitive_state = jnp.concatenate(
        [primitive_state, bxb[None, :], byb[None, :], bzb[None, :]], axis=0
    )

    jax_checkify.check(jnp.all(jnp.isfinite(primitive_state)),
                       "Final primitive_state contains NaN/Inf after concatenation")

    return primitive_state
