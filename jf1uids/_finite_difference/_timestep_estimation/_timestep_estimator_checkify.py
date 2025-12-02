# general imports
import jax.numpy as jnp
import jax
from functools import partial


# type checking imports
from jaxtyping import Array, Float
from beartype import beartype as typechecker
from typing import Union

# general jf1uids imports
from jf1uids._finite_difference._fluid_equations._eigen import _eigen_all_lambdas, _eigen_x
from jf1uids._finite_difference._fluid_equations._equations import conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd
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
def _cfl_time_step_fd(
    primitive_state,
    grid_spacing,
    dt_max,
    gamma,
    config,
    params,
    registered_variables,
    C_CFL=0.8,
):

    # -------------------------------------------------------
    # Basic physical sanity checks BEFORE transformations
    # -------------------------------------------------------
    # rho = primitive_state[registered_variables.density_index]
    # pres = primitive_state[registered_variables.pressure_index]

    # jax_checkify.check(jnp.all(rho > params.minimum_density * 0.01),
    #                    "Density too small → unstable gradients in CFL")
    # jax_checkify.check(jnp.all(pres > params.minimum_pressure * 0.01),
    #                    "Pressure too small → unstable gradients in CFL")

    # -------------------------------------------------------
    # Convert to conserved variables
    # -------------------------------------------------------
    conserved_state = conserved_state_from_primitive_mhd(
        primitive_state, gamma, registered_variables
    )

    jax_checkify.check(jnp.all(jnp.isfinite(conserved_state)),
                       "conserved_state contains non-finite values")

    # -------------------------------------------------------
    # Compute eigenvalues in X direction
    # -------------------------------------------------------
    lambda_x_all = _eigen_all_lambdas(
        conserved_state,
        params.minimum_density,
        params.minimum_pressure,
        gamma,
        registered_variables,
    )

    jax_checkify.check(jnp.all(jnp.isfinite(lambda_x_all)),
                       "Eigenvalues (X) contain NaN/Inf")

    jax_checkify.check(jnp.max(jnp.abs(lambda_x_all)) < 1e6,
                       "Eigenvalues (X) are too large → gradient explosion risk")

    lambda_x = jnp.max(jnp.abs(lambda_x_all))

    # -------------------------------------------------------
    # Compute eigenvalues in Y direction
    # -------------------------------------------------------
    qy = jnp.transpose(conserved_state, (0, 2, 1, 3))

    jax_checkify.check(jnp.all(jnp.isfinite(qy)),
                       "qy transpose produced non-finite values")

    lambda_y_all = _eigen_all_lambdas(
        qy,
        params.minimum_density,
        params.minimum_pressure,
        gamma,
        registered_variables,
    )

    jax_checkify.check(jnp.all(jnp.isfinite(lambda_y_all)),
                       "Eigenvalues (Y) contain NaN/Inf")

    jax_checkify.check(jnp.max(jnp.abs(lambda_y_all)) < 1e6,
                       "Eigenvalues (Y) too large → gradient explosion")

    lambda_y = jnp.max(jnp.abs(lambda_y_all))

    # -------------------------------------------------------
    # Compute eigenvalues in Z direction
    # -------------------------------------------------------
    qz = jnp.transpose(conserved_state, (0, 3, 2, 1))

    jax_checkify.check(jnp.all(jnp.isfinite(qz)),
                       "qz transpose produced non-finite values")

    lambda_z_all = _eigen_all_lambdas(
        qz,
        params.minimum_density,
        params.minimum_pressure,
        gamma,
        registered_variables,
    )

    jax_checkify.check(jnp.all(jnp.isfinite(lambda_z_all)),
                       "Eigenvalues (Z) contain NaN/Inf")

    jax_checkify.check(jnp.max(jnp.abs(lambda_z_all)) < 1e6,
                       "Eigenvalues (Z) too large → gradient explosion")

    lambda_z = jnp.max(jnp.abs(lambda_z_all))

    # -------------------------------------------------------
    # CFL denominator safety check
    # -------------------------------------------------------
    denom = lambda_x + lambda_y + lambda_z

    jax_checkify.check(jnp.all(jnp.abs(denom) > 1e-8),
                       "CFL denominator too small → dt → huge → unstable gradients")

    # -------------------------------------------------------
    # CFL time step
    # -------------------------------------------------------
    dt_cfl = C_CFL * grid_spacing / denom

    jax_checkify.check(jnp.isfinite(dt_cfl),
                       "dt_cfl is NaN/Inf")

    jax_checkify.check(jnp.all(dt_cfl > 0.0),
                       "dt_cfl is non-positive")

    jax_checkify.check(jnp.all(dt_cfl < dt_max * 10),
                       "dt_cfl is suspiciously larger than dt_max → likely division issue")

    dt_cfl = jnp.minimum(dt_cfl, dt_max)

    return dt_cfl
