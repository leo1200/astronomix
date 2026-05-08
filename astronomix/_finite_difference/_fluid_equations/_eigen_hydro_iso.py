"""
Computations of the eigenvalues and eigenvectors for the isothermal Euler
(hydrodynamics) equations.

Isothermal hydro has no energy equation. The sound speed cs is a fixed parameter.
The conserved variables are density and momenta only.

Waves:
  1D: u-cs, u+cs  (2 waves)
  2D: u-cs, u (shear y), u+cs  (3 waves)
  3D: u-cs, u (shear y), u (shear z), u+cs  (4 waves)

There is no entropy wave since there is no energy/pressure to evolve.
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Union

from astronomix._stencil_operations._stencil_operations import _shift
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables


def diff_safe_sqrt(x):
    epsilon = 1e-12
    x_safe = jnp.maximum(x, epsilon)
    return jnp.sqrt(x_safe)


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigenvalue_building_blocks(
    conserved_state,
    sound_speed,
    rhomin,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    density = conserved_state[registered_variables.density_index]

    if config.dimensionality == 1:
        momentum_x = conserved_state[registered_variables.momentum_index]
    else:
        momentum_x = conserved_state[registered_variables.momentum_index.x]


    rho = jnp.maximum(density, rhomin)
    velocity_x = momentum_x / rho

    return (velocity_x, sound_speed)


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigenvector_building_blocks(
    conserved_state,
    sound_speed,
    rhomin,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    rho = conserved_state[registered_variables.density_index]
    momentum_x = conserved_state[registered_variables.momentum_index.x]

    if config.dimensionality == 1:
        momentum_y = 0.0
        momentum_z = 0.0
    elif config.dimensionality == 2:
        momentum_y = conserved_state[registered_variables.momentum_index.y]
        momentum_z = 0.0
    elif config.dimensionality == 3:
        momentum_y = conserved_state[registered_variables.momentum_index.y]
        momentum_z = conserved_state[registered_variables.momentum_index.z]

    rho = jnp.maximum(rho, rhomin)
    velocity_x = momentum_x / rho
    velocity_y = momentum_y / rho
    velocity_z = momentum_z / rho

    def avg_x(arr):
        return 0.5 * (arr + _shift(arr, shift=-1, axis=0))

    # velocity_x_interface = avg_x(velocity_x)

    # if config.dimensionality == 1:
    #     velocity_y_interface = 0.0
    #     velocity_z_interface = 0.0
    # elif config.dimensionality == 2:
    #     velocity_y_interface = avg_x(velocity_y)
    #     velocity_z_interface = 0.0
    # elif config.dimensionality == 3:
    #     velocity_y_interface = avg_x(velocity_y)
    #     velocity_z_interface = avg_x(velocity_z)

    # average momenta approach
    rho_interface = avg_x(jnp.maximum(rho, rhomin))
    rho_interface = jnp.maximum(rho_interface, rhomin)
    velocity_x_interface = avg_x(momentum_x) / rho_interface

    if config.dimensionality == 1:
        velocity_y_interface = 0.0
        velocity_z_interface = 0.0
    elif config.dimensionality == 2:
        # velocity_y_interface = avg_x(velocity_y)
        velocity_y_interface = avg_x(momentum_y) / rho_interface
        velocity_z_interface = 0.0
    elif config.dimensionality == 3:
        # velocity_y_interface = avg_x(velocity_y)
        velocity_y_interface = avg_x(momentum_y) / rho_interface
        # velocity_z_interface = avg_x(velocity_z)
        velocity_z_interface = avg_x(momentum_z) / rho_interface

    cs2 = sound_speed ** 2
    cs2_inverse = jnp.where(cs2 > 0.0, 1.0 / cs2, 0.0)

    return (
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        sound_speed,
        cs2,
        cs2_inverse,
    )


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigen_R_col_hydro_iso_from_blocks(
    blocks,
    conserved_state,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    col: int,
):
    (
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        cs,
        cs2,
        cs2_inverse,
    ) = blocks

    density_index = registered_variables.density_index

    if config.dimensionality == 1:
        momentum_index_x = registered_variables.momentum_index
    else:
        momentum_index_x = registered_variables.momentum_index.x

    if config.dimensionality >= 2:
        momentum_index_y = registered_variables.momentum_index.y
    if config.dimensionality == 3:
        momentum_index_z = registered_variables.momentum_index.z

    def col_acoustic_minus():
        # u - cs
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(1.0)
        R = R.at[momentum_index_x].set(velocity_x_interface - cs)
        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(velocity_y_interface)
        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(velocity_z_interface)
        return R

    def col_shear_y():
        # shear y
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(0.0)
        R = R.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(1.0)
        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(0.0)
        return R

    def col_shear_z():
        # shear z
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(0.0)
        R = R.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(0.0)
        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(1.0)
        return R

    def col_acoustic_plus():
        # u + cs
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(1.0)
        R = R.at[momentum_index_x].set(velocity_x_interface + cs)
        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(velocity_y_interface)
        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(velocity_z_interface)
        return R

    if config.dimensionality == 1:
        R = jax.lax.switch(col, [col_acoustic_minus, col_acoustic_plus])
    elif config.dimensionality == 2:
        R = jax.lax.switch(col, [col_acoustic_minus, col_shear_y, col_acoustic_plus])
    elif config.dimensionality == 3:
        R = jax.lax.switch(col, [col_acoustic_minus, col_shear_y, col_shear_z, col_acoustic_plus])

    return R


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigen_L_row_hydro_iso_from_blocks(
    blocks,
    conserved_state,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    row: int,
):
    (
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        cs,
        cs2,
        cs2_inverse,
    ) = blocks

    density_index = registered_variables.density_index

    if config.dimensionality == 1:
        momentum_index_x = registered_variables.momentum_index
    else:
        momentum_index_x = registered_variables.momentum_index.x

    if config.dimensionality >= 2:
        momentum_index_y = registered_variables.momentum_index.y
    if config.dimensionality == 3:
        momentum_index_z = registered_variables.momentum_index.z

    def row_acoustic_minus():
        # u - cs
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(cs2 + velocity_x_interface * cs)
        L = L.at[momentum_index_x].set(-cs)
        if config.dimensionality >= 2:
            L = L.at[momentum_index_y].set(0.0)
        if config.dimensionality == 3:
            L = L.at[momentum_index_z].set(0.0)
        L = 0.5 * cs2_inverse * L
        return L

    def row_shear_y():
        # shear y
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(-velocity_y_interface)
        L = L.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            L = L.at[momentum_index_y].set(1.0)
        if config.dimensionality == 3:
            L = L.at[momentum_index_z].set(0.0)
        return L

    def row_shear_z():
        # shear z
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(-velocity_z_interface)
        L = L.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            L = L.at[momentum_index_y].set(0.0)
        if config.dimensionality == 3:
            L = L.at[momentum_index_z].set(1.0)
        return L

    def row_acoustic_plus():
        # u + cs
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(cs2 - velocity_x_interface * cs)
        L = L.at[momentum_index_x].set(cs)
        if config.dimensionality >= 2:
            L = L.at[momentum_index_y].set(0.0)
        if config.dimensionality == 3:
            L = L.at[momentum_index_z].set(0.0)
        L = 0.5 * cs2_inverse * L
        return L

    if config.dimensionality == 1:
        L = jax.lax.switch(row, [row_acoustic_minus, row_acoustic_plus])
    elif config.dimensionality == 2:
        L = jax.lax.switch(row, [row_acoustic_minus, row_shear_y, row_acoustic_plus])
    elif config.dimensionality == 3:
        L = jax.lax.switch(row, [row_acoustic_minus, row_shear_y, row_shear_z, row_acoustic_plus])

    return L


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigen_all_lambdas_hydro_iso(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    sound_speed: Union[float, jnp.ndarray],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    (velocity_x, cs) = _eigenvalue_building_blocks(
        conserved_state, sound_speed, rhomin, config, registered_variables,
    )

    if config.dimensionality == 1:
        return jnp.stack([velocity_x - cs, velocity_x + cs], axis=0)
    if config.dimensionality == 2:
        return jnp.stack([velocity_x - cs, velocity_x, velocity_x + cs], axis=0)
    if config.dimensionality == 3:
        return jnp.stack([velocity_x - cs, velocity_x, velocity_x, velocity_x + cs], axis=0)


def _eigen_lambdas_hydro_iso(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    sound_speed: Union[float, jnp.ndarray],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    mode: int,
):
    (velocity_x, cs) = _eigenvalue_building_blocks(
        conserved_state, sound_speed, rhomin, config, registered_variables,
    )

    def mode_minus():
        return velocity_x - cs

    def mode_contact():
        return velocity_x

    def mode_plus():
        return velocity_x + cs

    if config.dimensionality == 1:
        return jax.lax.switch(mode, [mode_minus, mode_plus])
    if config.dimensionality == 2:
        return jax.lax.switch(mode, [mode_minus, mode_contact, mode_plus])
    if config.dimensionality == 3:
        return jax.lax.switch(mode, [mode_minus, mode_contact, mode_contact, mode_plus])


def _eigen_R_col_hydro_iso(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    sound_speed: Union[float, jnp.ndarray],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    col: int,
):
    blocks = _eigenvector_building_blocks(
        conserved_state, sound_speed, rhomin, config, registered_variables
    )
    return _eigen_R_col_hydro_iso_from_blocks(blocks, conserved_state, config, registered_variables, col)


def _eigen_L_row_hydro_iso(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    sound_speed: Union[float, jnp.ndarray],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    row: int,
):
    blocks = _eigenvector_building_blocks(
        conserved_state, sound_speed, rhomin, config, registered_variables
    )
    return _eigen_L_row_hydro_iso_from_blocks(blocks, conserved_state, config, registered_variables, row)


# ---------------------------------------------------------------------------
# Component-tuple eigen helpers (no state-shape tensor materialization).
# Iso hydro: 1D=2 modes, 2D=3, 3D=4. WENO mode index → row index:
#   1D: [0, 3]    (u-cs, u+cs)
#   2D: [0, 1, 3] (u-cs, shear-y, u+cs)
#   3D: [0, 1, 2, 3] (u-cs, shear-y, shear-z, u+cs)
# ---------------------------------------------------------------------------

def _hydro_iso_indices(config, registered_variables):
    rv = registered_variables
    d_idx = rv.density_index
    if config.dimensionality == 1:
        mx_idx = rv.momentum_index
    else:
        mx_idx = rv.momentum_index.x
    my_idx = rv.momentum_index.y if config.dimensionality >= 2 else None
    mz_idx = rv.momentum_index.z if config.dimensionality == 3 else None
    return d_idx, mx_idx, my_idx, mz_idx


def _hydro_iso_row_for_mode(config, mode):
    if config.dimensionality == 1:
        return [0, 3][mode]
    if config.dimensionality == 2:
        return [0, 1, 3][mode]
    return [0, 1, 2, 3][mode]


def _eigen_L_pairs_hydro_iso_from_blocks(blocks, config, registered_variables, mode: int):
    (vx, vy, vz, cs, cs2, cs2_inv) = blocks
    d_idx, mx_idx, my_idx, mz_idx = _hydro_iso_indices(config, registered_variables)
    row = _hydro_iso_row_for_mode(config, mode)

    if row == 0:  # u - cs
        f = 0.5 * cs2_inv
        return [
            (f * (cs2 + vx * cs), d_idx),
            (-f * cs, mx_idx),
        ]
    if row == 1:  # shear y
        return [
            (-vy, d_idx),
            (1.0, my_idx),
        ]
    if row == 2:  # shear z
        return [
            (-vz, d_idx),
            (1.0, mz_idx),
        ]
    if row == 3:  # u + cs
        f = 0.5 * cs2_inv
        return [
            (f * (cs2 - vx * cs), d_idx),
            (f * cs, mx_idx),
        ]
    raise ValueError(f"Invalid row {row}")


def _eigen_R_pairs_hydro_iso_from_blocks(blocks, config, registered_variables, mode: int):
    (vx, vy, vz, cs, cs2, cs2_inv) = blocks
    d_idx, mx_idx, my_idx, mz_idx = _hydro_iso_indices(config, registered_variables)
    col = _hydro_iso_row_for_mode(config, mode)

    if col == 0:  # u - cs
        pairs = [
            (1.0, d_idx),
            (vx - cs, mx_idx),
        ]
        if config.dimensionality >= 2:
            pairs.append((vy, my_idx))
        if config.dimensionality == 3:
            pairs.append((vz, mz_idx))
        return pairs
    if col == 1:  # shear y
        return [
            (1.0, my_idx),
        ]
    if col == 2:  # shear z
        return [
            (1.0, mz_idx),
        ]
    if col == 3:  # u + cs
        pairs = [
            (1.0, d_idx),
            (vx + cs, mx_idx),
        ]
        if config.dimensionality >= 2:
            pairs.append((vy, my_idx))
        if config.dimensionality == 3:
            pairs.append((vz, mz_idx))
        return pairs
    raise ValueError(f"Invalid col {col}")