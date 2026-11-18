"""
Computations of the eigenvalues and eigenvectors for the Euler (hydrodynamics)
equations of a barotropic gas: isothermal, or polytropic.

A barotropic gas has no energy equation; the conserved variables are density
and momenta only. The sound speed cs is the isothermal constant, or for a
polytropic gas the local a with a^2 = dp / drho, at the cell (eigenvalues) or
at the averaged interface state (eigenvectors). The functions take the law as
``barotropic_law`` (``_equation_of_state.py``): the isothermal sound speed or
the ``PolytropicLaw``.

Waves:
  1D: u-cs, u+cs  (2 waves)
  2D: u-cs, u (shear y), u+cs  (3 waves)
  3D: u-cs, u (shear y), u (shear z), u+cs  (4 waves)

There is no entropy wave since there is no energy/pressure to evolve.
"""

# general
from functools import partial

# typing
from typing import Union

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._equation_of_state import law_sound_speed
from astronomix._stencil_operations._stencil_operations import _shift


def diff_safe_sqrt(x):
    """Square root with a small floor, so its derivative stays finite at x = 0.

    The derivative of sqrt(x) is 1 / (2 sqrt(x)), which blows up as x -> 0;
    clamping the argument to a tiny epsilon keeps the backward pass well-behaved.
    """
    epsilon = 1e-12
    x_safe = jnp.maximum(x, epsilon)
    return jnp.sqrt(x_safe)


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigenvalue_building_blocks(
    conserved_state,
    barotropic_law,
    rhomin,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Cell-centred velocity and sound speed entering the barotropic eigenvalues.

    Args:
        conserved_state: The conserved state, sweep direction first.
        barotropic_law: The isothermal sound speed or the ``PolytropicLaw``.
        rhomin: The density floor.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The normal velocity per cell and the sound speed.
    """
    density = conserved_state[registered_variables.density_index]

    if config.dimensionality == 1:
        momentum_x = conserved_state[registered_variables.momentum_index]
    else:
        momentum_x = conserved_state[registered_variables.momentum_index.x]

    rho = jnp.maximum(density, rhomin)
    velocity_x = momentum_x / rho
    sound_speed, _ = law_sound_speed(barotropic_law, rho)

    return (velocity_x, sound_speed)


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigenvector_building_blocks(
    conserved_state,
    barotropic_law,
    rhomin,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Interface quantities entering the barotropic eigenvectors.

    Args:
        conserved_state: The conserved state, sweep direction first.
        barotropic_law: The isothermal sound speed or the ``PolytropicLaw``.
        rhomin: The density floor.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The interface velocity components, the sound speed at the interface
        density, its square and the (zero-safe) inverse of the square.
    """
    rho = conserved_state[registered_variables.density_index]

    # In 1D the registry stores the single momentum index as a plain int.
    if config.dimensionality == 1:
        momentum_x = conserved_state[registered_variables.momentum_index]
        momentum_y = 0.0
        momentum_z = 0.0
    elif config.dimensionality == 2:
        momentum_x = conserved_state[registered_variables.momentum_index.x]
        momentum_y = conserved_state[registered_variables.momentum_index.y]
        momentum_z = 0.0
    elif config.dimensionality == 3:
        momentum_x = conserved_state[registered_variables.momentum_index.x]
        momentum_y = conserved_state[registered_variables.momentum_index.y]
        momentum_z = conserved_state[registered_variables.momentum_index.z]

    rho = jnp.maximum(rho, rhomin)

    # Periodic average from cell centres to interfaces.
    def avg_x(arr):
        return 0.5 * (arr + _shift(arr, shift=-1, axis=0))

    # Average the momenta (rather than the velocities) to the interface and divide
    # by the interface density; this keeps the interface velocity consistent with
    # the averaged conserved quantities.
    rho_interface = avg_x(jnp.maximum(rho, rhomin))
    rho_interface = jnp.maximum(rho_interface, rhomin)
    velocity_x_interface = avg_x(momentum_x) / rho_interface

    if config.dimensionality == 1:
        velocity_y_interface = 0.0
        velocity_z_interface = 0.0
    elif config.dimensionality == 2:
        velocity_y_interface = avg_x(momentum_y) / rho_interface
        velocity_z_interface = 0.0
    elif config.dimensionality == 3:
        velocity_y_interface = avg_x(momentum_y) / rho_interface
        velocity_z_interface = avg_x(momentum_z) / rho_interface

    sound_speed, cs2 = law_sound_speed(barotropic_law, rho_interface)
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
def _eigen_R_col_hydro_iso(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    barotropic_law,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    col: int,
):
    """
    One right eigenvector (column of R) of the barotropic flux Jacobian.

    Args:
        conserved_state: The conserved state, sweep direction first.
        rhomin: The density floor.
        barotropic_law: The isothermal sound speed or the ``PolytropicLaw``.
        config: The simulation configuration.
        registered_variables: The registered variables.
        col: The index of the characteristic mode, ordered by wave speed
            (u - cs, [u, u,] u + cs).

    Returns:
        The right eigenvector, shaped like ``conserved_state``.
    """
    (
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        cs,
        cs2,
        cs2_inverse,
    ) = _eigenvector_building_blocks(
        conserved_state,
        barotropic_law,
        rhomin,
        config,
        registered_variables,
    )

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
        # Acoustic mode u - cs.
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(1.0)
        R = R.at[momentum_index_x].set(velocity_x_interface - cs)
        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(velocity_y_interface)
        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(velocity_z_interface)
        return R

    def col_shear_y():
        # Shear mode along y.
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(0.0)
        R = R.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(1.0)
        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(0.0)
        return R

    def col_shear_z():
        # Shear mode along z.
        R = jnp.zeros_like(conserved_state)
        R = R.at[density_index].set(0.0)
        R = R.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            R = R.at[momentum_index_y].set(0.0)
        if config.dimensionality == 3:
            R = R.at[momentum_index_z].set(1.0)
        return R

    def col_acoustic_plus():
        # Acoustic mode u + cs.
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
        R = jax.lax.switch(
            col,
            [col_acoustic_minus, col_shear_y, col_shear_z, col_acoustic_plus],
        )

    return R


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigen_L_row_hydro_iso(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    barotropic_law,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    row: int,
):
    """
    One left eigenvector (row of L) of the barotropic flux Jacobian.

    Args:
        conserved_state: The conserved state, sweep direction first.
        rhomin: The density floor.
        barotropic_law: The isothermal sound speed or the ``PolytropicLaw``.
        config: The simulation configuration.
        registered_variables: The registered variables.
        row: The index of the characteristic mode, ordered by wave speed
            (u - cs, [u, u,] u + cs).

    Returns:
        The left eigenvector, shaped like ``conserved_state``.
    """
    (
        velocity_x_interface,
        velocity_y_interface,
        velocity_z_interface,
        cs,
        cs2,
        cs2_inverse,
    ) = _eigenvector_building_blocks(
        conserved_state,
        barotropic_law,
        rhomin,
        config,
        registered_variables,
    )

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
        # Acoustic mode u - cs.
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
        # Shear mode along y.
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(-velocity_y_interface)
        L = L.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            L = L.at[momentum_index_y].set(1.0)
        if config.dimensionality == 3:
            L = L.at[momentum_index_z].set(0.0)
        return L

    def row_shear_z():
        # Shear mode along z.
        L = jnp.zeros_like(conserved_state)
        L = L.at[density_index].set(-velocity_z_interface)
        L = L.at[momentum_index_x].set(0.0)
        if config.dimensionality >= 2:
            L = L.at[momentum_index_y].set(0.0)
        if config.dimensionality == 3:
            L = L.at[momentum_index_z].set(1.0)
        return L

    def row_acoustic_plus():
        # Acoustic mode u + cs.
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
        L = jax.lax.switch(
            row,
            [row_acoustic_minus, row_shear_y, row_shear_z, row_acoustic_plus],
        )

    return L


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _eigen_all_lambdas_hydro_iso(
    conserved_state,
    rhomin: Union[float, jnp.ndarray],
    barotropic_law,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    All barotropic eigenvalues per cell, stacked along a leading mode axis.

    Args:
        conserved_state: The conserved state, sweep direction first.
        rhomin: The density floor.
        barotropic_law: The isothermal sound speed or the ``PolytropicLaw``.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The eigenvalues (u - cs, [u, u,] u + cs) per cell.
    """
    (velocity_x, cs) = _eigenvalue_building_blocks(
        conserved_state,
        barotropic_law,
        rhomin,
        config,
        registered_variables,
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
    barotropic_law,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    mode: int,
):
    """
    One barotropic eigenvalue per cell.

    Args:
        conserved_state: The conserved state, sweep direction first.
        rhomin: The density floor.
        barotropic_law: The isothermal sound speed or the ``PolytropicLaw``.
        config: The simulation configuration.
        registered_variables: The registered variables.
        mode: The index of the characteristic mode, ordered by wave speed
            (u - cs, [u, u,] u + cs).

    Returns:
        The eigenvalue of the mode per cell.
    """
    (velocity_x, cs) = _eigenvalue_building_blocks(
        conserved_state,
        barotropic_law,
        rhomin,
        config,
        registered_variables,
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