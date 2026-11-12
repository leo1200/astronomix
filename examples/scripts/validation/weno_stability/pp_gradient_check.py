"""Gradients through the positivity-preserving WENO (forward and reverse mode).

A rough 3D state (hydro, ideal MHD, isothermal MHD), a few fixed steps, the
loss sum(rho^2) at the end; checks that jax.grad and jax.jvp are finite and
agree with a central finite difference along a random direction (x64).

    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/pp_gradient_check.py
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

# ruff: noqa: E402
import jax
import jax.numpy as jnp

from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    initialize_interface_fields,
    time_integration,
)
from astronomix.option_classes.simulation_config import IDEAL_GAS, ISOTHERMAL

n = 12
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
keys = jax.random.split(jax.random.PRNGKey(3), 5)
shape = (n, n, n)
rho = jnp.exp(0.8 * jax.random.normal(keys[0], shape))
pressure = jnp.exp(0.8 * jax.random.normal(keys[1], shape)) * 0.05
velocity = 0.8 * jax.random.normal(keys[2], (3,) + shape)
field = 0.4 * jax.random.normal(keys[3], (3,) + shape)
for case in ("hydro", "ideal_mhd", "iso_mhd"):
    mhd = case != "hydro"
    eos = ISOTHERMAL if case == "iso_mhd" else IDEAL_GAS
    config = SimulationConfig(
        equation_of_state=eos,
        dimensionality=3,
        num_cells=n,
        box_size=1.0,
        mhd=mhd,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        solver_config=FDSolverConfig(weno_positivity_preserving=True),
        fixed_timestep=True,
        num_timesteps=3,
    )
    params = SimulationParams(t_end=3e-3, gamma=5.0 / 3.0, isothermal_sound_speed=0.3)
    rv = get_registered_variables(config)
    fields = dict(config=config, registered_variables=rv, density=rho,
                  velocity_x=velocity[0], velocity_y=velocity[1], velocity_z=velocity[2])
    if eos == IDEAL_GAS:
        fields["gas_pressure"] = pressure
    if mhd:
        bx, by, bz = initialize_interface_fields(field[0], field[1], field[2])
        fields.update(magnetic_field_x=field[0], magnetic_field_y=field[1], magnetic_field_z=field[2],
                      interface_magnetic_field_x=bx, interface_magnetic_field_y=by, interface_magnetic_field_z=bz)
    state = construct_primitive_state(**fields)
    config = finalize_config(config, state.shape)

    def loss(density):
        final = time_integration(state.at[rv.density_index].set(density), config, params, rv)
        return jnp.sum(final[rv.density_index] ** 2)

    direction = jax.random.normal(keys[4], shape)
    gradient = jax.grad(loss)(rho)
    _, tangent = jax.jvp(loss, (rho,), (direction,))
    eps = 1e-6
    finite_difference = (loss(rho + eps * direction) - loss(rho - eps * direction)) / (2 * eps)
    reverse = float(jnp.sum(gradient * direction))
    print(f"{case:10s}: grad finite {bool(jnp.all(jnp.isfinite(gradient)))}; directional: reverse {reverse:+.6e} "
          f"forward {float(tangent):+.6e} finite-difference {float(finite_difference):+.6e}", flush=True)
