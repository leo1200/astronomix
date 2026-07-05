"""Validation for the ported thermal conduction (FD) on the refactor branch:
a sinusoidal temperature perturbation is damped *more* with conduction on than
off (isolating the conductive contribution), and d(final amplitude)/d(kappa) is
finite & nonzero (clean adjoint). 2D periodic ideal-gas box, runs on CPU.

Run: PYTHONPATH=<refactor worktree> JAX_PLATFORMS=cpu python tests/finite_difference/conduction_test.py
"""
import jax
import jax.numpy as jnp

from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FINITE_DIFFERENCE, IDEAL_GAS, PERIODIC_BOUNDARY,
    BoundarySettings, BoundarySettings1D, SimulationConfig, finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

N = 32
PERIODIC2D = BoundarySettings(
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)


def make_config(kappa_on, diff_mode=None):
    cfg = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS,
        dimensionality=2, num_cells=N, box_size=1.0, mhd=False,
        boundary_settings=PERIODIC2D, thermal_conduction=kappa_on,
    )
    if diff_mode is not None:
        cfg = cfg._replace(differentiation_mode=diff_mode, num_checkpoints=64)
    return cfg


def initial(config, rv):
    x = jnp.linspace(0, 1, N, endpoint=False)[:, None] * jnp.ones((N, N))
    rho = jnp.ones((N, N))
    p = 1.0 + 0.05 * jnp.sin(2 * jnp.pi * x)         # T = p/rho has a sinusoid in x
    zero = jnp.zeros((N, N))
    return construct_primitive_state(
        config=config, registered_variables=rv, density=rho,
        velocity_x=zero, velocity_y=zero, gas_pressure=p,
    )


def final_amp(kappa, conduction, t_end=0.02, diff_mode=None):
    config = make_config(conduction, diff_mode)
    rv = get_registered_variables(config)
    s = initial(config, rv)
    config = finalize_config(config, s.shape)
    params = SimulationParams(t_end=t_end, thermal_conductivity=kappa)
    f = time_integration(s, config, params, rv)
    p = f[rv.pressure_index]
    # amplitude of the kx=1 pressure mode (the conduction-damped sinusoid)
    return jnp.abs(jnp.fft.rfft(jnp.mean(p, axis=1))[1])


def main():
    amp_off = float(final_amp(0.0, conduction=False))    # hydro only
    amp_on = float(final_amp(0.1, conduction=True))       # hydro + conduction
    print(f"kx=1 pressure-mode amplitude: conduction OFF={amp_off:.5f}  ON(kappa=0.1)={amp_on:.5f}")
    assert jnp.isfinite(amp_on) and jnp.isfinite(amp_off)
    assert amp_on < amp_off - 1e-4, "conduction should damp the temperature perturbation more than hydro alone"

    g = jax.grad(lambda k: final_amp(k, conduction=True, diff_mode=BACKWARDS))(0.1)
    print(f"d(amplitude)/d(kappa) at kappa=0.1: {float(g):.4e}  (finite, <0 expected: more kappa -> more damping)")
    assert jnp.isfinite(g) and g < 0, "more conductivity should reduce the surviving amplitude"
    print("conduction validation: PASS")


if __name__ == "__main__":
    main()
