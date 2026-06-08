"""
Phase 0a verification (a): a sinusoidal temperature perturbation must decay
under thermal conduction at the analytic rate  chi * k^2  with the internal-
energy thermal diffusivity  chi = (gamma - 1) * kappa / rho.

Two independent checks:

  1. **Operator check** (exact, no time integration): on a sinusoidal
     temperature field the conductive energy source must equal
     kappa * laplacian(T) = -kappa * k^2 * (T - mean).  This isolates the
     spatial discretisation.

  2. **Decay-rate check** (full forward run): we evolve a small sinusoidal
     temperature (here: pressure-at-uniform-density) perturbation in the 1D
     finite-difference scheme in the *conduction-dominated* (overdamped)
     regime chi*k >> c_s, where the mode decays diffusively rather than
     oscillating acoustically, and measure the decay rate of the fundamental
     Fourier mode.  It must match chi * k^2.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    PERIODIC_BOUNDARY,
    BoundarySettings1D,
    SimulationConfig,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.time_stepping.time_integration import time_integration
from astronomix._physics_modules._conduction._conduction import fd_conduction_source


GAMMA = 5.0 / 3.0
RHO0 = 1.0
N = 64
L = 1.0


def operator_check():
    print("\n=== 1.  conduction operator check ===")
    kappa = 0.7
    k = 2.0 * np.pi / L  # fundamental mode

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS,
        dimensionality=1,
        num_cells=N,
        box_size=L,
        thermal_conduction=True,
        boundary_settings=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    x = helper_data.geometric_centers
    T = 2.0 + 0.1 * jnp.sin(k * x)        # temperature field, mean 2
    rho = jnp.full_like(x, RHO0)
    p = rho * T                            # T = p / rho

    state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=rho, velocity_x=jnp.zeros_like(x), gas_pressure=p,
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(gamma=GAMMA, thermal_conductivity=kappa)

    # pad the state with ghost cells the same way the integrator does, so the
    # periodic stencil sees the wrap-around neighbours
    from astronomix.time_stepping._utils import _pad
    from astronomix._geometry.boundaries import _boundary_handler
    state_pad = _pad(state, config)
    state_pad = _boundary_handler(state_pad, config, registered_variables, params)

    S = fd_conduction_source(state_pad, params, config, registered_variables)
    energy_src = np.asarray(S[registered_variables.energy_index])

    # strip ghost cells
    ng = config.num_ghost_cells
    energy_src = energy_src[ng:-ng] if ng > 0 else energy_src

    x_int = np.asarray(x)
    analytic = kappa * (-(k ** 2)) * (0.1 * np.sin(k * x_int))

    rel_err = np.max(np.abs(energy_src - analytic)) / np.max(np.abs(analytic))
    print(f"kappa = {kappa},  k = {k:.4f}")
    print(f"max relative error (source vs -kappa k^2 dT) = {rel_err:.3e}")
    ok = rel_err < 1e-3
    print("operator check:", "PASS" if ok else "FAIL")
    return ok


def decay_check():
    print("\n=== 2.  conduction decay-rate check (overdamped) ===")
    kappa = 5.0
    k = 2.0 * np.pi / L
    chi = (GAMMA - 1.0) * kappa / RHO0      # internal-energy thermal diffusivity
    rate_analytic = chi * k ** 2

    p0 = 1.0
    cs = np.sqrt(GAMMA * p0 / RHO0)
    print(f"chi = {chi:.4f},  chi*k = {chi*k:.2f}  vs  c_s = {cs:.2f}  "
          f"(want chi*k >> c_s for overdamped)")

    t_end = 0.004                            # ~ 0.5 / rate_analytic
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS,
        dimensionality=1,
        num_cells=N,
        box_size=L,
        thermal_conduction=True,
        boundary_settings=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        return_snapshots=False,
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    x = helper_data.geometric_centers
    eps = 1e-3
    rho = jnp.full_like(x, RHO0)
    p = p0 * (1.0 + eps * jnp.sin(k * x))    # sinusoidal T at uniform rho
    state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=rho, velocity_x=jnp.zeros_like(x), gas_pressure=p,
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(gamma=GAMMA, thermal_conductivity=kappa,
                              t_end=t_end, C_cfl=0.4)

    x_np = np.asarray(x)
    sin_mode = np.sin(k * x_np)

    def amplitude(st):
        T = np.asarray(st[registered_variables.pressure_index] /
                       st[registered_variables.density_index])
        return 2.0 / N * np.sum((T - T.mean()) * sin_mode)

    a0 = amplitude(state)
    final = time_integration(state, config, params, registered_variables)
    af = amplitude(final)

    rate_measured = -np.log(af / a0) / t_end
    rel_err = abs(rate_measured - rate_analytic) / rate_analytic
    print(f"analytic decay rate chi*k^2 = {rate_analytic:.2f}")
    print(f"measured decay rate         = {rate_measured:.2f}")
    print(f"relative error              = {rel_err:.3e}")
    ok = rel_err < 0.05
    print("decay check:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ok1 = operator_check()
    ok2 = decay_check()
    print("\nRESULT:", "PASS" if (ok1 and ok2) else "FAIL")
    assert ok1 and ok2, "thermal-conduction verification failed"
