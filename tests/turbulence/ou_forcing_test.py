"""Validation for the ported Ornstein-Uhlenbeck (and white) turbulent forcing
on the refactor branch: forcing injects kinetic energy, the no-forcing path is
unchanged, and the OU forward run is differentiable (clean state-independent
adjoint). Small 3D periodic hydro box, runs on CPU.

Run: PYTHONPATH=<refactor worktree> JAX_PLATFORMS=cpu python tests/turbulence/ou_forcing_test.py
"""
import jax
import jax.numpy as jnp

from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FINITE_DIFFERENCE, IDEAL_GAS, PERIODIC_BOUNDARY,
    BoundarySettings, BoundarySettings1D, SimulationConfig, finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

N = 16
PERIODIC = BoundarySettings(
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)


def make_config(forcing_cfg):
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS,
        dimensionality=3, num_cells=N, box_size=1.0, mhd=False,
        boundary_settings=PERIODIC, turbulent_forcing_config=forcing_cfg,
    )


def run(forcing_cfg, forcing_params, t_end=0.05):
    config = make_config(forcing_cfg)
    rv = get_registered_variables(config)
    rho = jnp.ones((N, N, N)); zero = jnp.zeros_like(rho)
    state = construct_primitive_state(
        config=config, registered_variables=rv, density=rho,
        velocity_x=zero, velocity_y=zero, velocity_z=zero, gas_pressure=rho,
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(t_end=t_end, turbulent_forcing_params=forcing_params)
    final = time_integration(state, config, params, rv)
    vx = final[rv.velocity_index.x]; vy = final[rv.velocity_index.y]; vz = final[rv.velocity_index.z]
    return float(jnp.sqrt(jnp.mean(vx**2 + vy**2 + vz**2)))


def main():
    none_cfg = TurbulentForcingConfig(turbulent_forcing=False)
    white_cfg = TurbulentForcingConfig(turbulent_forcing=True)
    ou_cfg = TurbulentForcingConfig(turbulent_forcing=True, ou_forcing=True)
    white_p = TurbulentForcingParams(energy_injection_rate=1.0)
    ou_p = TurbulentForcingParams(correlation_time=0.1, forcing_wavenumber=4.0, forcing_amplitude=2.0)

    v_none = run(none_cfg, TurbulentForcingParams())
    v_white = run(white_cfg, white_p)
    v_ou = run(ou_cfg, ou_p)
    print(f"v_rms: no-forcing={v_none:.3e}  white={v_white:.3e}  OU={v_ou:.3e}")
    assert v_none < 1e-6, f"no-forcing should stay quiescent, got {v_none}"
    assert jnp.isfinite(v_white) and v_white > 1e-3, "white forcing should inject energy"
    assert jnp.isfinite(v_ou) and v_ou > 1e-3, "OU forcing should inject energy"

    # gradient check: d(final KE)/d(forcing_amplitude) through the OU forward run
    def ke_of_amp(F0):
        # reverse-mode AD requires the checkpointed adaptive loop (BACKWARDS),
        # not the plain while_loop used by the default forward-mode path
        config = make_config(ou_cfg)._replace(
            differentiation_mode=BACKWARDS, num_checkpoints=64,
        )
        rv = get_registered_variables(config)
        rho = jnp.ones((N, N, N)); zero = jnp.zeros_like(rho)
        s = construct_primitive_state(
            config=config, registered_variables=rv, density=rho,
            velocity_x=zero, velocity_y=zero, velocity_z=zero, gas_pressure=rho,
        )
        config = finalize_config(config, s.shape)
        p = SimulationParams(
            t_end=0.05,
            turbulent_forcing_params=TurbulentForcingParams(
                correlation_time=0.1, forcing_wavenumber=4.0, forcing_amplitude=F0),
        )
        f = time_integration(s, config, p, rv)
        vx = f[rv.velocity_index.x]; vy = f[rv.velocity_index.y]; vz = f[rv.velocity_index.z]
        return jnp.mean(vx**2 + vy**2 + vz**2)

    g = jax.grad(ke_of_amp)(2.0)
    print(f"d(KE)/d(F0) at F0=2.0: {float(g):.4e}  (finite & >0 expected)")
    assert jnp.isfinite(g) and g > 0, "OU forcing gradient should be finite and positive"
    print("OU forcing validation: PASS")


if __name__ == "__main__":
    main()
