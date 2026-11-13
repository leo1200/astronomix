"""
Analytical-gradient correctness pytest (fast).

Runs a smooth small-amplitude 1D density wave through the finite-difference
solver, differentiates the quadratic cost J = 0.5 * sum(rho_P^2 + v_P^2) dV of
the final state with respect to the initial density / velocity via reverse-mode
AD, and checks it against the closed-form Fourier-space gradient of the
linearized Euler equations. This is the fast differentiability correctness
check; the full multi-dimensional study + figures lives in
``examples/scripts/differentiability/sensitivity.py``.
"""

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    CARTESIAN,
    BACKWARDS,
    PERIODIC_BOUNDARY,
)
from astronomix.option_classes.simulation_config import PERIODIC_ROLL

# astronomix containers
from astronomix import (
    BoundarySettings1D,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    get_helper_data,
    time_integration,
    get_registered_variables,
    construct_primitive_state,
    finalize_config,
)

# The exact analytic gradients require stable 64-bit precision to match the AD
# solver output; single precision would swamp the comparison with round-off.
pytestmark = pytest.mark.x64



def compute_analytic_gradients_fourier(rho_0, v0, L, c_s, rho_B, t_end):
    """Exact gradient of the quadratic cost for the 1D linearized Euler equations.

    The cost is J = 0.5 * sum |U|^2 dV on the final state. Because the linearized
    Euler operator is diagonal in Fourier space, the gradient is obtained in
    closed form from the S*S operator (S being the time-evolution operator),
    without any solver run.

    Args:
        rho_0: The initial density perturbation field (real space).
        v0: The initial velocity perturbation field (real space).
        L: The box size.
        c_s: The background sound speed.
        rho_B: The background density.
        t_end: The final time at which the cost is evaluated.

    Returns:
        A tuple (grad_rho, grad_v) of the real-space gradients of J with respect
        to the initial density and velocity perturbations.
    """
    num_cells = rho_0.shape[0]

    # Wavenumbers on the periodic grid and their magnitudes / dispersion relation.
    k = jnp.fft.fftfreq(num_cells, d=L / num_cells) * 2 * jnp.pi
    k_mag = jnp.abs(k)
    omega = c_s * k_mag
    k_hat = jnp.where(k_mag > 0, k / k_mag, 0.0)

    rho_0_hat = jnp.fft.fft(rho_0)
    v0_hat = jnp.fft.fft(v0)
    k_dot_v0_hat = k_hat * v0_hat

    # Closed-form S*S operator entries evaluated at t_end.
    cos_wt = jnp.cos(omega * t_end)
    sin_wt = jnp.sin(omega * t_end)
    rho_factor = cos_wt**2 + (c_s / rho_B) ** 2 * sin_wt**2
    cross_term = 1j * sin_wt * cos_wt * (rho_B / c_s - c_s / rho_B)
    v_parallel_factor = ((rho_B / c_s) ** 2 - 1.0) * sin_wt**2

    grad_rho_hat = rho_factor * rho_0_hat - cross_term * k_dot_v0_hat
    grad_v0_hat = (
        cross_term * rho_0_hat * k_hat
        + v0_hat
        + v_parallel_factor * k_dot_v0_hat * k_hat
    )

    # Only the real part is physical; the imaginary part is round-off.
    grad_rho = jnp.fft.ifft(grad_rho_hat).real
    grad_v0 = jnp.fft.ifft(grad_v0_hat).real
    return grad_rho, grad_v0


def get_config_and_params(num_cells, L, t_end):
    """Build the 1D periodic finite-difference configuration and parameters.

    Args:
        num_cells: The number of grid cells along the single spatial dimension.
        L: The box size.
        t_end: The final integration time.

    Returns:
        A tuple (config, params) of the simulation configuration and parameters.
    """
    config = SimulationConfig(
        geometry=CARTESIAN,
        progress_bar=False,
        boundary_handling=PERIODIC_ROLL,
        differentiation_mode=BACKWARDS,
        num_ghost_cells=0,
        mhd=False,
        dimensionality=1,
        box_size=L,
        num_cells=num_cells,
        boundary_settings=BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY,
            right_boundary=PERIODIC_BOUNDARY,
        ),
        return_snapshots=False,
    )
    params = SimulationParams(
        C_cfl=0.4,
        t_end=t_end,
        gamma=5 / 3,
        gravitational_constant=0.0,
    )
    return config, params


def run_forward_and_cost(rho_P0, v_P0, config, params, rho_B, c_s, P_B):
    """Integrate the wave forward and return the quadratic cost of the final state.

    The cost is J = 0.5 * sum(rho_P^2 + v_P^2) * dx, where rho_P and v_P are the
    density and velocity perturbations of the evolved state.

    Args:
        rho_P0: The initial density perturbation field.
        v_P0: The initial velocity perturbation field.
        config: The simulation configuration.
        params: The simulation parameters.
        rho_B: The background density.
        c_s: The background sound speed.
        P_B: The background pressure.

    Returns:
        The scalar cost J evaluated on the final state.
    """
    registered_variables = get_registered_variables(config)
    dx = config.box_size / config.num_cells

    rho = rho_B + rho_P0
    p = P_B + (c_s**2) * rho_P0
    initial_state = construct_primitive_state(
        config,
        registered_variables,
        density=rho,
        gas_pressure=p,
        velocity_x=v_P0,
    )
    config_final = finalize_config(config, initial_state.shape)
    final_state = time_integration(
        initial_state,
        config_final,
        params,
        registered_variables,
    )

    final_rho_P = final_state[registered_variables.density_index] - rho_B
    final_v_P = final_state[registered_variables.velocity_index]
    return 0.5 * jnp.sum(final_rho_P**2 + final_v_P**2) * dx


def test_analytical_gradient(N=64, tol=1e-3):
    """AD gradient of the 1D wave cost must match the exact Fourier gradient.

    Args:
        N: The number of grid cells.
        tol: The maximum allowed mean |AD - analytic| / eps for each field.
    """
    # Uniform background and a small perturbation amplitude eps so the dynamics
    # stay in the linear regime the analytic gradient assumes.
    rho_B, c_s, gamma = 1.0, 2.0, 5 / 3
    P_B = (c_s**2) * rho_B / gamma
    eps = 1e-6
    L, t_end = 1.0, 0.15

    config, params = get_config_and_params(N, L, t_end)
    helper_data = get_helper_data(config)

    # Smooth small-amplitude density wave; the velocity starts at rest.
    x = jnp.squeeze(helper_data.geometric_centers)
    k = 2 * jnp.pi * 2 / L
    rho_P0 = eps * jnp.sin(k * x)
    v_P0 = jnp.zeros_like(rho_P0)

    # The cost is defined per grid volume, so the raw AD gradient carries a
    # factor of dx that we divide out to compare against the sensitivity.
    dx = L / N
    cost_fn = lambda r, vx: run_forward_and_cost(r, vx, config, params, rho_B, c_s, P_B)
    grad_rho, grad_v = jax.grad(cost_fn, argnums=(0, 1))(rho_P0, v_P0)
    ad_grad_rho, ad_grad_v = grad_rho / dx, grad_v / dx

    ana_grad_rho, ana_grad_v = compute_analytic_gradients_fourier(
        rho_P0, v_P0, L, c_s, rho_B, t_end
    )

    l1_rho = float(jnp.mean(jnp.abs(ad_grad_rho - ana_grad_rho)) / eps)
    l1_v = float(jnp.mean(jnp.abs(ad_grad_v - ana_grad_v)) / eps)
    print(f"L1 error AD vs analytic  rho: {l1_rho:.3e}   v: {l1_v:.3e}")

    assert l1_rho < tol, f"density gradient L1 {l1_rho:.3e} exceeds {tol}"
    assert l1_v < tol, f"velocity gradient L1 {l1_v:.3e} exceeds {tol}"


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_analytical_gradient()
