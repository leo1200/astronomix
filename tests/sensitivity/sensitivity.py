"""
Unified N-dimensional Adjoint Sensitivity Test for Linearized Euler Equations
with exact analytical gradients derived via Fourier space.
Includes state evolution plotting and a convergence test for AD gradients.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# =======================

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Enable 64-bit precision for stable finite differences and exact AD
jax.config.update("jax_enable_x64", True)

from astronomix import CARTESIAN
from astronomix import get_helper_data, time_integration, get_registered_variables
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FINITE_DIFFERENCE, FINITE_VOLUME, NATIVE_JAX, PALLAS,
    PERIODIC_ROLL, SimulationConfig, finalize_config,
    PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    StaticFloatVector, StaticIntVector, GravityConfig, PositivityConfig
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.plotting_helpers.power_law_indicators import add_power_law_indicators

# ==============================================================================
# 1. Fourier-based N-Dimensional Analytical Gradients (from Theory Write-up)
# ==============================================================================
def compute_analytic_gradients_fourier(rho_0, v0_tuple, L, c_s, rho_B, t_end, mass_matrix='first_order'):
    """
    Computes the exact analytical gradients for the linearized Euler equations
    in N-dimensions using the exact Fourier space operator S*S derived in the theory write-up.

    The ``mass_matrix`` argument selects which inner product the discrete
    objective is taken in:

    - ``'first_order'``: J = 0.5 * Σ |U_bar|² dV  (lumped/midpoint quadrature on
      cell averages). The Fourier gradient is the unmodified Fourier multiplier
      acting on the DFT of the cell averages.
    - ``'fourier'``: J = 0.5 * ∫ u² dV using the spectrally-accurate Parseval
      quadrature on the implicit polynomial reconstruction. The same Fourier
      multiplier is divided by ``Π_d sinc(k_d h_d / 2)²``, which is the
      Fourier-space Jacobian of the cell-average ↔ point-value map.
    """
    dim = len(rho_0.shape)
    N_list = rho_0.shape

    # Generate frequencies for each dimension
    freqs = [jnp.fft.fftfreq(N_list[i], d=L/N_list[i]) * 2 * jnp.pi for i in range(dim)]

    # Create N-dimensional meshgrid of frequencies
    K = jnp.meshgrid(*freqs, indexing='ij')

    # Wavenumber magnitude and frequency
    k_sq = sum(k_i**2 for k_i in K)
    k_mag = jnp.sqrt(k_sq)
    omega = c_s * k_mag

    # Normalized wavenumber vector (handle k=0 safely)
    k_hat = [jnp.where(k_mag > 0, k_i / k_mag, 0.0) for k_i in K]

    # Transform initial conditions to Fourier space
    rho_0_hat = jnp.fft.fftn(rho_0)
    v0_hat = [jnp.fft.fftn(v) for v in v0_tuple]

    # Dot product: k_hat \cdot \hat{v}_0
    k_dot_v0_hat = sum(k_hat[i] * v0_hat[i] for i in range(dim))

    # Evaluate S*S Operator Multipliers
    cos_wt = jnp.cos(omega * t_end)
    sin_wt = jnp.sin(omega * t_end)

    rho_factor = cos_wt**2 + (c_s / rho_B)**2 * sin_wt**2
    cross_term = 1j * sin_wt * cos_wt * (rho_B / c_s - c_s / rho_B)
    v_parallel_factor = ((rho_B / c_s)**2 - 1.0) * sin_wt**2

    # Gradients in Fourier Space (cell-average → cell-average operator)
    grad_rho_hat = rho_factor * rho_0_hat - cross_term * k_dot_v0_hat
    grad_v0_hat = [
        cross_term * rho_0_hat * k_hat[i] + v0_hat[i] + v_parallel_factor * k_dot_v0_hat * k_hat[i]
        for i in range(dim)
    ]

    if mass_matrix == 'fourier':
        # Sinc² correction: AD with the high-order mass matrix on cell averages
        # produces a per-cell-average gradient equal to (S*S U_bar) / sinc² in
        # Fourier space, because both forward and adjoint maps go through the
        # cell-average ↔ point-value transform sinc(k h / 2).
        h_list = [L / N_list[i] for i in range(dim)]
        sinc_total = None
        for d in range(dim):
            freq_d = jnp.fft.fftfreq(N_list[d], d=h_list[d])
            s = jnp.sinc(freq_d * h_list[d])
            shape = [1] * dim
            shape[d] = N_list[d]
            s = s.reshape(shape)
            sinc_total = s if sinc_total is None else sinc_total * s
        inv_sinc_sq = 1.0 / (sinc_total ** 2)
        grad_rho_hat = grad_rho_hat * inv_sinc_sq
        grad_v0_hat = [gv * inv_sinc_sq for gv in grad_v0_hat]
    elif mass_matrix != 'first_order':
        raise ValueError(f"Unknown mass_matrix option: {mass_matrix!r}")

    # Inverse FFT back to spatial domain
    grad_rho = jnp.fft.ifftn(grad_rho_hat).real
    grad_v0 = [jnp.fft.ifftn(gv).real for gv in grad_v0_hat]

    return grad_rho, grad_v0

# ==============================================================================
# 2. Universal Config and IC Generation
# ==============================================================================
def get_config_and_params(dim, N, L, t_end, solver_mode=FINITE_DIFFERENCE):
    if dim == 1:
        box_size = L
        num_cells = N
        boundary_settings = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
    elif dim == 2:
        box_size = StaticFloatVector(x=L, y=L)
        num_cells = StaticIntVector(x=N, y=N)
        boundary_settings = BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
        )
    elif dim == 3:
        box_size = StaticFloatVector(x=L, y=L, z=L)
        num_cells = StaticIntVector(x=N, y=N, z=N)
        boundary_settings = BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
        )

    config = SimulationConfig(
        solver_mode=solver_mode,
        geometry=CARTESIAN,
        progress_bar=False,
        boundary_handling=PERIODIC_ROLL,
        differentiation_mode=BACKWARDS,
        num_ghost_cells=0,
        mhd=False,
        dimensionality=dim,
        box_size=box_size,
        num_cells=num_cells,
        boundary_settings=boundary_settings,
        return_snapshots=False,
    )
    
    params = SimulationParams(
        C_cfl=0.4,
        t_end=t_end,
        gamma=5/3,
        gravitational_constant=0.0
    )
    
    return config, params

def generate_ic(helper_data, dim, ic_type, eps, L):
    """ Generates coordinate axes and the Initial Conditions (Gaussian or Wave) """
    if dim == 1:
        coords = [jnp.squeeze(helper_data.geometric_centers)]
    elif dim == 2:
        centers = jnp.transpose(helper_data.geometric_centers, (2, 0, 1))
        coords = [centers[0], centers[1]]
    elif dim == 3:
        centers = jnp.transpose(helper_data.geometric_centers, (3, 0, 1, 2))
        coords = [centers[0], centers[1], centers[2]]

    if ic_type == 'wave':
        # Create an angled wave depending on dimensions
        k_vec = jnp.array([2 * jnp.pi * (i+1) * 2 / L for i in range(dim)])
        phase = sum(k_vec[i] * coords[i] for i in range(dim))
        rho_P0 = eps * jnp.sin(phase)
    elif ic_type == 'gaussian':
        # Centered Gaussian blob
        r_sq = sum((coords[i] - L/2)**2 for i in range(dim))
        sigma = 0.1 * L
        rho_P0 = eps * jnp.exp(-r_sq / (2 * sigma**2))
        
    v_P0 = [jnp.zeros_like(rho_P0) for _ in range(dim)]
    return coords, rho_P0, v_P0

# ==============================================================================
# 3. Simulator Target & Automatic Differentiation
# ==============================================================================
def _spectral_squared_integral(u, dx_vec):
    """
    Spectrally-accurate integral of u(x)^2 over a uniform periodic domain,
    given u as cell averages of a smooth function.

    Cell averages and underlying Fourier coefficients are related by
        U_bar_k = u_hat_k * prod_d sinc(k_d * dx_d / 2),
    so dividing the DFT of the cell averages by this sinc factor recovers
    the Fourier coefficients of the reconstructed (point-valued) field.
    Parseval then gives the integral of the squared reconstruction. For
    the 5th-order WENO reconstruction on smooth periodic data this acts
    as the matching high-order mass matrix.
    """
    N_vec = u.shape
    dim = len(N_vec)
    N_total = 1
    L_total = 1.0
    for d in range(dim):
        N_total *= N_vec[d]
        L_total *= dx_vec[d] * N_vec[d]

    U_hat = jnp.fft.fftn(u)

    sinc_corr = None
    for d in range(dim):
        freq = jnp.fft.fftfreq(N_vec[d], d=dx_vec[d])
        # jnp.sinc(x) = sin(pi*x)/(pi*x); k*dx/2 = pi*freq*dx
        s = jnp.sinc(freq * dx_vec[d])
        shape = [1] * dim
        shape[d] = N_vec[d]
        s = s.reshape(shape)
        sinc_corr = s if sinc_corr is None else sinc_corr * s

    u_hat = U_hat / sinc_corr
    return (L_total / (N_total ** 2)) * jnp.sum(u_hat.real ** 2 + u_hat.imag ** 2)


def run_forward_and_cost(rho_P0, v_P0_tuple, config, params, rho_B, c_s, P_B, mass_matrix='first_order'):
    registered_variables = get_registered_variables(config)
    dim = config.dimensionality

    # Calculate cell volume
    L_vec = [config.box_size] if dim == 1 else ([config.box_size.x, config.box_size.y, config.box_size.z][:dim])
    N_vec = [config.num_cells] if dim == 1 else ([config.num_cells.x, config.num_cells.y, config.num_cells.z][:dim])
    dx_vec = [L_vec[i]/N_vec[i] for i in range(dim)]
    dV = jnp.prod(jnp.array(dx_vec))

    rho = rho_B + rho_P0
    p = P_B + (c_s**2) * rho_P0

    kwargs = {"density": rho, "gas_pressure": p, "velocity_x": v_P0_tuple[0]}
    if dim > 1: kwargs["velocity_y"] = v_P0_tuple[1]
    if dim > 2: kwargs["velocity_z"] = v_P0_tuple[2]

    initial_state = construct_primitive_state(config, registered_variables, **kwargs)
    config_final = finalize_config(config, initial_state.shape)

    final_state = time_integration(initial_state, config_final, params, registered_variables)

    final_rho_P = final_state[registered_variables.density_index] - rho_B
    if dim == 1:
        final_v_P = [final_state[registered_variables.velocity_index]]
    else:
        final_v_P = [final_state[registered_variables.velocity_index.x]]
    if dim > 1: final_v_P.append(final_state[registered_variables.velocity_index.y])
    if dim > 2: final_v_P.append(final_state[registered_variables.velocity_index.z])

    # Objective Function: J = 0.5 * integral(rho_P^2 + v_P^2) dV
    if mass_matrix == 'first_order':
        # Lumped (midpoint) mass matrix on cell averages.
        v_sq = sum(v**2 for v in final_v_P)
        J = 0.5 * jnp.sum(final_rho_P**2 + v_sq) * dV
    elif mass_matrix == 'fourier':
        # Sinc-corrected spectral quadrature — matches the high-order
        # polynomial reconstruction of WENO in the smooth limit.
        J = 0.5 * _spectral_squared_integral(final_rho_P, dx_vec)
        for v in final_v_P:
            J = J + 0.5 * _spectral_squared_integral(v, dx_vec)
    else:
        raise ValueError(f"Unknown mass_matrix option: {mass_matrix!r}")
    return J

def get_ad_gradients(rho_P0, v_P0_tuple, config, params, rho_B, c_s, P_B, mass_matrix='first_order'):
    """ Compiles and extracts JAX continuous Fréchet gradients via AD """
    dim = config.dimensionality
    L_vec = [config.box_size] if dim == 1 else ([config.box_size.x, config.box_size.y, config.box_size.z][:dim])
    N_vec = [config.num_cells] if dim == 1 else ([config.num_cells.x, config.num_cells.y, config.num_cells.z][:dim])
    dV = jnp.prod(jnp.array([L_vec[i]/N_vec[i] for i in range(dim)]))

    if dim == 1:
        cost_fn = lambda r, vx: run_forward_and_cost(r, (vx,), config, params, rho_B, c_s, P_B, mass_matrix=mass_matrix)
        grad_fn = jax.grad(cost_fn, argnums=(0, 1))
        grads = grad_fn(rho_P0, v_P0_tuple[0])
        return grads[0]/dV, [grads[1]/dV]
    elif dim == 2:
        cost_fn = lambda r, vx, vy: run_forward_and_cost(r, (vx, vy), config, params, rho_B, c_s, P_B, mass_matrix=mass_matrix)
        grad_fn = jax.grad(cost_fn, argnums=(0, 1, 2))
        grads = grad_fn(rho_P0, v_P0_tuple[0], v_P0_tuple[1])
        return grads[0]/dV, [grads[1]/dV, grads[2]/dV]
    elif dim == 3:
        cost_fn = lambda r, vx, vy, vz: run_forward_and_cost(r, (vx, vy, vz), config, params, rho_B, c_s, P_B, mass_matrix=mass_matrix)
        grad_fn = jax.grad(cost_fn, argnums=(0, 1, 2, 3))
        grads = grad_fn(rho_P0, v_P0_tuple[0], v_P0_tuple[1], v_P0_tuple[2])
        return grads[0]/dV, [grads[1]/dV, grads[2]/dV, grads[3]/dV]

# ==============================================================================
# 4. Running & Plotting Individual Setups
# ==============================================================================
def run_adjoint_test(dim, ic_type, compare_backends=False):
    print(f"\n{'-'*50}\nRunning {dim}D Adjoint Sensitivity Test: {ic_type.upper()}\n{'-'*50}")
    
    rho_B, c_s, gamma = 1.0, 2.0, 5/3
    P_B = (c_s**2) * rho_B / gamma
    eps = 1e-6
    
    # Establish a "sufficiently long" simulation time depending on box size
    if ic_type == 'gaussian':
        L, t_end = 10.0, 1.5  # Time for Gaussian blob to clearly separate into two waves
    else:
        L, t_end = 1.0, 0.15  # Time for phase shift in the simple wave
        
    N = 64
    
    config, params = get_config_and_params(dim, N, L, t_end)
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    coords, rho_P0, v_P0 = generate_ic(helper_data, dim, ic_type, eps, L)
    
    os.makedirs("figures", exist_ok=True)
    
    # --------------------------------------------------------------------------
    # Forward Pass Simulation (To plot Initial vs. Final State Evolution)
    # --------------------------------------------------------------------------
    print(f"Running forward simulation to t={t_end} for state evolution plot...")
    rho = rho_B + rho_P0
    p = P_B + (c_s**2) * rho_P0
    
    kwargs = {"density": rho, "gas_pressure": p, "velocity_x": v_P0[0]}
    if dim > 1: kwargs["velocity_y"] = v_P0[1]
    if dim > 2: kwargs["velocity_z"] = v_P0[2]
    
    initial_state = construct_primitive_state(config, registered_variables, **kwargs)
    config_final = finalize_config(config, initial_state.shape)
    final_state = time_integration(initial_state, config_final, params, registered_variables)
    
    # Extract states for plotting
    rho_final = final_state[registered_variables.density_index] - rho_B
    if dim == 1:
        v_final = [final_state[registered_variables.velocity_index]]
    else:
        v_final = [final_state[registered_variables.velocity_index.x]]

    if dim > 1: v_final.append(final_state[registered_variables.velocity_index.y])
    if dim > 2: v_final.append(final_state[registered_variables.velocity_index.z])

    # --------------------------------------------------------------------------
    # Compute Gradients
    # --------------------------------------------------------------------------
    print("Computing AD gradients through simulation...")
    # Backends overlaid in the gradient plot (FV (Pallas) left out).  FD (Pallas)
    # uses the wired native Pallas adjoint (periodic BCs here, so the fast GPU
    # adjoint kernel applies); FD/FV (JAX) use the native backward.  All three
    # should land on the exact Fourier gradient.
    # Distinct backend colours (FV green here to avoid clashing with the orange
    # exact-gradient line); FD (JAX) 'o' and FD (Pallas) 'x' on top stay visible.
    grad_backend_specs = [
        ("FD (JAX)",    FINITE_DIFFERENCE, NATIVE_JAX, dict(color='cornflowerblue', marker='o', s=20, zorder=2)),
        ("FD (Pallas)", FINITE_DIFFERENCE, PALLAS,     dict(color='darkviolet',     marker='x', s=26, zorder=3, linewidths=1.3)),
        ("FV (JAX)",    FINITE_VOLUME,     NATIVE_JAX, dict(color='tab:green',      marker='.', s=18, zorder=1)),
    ]
    ad_grads_by_backend = {}
    if compare_backends:
        for lab, sm, be, _st in grad_backend_specs:
            cfg_b, params_b = get_config_and_params(dim, N, L, t_end, solver_mode=sm)
            if be == PALLAS:
                cfg_b = cfg_b._replace(backend=PALLAS, pallas_use_triton=True, pallas_interpret=False)
            ad_grads_by_backend[lab] = get_ad_gradients(rho_P0, v_P0, cfg_b, params_b, rho_B, c_s, P_B)
        ad_grad_rho, ad_grad_v = ad_grads_by_backend[grad_backend_specs[0][0]]
    else:
        ad_grad_rho, ad_grad_v = get_ad_gradients(rho_P0, v_P0, config, params, rho_B, c_s, P_B)

    print("Computing Exact Analytical gradients (Fourier method)...")
    ana_grad_rho, ana_grad_v = compute_analytic_gradients_fourier(rho_P0, v_P0, L, c_s, rho_B, params.t_end)
    
    # --------------------------------------------------------------------------
    # Prepare Plot Arrays (Collapsing N-Dimensional Tensors to 1D Slices)
    # --------------------------------------------------------------------------
    ad_rho_norm, ana_rho_norm = ad_grad_rho / eps, ana_grad_rho / eps
    ad_v_norm = [adv / eps for adv in ad_grad_v]
    ana_v_norm = [anv / eps for anv in ana_grad_v]
    
    rho_init_norm = rho_P0 / eps
    rho_final_norm = rho_final / eps
    v_init_norm = [vi / eps for vi in v_P0]
    v_final_norm = [vf / eps for vf in v_final]

    if ic_type == 'wave':
        # Scatter onto 1D phase cycle layout
        k_vec = jnp.array([2 * jnp.pi * (i+1) * 2 / L for i in range(dim)])
        phase = sum(k_vec[i] * coords[i] for i in range(dim))
        phase_1d = phase.flatten() / (2 * jnp.pi)
        
        k_hat = k_vec / jnp.linalg.norm(k_vec)
        
        # Flatten all arrays
        x_plot = phase_1d
        y_ad_rho, y_ana_rho = ad_rho_norm.flatten(), ana_rho_norm.flatten()
        y_rho_init, y_rho_final = rho_init_norm.flatten(), rho_final_norm.flatten()
        
        # Project velocities longitudinally along the wave vector
        y_ad_v = sum(ad_v_norm[i] * k_hat[i] for i in range(dim)).flatten()
        y_ana_v = sum(ana_v_norm[i] * k_hat[i] for i in range(dim)).flatten()
        y_v_init = sum(v_init_norm[i] * k_hat[i] for i in range(dim)).flatten()
        y_v_final = sum(v_final_norm[i] * k_hat[i] for i in range(dim)).flatten()
        
        x_label = r'Wave Phase cycles ($\vec{k} \cdot \vec{x} / 2\pi$)'
        v_title = r'(Longitudinal Projection $v_\parallel$)'
        
        sort_idx = jnp.argsort(x_plot)
        x_line, y_ana_rho_line, y_ana_v_line = x_plot[sort_idx], y_ana_rho[sort_idx], y_ana_v[sort_idx]
        
    else:
        # Take an exact central slice out of the multidimensional tensor
        x_axis = coords[0][:, 0, 0] if dim == 3 else (coords[0][:, 0] if dim == 2 else coords[0])
        idx = () if dim == 1 else ((slice(None), N//2) if dim == 2 else (slice(None), N//2, N//2))
        
        x_plot, x_line = x_axis, x_axis
        y_ad_rho, y_ana_rho = ad_rho_norm[idx], ana_rho_norm[idx]
        y_rho_init, y_rho_final = rho_init_norm[idx], rho_final_norm[idx]
        
        y_ad_v, y_ana_v = ad_v_norm[0][idx], ana_v_norm[0][idx]
        y_v_init, y_v_final = v_init_norm[0][idx], v_final_norm[0][idx]
        
        y_ana_rho_line, y_ana_v_line = y_ana_rho, y_ana_v
        
        x_label = 'Spatial Coordinate $x$'
        v_title = r'(Velocity $v_x$ Slice)'

    # Apply the same normalisation + 1D slice used for the default backend to
    # each backend's AD gradient, so the gradient plot can overlay all three.
    if compare_backends:
        def _slice_ad(adr, adv):
            adr_n = adr / eps
            adv_n = [a / eps for a in adv]
            if ic_type == 'wave':
                yr = adr_n.flatten()
                yv = sum(adv_n[i] * k_hat[i] for i in range(dim)).flatten()
            else:
                yr = adr_n[idx]
                yv = adv_n[0][idx]
            return yr, yv
        y_ad_by_backend = {lab: _slice_ad(*ad_grads_by_backend[lab])
                           for lab, *_ in grad_backend_specs}

    # --------------------------------------------------------------------------
    # Plot 1: State Evolution (Initial vs Final)
    # --------------------------------------------------------------------------
    fig_state, axs_state = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # State: Density
    plot_fn = axs_state[0].scatter if ic_type == 'wave' else axs_state[0].plot
    plot_args = {'s': 15, 'alpha': 0.7} if ic_type == 'wave' else {'marker': 'o', 'linewidth': 2}
    plot_fn(x_plot, y_rho_init, label='Initial State (t=0)', **plot_args)
    plot_fn(x_plot, y_rho_final, label=f'Final State (t={t_end})', **plot_args)
    
    axs_state[0].set_title(rf'{dim}D {ic_type.capitalize()} State Evolution: Density Perturbation $\rho_P$')
    axs_state[0].set_ylabel(r'Amplitude (scaled by $1/\epsilon$)')
    axs_state[0].legend()
    axs_state[0].grid(True, linestyle='--', alpha=0.6)
    
    # State: Velocity
    plot_fn = axs_state[1].scatter if ic_type == 'wave' else axs_state[1].plot
    plot_fn(x_plot, y_v_init, label='Initial State (t=0)', **plot_args)
    plot_fn(x_plot, y_v_final, label=f'Final State (t={t_end})', **plot_args)
    
    axs_state[1].set_title(rf'{dim}D {ic_type.capitalize()} State Evolution: Velocity ' + v_title)
    axs_state[1].set_xlabel(x_label)
    axs_state[1].set_ylabel(r'Amplitude (scaled by $1/\epsilon$)')
    axs_state[1].legend()
    axs_state[1].grid(True, linestyle='--', alpha=0.6)
    
    fig_state.tight_layout()
    state_plot_path = f'figures/state_evolution_{dim}d_{ic_type}.png'
    fig_state.savefig(state_plot_path, bbox_inches='tight')
    plt.close(fig_state)

    # --------------------------------------------------------------------------
    # Plot 2: Gradients (AD vs Analytic)
    # --------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Grad: Density
    if compare_backends:
        for lab, _sm, _be, st in grad_backend_specs:
            axs[0].scatter(x_plot, y_ad_by_backend[lab][0], alpha=0.55,
                           label=f'AD: {lab}', **st)
    else:
        axs[0].scatter(x_plot, y_ad_rho, s=20, color='blue', alpha=0.5, label='JAX AD Gradients')
    axs[0].plot(x_line, y_ana_rho_line, '-', color='orange', linewidth=3, label='Exact Fourier Gradient')
    axs[0].set_title(rf'{dim}D {ic_type.capitalize()} Sensitivity to Initial Density: $\partial_{{U_0}} J$ for $\rho$')
    axs[0].set_ylabel('Gradient Amplitude')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # Grad: Velocity
    if compare_backends:
        for lab, _sm, _be, st in grad_backend_specs:
            axs[1].scatter(x_plot, y_ad_by_backend[lab][1], alpha=0.55,
                           label=f'AD: {lab}', **st)
    else:
        axs[1].scatter(x_plot, y_ad_v, s=20, color='green', alpha=0.5, label='JAX AD Gradients')
    axs[1].plot(x_line, y_ana_v_line, '-', color='red', linewidth=3, label='Exact Fourier Gradient')
    axs[1].set_title(rf'{dim}D {ic_type.capitalize()} Sensitivity to Initial Velocity: $\partial_{{U_0}} J$ ' + v_title)
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel('Gradient Amplitude')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    grad_plot_path = f'figures/gradient_{dim}d_{ic_type}.svg'
    fig.savefig(grad_plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Success! Evolution & Gradient plots saved to figures/ for {dim}D {ic_type}")

# ==============================================================================
# 5. Convergence Test of AD Gradients (cross-simulator evaluation)
# ==============================================================================
def run_gradient_convergence_test():
    print(f"\n{'-'*50}\nRunning 1D Gradient AD Convergence Test\n{'-'*50}")

    N_values = [16, 32, 64, 128, 256]

    # Each backend is (label, solver_mode, backend_id).  In BACKWARDS mode the FD
    # (Pallas) gradient is produced by the native Pallas adjoint kernel
    # (pallas_vjp_call), which equals the native gradient to ~1e-8 — far below
    # the discretisation error this convergence test resolves, so the FD (JAX)
    # and FD (Pallas) curves overlap.  FV (Pallas) is omitted.
    backends = [
        ("Finite Difference (JAX)",  FINITE_DIFFERENCE, NATIVE_JAX),
        ("Finite Volume (JAX)",      FINITE_VOLUME,     NATIVE_JAX),
        ("Finite Difference (Pallas)", FINITE_DIFFERENCE, PALLAS),
    ]

    errors_dict = {label: [] for (label, _, _) in backends}

    rho_B, c_s, gamma = 1.0, 2.0, 5/3
    P_B = (c_s**2) * rho_B / gamma
    eps = 1e-6
    L, t_end = 1.0, 0.15
    dim = 1

    for label, solver_mode, backend in backends:
        print(f"Testing Solver: {label}...")
        for N in N_values:
            config, params = get_config_and_params(dim, N, L, t_end, solver_mode=solver_mode)
            if backend == PALLAS:
                # Block shape must divide N. Use the largest power-of-two
                # block that fits, capped at 128.
                bx = 1
                while bx * 2 <= min(N, 128) and N % (bx * 2) == 0:
                    bx *= 2
                config = config._replace(
                    backend=PALLAS,
                    pallas_block_shape=(bx, 1, 1),
                    pallas_interpret=False,
                    pallas_use_triton=True,
                )
            helper_data = get_helper_data(config)

            _, rho_P0, v_P0 = generate_ic(helper_data, dim, 'wave', eps, L)

            ad_grad_rho, ad_grad_v = get_ad_gradients(rho_P0, v_P0, config, params, rho_B, c_s, P_B)
            ana_grad_rho, ana_grad_v = compute_analytic_gradients_fourier(rho_P0, v_P0, L, c_s, rho_B, params.t_end)

            ad_norm_r, ana_norm_r = ad_grad_rho / eps, ana_grad_rho / eps
            ad_norm_v, ana_norm_v = ad_grad_v[0] / eps, ana_grad_v[0] / eps

            err_rho = jnp.mean(jnp.abs(ad_norm_r - ana_norm_r))
            err_vx = jnp.mean(jnp.abs(ad_norm_v - ana_norm_v))
            total_l1_error = (err_rho + err_vx) / 2.0

            errors_dict[label].append(total_l1_error)
            print(f"  -> N={N:3d}: L1 Error = {total_l1_error:.2e}")

    fig_err, ax_err = plt.subplots(1, 1, figsize=(8, 6))
    N_arr = np.array(N_values)

    # Consistent backend colours (see the other figures): FD (JAX) light blue,
    # FD (Pallas) violet, FV (JAX) orange; FD (JAX) thick solid / 'o',
    # FD (Pallas) thinner dashed / 'x' on top, so both stay visible on overlap.
    styles = {
        "Finite Difference (JAX)":    dict(color='cornflowerblue', marker='o', linestyle='-',  linewidth=3.0, markersize=6, zorder=2),
        "Finite Volume (JAX)":        dict(color='tab:orange',     marker='o', linestyle='-',  linewidth=2.2, markersize=6, zorder=2),
        "Finite Difference (Pallas)": dict(color='darkviolet',     marker='x', linestyle='--', linewidth=1.8, markersize=7, zorder=3),
    }
    for label, _, _ in backends:
        ax_err.loglog(N_arr, errors_dict[label], label=label, **styles[label])

    # Short reference-slope triangles placed next to the curves they describe:
    # the -2 line tracks the (shallow) finite-volume row and the -5 line tracks
    # the (steep) finite-difference row. Keep x_span ~ one octave so the
    # indicators stay inside the plotted N range instead of running off-axis.
    add_power_law_indicators(
        ax_err,
        anchor=(48.0, 1.0),
        exponents=[-2, -5],
        scales=[0.6, 8e-4],
        x_span=2.0,
        x_label='N',
        text_kwargs=dict(fontsize=11, ha='left', va='center'),
    )

    ax_err.set_xlabel('Grid Resolution N', fontsize=12)
    ax_err.set_ylabel(r'Average $L_1$ Error ($\partial J/\partial U_0$ vs Exact)', fontsize=12)
    ax_err.set_title('AD Gradient Convergence to Exact Analytical Fourier Operator', fontsize=14)
    # Show only the integer N ticks; suppress the default log minor-tick labels
    # (2x10^1, 3x10^1, ...) that otherwise overlap the 16/32/64/... labels.
    import matplotlib.ticker as mticker
    ax_err.set_xticks(N_values)
    ax_err.set_xticklabels([str(n) for n in N_values])
    ax_err.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_err.legend(loc='lower left', fontsize=9)
    ax_err.grid(True, which="both", ls="-", alpha=0.2)

    os.makedirs("figures", exist_ok=True)
    fig_err.tight_layout()
    plot_path = "figures/gradient_convergence_test.svg"
    fig_err.savefig(plot_path)
    plt.close(fig_err)
    print(f"Success! Convergence plot saved to {plot_path}")

# ==============================================================================
# 6. Mass Matrix Comparison (3D, low resolution)
# ==============================================================================
def run_3d_mass_matrix_comparison(N=16):
    """ Compare AD gradients computed with the first-order (lumped) mass matrix
        vs the high-order Fourier mass matrix on a 3D smooth wave at low N,
        with the exact analytical Fourier gradient as reference. """
    print(f"\n{'-'*50}\nRunning 3D Mass Matrix Comparison (N={N}, FD/WENO)\n{'-'*50}")

    rho_B, c_s, gamma = 1.0, 2.0, 5/3
    P_B = (c_s**2) * rho_B / gamma
    eps = 1e-6
    L, t_end = 1.0, 0.15
    dim = 3

    config, params = get_config_and_params(dim, N, L, t_end, solver_mode=FINITE_DIFFERENCE)
    helper_data = get_helper_data(config)
    coords, rho_P0, v_P0 = generate_ic(helper_data, dim, 'wave', eps, L)

    print("AD gradient with first-order (lumped) mass matrix...")
    ad_grad_rho_lo, ad_grad_v_lo = get_ad_gradients(
        rho_P0, v_P0, config, params, rho_B, c_s, P_B, mass_matrix='first_order')
    print("AD gradient with high-order (Fourier) mass matrix...")
    ad_grad_rho_hi, ad_grad_v_hi = get_ad_gradients(
        rho_P0, v_P0, config, params, rho_B, c_s, P_B, mass_matrix='fourier')
    print("Analytical Fourier gradients (matched mass matrices)...")
    ana_grad_rho_lo, ana_grad_v_lo = compute_analytic_gradients_fourier(
        rho_P0, v_P0, L, c_s, rho_B, params.t_end, mass_matrix='first_order')
    ana_grad_rho_hi, ana_grad_v_hi = compute_analytic_gradients_fourier(
        rho_P0, v_P0, L, c_s, rho_B, params.t_end, mass_matrix='fourier')

    # Project velocity gradient longitudinally along the wave direction.
    k_vec = jnp.array([2 * jnp.pi * (i+1) * 2 / L for i in range(dim)])
    k_hat_vec = k_vec / jnp.linalg.norm(k_vec)
    phase = sum(k_vec[i] * coords[i] for i in range(dim))
    phase_1d = (phase.flatten() / (2 * jnp.pi))

    def _flatten(scaled):
        return (scaled / eps).flatten()

    def _proj(v_list):
        return sum(v_list[i] * k_hat_vec[i] for i in range(dim))

    y_ad_rho_lo = _flatten(ad_grad_rho_lo)
    y_ad_rho_hi = _flatten(ad_grad_rho_hi)
    y_ana_rho_lo = _flatten(ana_grad_rho_lo)
    y_ana_rho_hi = _flatten(ana_grad_rho_hi)

    y_ad_v_lo = _flatten(_proj(ad_grad_v_lo))
    y_ad_v_hi = _flatten(_proj(ad_grad_v_hi))
    y_ana_v_lo = _flatten(_proj(ana_grad_v_lo))
    y_ana_v_hi = _flatten(_proj(ana_grad_v_hi))

    sort_idx = jnp.argsort(phase_1d)
    x_line = phase_1d[sort_idx]
    y_ana_rho_lo_line = y_ana_rho_lo[sort_idx]
    y_ana_rho_hi_line = y_ana_rho_hi[sort_idx]
    y_ana_v_lo_line = y_ana_v_lo[sort_idx]
    y_ana_v_hi_line = y_ana_v_hi[sort_idx]

    err_rho_lo = float(jnp.mean(jnp.abs(y_ad_rho_lo - y_ana_rho_lo)))
    err_rho_hi = float(jnp.mean(jnp.abs(y_ad_rho_hi - y_ana_rho_hi)))
    err_v_lo = float(jnp.mean(jnp.abs(y_ad_v_lo - y_ana_v_lo)))
    err_v_hi = float(jnp.mean(jnp.abs(y_ad_v_hi - y_ana_v_hi)))
    print(f"  L1 error AD vs matched analytical  rho:  lumped MM = {err_rho_lo:.3e},  Fourier MM = {err_rho_hi:.3e}")
    print(f"  L1 error AD vs matched analytical  v||:  lumped MM = {err_v_lo:.3e},  Fourier MM = {err_v_hi:.3e}")

    sep_rho = float(jnp.mean(jnp.abs(y_ana_rho_hi - y_ana_rho_lo)))
    sep_v = float(jnp.mean(jnp.abs(y_ana_v_hi - y_ana_v_lo)))
    print(f"  Separation between MMs (analytical lines)  rho: {sep_rho:.3e},  v||: {sep_v:.3e}")

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].scatter(phase_1d, y_ad_rho_lo, s=18, color='steelblue', alpha=0.45,
                   label='AD: first-order (lumped) mass matrix')
    axs[0].scatter(phase_1d, y_ad_rho_hi, s=18, color='seagreen', alpha=0.6, marker='^',
                   label='AD: high-order Fourier mass matrix')
    axs[0].plot(x_line, y_ana_rho_lo_line, '-', color='orange', linewidth=2.0,
                label='Analytical (lumped MM target)')
    axs[0].plot(x_line, y_ana_rho_hi_line, '--', color='firebrick', linewidth=2.0,
                label='Analytical (Fourier MM target)')
    axs[0].set_title(rf'3D wave (N={N}, 5th-order WENO): $\partial J/\partial \rho_0$')
    axs[0].set_ylabel('Gradient amplitude')
    axs[0].legend(loc='best', fontsize=9)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    axs[1].scatter(phase_1d, y_ad_v_lo, s=18, color='steelblue', alpha=0.45,
                   label='AD: first-order (lumped) mass matrix')
    axs[1].scatter(phase_1d, y_ad_v_hi, s=18, color='seagreen', alpha=0.6, marker='^',
                   label='AD: high-order Fourier mass matrix')
    axs[1].plot(x_line, y_ana_v_lo_line, '-', color='orange', linewidth=2.0,
                label='Analytical (lumped MM target)')
    axs[1].plot(x_line, y_ana_v_hi_line, '--', color='firebrick', linewidth=2.0,
                label='Analytical (Fourier MM target)')
    axs[1].set_title(rf'3D wave (N={N}, 5th-order WENO): $\partial J/\partial v_\parallel$ (longitudinal)')
    axs[1].set_xlabel(r'Wave phase cycles ($\vec{k} \cdot \vec{x} / 2\pi$)')
    axs[1].set_ylabel('Gradient amplitude')
    axs[1].legend(loc='best', fontsize=9)
    axs[1].grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plot_path = f"figures/mass_matrix_comparison_3d_N{N}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Success! Plot saved to {plot_path}")

# ==============================================================================
# Execution
# ==============================================================================
if __name__ == "__main__":
    # Test Gradients in all dimensional setups mapped onto Wave and Gaussian cases
    for dimensionality in [1, 2, 3]:
        for condition in ['wave', 'gaussian']:
            run_adjoint_test(
                dimensionality, condition,
                compare_backends=(dimensionality == 3 and condition == 'gaussian'),
            )

    # Test continuous convergence across solvers
    run_gradient_convergence_test()

    # Mass-matrix comparison on a low-resolution 3D test. N<48 with the
    # k_vec=(4π, 8π, 12π) IC pushes the k_z=12π mode past WENO's stability
    # margin in 3D and the run blows up; N=48 is the smallest stable point.
    run_3d_mass_matrix_comparison(N=48)