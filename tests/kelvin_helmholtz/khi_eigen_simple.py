"""
# Kelvin-Helmholtz instability in 2D

We analyze the Kelvin-Helmholtz instability across 
Mach and Reynolds numbers.

## Setup

Where not indicated differently, 
our general setup follows Mandelker et al, 2016,
https://arxiv.org/pdf/1606.06289. Note that they
use different coordinate conventions, 

- their x-axis (transverse) is our y-axis
- their z-axis (flow direction) is our x-axis

### Baseline setup

Consider a dense slab with radius R_s flowing 
through a dilute background medium along the x-axis. 

- the simulation domain is [0, 1] x [0, 1] with periodic boundaries at x = 0 and x = 1
  and periodic (open in Mandelker et al.) boundaries at y = 0 and y = 1
- R_s = 1/160 in Mandelker et al.
- the slab is centered at y_c = 0.5
- slab and background are ideal gases with γ = 5/3 and P = 1 everywhere
- the background has density ρ_b = 1 and velocity v_b = 0
- the slab has density ρ_s = χ and velocity v_s = M_b * c_b * x̂, 
  where c_b = sqrt(γ * P / ρ_b) = sqrt(5/3) and M_b is the Mach 
  number of the hot dilute background flow relative to the slab
- diffusion can be included, the kinematic viscosity ν is set

Truly discontinuous density and velocity 
profiles lead to numerical noise at the grid-scale,
such that we use a smoothed transition

f(y) = f_b + 0.25 * (f_s - f_b) * (1 + tanh((R_s - (y - y_c)) / σ)) * (1 + tanh((R_s + (y - y_c)) / σ))

with smoothing length σ, e.g. σ = λ / 102, 
where λ is the wavelength of the perturbation.

Roediger et al. 2013 only use smoothing for the highest
Re run, they then smooth with σ being 1 to 2% of λ.

### Perturbation

There are multiple ways to perturb this system.

#### Velocity only perturbation

Following Sec. 3.1 in Roediger et al. 2013, we can perturb

v_y = v_0 sin(k * x) * [exp(-((y - y_c) - R_s)^2 / (σ_y^2)) + exp(-((y - y_c) + R_s)^2 / (σ_y^2))]

with 

- k = 2 * π / λ
- σ_y^2 = 0.3 * λ
- v_0 = 0.1 * v_s (the slab and also the shear velocity)

#### Pressure only perturbation (Sec. 3.3 in Mandelker et al.)

We can perturb the pressure with

P_1 = A * cos(k * x) * [exp(-((y - y_c) - R_s)^2 / (2 Σ^2)) + exp(-((y - y_c) + R_s)^2 / (2 Σ^2))]

e.g. with the same A and k as above and e.g. Σ = 5 * σ or

- A = 0.05
- Σ = 5 * σ
- k = 2 * π / λ, e.g. λ = R_s or λ = 2 * R_s

#### Eigenmode perturbations

##### Mandelker et al. analytical approach

Following Sec. 2.1 and 2.3 in Mandelker et al., by inserting
small perturbations into the Euler equations, one can relate
the perturbations in velocity, density, and pressure
to each other.

Denote \partial_y f as f' and k_x = k * cos(θ).

Given a pressure perturbation P_1(x, y),

ρ_1 = -1 / (k^2 (v_k - ω / k)^2) * [P_1'' - 2v_k' / (v_k - ω / k) * P_1' - k^2 P_1]
v_{1x} = -cos(θ) / (ρ_0 (v_k - ω / k)) * [v_k' / (k^2 cos^2(θ) (v_k - ω / k)) * P_1' + P_1]
v_{1y} = i / (ρ_0 k (v_k - ω / k)) * P_1'

where 

- k is the wavenumber of the perturbation
- θ is the angle of the perturbation wavevector with respect to the flow direction (x-axis)
- v_k = v_0 \cdot k = v_0 * cos(θ) is the component of the background 
  velocity in the direction of the perturbation wavevector
- ω is the complex frequency of the perturbation

In our case of the slab flowing along the x-axis, we have

- subscript s denotes the slab and subscript b denotes the background
- θ = 0 (perturbation aligned with the flow)
- v_k = v_0 = const. -> v_k' = 0

so

ρ_1 = -1 / (k^2 (v_0 - ω / k)^2) * [P_1'' - k^2 P_1]
v_{1x} = -1 / (ρ_0 (v_0 - ω / k)) * P_1
v_{1y} = i / (ρ_0 k (v_0 - ω / k)) * P_1'

For the slab, one finds the pressure perturbation P_1(y)
to be

         { A * exp(-q_b * (y - y_c - R_s)) for y > y_c + R_s
P_1(y) = | A * S(q_s * (y - y_c)) / S(q_s * R_s) for y_c - R_s <= y <= y_c + R_s
         { ±A * exp(q_b * (y - y_c + R_s)) for y < y_c - R_s

where

- S = cosh for pinch modes (plus sign in the bottom line)
- S = sinh for sinusoidal modes (minus sign in the bottom line)

where

- q_b = k * sqrt(1 - (Δv_b / c_b)^2)
- q_s = k * sqrt(1 - (Δv_s / c_s)^2)

where

- Δv_b = v_b - ω/k
- Δv_s = v_s - ω/k

Now we are still missing the complex frequency ω.
It follows from solving Eq. 25 in Mandelker et al.

(ω - k * v_b)^2 / (ω - k * v_s)^2 = -ρ_s / ρ_b * q_b / q_s * T(q_s * R_s)

where

- T = tanh for sinusoidal modes
- T = coth for pinch modes

This is solved by numerical root finding, under 
the constraint that Re(q_b) > 0 and 
Re(q_s) > 0 (wave decays towards
the edge).

The characteristic time of perturbation growth in 
the linear regime is the Kelvin-Helmholtz time 
k_KH = 1 / Im(ω).

Smoothing is applied to all initial perturbations 
(not fully consistent, but sufficient).

For more details, see the Mandelker et al. paper.

##### Purely numerical eigenmodes (optional)

One might alternatively directly linearize the Euler equations
around the background flow and numerically solve the resulting
eigenvalues problem. We might use JAX autodiff to linearize
the RHS of our fluid solver directly and solve for the eigenmodes
of the resulting linear operator, this allows us to allows us
to find the eigenmodes of our numerical scheme with the smoothed
initial conditions, and is flexible regarding the inclusion of 
additional physics like viscosity.

Ideally, one would implement both approaches 
and check for consistency.

## Criticality

### Critical Mach number

The critical Mach number is given by

M_crit = (1 + χ^{-1/3})^{3/2}  (Eq. 22 in Mandelker et al.)

where χ = ρ_s / ρ_b is the density contrast 
between the slab and the background.

- M_b < M_crit: surface modes grow rapidly, stream disrupts
- M_b > M_crit: surface modes suppressed, stream remains stable

### Critical Reynolds number

Roediger et al. 2013 (https://arxiv.org/pdf/1309.2635) empirically found

  Re_crit = 880 / Δ       (Eq. 22, for constant kinematic viscosity)
  Re_0 = 1320 / sqrt(Δ)   (Eq. 23)
  Δ = (ρ_cold + ρ_hot)² / (ρ_cold ρ_hot)

with the viscous growth time (Eq. 21):

  τ_KH,visc = τ_KH,inv × [1 + Re₀ / (Re − Re_crit)]

while the non-viscous growth time is

	t_kh = jnp.sqrt(Delta) / (2 * jnp.pi) * wavelength / v_slab

where approximately for

- Re >> Re_crit: viscosity negligible, usual KHI growth
- Re <  Re_crit:  viscosity dominates, KHI suppressed

However, suppresion by viscosity is a gradual effect and
it is hard to define what "suppressed" means in practice.

## Experimental diagnostics

### Growth time
Consider the maximum (or minimum) v_y, v_ymax as a function of time.

- until ~ 1 t_kh, v_ymax reflects sound waves from the initialization and we ignore it
- from ~ 1 t_kh to ~ 5 t_kh, we measure the growth time t_kh_measured by fitting
    v_ymax(t) = v_ymax(t=0) * exp(t / t_kh_measured)

"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

from functools import partial

# typing
from dataclasses import dataclass
from typing import NamedTuple

# numerics
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, ArpackNoConvergence
import scipy.sparse as sp
import scipy.linalg as la

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib as mpl

# astronomix
from astronomix import (
    SimulationConfig,
    get_helper_data,
    SimulationParams,
    time_integration,
    construct_primitive_state,
    get_registered_variables,
)
from astronomix._fluid_equations._equations import conserved_state_from_primitive, primitive_state_from_conserved
from astronomix._finite_difference._interface_fluxes._weno import _weno_flux_x, _weno_flux_y
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix._geometry.boundaries import _boundary_handler
from astronomix.time_stepping._utils import _pad, _unpad
from astronomix.option_classes.simulation_config import (
    CONSERVATIVE_GAS_STATE,
    DYNAMIC_VISCOSITY,
    FINITE_VOLUME,
    KINEMATIC_VISCOSITY,
    MINMOD,
    MUSCL,
    OPEN_BOUNDARY,
    SPLIT,
    UNSPLIT,
    SnapshotSettings,
    finalize_config,
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)
from astronomix.analysis_helpers.jacobians import single_xmode_jacobian2Dt, single_xmode_rhs_jacobian2D
from astronomix.plotting_helpers.inset_box import add_inset_box

SINGLE_INTERFACE = 0
SLAB = 1
interface_mode = SINGLE_INTERFACE

ROEDIGER = 0
TIRSO = 1
paper_mode = TIRSO

VELOCITY_PERTURBATION = 0
PRESSURE_PERTURBATION = 1
EIGENMODE_PERTURBATION = 2

# global parameters
y_center = 0.5
background_density = 1.0
pressure = 1.0 # uniform pressure everywhere
background_velocity = 0.0
# box_size = 1.0
box_size = 1.0
gamma = 5/3

num_cells = 300

@dataclass
class PerturbationSetup:

	#: The amplitude of the perturbation.
	amplitude: float

	#: The wavelength of the perturbation.
	wavelength: float

@dataclass
class PressurePerturbationSetup(PerturbationSetup):

	#: The width of the Gaussian envelope for the pressure perturbation.
	gaussian_width: float # Σ

class KHISetup(NamedTuple):

    #: The number of cells in the x and y directions.
    num_cells: int

    #: Simulation duration.
    simulation_time: float

    #: The perturbation type.
    perturbation_type: int

    #: The perturbation setup.
    perturbation_setup: PerturbationSetup

    #: Switch for whether to include diffusion (viscosity) or not.
    diffusion: bool

    #: The kinematic viscosity of the fluid, if diffusion is included.
    viscosity: float

    #: The Mach number of the slab relative to the background.
    mach_number: float # M_b

    #: The density contrast between the slab and the background.
    density_contrast: float # χ

    #: The radius (half-width) of the slab.
    slab_radius: float # R_s

    #: The smoothing length for the initial conditions.
    smoothing_length: float # σ
        
    #: Whether to use smooth profiles for the initial conditions or not.
    smooth_profiles: bool = True

def slab_profile(f_b, f_s, Y, y_center, slab_radius, smoothing_length):
    """Tanh transition from f_b (background) to f_s (stream)."""

	# f(y) = f_b + 0.25 * (f_s - f_b) * (1 + tanh((R_s - (y - y_c)) / σ)) * (1 + tanh((R_s + (y - y_c)) / σ))
    return f_b + 0.25 * (f_s - f_b) * (
		(1 + jnp.tanh((slab_radius - (Y - y_center)) / smoothing_length)) *
		(1 + jnp.tanh((slab_radius + (Y - y_center)) / smoothing_length))
	)

def single_interface(f_l, f_u, Y, y_center, smoothing_length):
	return 0.5 * (f_l * (1 - jnp.tanh((Y - y_center) / smoothing_length)) + f_u * (1 + jnp.tanh((Y - y_center) / smoothing_length)))

def velocity_perturbation(cell_centers, slab_radius, perturbation_setup):
	# NOTE: here without smoothing
	k = 2 * jnp.pi / perturbation_setup.wavelength
	X = cell_centers[:, :, 0] # x-coordinates of cell centers
	Y = cell_centers[:, :, 1] # y-coordinates of cell centers
	sigma_y = 0.3 * perturbation_setup.wavelength
	vy_pert = perturbation_setup.amplitude * jnp.sin(k * X) * (
		jnp.exp(-((Y - y_center) - slab_radius)**2 / (sigma_y**2)) + 
		jnp.exp(-((Y - y_center) + slab_radius)**2 / (sigma_y**2))
	)
	return vy_pert

def single_interface_velocity_perturbation(cell_centers, perturbation_setup):
	# v_y = A * sin(k * x) * exp(-((y - y_c) / σ)^2)
	k = 2 * jnp.pi / perturbation_setup.wavelength
	X = cell_centers[:, :, 0] # x-coordinates of cell centers
	Y = cell_centers[:, :, 1] # y-coordinates of cell centers
	sigma = 0.2 * perturbation_setup.wavelength
	vy_pert = perturbation_setup.amplitude * jnp.sin(k * X) * jnp.exp(-((Y - y_center) / sigma)**2)
	return vy_pert

def pressure_perturbation(cell_centers, slab_radius, y_center, perturbation_setup):
	# P_1 = A * cos(k * x) * [exp(-((y - y_c) - R_s)^2 / (2 Σ^2)) + exp(-((y - y_c) + R_s)^2 / (2 Σ^2))]
	k = 2 * jnp.pi / perturbation_setup.wavelength
	X = cell_centers[:, :, 0] # x-coordinates of cell centers
	Y = cell_centers[:, :, 1] # y-coordinates of cell centers
	P_1 = perturbation_setup.amplitude * jnp.cos(k * X) * (
		jnp.exp(-((Y - y_center) - slab_radius)**2 / (2 * perturbation_setup.gaussian_width**2)) + 
		jnp.exp(-((Y - y_center) + slab_radius)**2 / (2 * perturbation_setup.gaussian_width**2))
	)
	return P_1

def setup_khi_unperturbed_base(setup: KHISetup, return_snapshots = False):
	
    # set up the simulation configuration
    config = SimulationConfig(
        solver_mode = FINITE_DIFFERENCE,
        progress_bar = True,
        dimensionality = 2,
        box_size = box_size,
        num_cells = setup.num_cells,
        diffusion = setup.diffusion,
        viscosity_type = DYNAMIC_VISCOSITY, # KINEMATIC_VISCOSITY,
        boundary_settings = BoundarySettings(
            x = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),   # flow dir
            y = BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),           # transverse
        ),
        return_snapshots = return_snapshots,
        num_snapshots = 300,
        snapshot_settings = SnapshotSettings(
            return_states = True,
        )
    )

    # set up the simulation parameters
    params = SimulationParams(
        viscosity = setup.viscosity,
        t_end = setup.simulation_time,
        C_cfl = 1.5 if config.solver_mode == FINITE_DIFFERENCE else 0.4, 
        gamma = gamma,
    )

    # helper data
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # construct the unperturbed initial conditions
    cell_centers = helper_data.geometric_centers
    Y = cell_centers[:, :, 1] # y-coordinates of cell centers
    stream_density = setup.density_contrast * background_density

    c_background = jnp.sqrt(params.gamma * pressure / background_density)

    smooth_profiles = setup.smooth_profiles

    if smooth_profiles:
        if interface_mode == SINGLE_INTERFACE:

            density = single_interface(
                f_l = stream_density,
                f_u = background_density,
                Y = Y,
                y_center = y_center,
                smoothing_length = setup.smoothing_length,
            )

            v_mach = setup.mach_number * c_background
            v_x = single_interface(
                f_l = -v_mach / 2,
                f_u = v_mach / 2,
                Y = Y,
                y_center = y_center,
                smoothing_length = setup.smoothing_length,
            )
        
        elif interface_mode == SLAB:
            density = slab_profile(
                f_b = background_density,
                f_s = stream_density,
                Y = Y,
                y_center = y_center,
                slab_radius = setup.slab_radius,
                smoothing_length = setup.smoothing_length,
            )

            v_x = slab_profile(
                f_b = background_velocity,
                f_s = setup.mach_number * c_background,
                Y = Y,
                y_center = y_center,
                slab_radius = setup.slab_radius,
                smoothing_length = setup.smoothing_length,
            )
    else:
        if interface_mode == SLAB:
            density = jnp.where(
                jnp.abs(Y - y_center) <= setup.slab_radius,
                stream_density,
                background_density
            )

            v_x = jnp.where(
                jnp.abs(Y - y_center) <= setup.slab_radius,
                setup.mach_number * c_background,
                background_velocity
            )
        else:
            density = jnp.where(
                Y < y_center,
                stream_density,
                background_density
            )
            v_mach = setup.mach_number * c_background
            v_x = jnp.where(
                Y < y_center,
                -v_mach / 2,
                v_mach / 2
            )

    v_y = jnp.zeros_like(v_x)

    P = jnp.full_like(v_x, pressure)

    primitive_state_unperturbed = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = density,
        velocity_x = v_x,
        velocity_y = v_y,
        gas_pressure = P,
    )

    config = finalize_config(config, primitive_state_unperturbed.shape)

    return primitive_state_unperturbed, config, params, registered_variables, helper_data

def perturb_khi(primitive_state_unperturbed, setup, config, params, helper_data, registered_variables):
	
    cell_centers = helper_data.geometric_centers

    # add perturbation
    if setup.perturbation_type == VELOCITY_PERTURBATION:
        if interface_mode == SLAB:
            vy_pert = velocity_perturbation(
                cell_centers = cell_centers,
                slab_radius = setup.slab_radius,
                perturbation_setup = setup.perturbation_setup,
            )
            primitive_state = primitive_state_unperturbed.at[
                registered_variables.velocity_index.y
            ].add(vy_pert)

        elif interface_mode == SINGLE_INTERFACE:
            vy_pert = single_interface_velocity_perturbation(
                cell_centers = cell_centers,
                perturbation_setup = setup.perturbation_setup
            )
            primitive_state = primitive_state_unperturbed.at[
                registered_variables.velocity_index.y
            ].add(vy_pert)

    elif setup.perturbation_type == PRESSURE_PERTURBATION:
        if interface_mode == SLAB:
            P_pert = pressure_perturbation(
                cell_centers = cell_centers,
                slab_radius = setup.slab_radius,
                y_center = y_center,
                perturbation_setup = setup.perturbation_setup,
            )
            primitive_state = primitive_state_unperturbed.at[
                registered_variables.pressure_index
            ].add(P_pert)
        elif interface_mode == SINGLE_INTERFACE:
            raise NotImplementedError("Single interface with pressure perturbation not implemented yet.")
        
    elif setup.perturbation_type == EIGENMODE_PERTURBATION:
        primitive_state, dom_eval = find_numerical_eigenmode(
        	primitive_state_unperturbed,
        	config,
        	params,
        	registered_variables,
        	helper_data,
        	setup
        )
        growth_rate = float(jnp.real(dom_eval))
        phase_freq = float(jnp.imag(dom_eval))
        print(f"🎯 Exact Complex Eigenmode | Growth Rate σ: {growth_rate:.4e} | Phase ω_R: {phase_freq:.4e}")
    else:
        raise NotImplementedError("Only velocity and pressure perturbations are implemented so far.")
    
    return primitive_state

def simulate_khi(setup: KHISetup, return_snapshots = False):

    (
        primitive_state_unperturbed,
        config,
        params,
        registered_variables,
        helper_data,
    ) = setup_khi_unperturbed_base(setup, return_snapshots)
	
    primitive_state = perturb_khi(primitive_state_unperturbed, setup, config, params, helper_data, registered_variables)

    # finalize the simulation configuration
    config = finalize_config(config, primitive_state.shape)

    # run the simulation
    result = time_integration(
        primitive_state,
        config, 
        params,
        registered_variables
    )

    return result, helper_data, registered_variables


# -------------------------------------------------------------
# ================ ↓ Eigenmode initialization ↓ ===============
# -------------------------------------------------------------
def find_numerical_eigenmode(
    primitive_state_unperturbed,
    config,
    params,
    registered_variables,
    helper_data,
    setup: KHISetup,
    assembly_batch_size=4,
    require_growth=True,
    print_candidates=12,
):
    """
    Compute a numerical KH eigenmode from the full AD Fourier-block Jacobian.

    Uses:
      - single_xmode_rhs_jacobian2D(...) for the raw +kx simulator Jacobian,
      - jnp.linalg.eig(...) for the full dense eigensystem,
      - physical KH scoring based on growth, localization, edge power,
        roughness, and high-ky content.
    """

    print(
        f"⏳ Extracting raw-AD KH eigenmode at "
        f"λx={setup.perturbation_setup.wavelength} ..."
    )

    config = finalize_config(config, primitive_state_unperturbed.shape)
    gamma = params.gamma

    rho_i = registered_variables.density_index
    momx_i = registered_variables.velocity_index.x
    momy_i = registered_variables.velocity_index.y
    energy_i = registered_variables.pressure_index

    cons0 = conserved_state_from_primitive(
        primitive_state_unperturbed,
        gamma,
        config,
        registered_variables,
    )

    nvar, Nx, Ny = cons0.shape

    Lx = float(config.box_size.x)
    Ly = float(config.box_size.y) if hasattr(config.box_size, "y") else Lx

    m_float = Lx / float(setup.perturbation_setup.wavelength)
    m = int(round(m_float))

    if abs(m_float - m) > 1e-12:
        raise ValueError(
            f"wavelength={setup.perturbation_setup.wavelength} is not "
            f"commensurate with Lx={Lx}. Got Lx / wavelength = {m_float}."
        )

    kx = 2.0 * jnp.pi * m / Lx

    X = helper_data.geometric_centers[:, :, 0]
    Y = helper_data.geometric_centers[:, :, 1]
    y = Y[0, :]
    dy = y[1] - y[0]

    rho0 = primitive_state_unperturbed[rho_i, 0, :]
    u0 = primitive_state_unperturbed[momx_i, 0, :]
    v0 = primitive_state_unperturbed[momy_i, 0, :]

    # ------------------------------------------------------------------
    # Full AD Fourier-block Jacobian and eigensystem.
    # ------------------------------------------------------------------
    print("  Building full AD Fourier-block Jacobian ...")
    J = single_xmode_rhs_jacobian2D(
        primitive_state_unperturbed,
        config,
        params,
        registered_variables,
        helper_data,
        wavelength=setup.perturbation_setup.wavelength,
        assembly_batch_size=assembly_batch_size,
    )

    print("  Solving dense JAX eigenproblem ...")
    eigvals, eigvecs = jnp.linalg.eig(J)

    # eigvecs[:, i] is eigenvector i. Convert to modes[i, var, y].
    modes = eigvecs.T.reshape((-1, nvar, Ny))

    # ------------------------------------------------------------------
    # Conserved -> primitive perturbations.
    # ------------------------------------------------------------------
    def conserved_to_primitive_hat(qhat):
        drho = qhat[:, rho_i, :]
        dmx = qhat[:, momx_i, :]
        dmy = qhat[:, momy_i, :]
        dE = qhat[:, energy_i, :]

        du = dmx / rho0[None, :] - u0[None, :] * drho / rho0[None, :]
        dv = dmy / rho0[None, :] - v0[None, :] * drho / rho0[None, :]

        dp = (gamma - 1.0) * (
            dE
            - 0.5 * drho * (u0[None, :] ** 2 + v0[None, :] ** 2)
            - rho0[None, :] * (u0[None, :] * du + v0[None, :] * dv)
        )

        return drho, du, dv, dp

    drho, du, dv, dp = conserved_to_primitive_hat(modes)

    # ------------------------------------------------------------------
    # KH scoring diagnostics.
    # ------------------------------------------------------------------
    if interface_mode == SLAB:
        envelope = (
            jnp.exp(-kx * jnp.abs(y - (y_center - setup.slab_radius)))
            + jnp.exp(-kx * jnp.abs(y - (y_center + setup.slab_radius)))
        )
    else:
        envelope = jnp.exp(-kx * jnp.abs(y - y_center))

    envelope = envelope / jnp.max(envelope)

    y_min = jnp.min(y)
    y_max = jnp.max(y)
    dist_to_edge = jnp.minimum(y - y_min, y_max - y)
    edge_mask = dist_to_edge < 0.12 * Ly

    diagnostic_max_y_mode = int(
        min(
            Ny // 2,
            max(24, int(jnp.ceil(2.0 * Ly / setup.smoothing_length))),
        )
    )

    ky_modes = jnp.round(jnp.fft.fftfreq(Ny, d=1.0 / Ny)).astype(jnp.int32)
    high_ky_mask = jnp.abs(ky_modes) > diagnostic_max_y_mode // 2

    def ddy_centered(z):
        dz = jnp.zeros_like(z)
        dz = dz.at[:, 1:-1].set((z[:, 2:] - z[:, :-2]) / (2.0 * dy))
        dz = dz.at[:, 0].set((z[:, 1] - z[:, 0]) / dy)
        dz = dz.at[:, -1].set((z[:, -1] - z[:, -2]) / dy)
        return dz

    dv_power = jnp.sum(jnp.abs(dv) ** 2, axis=1) + 1e-30
    dp_power = jnp.sum(jnp.abs(dp) ** 2, axis=1) + 1e-30

    localization = jnp.sum(jnp.abs(dv) ** 2 * envelope[None, :], axis=1) / dv_power
    edge_power = jnp.sum(jnp.abs(dv[:, edge_mask]) ** 2, axis=1) / dv_power

    ddv_dy = ddy_centered(dv)
    roughness = (
        jnp.sqrt(jnp.sum(jnp.abs(ddv_dy) ** 2, axis=1) / dv_power)
        / jnp.maximum(kx, 1e-30)
    )

    dv_fft = jnp.fft.fft(dv, axis=1)
    dp_fft = jnp.fft.fft(dp, axis=1)

    high_ky_power = (
        jnp.sum(jnp.abs(dv_fft[:, high_ky_mask]) ** 2, axis=1)
        / (jnp.sum(jnp.abs(dv_fft) ** 2, axis=1) + 1e-30)
    )

    pressure_high_ky_power = (
        jnp.sum(jnp.abs(dp_fft[:, high_ky_mask]) ** 2, axis=1)
        / (jnp.sum(jnp.abs(dp_fft) ** 2, axis=1) + 1e-30)
    )

    finite = (
        jnp.isfinite(jnp.real(eigvals))
        & jnp.isfinite(jnp.imag(eigvals))
        & jnp.isfinite(localization)
        & jnp.isfinite(edge_power)
        & jnp.isfinite(roughness)
        & jnp.isfinite(high_ky_power)
        & jnp.isfinite(pressure_high_ky_power)
    )

    growth_ok = jnp.real(eigvals) > 0.0 if require_growth else jnp.ones_like(finite)

    is_kh_like = (
        finite
        & growth_ok
        & (localization > 0.20)
        & (edge_power < 0.12)
        & (roughness < 8.0)
        & (high_ky_power < 0.25)
        & (pressure_high_ky_power < 0.40)
    )

    strict_score = (
        jnp.real(eigvals)
        + 0.15 * localization
        - 2.0 * edge_power
        - 0.04 * jnp.maximum(roughness - 2.0, 0.0)
        - 0.75 * high_ky_power
        - 0.35 * pressure_high_ky_power
    )

    strict_score = jnp.where(is_kh_like, strict_score, -1e30)

    relaxed_score = jnp.where(
        finite & growth_ok,
        (
            jnp.real(eigvals)
            + 0.25 * localization
            - 1.5 * edge_power
            - 0.02 * jnp.maximum(roughness - 3.0, 0.0)
            - 0.25 * high_ky_power
            - 0.15 * pressure_high_ky_power
        ),
        -1e30,
    )

    use_strict = bool(jnp.any(is_kh_like))
    if not use_strict:
        print(
            "  Warning: no mode passed the strict KH filters. "
            "Relaxing filters and selecting the best localized mode."
        )

    score = strict_score if use_strict else relaxed_score

    best_idx = int(jnp.argmax(score))
    lam = eigvals[best_idx]
    qhat = modes[best_idx]

    # ------------------------------------------------------------------
    # Print candidates.
    # ------------------------------------------------------------------
    order = jnp.argsort(score)[::-1]

    print("  Top raw-AD eigenmode candidates:")
    for rank in range(min(print_candidates, eigvals.shape[0])):
        i = int(order[rank])
        marker = " <--- selected" if i == best_idx else ""
        print(
            f"    {rank:02d}: idx={i:04d}, "
            f"lambda={float(jnp.real(eigvals[i])):+.6e} "
            f"{float(jnp.imag(eigvals[i])):+.6e}i, "
            f"loc={float(localization[i]):.3f}, "
            f"edge={float(edge_power[i]):.3e}, "
            f"rough={float(roughness[i]):.3f}, "
            f"highky={float(high_ky_power[i]):.3e}, "
            f"p_highky={float(pressure_high_ky_power[i]):.3e}, "
            f"score={float(score[i]):+.6e}"
            f"{marker}"
        )

    # ------------------------------------------------------------------
    # Phase-align and scale eigenmode.
    # ------------------------------------------------------------------
    drho_b = qhat[rho_i]
    dmy_b = qhat[momy_i]
    dv_b = dmy_b / rho0 - v0 * drho_b / rho0

    anchor = int(jnp.argmax(jnp.abs(dv_b) * envelope))
    qhat = qhat * jnp.exp(-1j * jnp.angle(dv_b[anchor]))

    drho_b = qhat[rho_i]
    dmy_b = qhat[momy_i]
    dv_b = dmy_b / rho0 - v0 * drho_b / rho0

    phase = jnp.exp(1j * kx * X).astype(qhat.dtype)
    dv_real_space = jnp.real(dv_b[None, :] * phase)

    max_vy = jnp.max(jnp.abs(dv_real_space))

    if float(max_vy) <= 0.0 or not bool(jnp.isfinite(max_vy)):
        raise RuntimeError("Selected AD eigenmode has invalid transverse velocity amplitude.")

    eps = setup.perturbation_setup.amplitude / max_vy

    delta_cons = eps * jnp.real(qhat[:, None, :] * phase[None, :, :])
    cons_perturbed = cons0 + delta_cons.astype(cons0.dtype)

    primitive_state_perturbed = primitive_state_from_conserved(
        cons_perturbed,
        gamma,
        config,
        registered_variables,
    )

    min_rho = float(jnp.min(primitive_state_perturbed[rho_i]))
    min_p = float(jnp.min(primitive_state_perturbed[registered_variables.pressure_index]))

    if min_rho <= 0.0 or min_p <= 0.0:
        raise RuntimeError(
            f"Eigenmode amplitude too large: min rho={min_rho:.3e}, "
            f"min p={min_p:.3e}. Reduce perturbation_setup.amplitude."
        )

    # Residual diagnostic using the dense JAX Jacobian.
    q_flat = qhat.reshape(-1)
    Lq = J @ q_flat
    residual = jnp.linalg.norm(Lq - lam * q_flat) / (
        jnp.linalg.norm(Lq) + jnp.abs(lam) * jnp.linalg.norm(q_flat) + 1e-30
    )

    growth_time = jnp.where(jnp.real(lam) > 0.0, 1.0 / jnp.real(lam), jnp.inf)

    slab_density = setup.density_contrast * background_density
    delta = (slab_density + background_density) ** 2 / (
        slab_density * background_density
    )
    c_background = jnp.sqrt(gamma * pressure / background_density)
    v_shear = setup.mach_number * c_background
    t_kh = jnp.sqrt(delta) * setup.perturbation_setup.wavelength / v_shear

    print(
        "🎯 Raw-AD KH eigenmode selected: "
        f"lambda = {float(jnp.real(lam)):.6e} "
        f"+ {float(jnp.imag(lam)):.6e} i, "
        f"growth time = {float(growth_time):.6e}"
    )
    print(f"  Eigen residual: {float(residual):.3e}")
    print(
        "  Expected slope in ln(A/A0) vs t/t_KH: "
        f"{float(jnp.real(lam) * t_kh):.6e}"
    )

    return primitive_state_perturbed, lam

# -------------------------------------------------------------
# ================ ↑ Eigenmode initialization ↑ ===============
# -------------------------------------------------------------




# -------------------------------------------------------------
# ===================== ↓ Eigenmode test ↓ ====================
# -------------------------------------------------------------

def eigenvalue_plot():

    density_contrast = 10.0
    mach_number = 0.5
    diffusion = False
    kinematic_viscosity = 0.0

    c_background = float(jnp.sqrt(gamma * pressure / background_density))
    wavelength = box_size / 2
    amplitude = mach_number * c_background / 20
    smoothing_length = wavelength / 10
    slab_radius = 0.1

    # KHI timescale for reference
    slab_density = density_contrast * background_density
    Delta = float((slab_density + background_density)**2 / (slab_density * background_density))
    v_slab = mach_number * c_background
    t_kh = float(jnp.sqrt(Delta) * wavelength / v_slab)

    # Simulate up to 1.5 tau_KH to stay in the clean linear regime
    simulation_time = 1.5 * t_kh

    # either simulation or RHS jacobian
    simulation_jacobian = False
    # t end simulation jacobian
    t_end_jac = 0.1 * t_kh

    setup = KHISetup(
        num_cells=num_cells,
        simulation_time=simulation_time,
        perturbation_type=VELOCITY_PERTURBATION,
        perturbation_setup=PressurePerturbationSetup(
            amplitude=amplitude,
            wavelength=wavelength,
            gaussian_width=5 * smoothing_length,
        ),
        diffusion=diffusion,
        viscosity=kinematic_viscosity,
        mach_number=mach_number,
        density_contrast=density_contrast,
        slab_radius=slab_radius,
        smoothing_length=smoothing_length,
        smooth_profiles=False
    )

    # set up the unperturbed base state and config
    (
        khi_unperturbed,
        config,
        params,
        registered_variables,
        helper_data,
    ) = setup_khi_unperturbed_base(setup)

    setup = setup._replace(smooth_profiles=True)
    (
        khi_unperturbed_smoothed,
        _,
        _,
        _,
        _,
    ) = setup_khi_unperturbed_base(setup)

    # for testing set the whole state to be the upper
    # or lower state, assumes single interface mode
    upper_state = jnp.repeat(khi_unperturbed[:, :, -1][:, :, None], khi_unperturbed.shape[2], axis=2)
    lower_state = jnp.repeat(khi_unperturbed[:, :, 0][:, :, None], khi_unperturbed.shape[2], axis=2)
    # make plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    imaginary_range = (-650, 650)
    real_range = (-450, 50)

    def effective_ky_from_eigenvectors(eigenvectors, state):
        """
        eigenvectors[:, i] is one qhat_i(y).

        Returns k_y^eff / k_y^Nyquist for each eigenvector, estimated from the
        FFT power spectrum in y. This detects Nyquist/checkerboard modes correctly,
        unlike a centered-gradient estimate.
        """
        nvar, Nx, Ny = state.shape

        Y = helper_data.geometric_centers[:, :, 1]
        y = Y[0, :]
        dy = y[1] - y[0]

        modes = eigenvectors.T.reshape((-1, nvar, Ny))

        # Option A: use full conserved eigenvector.
        q_for_spectrum = modes

        # Option B, often cleaner for KH/acoustic interpretation:
        # use primitive transverse velocity only.
        # rho_i = registered_variables.density_index
        # momy_i = registered_variables.velocity_index.y
        # rho0 = state[rho_i, 0, :]
        # v0 = state[momy_i, 0, :]
        # drho = modes[:, rho_i, :]
        # dmy = modes[:, momy_i, :]
        # dvy = dmy / rho0[None, :] - v0[None, :] * drho / rho0[None, :]
        # q_for_spectrum = dvy[:, None, :]

        q_fft = jnp.fft.fft(q_for_spectrum, axis=-1)

        ky = 2.0 * jnp.pi * jnp.fft.fftfreq(Ny, d=dy)
        ky2 = ky**2

        power = jnp.sum(jnp.abs(q_fft) ** 2, axis=1)  # shape: (num_modes, Ny)

        ky_eff = jnp.sqrt(
            jnp.sum(power * ky2[None, :], axis=1)
            / (jnp.sum(power, axis=1) + 1e-30)
        )

        ky_nyquist = jnp.pi / dy

        return ky_eff / ky_nyquist

    last_scatter = None

    for i, label, state in zip(
        range(4),
        ["upper state", "lower state", "khi unperturbed", "khi unperturbed smoothed"],
        [upper_state, lower_state, khi_unperturbed, khi_unperturbed_smoothed]
    ):
        
        if not simulation_jacobian:
            J = single_xmode_rhs_jacobian2D(
                state,
                config,
                params,
                registered_variables,
                helper_data,
                wavelength=wavelength,
            )
        else:
            J = single_xmode_jacobian2Dt(
                state,
                config,
                params._replace(t_end=t_end_jac),
                registered_variables,
                helper_data,
                wavelength=wavelength,
            )

        eigenvalues, eigenvectors = jnp.linalg.eig(J)

        ky_eff_norm = effective_ky_from_eigenvectors(eigenvectors, state)

        ax = axs[i // 2, i % 2]

        last_scatter = ax.scatter(
            jnp.real(eigenvalues),
            jnp.imag(eigenvalues),
            c=ky_eff_norm,
            s=10,
            cmap="plasma",
            vmin=0.0,
            vmax=1.0,
            alpha=0.8,
            linewidths=0,
        )

        ax.set_title(label)
        ax.set_xlim(real_range)
        ax.set_ylim(imaginary_range)
        ax.set_xlabel(r"$\operatorname{Re}(\lambda)$")
        ax.set_ylabel(r"$\operatorname{Im}(\lambda)$")
        ax.grid(True)

    # find the khi-eigenmode, mark it in the ax[1, 1] plot
    # and make an inset plot to zoom in on it
    _, lambda_khi = find_numerical_eigenmode(
        khi_unperturbed_smoothed,
        config,
        params,
        registered_variables,
        helper_data,
        setup,
    )
    ax = axs[1, 1]
    # mark the khi mode with a star
    ax.scatter(
        jnp.real(lambda_khi),
        jnp.imag(lambda_khi),
        marker="*",
        s=100,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label=f"Selected KH eigenmode, λ={jnp.real(lambda_khi):.1f} + {jnp.imag(lambda_khi):.1f}i",
        zorder=10,
    )
    ax.legend(loc="upper left")

    # inset axes
    add_inset_box(ax, x1=-2.0, x2=2.0, y1=0.0, y2=3.0, loc='lower left', connect_loc1=2, connect_loc2=4)

    cbar = fig.colorbar(last_scatter, ax=axs.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label(r"$k_y^\mathrm{eff}/k_{y,\mathrm{Nyq}}$")

    fig.savefig("figures/eigenvalue_spectrum.png", dpi=600, bbox_inches="tight")
    print("Saved figures/eigenvalue_spectrum.png")


def eigenvalue_plot_simulation():

    density_contrast = 10.0
    mach_number = 0.5
    diffusion = False
    kinematic_viscosity = 0.0

    c_background = float(jnp.sqrt(gamma * pressure / background_density))
    wavelength = box_size / 2
    amplitude = mach_number * c_background / 20
    smoothing_length = wavelength / 10
    slab_radius = 0.1

    # KHI timescale for reference
    slab_density = density_contrast * background_density
    Delta = float((slab_density + background_density)**2 / (slab_density * background_density))
    v_slab = mach_number * c_background
    t_kh = float(jnp.sqrt(Delta) * wavelength / v_slab)

    # t end simulation jacobian
    t_end_jac = 0.1 * t_kh
    print(f"KH timescale: {t_kh:.3e}, t_end for jacobian: {t_end_jac:.3e}")

    setup = KHISetup(
        num_cells=num_cells,
        simulation_time=t_end_jac,
        perturbation_type=VELOCITY_PERTURBATION,
        perturbation_setup=PressurePerturbationSetup(
            amplitude=amplitude,
            wavelength=wavelength,
            gaussian_width=5 * smoothing_length,
        ),
        diffusion=diffusion,
        viscosity=kinematic_viscosity,
        mach_number=mach_number,
        density_contrast=density_contrast,
        slab_radius=slab_radius,
        smoothing_length=smoothing_length,
        smooth_profiles=False
    )

    # set up the unperturbed base state and config
    (
        khi_unperturbed,
        config,
        params,
        registered_variables,
        helper_data,
    ) = setup_khi_unperturbed_base(setup)

    setup = setup._replace(smooth_profiles=True)
    (
        khi_unperturbed_smoothed,
        _,
        _,
        _,
        _,
    ) = setup_khi_unperturbed_base(setup)


    # make plots
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # imaginary_range = (-650, 650)
    # real_range = (-450, 50)

    def effective_ky_from_eigenvectors(eigenvectors, state):
        """
        eigenvectors[:, i] is one qhat_i(y).

        Returns k_y^eff / k_y^Nyquist for each eigenvector, estimated from the
        FFT power spectrum in y. This detects Nyquist/checkerboard modes correctly,
        unlike a centered-gradient estimate.
        """
        nvar, Nx, Ny = state.shape

        Y = helper_data.geometric_centers[:, :, 1]
        y = Y[0, :]
        dy = y[1] - y[0]

        modes = eigenvectors.T.reshape((-1, nvar, Ny))

        # Option A: use full conserved eigenvector.
        q_for_spectrum = modes

        # Option B, often cleaner for KH/acoustic interpretation:
        # use primitive transverse velocity only.
        # rho_i = registered_variables.density_index
        # momy_i = registered_variables.velocity_index.y
        # rho0 = state[rho_i, 0, :]
        # v0 = state[momy_i, 0, :]
        # drho = modes[:, rho_i, :]
        # dmy = modes[:, momy_i, :]
        # dvy = dmy / rho0[None, :] - v0[None, :] * drho / rho0[None, :]
        # q_for_spectrum = dvy[:, None, :]

        q_fft = jnp.fft.fft(q_for_spectrum, axis=-1)

        ky = 2.0 * jnp.pi * jnp.fft.fftfreq(Ny, d=dy)
        ky2 = ky**2

        power = jnp.sum(jnp.abs(q_fft) ** 2, axis=1)  # shape: (num_modes, Ny)

        ky_eff = jnp.sqrt(
            jnp.sum(power * ky2[None, :], axis=1)
            / (jnp.sum(power, axis=1) + 1e-30)
        )

        ky_nyquist = jnp.pi / dy

        return ky_eff / ky_nyquist

    last_scatter = None

    state = khi_unperturbed_smoothed

    # J = single_xmode_jacobian2Dt(
    #     state,
    #     config,
    #     params._replace(t_end=t_end_jac),
    #     registered_variables,
    #     helper_data,
    #     wavelength=wavelength,
    # )

    # save J
    # jnp.save("simulation_jacobian.npy", J)

    J = jnp.load("simulation_jacobian.npy")

    eigenvalues, eigenvectors = jnp.linalg.eig(J)

    ky_eff_norm = effective_ky_from_eigenvectors(eigenvectors, state)

    last_scatter = ax.scatter(
        jnp.real(eigenvalues),
        jnp.imag(eigenvalues),
        c=ky_eff_norm,
        s=10,
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
        alpha=0.8,
        linewidths=0,
    )

    ax.set_xlabel(r"$\operatorname{Re}(\mu)$")
    ax.set_ylabel(r"$\operatorname{Im}(\mu)$")
    ax.grid(True)

    # rhs eigenmode, not simulation jacobian
    _, lambda_khi = find_numerical_eigenmode(
        khi_unperturbed_smoothed,
        config,
        params,
        registered_variables,
        helper_data,
        setup,
    )

    # Convert generator eigenvalue to finite-time amplification factor.
    mu_khi = jnp.exp(lambda_khi * t_end_jac)

    lambda_khi_re = float(jnp.real(lambda_khi))
    lambda_khi_im = float(jnp.imag(lambda_khi))
    mu_khi_re = float(jnp.real(mu_khi))
    mu_khi_im = float(jnp.imag(mu_khi))
    mu_khi_abs = float(jnp.abs(mu_khi))

    ax.scatter(
        mu_khi_re,
        mu_khi_im,
        marker="*",
        s=140,
        facecolors="none",
        edgecolors="black",
        linewidths=1.6,
        label=(
            rf"KH mode: $|\mu|={mu_khi_abs:.3f}$"
            "\n"
            rf"$\lambda={lambda_khi_re:.2f}{lambda_khi_im:+.2f}i$"
        ),
        zorder=10,
    )

    ax.set_ylim(-2.0, 2.0)
    ax.set_xlim(0.5, 5.0)

    # inset axes
    # add_inset_box(ax, x1=-2.0, x2=2.0, y1=0.0, y2=3.0, loc='lower left', connect_loc1=2, connect_loc2=4)

    cbar = fig.colorbar(last_scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$k_y^\mathrm{eff}/k_{y,\mathrm{Nyq}}$")

    fig.savefig("figures/eigenvalue_spectrum_t.png", dpi=600, bbox_inches="tight")
    print("Saved figures/eigenvalue_spectrum_t.png")

def eigenmode_growth_test():
    
    print("\n🚀 Starting Eigenmode Transient Test...")

    density_contrast = 10.0
    mach_number = 0.5
    diffusion = False
    kinematic_viscosity = 0.0

    c_background = float(jnp.sqrt(gamma * pressure / background_density))
    wavelength = box_size / 2
    amplitude = mach_number * c_background / 20
    smoothing_length = wavelength / 10
    slab_radius = 0.1

    # KHI timescale for reference
    slab_density = density_contrast * background_density
    Delta = float((slab_density + background_density)**2 / (slab_density * background_density))
    v_slab = mach_number * c_background
    t_kh = float(jnp.sqrt(Delta) * wavelength / v_slab)

    # Simulate up to 1.5 tau_KH to stay in the clean linear regime
    simulation_time = 1.5 * t_kh

    setup_vel = KHISetup(
        num_cells=num_cells,
        simulation_time=simulation_time,
        perturbation_type=VELOCITY_PERTURBATION,
        perturbation_setup=PressurePerturbationSetup(
            amplitude=amplitude,
            wavelength=wavelength,
            gaussian_width=5 * smoothing_length,
        ),
        diffusion=diffusion,
        viscosity=kinematic_viscosity,
        mach_number=mach_number,
        density_contrast=density_contrast,
        slab_radius=slab_radius,
        smoothing_length=smoothing_length,
    )

    setup_eig = setup_vel._replace(perturbation_type=EIGENMODE_PERTURBATION)

    print("\n--- Running Standard Velocity Initialization ---")
    result_vel, helper_data, reg_vars = simulate_khi(setup_vel, return_snapshots=True)

    print("\n--- Running Pure JAX Eigenmode Initialization ---")
    result_eig, _, _ = simulate_khi(setup_eig, return_snapshots=True)

    # get the eigenmode growth rate for reference
    # set up the unperturbed base state and config
    (
        khi_unperturbed,
        config,
        params,
        registered_variables,
        helper_data,
    ) = setup_khi_unperturbed_base(setup_eig)
    _, lambda_khi = find_numerical_eigenmode(
        khi_unperturbed,
        config,
        params,
        registered_variables,
        helper_data,
        setup_eig,
    )
    growth_rate_khi = float(jnp.real(lambda_khi))
    print(f"Reference KH eigenmode growth rate: {growth_rate_khi:.6e}")

    # --- Analysis: Measure Mode Amplitude ---
    # We project v_y onto the spatial Fourier basis to extract the envelope amplitude,
    # avoiding phase oscillations as the wave travels.
    X = helper_data.geometric_centers[:, :, 0]
    k = 2 * jnp.pi / wavelength
    sin_kx = jnp.sin(k * X)
    cos_kx = jnp.cos(k * X)

    def get_mode_amplitude(result_obj):
        amplitudes = []
        for state in result_obj.states:
            vy_field = state[reg_vars.velocity_index.y]
            
            # Project onto sin and cos to capture any phase
            proj_sin = jnp.mean(vy_field * sin_kx, axis=0) * 2  
            proj_cos = jnp.mean(vy_field * cos_kx, axis=0) * 2  
            
            # Magnitude of the Fourier coefficient envelope, take peak across y
            amp_envelope = jnp.sqrt(proj_sin**2 + proj_cos**2)
            amplitudes.append(float(jnp.max(amp_envelope)))
            
        return jnp.array(amplitudes), jnp.array(result_obj.time_points)

    amp_vel, t_vel = get_mode_amplitude(result_vel)
    amp_eig, t_eig = get_mode_amplitude(result_eig)

    # Normalize to t=0 amplitude and convert to log scale for exponential plot
    ln_vel = jnp.log(amp_vel / amp_vel[0])
    ln_eig = jnp.log(amp_eig / amp_eig[0])

    # --- Plot 1: Transient Analysis ---
    fig_line, ax_line = plt.subplots(figsize=(8, 6))
    ax_line.plot(t_vel / t_kh, ln_vel, label="ad-hoc velocity perturbation", linestyle="--", color="red", linewidth=2)
    ax_line.plot(t_eig / t_kh, ln_eig, label="selected eigenmode", linestyle="-", color="blue", linewidth=2)

    # Reference line for expected growth rate
    t_ref = jnp.linspace(0, simulation_time, 100)
    ax_line.plot(t_ref / t_kh, growth_rate_khi * t_ref, label=f"expected growth, Re[λ] = {growth_rate_khi:.2e}", linestyle=":", color="black", linewidth=1.5)
    ax_line.set_ylim(top = 2.0)

    ax_line.set_xlabel(r"Time ($t / \tau_{KH}$)", fontsize=14)
    ax_line.set_ylabel(r"$\ln(A(t) / A_0)$", fontsize=14)
    ax_line.set_title("KHI growth: elimination of initialization transients", fontsize=14, fontweight="bold")
    ax_line.legend(fontsize=12)
    ax_line.grid(True, alpha=0.3)
    fig_line.tight_layout()
    fig_line.savefig("figures/eigenmode_transient_test.png", dpi=150)
    print("Saved figures/eigenmode_transient_test.png")

    # --- Plot 2: Visual comparison of the final states ---
    fig_state, axs = plt.subplots(2, 3, figsize=(15, 9))
    extent = [0, box_size, 0, box_size]

    final_state_vel = result_vel.states[-1]
    final_state_eig = result_eig.states[-1]

    # Setup A (Velocity)
    im0 = axs[0, 0].imshow(final_state_vel[reg_vars.density_index].T, cmap='viridis', origin='lower', extent=extent, norm=LogNorm())
    im1 = axs[0, 1].imshow(final_state_vel[reg_vars.velocity_index.y].T, cmap='RdBu_r', origin='lower', extent=extent)
    im2 = axs[0, 2].imshow(final_state_vel[reg_vars.pressure_index].T, cmap='magma', origin='lower', extent=extent)

    # Setup B (Eigenmode)
    im3 = axs[1, 0].imshow(final_state_eig[reg_vars.density_index].T, cmap='viridis', origin='lower', extent=extent, norm=LogNorm())
    im4 = axs[1, 1].imshow(final_state_eig[reg_vars.velocity_index.y].T, cmap='RdBu_r', origin='lower', extent=extent)
    im5 = axs[1, 2].imshow(final_state_eig[reg_vars.pressure_index].T, cmap='magma', origin='lower', extent=extent)

    axs[0, 0].set_ylabel("ad-hoc velocity", fontsize=12, fontweight='bold')
    axs[1, 0].set_ylabel("eigenmode", fontsize=12, fontweight='bold')

    titles = ['Density (log)', r'Transverse Velocity ($v_y$)', 'Pressure']
    for col in range(3):
        axs[0, col].set_title(titles[col], fontsize=13)

    for i, ax in enumerate(axs.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        fig_state.colorbar([im0, im1, im2, im3, im4, im5][i], cax=cax)

    fig_state.suptitle(f"Final State Comparison at $t = {simulation_time:.2f}$", fontsize=16, fontweight='bold')
    fig_state.tight_layout()
    fig_state.savefig("figures/eigenmode_final_states.png", dpi=150)
    print("Saved figures/eigenmode_final_states.png")

# -------------------------------------------------------------
# ===================== ↑ Eigenmode test ↑ ====================
# -------------------------------------------------------------


if __name__ == "__main__":
    # eigenvalue_plot()
    eigenmode_growth_test()
    # eigenvalue_plot_simulation()