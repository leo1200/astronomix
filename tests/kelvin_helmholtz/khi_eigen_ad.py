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

# -------------------------------------------------------------
# ================ ↓ Eigenmode initialization ↓ ===============
# -------------------------------------------------------------

@partial(jax.jit, static_argnames=["config", "registered_variables"])
def astronomix_rhs_2D(conserved_state, params, config, registered_variables):
	"""
	Computes the right-hand side (RHS) of the hydro equations for a given stage.
	The `k2_coeff` scales the timestep `dt` for the current RK stage.
	"""

	# pad the state with ghost cells according to the boundary conditions
	conserved_state = _pad(conserved_state, config)

	# apply boundary conditions on the padded state
	conserved_state = _boundary_handler(
        conserved_state, config, registered_variables, params, CONSERVATIVE_GAS_STATE
    )

	grid_spacing = config.grid_spacing

	# Calculate fluxes based on the state of the current stage
	dF_x = _weno_flux_x(conserved_state, params, config, registered_variables)
	dF_y = _weno_flux_y(conserved_state, params, config, registered_variables)
	
	rhs_q = -1 / grid_spacing * (
		(dF_x - _shift(dF_x, 1, axis=1))
		+ (dF_y - _shift(dF_y, 1, axis=2))
	)

	# unpad the RHS to remove the ghost cells
	rhs_q = _unpad(rhs_q, config)

	return rhs_q

# def compute_numerical_eigenmode(
#     primitive_state_unperturbed,
#     config,
#     params,
#     registered_variables,
#     helper_data,
#     setup: KHISetup,
#     max_y_mode=None,
#     filter_alpha=8.0,
#     filter_order=8,
#     assembly_batch_size=8,
#     print_candidates=12,
# ):
#     """
#     Robust AD-based KH eigenmode extraction.

#     This relies on the actual simulator RHS through JAX JVPs, but avoids the
#     previous Arnoldi failure mode by constructing the reduced Fourier-block
#     tangent operator

#         L_k^sim = P_y P_k J_sim(U0) P_k P_y

#     where P_k projects onto the requested streamwise Fourier mode and P_y is a
#     smooth low-pass filter in the transverse direction.

#     The eigenproblem is solved in conserved variables. Modes are scored using
#     primitive perturbations reconstructed from the conserved eigenvectors.
#     """

#     print(
#         f"⏳ Extracting AD/simulator KH eigenmode at "
#         f"λ={setup.perturbation_setup.wavelength} ..."
#     )

#     eigen_config = finalize_config(config, primitive_state_unperturbed.shape)

#     gamma = params.gamma

#     rho_i = registered_variables.density_index
#     momx_i = registered_variables.velocity_index.x
#     momy_i = registered_variables.velocity_index.y
#     energy_i = registered_variables.pressure_index

#     cons0 = conserved_state_from_primitive(
#         primitive_state_unperturbed,
#         gamma,
#         eigen_config,
#         registered_variables,
#     )

#     nvar, Nx, Ny = cons0.shape
#     n = nvar * Ny

#     m_float = box_size / setup.perturbation_setup.wavelength
#     m = int(np.round(float(m_float)))

#     if not np.isclose(float(m_float), m, rtol=1e-12, atol=1e-12):
#         raise ValueError(
#             f"wavelength={setup.perturbation_setup.wavelength} is not commensurate "
#             f"with box_size={box_size}. Got box_size / wavelength = {m_float}."
#         )

#     kx = 2.0 * np.pi * m / box_size

#     X = helper_data.geometric_centers[:, :, 0]
#     Y = helper_data.geometric_centers[:, :, 1]

#     y_np = np.asarray(Y[0, :], dtype=np.float64)
#     dy = float(y_np[1] - y_np[0])

#     phase = jnp.exp(1j * kx * X).astype(jnp.complex64)
#     phase_conj = jnp.conj(phase)

#     rho0 = np.asarray(primitive_state_unperturbed[rho_i, 0, :], dtype=np.float64)
#     u0 = np.asarray(primitive_state_unperturbed[momx_i, 0, :], dtype=np.float64)
#     v0 = np.asarray(primitive_state_unperturbed[momy_i, 0, :], dtype=np.float64)

#     # Conservative default: keep the KH/interface scales, remove grid-scale modes.
#     # For Ny=300, smoothing_length=0.05, this gives about 40 modes.
#     if max_y_mode is None:
#         max_y_mode = int(
#             min(
#                 Ny // 2,
#                 max(
#                     24,
#                     np.ceil(2.0 * box_size / setup.smoothing_length),
#                 ),
#             )
#         )

#     max_y_mode = int(max_y_mode)
#     print(f"  Using transverse low-pass filter: max_y_mode={max_y_mode}")

#     ky_modes = jnp.round(jnp.fft.fftfreq(Ny, d=1.0 / Ny)).astype(jnp.float32)
#     eta = jnp.abs(ky_modes) / float(max_y_mode)

#     # Smooth spectral filter. Modes above max_y_mode are zeroed.
#     filter_y = jnp.where(
#         eta <= 1.0,
#         jnp.exp(-filter_alpha * eta**filter_order),
#         0.0,
#     ).astype(jnp.complex64)

#     def project_y_jax(qhat):
#         qfft = jnp.fft.fft(qhat, axis=-1)
#         qfft = qfft * filter_y[None, :]
#         return jnp.fft.ifft(qfft, axis=-1)

#     filter_y_np = np.asarray(filter_y, dtype=np.complex128)

#     def project_y_np(qhat):
#         qfft = np.fft.fft(qhat, axis=-1)
#         qfft = qfft * filter_y_np[None, :]
#         return np.fft.ifft(qfft, axis=-1)

#     @jax.jit
#     def linear_rhs_real(delta_cons_real):
#         _, Jv = jax.jvp(
#             lambda c: astronomix_rhs_2D(
#                 c,
#                 params,
#                 eigen_config,
#                 registered_variables,
#             ),
#             (cons0,),
#             (delta_cons_real,),
#         )
#         return Jv

#     @jax.jit
#     def apply_projected_Lk_one(qhat_flat):
#         """
#         qhat_flat: conserved Fourier amplitudes, flattened shape nvar * Ny.

#         Returns:
#             P_y P_k J_sim P_k P_y qhat.
#         """
#         qhat = qhat_flat.reshape((nvar, Ny))
#         qhat = project_y_jax(qhat)

#         delta_cons_complex = qhat[:, None, :] * phase[None, :, :]

#         J_delta_complex = (
#             linear_rhs_real(jnp.real(delta_cons_complex))
#             + 1j * linear_rhs_real(jnp.imag(delta_cons_complex))
#         )

#         J_hat = jnp.mean(
#             J_delta_complex * phase_conj[None, :, :],
#             axis=1,
#         )

#         J_hat = project_y_jax(J_hat)

#         return J_hat.reshape(-1)

#     apply_projected_Lk_batch = jax.jit(jax.vmap(apply_projected_Lk_one))

#     # ------------------------------------------------------------------
#     # Assemble dense reduced AD Jacobian.
#     # ------------------------------------------------------------------
#     print(f"  Assembling dense reduced AD Jacobian of size {n} x {n} ...")

#     A = np.zeros((n, n), dtype=np.complex128)

#     eye = np.eye(n, dtype=np.complex128)

#     for start in range(0, n, assembly_batch_size):
#         stop = min(start + assembly_batch_size, n)

#         basis_batch = eye[start:stop]
#         basis_batch_jax = jnp.asarray(basis_batch, dtype=jnp.complex64)

#         out_batch = apply_projected_Lk_batch(basis_batch_jax)
#         out_batch_np = np.asarray(out_batch, dtype=np.complex128)

#         A[:, start:stop] = out_batch_np.T

#         if start == 0 or stop == n or (start // assembly_batch_size) % 25 == 0:
#             print(f"    assembled columns {stop}/{n}")

#     # ------------------------------------------------------------------
#     # Dense eigensolve.
#     # ------------------------------------------------------------------
#     print("  Solving dense reduced AD eigenproblem ...")
#     eigvals, eigvecs = la.eig(A)

#     # ------------------------------------------------------------------
#     # Mode scoring in primitive variables.
#     # ------------------------------------------------------------------
#     if interface_mode == SLAB:
#         envelope = (
#             np.exp(-kx * np.abs(y_np - (y_center - setup.slab_radius)))
#             + np.exp(-kx * np.abs(y_np - (y_center + setup.slab_radius)))
#         )
#     else:
#         envelope = np.exp(-kx * np.abs(y_np - y_center))

#     envelope = envelope / np.max(envelope)

#     y_min = float(np.min(y_np))
#     y_max = float(np.max(y_np))
#     dist_to_edge = np.minimum(y_np - y_min, y_max - y_np)

#     edge_width = 0.12 * box_size
#     edge_mask = dist_to_edge < edge_width

#     ky_modes_np = np.round(np.fft.fftfreq(Ny, d=1.0 / Ny)).astype(int)
#     high_ky_mask = np.abs(ky_modes_np) > max_y_mode // 2

#     def conserved_to_primitive_hat(qhat):
#         drho = qhat[rho_i]
#         dmx = qhat[momx_i]
#         dmy = qhat[momy_i]
#         dE = qhat[energy_i]

#         du = dmx / rho0 - u0 * drho / rho0
#         dv = dmy / rho0 - v0 * drho / rho0

#         dp = (gamma - 1.0) * (
#             dE
#             - 0.5 * drho * (u0**2 + v0**2)
#             - rho0 * (u0 * du + v0 * dv)
#         )

#         return drho, du, dv, dp

#     candidates = []

#     for i in range(len(eigvals)):
#         lam_i = eigvals[i]
#         qhat_i = eigvecs[:, i].reshape((nvar, Ny))
#         qhat_i = project_y_np(qhat_i)

#         drho, du, dv, dp = conserved_to_primitive_hat(qhat_i)

#         dv_power = np.sum(np.abs(dv) ** 2) + 1e-300
#         dp_power = np.sum(np.abs(dp) ** 2) + 1e-300

#         localization = float(np.sum(np.abs(dv) ** 2 * envelope) / dv_power)
#         edge_power = float(np.sum(np.abs(dv[edge_mask]) ** 2) / dv_power)

#         ddv_dy = np.gradient(dv, dy, edge_order=1)
#         roughness = float(
#             np.sqrt(np.sum(np.abs(ddv_dy) ** 2) / dv_power) / max(kx, 1e-300)
#         )

#         dv_fft = np.fft.fft(dv)
#         high_ky_power = float(
#             np.sum(np.abs(dv_fft[high_ky_mask]) ** 2)
#             / (np.sum(np.abs(dv_fft) ** 2) + 1e-300)
#         )

#         dp_fft = np.fft.fft(dp)
#         pressure_high_ky_power = float(
#             np.sum(np.abs(dp_fft[high_ky_mask]) ** 2)
#             / (np.sum(np.abs(dp_fft) ** 2) + 1e-300)
#         )

#         finite = (
#             np.isfinite(lam_i.real)
#             and np.isfinite(lam_i.imag)
#             and np.isfinite(localization)
#             and np.isfinite(edge_power)
#             and np.isfinite(roughness)
#             and np.isfinite(high_ky_power)
#             and np.isfinite(pressure_high_ky_power)
#         )

#         is_growing = lam_i.real > 0.0

#         # These filters reject exactly the kind of mode that produced the
#         # fine pressure ripples and immediate amplitude collapse.
#         is_kh_like = (
#             finite
#             and is_growing
#             and localization > 0.20
#             and edge_power < 0.12
#             and roughness < 8.0
#             and high_ky_power < 0.15
#             and pressure_high_ky_power < 0.25
#         )

#         score = (
#             lam_i.real
#             + 0.10 * localization
#             - 2.0 * edge_power
#             - 0.05 * max(roughness - 2.0, 0.0)
#             - 1.0 * high_ky_power
#             - 0.5 * pressure_high_ky_power
#         )

#         if not is_kh_like:
#             score -= 1e6

#         candidates.append(
#             {
#                 "index": i,
#                 "lambda": lam_i,
#                 "score": score,
#                 "localization": localization,
#                 "edge_power": edge_power,
#                 "roughness": roughness,
#                 "high_ky_power": high_ky_power,
#                 "pressure_high_ky_power": pressure_high_ky_power,
#                 "is_kh_like": is_kh_like,
#             }
#         )

#     if not any(c["is_kh_like"] for c in candidates):
#         print(
#             "  Warning: no AD eigenmode passed the strict KH filters. "
#             "Relaxing filters and selecting the best smooth localized growing mode."
#         )

#         for c in candidates:
#             lam_i = c["lambda"]

#             if lam_i.real > 0.0 and np.isfinite(lam_i.real):
#                 c["score"] = (
#                     lam_i.real
#                     + 0.20 * c["localization"]
#                     - 1.5 * c["edge_power"]
#                     - 0.03 * max(c["roughness"] - 3.0, 0.0)
#                     - 0.5 * c["high_ky_power"]
#                     - 0.25 * c["pressure_high_ky_power"]
#                 )
#             else:
#                 c["score"] = -1e6

#     best = max(candidates, key=lambda c: c["score"])
#     best_idx = best["index"]

#     lam = eigvals[best_idx]
#     qhat = eigvecs[:, best_idx].reshape((nvar, Ny))
#     qhat = project_y_np(qhat)

#     print("  Candidate AD eigenvalues:")
#     ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)

#     for rank, c in enumerate(ordered[:print_candidates]):
#         marker = " <--- selected" if c["index"] == best_idx else ""
#         lam_i = c["lambda"]

#         print(
#             f"    {rank:02d}: idx={c['index']:04d}, "
#             f"lambda={lam_i.real:+.6e} {lam_i.imag:+.6e}i, "
#             f"loc={c['localization']:.3f}, "
#             f"edge={c['edge_power']:.3e}, "
#             f"rough={c['roughness']:.3f}, "
#             f"highky={c['high_ky_power']:.3e}, "
#             f"p_highky={c['pressure_high_ky_power']:.3e}, "
#             f"score={c['score']:+.6e}"
#             f"{marker}"
#         )

#     # ------------------------------------------------------------------
#     # Phase-align and scale perturbation.
#     # ------------------------------------------------------------------
#     drho, du, dv, dp = conserved_to_primitive_hat(qhat)

#     anchor = int(np.argmax(np.abs(dv) * envelope))
#     phase_align = np.exp(-1j * np.angle(dv[anchor]))
#     qhat = qhat * phase_align

#     drho, du, dv, dp = conserved_to_primitive_hat(qhat)

#     phase_np = np.asarray(phase, dtype=np.complex128)
#     dv_real_space = np.real(dv[None, :] * phase_np)

#     max_vy = np.max(np.abs(dv_real_space))

#     if max_vy <= 0.0 or not np.isfinite(max_vy):
#         raise RuntimeError("Selected AD eigenmode has invalid transverse velocity amplitude.")

#     eps = setup.perturbation_setup.amplitude / max_vy

#     delta_cons = eps * np.real(qhat[:, None, :] * phase_np[None, :, :])

#     cons_perturbed = cons0 + jnp.asarray(delta_cons, dtype=cons0.dtype)

#     primitive_state_perturbed = primitive_state_from_conserved(
#         cons_perturbed,
#         gamma,
#         eigen_config,
#         registered_variables,
#     )

#     min_rho = float(jnp.min(primitive_state_perturbed[rho_i]))
#     min_p = float(jnp.min(primitive_state_perturbed[registered_variables.pressure_index]))

#     if min_rho <= 0.0 or min_p <= 0.0:
#         raise RuntimeError(
#             f"AD eigenmode amplitude too large: min rho={min_rho:.3e}, "
#             f"min p={min_p:.3e}. Reduce perturbation_setup.amplitude."
#         )

#     growth_time = 1.0 / lam.real if lam.real > 0.0 else np.inf

#     slab_density = setup.density_contrast * background_density
#     delta = (slab_density + background_density) ** 2 / (
#         slab_density * background_density
#     )
#     c_background = float(np.sqrt(gamma * pressure / background_density))
#     v_shear = setup.mach_number * c_background
#     t_kh_tirso = float(
#         np.sqrt(delta) * setup.perturbation_setup.wavelength / v_shear
#     )

#     print(
#         "🎯 AD/simulator KH eigenmode selected: "
#         f"lambda = {lam.real:.6e} + {lam.imag:.6e} i, "
#         f"growth time = {growth_time:.6e}"
#     )
#     print(
#         f"  Expected slope in ln(A/A0) vs t/t_KH: "
#         f"lambda.real * t_KH = {lam.real * t_kh_tirso:.6e}"
#     )

#     return primitive_state_perturbed, lam

def compute_numerical_eigenmode(
    primitive_state_unperturbed,
    config,
    params,
    registered_variables,
    helper_data,
    setup: KHISetup,
    assembly_batch_size=8,
    require_growth=True,
    print_candidates=12,
):
    """
    Simplest robust AD-based KH eigenmode initializer.

    This computes the raw simulator tangent operator in a single +kx Fourier block:

        delta U(x, y, t) = U_hat(y) exp(i kx x) exp(lambda t)

    using JAX JVPs through the Astronomix RHS. It then dense-solves the reduced
    eigenproblem and selects the smooth, interface-localized KH-like mode.

    No hand-coded linearized Euler equations.
    No transverse low-pass filtering.
    No Arnoldi.
    """

    print(
        f"⏳ Extracting raw-AD KH eigenmode at "
        f"λx={setup.perturbation_setup.wavelength} ..."
    )

    eigen_config = finalize_config(config, primitive_state_unperturbed.shape)

    gamma = params.gamma

    rho_i = registered_variables.density_index
    momx_i = registered_variables.velocity_index.x
    momy_i = registered_variables.velocity_index.y
    energy_i = registered_variables.pressure_index

    cons0 = conserved_state_from_primitive(
        primitive_state_unperturbed,
        gamma,
        eigen_config,
        registered_variables,
    )

    nvar, Nx, Ny = cons0.shape
    n = nvar * Ny

    # Streamwise Fourier mode.
    m_float = box_size / setup.perturbation_setup.wavelength
    m = int(np.round(float(m_float)))

    if not np.isclose(float(m_float), m, rtol=1e-12, atol=1e-12):
        raise ValueError(
            f"wavelength={setup.perturbation_setup.wavelength} is not commensurate "
            f"with box_size={box_size}. Got box_size / wavelength = {m_float}."
        )

    kx = 2.0 * np.pi * m / box_size

    X = helper_data.geometric_centers[:, :, 0]
    Y = helper_data.geometric_centers[:, :, 1]

    y = np.asarray(Y[0, :], dtype=np.float64)
    dy = float(y[1] - y[0])

    phase = jnp.exp(1j * kx * X).astype(jnp.complex64)
    phase_conj = jnp.conj(phase)

    rho0 = np.asarray(primitive_state_unperturbed[rho_i, 0, :], dtype=np.float64)
    u0 = np.asarray(primitive_state_unperturbed[momx_i, 0, :], dtype=np.float64)
    v0 = np.asarray(primitive_state_unperturbed[momy_i, 0, :], dtype=np.float64)

    # ------------------------------------------------------------------
    # Simulator tangent action via AD.
    # ------------------------------------------------------------------
    @jax.jit
    def linear_rhs_real(delta_cons_real):
        _, Jv = jax.jvp(
            lambda c: astronomix_rhs_2D(
                c,
                params,
                eigen_config,
                registered_variables,
            ),
            (cons0,),
            (delta_cons_real,),
        )
        return Jv

    @jax.jit
    def apply_ad_Lk_one(qhat_flat):
        """
        Apply raw AD Fourier-block operator.

        qhat_flat: flattened conserved Fourier amplitude, shape nvar * Ny.
        """
        qhat = qhat_flat.reshape((nvar, Ny))

        # Reconstruct full 2D complex perturbation:
        # delta q(x, y) = qhat(y) exp(i kx x)
        delta_cons_complex = qhat[:, None, :] * phase[None, :, :]

        # Complex JVP from two real JVPs.
        J_delta_complex = (
            linear_rhs_real(jnp.real(delta_cons_complex))
            + 1j * linear_rhs_real(jnp.imag(delta_cons_complex))
        )

        # Project back onto +kx.
        J_hat = jnp.mean(
            J_delta_complex * phase_conj[None, :, :],
            axis=1,
        )

        return J_hat.reshape(-1)

    apply_ad_Lk_batch = jax.jit(jax.vmap(apply_ad_Lk_one))

    # ------------------------------------------------------------------
    # Assemble dense reduced AD Jacobian.
    # ------------------------------------------------------------------
    print(f"  Assembling raw AD Fourier-block Jacobian: {n} x {n}")

    A = np.zeros((n, n), dtype=np.complex128)
    eye = np.eye(n, dtype=np.complex128)

    for start in range(0, n, assembly_batch_size):
        stop = min(start + assembly_batch_size, n)

        basis_batch = eye[start:stop]
        basis_batch_jax = jnp.asarray(basis_batch, dtype=jnp.complex64)

        out_batch = apply_ad_Lk_batch(basis_batch_jax)
        out_batch_np = np.asarray(out_batch, dtype=np.complex128)

        # Each batch item is A @ e_j, so these are columns.
        A[:, start:stop] = out_batch_np.T

        if start == 0 or stop == n or (start // assembly_batch_size) % 25 == 0:
            print(f"    assembled columns {stop}/{n}")

    # ------------------------------------------------------------------
    # Dense eigensolve.
    # ------------------------------------------------------------------
    print("  Solving dense raw-AD eigenproblem ...")
    eigvals, eigvecs = la.eig(A)

    # ------------------------------------------------------------------
    # Convert conserved Fourier amplitudes to primitive perturbations.
    # ------------------------------------------------------------------
    def conserved_to_primitive_hat(qhat):
        drho = qhat[rho_i]
        dmx = qhat[momx_i]
        dmy = qhat[momy_i]
        dE = qhat[energy_i]

        du = dmx / rho0 - u0 * drho / rho0
        dv = dmy / rho0 - v0 * drho / rho0

        dp = (gamma - 1.0) * (
            dE
            - 0.5 * drho * (u0**2 + v0**2)
            - rho0 * (u0 * du + v0 * dv)
        )

        return drho, du, dv, dp

    # ------------------------------------------------------------------
    # Physical KH-mode selection.
    # ------------------------------------------------------------------
    if interface_mode == SLAB:
        envelope = (
            np.exp(-kx * np.abs(y - (y_center - setup.slab_radius)))
            + np.exp(-kx * np.abs(y - (y_center + setup.slab_radius)))
        )
    else:
        envelope = np.exp(-kx * np.abs(y - y_center))

    envelope = envelope / np.max(envelope)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    dist_to_edge = np.minimum(y - y_min, y_max - y)
    edge_width = 0.12 * box_size
    edge_mask = dist_to_edge < edge_width

    ky_modes_np = np.round(np.fft.fftfreq(Ny, d=1.0 / Ny)).astype(int)

    # Used only as a diagnostic/penalty, not as a hard projection.
    diagnostic_max_y_mode = int(
        min(
            Ny // 2,
            max(24, np.ceil(2.0 * box_size / setup.smoothing_length)),
        )
    )
    high_ky_mask = np.abs(ky_modes_np) > diagnostic_max_y_mode // 2

    candidates = []

    for i in range(len(eigvals)):
        lam_i = eigvals[i]
        qhat_i = eigvecs[:, i].reshape((nvar, Ny))

        drho, du, dv, dp = conserved_to_primitive_hat(qhat_i)

        dv_power = np.sum(np.abs(dv) ** 2) + 1e-300
        dp_power = np.sum(np.abs(dp) ** 2) + 1e-300

        localization = float(np.sum(np.abs(dv) ** 2 * envelope) / dv_power)
        edge_power = float(np.sum(np.abs(dv[edge_mask]) ** 2) / dv_power)

        ddv_dy = np.gradient(dv, dy, edge_order=1)
        roughness = float(
            np.sqrt(np.sum(np.abs(ddv_dy) ** 2) / dv_power) / max(kx, 1e-300)
        )

        dv_fft = np.fft.fft(dv)
        high_ky_power = float(
            np.sum(np.abs(dv_fft[high_ky_mask]) ** 2)
            / (np.sum(np.abs(dv_fft) ** 2) + 1e-300)
        )

        dp_fft = np.fft.fft(dp)
        pressure_high_ky_power = float(
            np.sum(np.abs(dp_fft[high_ky_mask]) ** 2)
            / (np.sum(np.abs(dp_fft) ** 2) + 1e-300)
        )

        finite = (
            np.isfinite(lam_i.real)
            and np.isfinite(lam_i.imag)
            and np.isfinite(localization)
            and np.isfinite(edge_power)
            and np.isfinite(roughness)
            and np.isfinite(high_ky_power)
            and np.isfinite(pressure_high_ky_power)
        )

        growth_ok = lam_i.real > 0.0 if require_growth else True

        is_kh_like = (
            finite
            and growth_ok
            and localization > 0.20
            and edge_power < 0.12
            and roughness < 8.0
            and high_ky_power < 0.25
            and pressure_high_ky_power < 0.40
        )

        score = (
            lam_i.real
            + 0.15 * localization
            - 2.0 * edge_power
            - 0.04 * max(roughness - 2.0, 0.0)
            - 0.75 * high_ky_power
            - 0.35 * pressure_high_ky_power
        )

        if not is_kh_like:
            score -= 1e6

        candidates.append(
            {
                "index": i,
                "lambda": lam_i,
                "score": score,
                "localization": localization,
                "edge_power": edge_power,
                "roughness": roughness,
                "high_ky_power": high_ky_power,
                "pressure_high_ky_power": pressure_high_ky_power,
                "is_kh_like": is_kh_like,
            }
        )

    if not any(c["is_kh_like"] for c in candidates):
        print(
            "  Warning: no mode passed the strict KH filters. "
            "Relaxing filters and selecting the best localized mode."
        )

        for c in candidates:
            lam_i = c["lambda"]
            growth_ok = lam_i.real > 0.0 if require_growth else True

            if growth_ok and np.isfinite(lam_i.real):
                c["score"] = (
                    lam_i.real
                    + 0.25 * c["localization"]
                    - 1.5 * c["edge_power"]
                    - 0.02 * max(c["roughness"] - 3.0, 0.0)
                    - 0.25 * c["high_ky_power"]
                    - 0.15 * c["pressure_high_ky_power"]
                )
            else:
                c["score"] = -1e6

    ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
    best = ordered[0]

    best_idx = best["index"]
    lam = eigvals[best_idx]
    qhat = eigvecs[:, best_idx].reshape((nvar, Ny))

    print("  Top raw-AD eigenmode candidates:")
    for rank, c in enumerate(ordered[:print_candidates]):
        marker = " <--- selected" if c["index"] == best_idx else ""
        lam_i = c["lambda"]

        print(
            f"    {rank:02d}: idx={c['index']:04d}, "
            f"lambda={lam_i.real:+.6e} {lam_i.imag:+.6e}i, "
            f"loc={c['localization']:.3f}, "
            f"edge={c['edge_power']:.3e}, "
            f"rough={c['roughness']:.3f}, "
            f"highky={c['high_ky_power']:.3e}, "
            f"p_highky={c['pressure_high_ky_power']:.3e}, "
            f"score={c['score']:+.6e}"
            f"{marker}"
        )

    # ------------------------------------------------------------------
    # Phase-align and scale eigenmode.
    # ------------------------------------------------------------------
    drho, du, dv, dp = conserved_to_primitive_hat(qhat)

    anchor = int(np.argmax(np.abs(dv) * envelope))
    phase_align = np.exp(-1j * np.angle(dv[anchor]))

    qhat = qhat * phase_align
    drho, du, dv, dp = conserved_to_primitive_hat(qhat)

    phase_np = np.asarray(phase, dtype=np.complex128)

    dv_real_space = np.real(dv[None, :] * phase_np)
    max_vy = np.max(np.abs(dv_real_space))

    if max_vy <= 0.0 or not np.isfinite(max_vy):
        raise RuntimeError("Selected AD eigenmode has invalid transverse velocity amplitude.")

    eps = setup.perturbation_setup.amplitude / max_vy

    delta_cons = eps * np.real(qhat[:, None, :] * phase_np[None, :, :])

    cons_perturbed = cons0 + jnp.asarray(delta_cons, dtype=cons0.dtype)

    primitive_state_perturbed = primitive_state_from_conserved(
        cons_perturbed,
        gamma,
        eigen_config,
        registered_variables,
    )

    min_rho = float(jnp.min(primitive_state_perturbed[rho_i]))
    min_p = float(jnp.min(primitive_state_perturbed[registered_variables.pressure_index]))

    if min_rho <= 0.0 or min_p <= 0.0:
        raise RuntimeError(
            f"Eigenmode amplitude too large: min rho={min_rho:.3e}, "
            f"min p={min_p:.3e}. Reduce perturbation_setup.amplitude."
        )

    # Optional residual diagnostic.
    qhat_flat = jnp.asarray(qhat.reshape(-1), dtype=jnp.complex64)
    Lq = np.asarray(apply_ad_Lk_one(qhat_flat), dtype=np.complex128)
    q_np = qhat.reshape(-1)

    residual = np.linalg.norm(Lq - lam * q_np) / (
        np.linalg.norm(Lq) + abs(lam) * np.linalg.norm(q_np) + 1e-300
    )

    growth_time = 1.0 / lam.real if lam.real > 0.0 else np.inf

    slab_density = setup.density_contrast * background_density
    delta = (slab_density + background_density) ** 2 / (
        slab_density * background_density
    )
    c_background = float(np.sqrt(gamma * pressure / background_density))
    v_shear = setup.mach_number * c_background
    t_kh = float(
        np.sqrt(delta) * setup.perturbation_setup.wavelength / v_shear
    )

    print(
        "🎯 Raw-AD KH eigenmode selected: "
        f"lambda = {lam.real:.6e} + {lam.imag:.6e} i, "
        f"growth time = {growth_time:.6e}"
    )
    print(f"  Eigen residual: {residual:.3e}")
    print(f"  Expected slope in ln(A/A0) vs t/t_KH: {lam.real * t_kh:.6e}")

    return primitive_state_perturbed, lam

# def compute_numerical_eigenmode(
#     primitive_state_unperturbed,
#     config,
#     params,
#     registered_variables,
#     helper_data,
#     setup: KHISetup,
#     k_eigs=4,
#     tol=1e-8,
#     maxiter=2000,
# ):
#     print(f"⏳ Extracting rightmost AD eigenmode at λx={setup.perturbation_setup.wavelength} ...")

#     config = finalize_config(config, primitive_state_unperturbed.shape)
#     gamma = params.gamma

#     rho_i = registered_variables.density_index
#     momx_i = registered_variables.velocity_index.x
#     momy_i = registered_variables.velocity_index.y
#     energy_i = registered_variables.pressure_index  # conserved energy lives at pressure slot

#     cons0 = conserved_state_from_primitive(
#         primitive_state_unperturbed, gamma, config, registered_variables
#     )

#     nvar, Nx, Ny = cons0.shape
#     n = nvar * Ny

#     m = int(round(float(box_size / setup.perturbation_setup.wavelength)))
#     kx = 2.0 * np.pi * m / box_size

#     X = helper_data.geometric_centers[:, :, 0]
#     Y = helper_data.geometric_centers[:, :, 1]
#     phase = jnp.exp(1j * kx * X).astype(jnp.complex64)
#     phase_conj = jnp.conj(phase)

#     rho0 = np.asarray(primitive_state_unperturbed[rho_i, 0, :])
#     u0 = np.asarray(primitive_state_unperturbed[momx_i, 0, :])
#     v0 = np.asarray(primitive_state_unperturbed[momy_i, 0, :])
#     y = np.asarray(Y[0, :])

#     @jax.jit
#     def J_real(dq):
#         _, Jdq = jax.jvp(
#             lambda q: astronomix_rhs_2D(q, params, config, registered_variables),
#             (cons0,),
#             (dq,),
#         )
#         return Jdq

#     @jax.jit
#     def Lk(qhat_flat):
#         qhat = qhat_flat.reshape((nvar, Ny))
#         dq = qhat[:, None, :] * phase[None, :, :]

#         Jdq = J_real(jnp.real(dq)) + 1j * J_real(jnp.imag(dq))
#         Jhat = jnp.mean(Jdq * phase_conj[None, :, :], axis=1)

#         return Jhat.reshape(-1)

#     def matvec(v):
#         return np.asarray(Lk(jnp.asarray(v, dtype=jnp.complex64)), dtype=np.complex128)

#     A = LinearOperator((n, n), matvec=matvec, dtype=np.complex128)

#     # Bias ARPACK toward the KH mode with an interface-localized transverse momentum seed.
#     if interface_mode == SLAB:
#         envelope = (
#             np.exp(-kx * np.abs(y - (y_center - setup.slab_radius)))
#             + np.exp(-kx * np.abs(y - (y_center + setup.slab_radius)))
#         )
#     else:
#         envelope = np.exp(-kx * np.abs(y - y_center))

#     v0_hat = np.zeros((nvar, Ny), dtype=np.complex128)
#     v0_hat[momy_i] = rho0 * envelope

#     eigvals, eigvecs = eigs(
#         A,
#         k=k_eigs,
#         which="LR",
#         v0=v0_hat.reshape(-1),
#         tol=tol,
#         maxiter=maxiter,
#     )

#     best = int(np.argmax(eigvals.real))
#     lam = eigvals[best]
#     qhat = eigvecs[:, best].reshape((nvar, Ny))

#     def conserved_to_dv(q):
#         drho = q[rho_i]
#         dmy = q[momy_i]
#         return dmy / rho0 - v0 * drho / rho0

#     dv = conserved_to_dv(qhat)

#     anchor = int(np.argmax(np.abs(dv) * envelope))
#     qhat *= np.exp(-1j * np.angle(dv[anchor]))
#     dv = conserved_to_dv(qhat)

#     phase_np = np.asarray(phase, dtype=np.complex128)
#     max_vy = np.max(np.abs(np.real(dv[None, :] * phase_np)))

#     eps = setup.perturbation_setup.amplitude / (max_vy + 1e-300)

#     cons_perturbed = cons0 + jnp.asarray(
#         eps * np.real(qhat[:, None, :] * phase_np[None, :, :]),
#         dtype=cons0.dtype,
#     )

#     primitive_state_perturbed = primitive_state_from_conserved(
#         cons_perturbed, gamma, config, registered_variables
#     )

#     print(
#         f"🎯 Rightmost AD eigenmode: "
#         f"lambda = {lam.real:.6e} + {lam.imag:.6e} i"
#     )

#     return primitive_state_perturbed, lam

def plot_khi_spectra_comparison(
    primitive_state_unperturbed,
    config,
    params,
    registered_variables,
    helper_data,
    setup: KHISetup,
    filename="figures/khi_spectra_comparison_two_panel.png",
    assembly_batch_size=8,
    require_growth=True,
    euler_sponge=True,
    print_candidates=8,
    xlim=None,
    ylim=None,
    zoom_xlim=None,
    zoom_ylim=None,
    show_insets=True,
):
    """
    Two-panel KHI spectrum comparison:

        1. hand-linearized primitive Euler operator,
        2. raw AD/simulator tangent operator in the +kx Fourier block.

    The AD panel uses JAX JVPs through astronomix_rhs_2D and does not use
    transverse low-pass filtering.
    """

    print("📊 Building two-panel KHI spectrum comparison figure ...")

    eigen_config = finalize_config(config, primitive_state_unperturbed.shape)

    gamma = params.gamma

    rho_i = registered_variables.density_index
    momx_i = registered_variables.velocity_index.x
    momy_i = registered_variables.velocity_index.y
    energy_i = registered_variables.pressure_index

    cons0 = conserved_state_from_primitive(
        primitive_state_unperturbed,
        gamma,
        eigen_config,
        registered_variables,
    )

    nvar, Nx, Ny = cons0.shape
    n_ad = nvar * Ny

    # Streamwise Fourier mode.
    m_float = box_size / setup.perturbation_setup.wavelength
    m = int(np.round(float(m_float)))

    if not np.isclose(float(m_float), m, rtol=1e-12, atol=1e-12):
        raise ValueError(
            f"wavelength={setup.perturbation_setup.wavelength} is not commensurate "
            f"with box_size={box_size}. Got box_size / wavelength = {m_float}."
        )

    kx = 2.0 * np.pi * m / box_size

    X = helper_data.geometric_centers[:, :, 0]
    Y = helper_data.geometric_centers[:, :, 1]

    y = np.asarray(Y[0, :], dtype=np.float64)
    dy = float(y[1] - y[0])

    rho0 = np.asarray(primitive_state_unperturbed[rho_i, 0, :], dtype=np.float64)
    u0 = np.asarray(primitive_state_unperturbed[momx_i, 0, :], dtype=np.float64)
    v0 = np.asarray(primitive_state_unperturbed[momy_i, 0, :], dtype=np.float64)
    p0 = np.asarray(
        primitive_state_unperturbed[registered_variables.pressure_index, 0, :],
        dtype=np.float64,
    )

    phase = jnp.exp(1j * kx * X).astype(jnp.complex64)
    phase_conj = jnp.conj(phase)

    # ------------------------------------------------------------------
    # Linearized primitive Euler reference operator.
    # q = [rho1, u1, v1, p1].
    # ------------------------------------------------------------------
    def first_derivative_matrix(n, dy):
        lower = -0.5 * np.ones(n - 1) / dy
        upper = 0.5 * np.ones(n - 1) / dy

        D = sp.diags(
            diagonals=[lower, upper],
            offsets=[-1, 1],
            shape=(n, n),
            dtype=np.complex128,
            format="lil",
        )

        # One-sided boundary derivative closure.
        D[0, 0] = -1.0 / dy
        D[0, 1] = 1.0 / dy

        D[n - 1, n - 2] = -1.0 / dy
        D[n - 1, n - 1] = 1.0 / dy

        return D.tocsr()

    def build_linearized_euler_matrix():
        D = first_derivative_matrix(Ny, dy)

        drho0_dy = np.asarray(D @ rho0, dtype=np.complex128)
        du0_dy = np.asarray(D @ u0, dtype=np.complex128)
        dp0_dy = np.asarray(D @ p0, dtype=np.complex128)

        I = sp.identity(Ny, dtype=np.complex128, format="csr")
        Z = sp.csr_matrix((Ny, Ny), dtype=np.complex128)

        R = sp.diags(rho0, 0, dtype=np.complex128, format="csr")
        invR = sp.diags(1.0 / rho0, 0, dtype=np.complex128, format="csr")
        U = sp.diags(u0, 0, dtype=np.complex128, format="csr")
        P0 = sp.diags(p0, 0, dtype=np.complex128, format="csr")

        R_y = sp.diags(drho0_dy, 0, dtype=np.complex128, format="csr")
        U_y = sp.diags(du0_dy, 0, dtype=np.complex128, format="csr")
        P_y_diag = sp.diags(dp0_dy, 0, dtype=np.complex128, format="csr")

        if euler_sponge:
            y_min = float(np.min(y))
            y_max = float(np.max(y))
            dist_to_edge = np.minimum(y - y_min, y_max - y)

            sponge_width = 0.15 * box_size
            ramp = np.clip((sponge_width - dist_to_edge) / sponge_width, 0.0, 1.0)

            cmax = float(np.max(np.sqrt(gamma * p0 / rho0)))
            umax = float(np.max(np.abs(u0)))
            sponge_strength = 2.0 * (cmax + umax) * kx

            sponge = sponge_strength * ramp**2
            S = sp.diags(sponge, 0, dtype=np.complex128, format="csr")
        else:
            S = Z

        adv = -1j * kx * U - S

        # Linearized compressible primitive Euler:
        #
        # rho_t = -u0 i k rho - rho0 i k u - rho0_y v - rho0 v_y
        # u_t   = -u0 i k u   - u0_y v   - i k p / rho0
        # v_t   = -u0 i k v               - p_y / rho0
        # p_t   = -u0 i k p   - p0_y v    - gamma p0 (i k u + v_y)
        #
        # Includes p0_y correction in y-momentum, although p0 is constant here.

        A_rho_rho = adv
        A_rho_u = -1j * kx * R
        A_rho_v = -(R_y + R @ D)
        A_rho_p = Z

        A_u_rho = Z
        A_u_u = adv
        A_u_v = -U_y
        A_u_p = -1j * kx * invR

        A_v_rho = sp.diags(dp0_dy / rho0**2, 0, dtype=np.complex128, format="csr")
        A_v_u = Z
        A_v_v = adv
        A_v_p = -(invR @ D)

        A_p_rho = Z
        A_p_u = -gamma * 1j * kx * P0
        A_p_v = -(P_y_diag + gamma * P0 @ D)
        A_p_p = adv

        A = sp.bmat(
            [
                [A_rho_rho, A_rho_u, A_rho_v, A_rho_p],
                [A_u_rho,   A_u_u,   A_u_v,   A_u_p],
                [A_v_rho,   A_v_u,   A_v_v,   A_v_p],
                [A_p_rho,   A_p_u,   A_p_v,   A_p_p],
            ],
            format="csr",
            dtype=np.complex128,
        )

        return A.toarray()

    print("  Building linearized Euler matrix ...")
    A_euler = build_linearized_euler_matrix()

    print("  Solving Euler eigenproblem ...")
    eigvals_euler, eigvecs_euler = la.eig(A_euler)

    # ------------------------------------------------------------------
    # Raw AD/simulator Fourier-block tangent operator.
    # ------------------------------------------------------------------
    @jax.jit
    def linear_rhs_real(delta_cons_real):
        _, Jv = jax.jvp(
            lambda c: astronomix_rhs_2D(
                c,
                params,
                eigen_config,
                registered_variables,
            ),
            (cons0,),
            (delta_cons_real,),
        )
        return Jv

    @jax.jit
    def apply_ad_Lk_one(qhat_flat):
        qhat = qhat_flat.reshape((nvar, Ny))

        delta_cons_complex = qhat[:, None, :] * phase[None, :, :]

        J_delta_complex = (
            linear_rhs_real(jnp.real(delta_cons_complex))
            + 1j * linear_rhs_real(jnp.imag(delta_cons_complex))
        )

        J_hat = jnp.mean(
            J_delta_complex * phase_conj[None, :, :],
            axis=1,
        )

        return J_hat.reshape(-1)

    apply_ad_Lk_batch = jax.jit(jax.vmap(apply_ad_Lk_one))

    print(f"  Assembling raw AD reduced Jacobian of size {n_ad} x {n_ad} ...")

    A_ad_raw = np.zeros((n_ad, n_ad), dtype=np.complex128)
    eye_ad = np.eye(n_ad, dtype=np.complex128)

    for start in range(0, n_ad, assembly_batch_size):
        stop = min(start + assembly_batch_size, n_ad)

        basis_batch = eye_ad[start:stop]
        basis_batch_jax = jnp.asarray(basis_batch, dtype=jnp.complex64)

        out_batch = apply_ad_Lk_batch(basis_batch_jax)
        out_batch_np = np.asarray(out_batch, dtype=np.complex128)

        A_ad_raw[:, start:stop] = out_batch_np.T

        if start == 0 or stop == n_ad or (start // assembly_batch_size) % 25 == 0:
            print(f"    assembled columns {stop}/{n_ad}")

    print("  Solving raw AD eigenproblem ...")
    eigvals_ad_raw, eigvecs_ad_raw = la.eig(A_ad_raw)

    # ------------------------------------------------------------------
    # Mode diagnostics and KH-like selection.
    # ------------------------------------------------------------------
    if interface_mode == SLAB:
        envelope = (
            np.exp(-kx * np.abs(y - (y_center - setup.slab_radius)))
            + np.exp(-kx * np.abs(y - (y_center + setup.slab_radius)))
        )
    else:
        envelope = np.exp(-kx * np.abs(y - y_center))

    envelope = envelope / np.max(envelope)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    dist_to_edge = np.minimum(y - y_min, y_max - y)

    edge_width = 0.12 * box_size
    edge_mask = dist_to_edge < edge_width

    diagnostic_max_y_mode = int(
        min(
            Ny // 2,
            max(24, np.ceil(2.0 * box_size / setup.smoothing_length)),
        )
    )

    ky_modes_np = np.round(np.fft.fftfreq(Ny, d=1.0 / Ny)).astype(int)
    high_ky_mask = np.abs(ky_modes_np) > diagnostic_max_y_mode // 2

    def conserved_to_primitive_hat(qhat):
        drho = qhat[rho_i]
        dmx = qhat[momx_i]
        dmy = qhat[momy_i]
        dE = qhat[energy_i]

        du = dmx / rho0 - u0 * drho / rho0
        dv = dmy / rho0 - v0 * drho / rho0

        dp = (gamma - 1.0) * (
            dE
            - 0.5 * drho * (u0**2 + v0**2)
            - rho0 * (u0 * du + v0 * dv)
        )

        return drho, du, dv, dp

    def primitive_components_from_vector(vec, kind):
        if kind == "euler":
            q = vec.reshape((4, Ny))
            return q[0], q[1], q[2], q[3]

        if kind == "ad":
            q = vec.reshape((nvar, Ny))
            return conserved_to_primitive_hat(q)

        raise ValueError(f"Unknown kind={kind}")

    def score_spectrum(eigvals, eigvecs, kind):
        candidates = []

        for i in range(len(eigvals)):
            lam_i = eigvals[i]
            vec_i = eigvecs[:, i]

            drho, du, dv, dp = primitive_components_from_vector(vec_i, kind)

            dv_power = np.sum(np.abs(dv) ** 2) + 1e-300

            localization = float(np.sum(np.abs(dv) ** 2 * envelope) / dv_power)
            edge_power = float(np.sum(np.abs(dv[edge_mask]) ** 2) / dv_power)

            ddv_dy = np.gradient(dv, dy, edge_order=1)
            roughness = float(
                np.sqrt(np.sum(np.abs(ddv_dy) ** 2) / dv_power) / max(kx, 1e-300)
            )

            dv_fft = np.fft.fft(dv)
            high_ky_power = float(
                np.sum(np.abs(dv_fft[high_ky_mask]) ** 2)
                / (np.sum(np.abs(dv_fft) ** 2) + 1e-300)
            )

            dp_fft = np.fft.fft(dp)
            pressure_high_ky_power = float(
                np.sum(np.abs(dp_fft[high_ky_mask]) ** 2)
                / (np.sum(np.abs(dp_fft) ** 2) + 1e-300)
            )

            finite = (
                np.isfinite(lam_i.real)
                and np.isfinite(lam_i.imag)
                and np.isfinite(localization)
                and np.isfinite(edge_power)
                and np.isfinite(roughness)
                and np.isfinite(high_ky_power)
                and np.isfinite(pressure_high_ky_power)
            )

            growth_ok = lam_i.real > 0.0 if require_growth else True

            is_kh_like = (
                finite
                and growth_ok
                and localization > 0.20
                and edge_power < 0.12
                and roughness < 8.0
                and high_ky_power < 0.25
                and pressure_high_ky_power < 0.40
            )

            score = (
                lam_i.real
                + 0.15 * localization
                - 2.0 * edge_power
                - 0.04 * max(roughness - 2.0, 0.0)
                - 0.75 * high_ky_power
                - 0.35 * pressure_high_ky_power
            )

            if not is_kh_like:
                score -= 1e6

            candidates.append(
                {
                    "index": i,
                    "lambda": lam_i,
                    "score": score,
                    "localization": localization,
                    "edge_power": edge_power,
                    "roughness": roughness,
                    "high_ky_power": high_ky_power,
                    "pressure_high_ky_power": pressure_high_ky_power,
                    "is_kh_like": is_kh_like,
                }
            )

        if not any(c["is_kh_like"] for c in candidates):
            print(f"  Warning: no strict KH-like mode found for {kind}; relaxing filters.")

            for c in candidates:
                lam_i = c["lambda"]
                growth_ok = lam_i.real > 0.0 if require_growth else True

                if growth_ok and np.isfinite(lam_i.real):
                    c["score"] = (
                        lam_i.real
                        + 0.25 * c["localization"]
                        - 1.5 * c["edge_power"]
                        - 0.02 * max(c["roughness"] - 3.0, 0.0)
                        - 0.25 * c["high_ky_power"]
                        - 0.15 * c["pressure_high_ky_power"]
                    )
                else:
                    c["score"] = -1e6

        best = max(candidates, key=lambda c: c["score"])
        ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)

        return best, ordered, candidates

    best_euler, ordered_euler, cand_euler = score_spectrum(
        eigvals_euler,
        eigvecs_euler,
        kind="euler",
    )

    best_ad_raw, ordered_ad_raw, cand_ad_raw = score_spectrum(
        eigvals_ad_raw,
        eigvecs_ad_raw,
        kind="ad",
    )

    def print_top(label, ordered):
        print(f"\n  Top candidates: {label}")
        for rank, c in enumerate(ordered[:print_candidates]):
            lam_i = c["lambda"]
            print(
                f"    {rank:02d}: idx={c['index']:04d}, "
                f"lambda={lam_i.real:+.6e} {lam_i.imag:+.6e}i, "
                f"loc={c['localization']:.3f}, "
                f"edge={c['edge_power']:.3e}, "
                f"rough={c['roughness']:.3f}, "
                f"highky={c['high_ky_power']:.3e}, "
                f"p_highky={c['pressure_high_ky_power']:.3e}, "
                f"score={c['score']:+.6e}"
            )

    print_top("Euler", ordered_euler)
    print_top("raw AD simulator", ordered_ad_raw)

    lam_euler = best_euler["lambda"]
    lam_ad = best_ad_raw["lambda"]
    rel_diff = abs(lam_ad - lam_euler) / (abs(lam_euler) + 1e-300)

    print(
        "\n  Selected eigenvalue comparison:"
        f"\n    Euler: {lam_euler.real:+.8e} {lam_euler.imag:+.8e}i"
        f"\n    AD:    {lam_ad.real:+.8e} {lam_ad.imag:+.8e}i"
        f"\n    relative difference: {rel_diff:.3e}"
    )

    # ------------------------------------------------------------------
    # Axis limits.
    # ------------------------------------------------------------------
    if xlim is None or ylim is None:
        all_vals = np.concatenate([eigvals_euler, eigvals_ad_raw])
        finite = np.isfinite(all_vals.real) & np.isfinite(all_vals.imag)
        all_vals = all_vals[finite]

        re = all_vals.real
        im = all_vals.imag

        re_lo, re_hi = np.percentile(re, [1.0, 99.0])
        im_lo, im_hi = np.percentile(im, [1.0, 99.0])

        selected_vals = np.array([lam_euler, lam_ad], dtype=np.complex128)

        re_lo = min(re_lo, np.min(selected_vals.real))
        re_hi = max(re_hi, np.max(selected_vals.real))
        im_lo = min(im_lo, np.min(selected_vals.imag))
        im_hi = max(im_hi, np.max(selected_vals.imag))

        re_pad = 0.08 * max(re_hi - re_lo, 1e-12)
        im_pad = 0.08 * max(im_hi - im_lo, 1e-12)

        if xlim is None:
            xlim = (re_lo - re_pad, re_hi + re_pad)

        if ylim is None:
            ylim = (im_lo - im_pad, im_hi + im_pad)

    if zoom_xlim is None or zoom_ylim is None:
        selected_vals = np.array([lam_euler, lam_ad], dtype=np.complex128)

        zoom_center_re = float(np.mean(selected_vals.real))
        zoom_center_im = float(np.mean(selected_vals.imag))

        if zoom_xlim is None:
            zoom_xlim = (zoom_center_re - 2.0, zoom_center_re + 2.0)

        if zoom_ylim is None:
            zoom_ylim = (zoom_center_im - 8.0, zoom_center_im + 8.0)

    # ------------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11.8, 5.0), constrained_layout=False)

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.045],
        wspace=0.08,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
    cax = fig.add_subplot(gs[0, 2])

    axes = [ax0, ax1]

    norm_loc = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap_loc = plt.cm.viridis

    def draw_spectrum(
        ax,
        eigvals,
        candidates,
        best,
        *,
        title=None,
        annotate=True,
        add_labels=False,
        marker_scale=1.0,
    ):
        finite = np.isfinite(eigvals.real) & np.isfinite(eigvals.imag)
        vals = eigvals[finite]

        ax.scatter(
            vals.real,
            vals.imag,
            s=8.0 * marker_scale,
            c="0.75",
            alpha=0.45,
            linewidths=0,
            label="all modes" if add_labels else None,
            rasterized=True,
        )

        good_candidates = [c for c in candidates if c["is_kh_like"]]

        if len(good_candidates) > 0:
            good_vals = np.array(
                [eigvals[c["index"]] for c in good_candidates],
                dtype=np.complex128,
            )
            good_loc = np.array(
                [c["localization"] for c in good_candidates],
                dtype=np.float64,
            )

            ax.scatter(
                good_vals.real,
                good_vals.imag,
                s=24.0 * marker_scale,
                c=good_loc,
                cmap=cmap_loc,
                norm=norm_loc,
                alpha=0.95,
                linewidths=0.25,
                edgecolors="k",
                label="KH-like candidates" if add_labels else None,
                rasterized=True,
            )

        lam_best = best["lambda"]

        ax.scatter(
            [lam_best.real],
            [lam_best.imag],
            marker="*",
            s=260.0 * marker_scale,
            c="crimson",
            edgecolors="k",
            linewidths=0.8,
            zorder=10,
            label="selected mode" if add_labels else None,
        )

        ax.axvline(0.0, color="k", linewidth=0.8, alpha=0.35)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.20)

        if title is not None:
            ax.set_title(title, fontsize=12)

        if annotate:
            annotation = (
                rf"$\lambda_\star$"
                "\n"
                rf"$= {lam_best.real:.3e} {lam_best.imag:+.3e}\,\mathrm{{i}}$"
            )

            ax.text(
                0.03,
                0.97,
                annotation,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.82,
                    linewidth=0.6,
                    edgecolor="0.4",
                ),
            )

        ax.grid(True, alpha=0.25)

    panels = [
        (axes[0], eigvals_euler, cand_euler, best_euler, "Linearized Euler"),
        (axes[1], eigvals_ad_raw, cand_ad_raw, best_ad_raw, "AD simulator Jacobian"),
    ]

    for i, (ax, eigvals, candidates, best, title) in enumerate(panels):
        draw_spectrum(
            ax,
            eigvals,
            candidates,
            best,
            title=title,
            annotate=True,
            add_labels=(i == 1),
            marker_scale=1.0,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\operatorname{Re}(\lambda)$")

        if show_insets:
            axins = inset_axes(
                ax,
                width="43%",
                height="43%",
                loc="lower left",
                borderpad=1.2,
            )

            draw_spectrum(
                axins,
                eigvals,
                candidates,
                best,
                title=None,
                annotate=False,
                add_labels=False,
                marker_scale=0.45,
            )

            axins.set_xlim(*zoom_xlim)
            axins.set_ylim(*zoom_ylim)

            axins.tick_params(
                axis="both",
                which="both",
                labelsize=7,
                length=2,
                pad=1,
            )

            axins.grid(True, alpha=0.25)

            mark_inset(
                ax,
                axins,
                loc1=2,
                loc2=4,
                fc="none",
                ec="0.25",
                lw=0.8,
                alpha=0.8,
            )

    axes[0].set_ylabel(r"$\operatorname{Im}(\lambda)$")
    plt.setp(axes[1].get_yticklabels(), visible=False)

    sm = mpl.cm.ScalarMappable(norm=norm_loc, cmap=cmap_loc)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("interface localization", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    handles, labels = axes[1].get_legend_handles_labels()
    if len(handles) > 0:
        axes[1].legend(
            handles,
            labels,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )

    fig.suptitle(
        rf"Spectra at $k_x = {kx:.3f}$, "
        rf"$\lambda_x = {setup.perturbation_setup.wavelength:.3f}$",
        fontsize=13,
        y=0.98,
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.955,
        bottom=0.16,
        top=0.84,
    )

    fig.savefig(filename, dpi=220, bbox_inches="tight")
    print(f"Saved {filename}")

    return {
        "fig": fig,
        "axes": axes,
        "kx": kx,
        "zoom_xlim": zoom_xlim,
        "zoom_ylim": zoom_ylim,
        "relative_eigenvalue_difference": rel_diff,
        "euler": {
            "matrix": A_euler,
            "eigvals": eigvals_euler,
            "eigvecs": eigvecs_euler,
            "best": best_euler,
            "candidates": cand_euler,
        },
        "ad_raw": {
            "matrix": A_ad_raw,
            "eigvals": eigvals_ad_raw,
            "eigvecs": eigvecs_ad_raw,
            "best": best_ad_raw,
            "candidates": cand_ad_raw,
        },
    }

# -------------------------------------------------------------
# ================ ↑ Eigenmode initialization ↑ ===============
# -------------------------------------------------------------


def simulate_khi(setup: KHISetup, return_snapshots = False, plot_spectra_comparison = False):

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

	smooth_profiles = True

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
			raise NotImplementedError("Single interface without smoothing not implemented yet.")

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

	# eigen-spectrum comparison
	if plot_spectra_comparison:
		plot_data = plot_khi_spectra_comparison(
			primitive_state_unperturbed=primitive_state_unperturbed,
			config=config,
			params=params,
			registered_variables=registered_variables,
			helper_data=helper_data,
			setup=setup,
			filename="figures/khi_spectra_comparison.png",
		)

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
		primitive_state, dom_eval = compute_numerical_eigenmode(
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

def example_setup_run(
	density_contrast,
	Re_or_nu,
	mach_number,
	adapt_simulation_time = False,
	return_snapshots = True,
	nu_specified = False
):

	simulation_time = 2.0
	slab_radius = 0.1

	if nu_specified:
		kinematic_viscosity = Re_or_nu
		diffusion = True if kinematic_viscosity > 0 else False
	else:
		diffusion = True if Re_or_nu != float("inf") else False
		reynolds_number = Re_or_nu

	slab_density = density_contrast * background_density
	c_background = jnp.sqrt(gamma * pressure / background_density)
	c_slab = float(jnp.sqrt(gamma * pressure / slab_density))

	M_crit = (1 + density_contrast**(-1/3))**(3/2)
	v_slab = mach_number * c_background

	Delta = (slab_density + background_density)**2 / (slab_density * background_density)
	Re_crit = 880 / Delta
	Re = Re_or_nu

	# in the Tirso paper
	if paper_mode == TIRSO:
		perturbation_type = VELOCITY_PERTURBATION
		wavelength = box_size / 2
		amplitude = mach_number * c_background / 20
		smoothing_length = wavelength / 10

	if paper_mode == ROEDIGER:
		perturbation_type = VELOCITY_PERTURBATION
		wavelength = box_size / 4
		amplitude = 0.1 * v_slab
		smoothing_length = wavelength / 102

	if not nu_specified:
		kinematic_viscosity = wavelength * v_slab / Re

	if adapt_simulation_time:
		# KHI growth time from Eq. 2 in Roediger et al 2013
		# t_kh = jnp.sqrt(Delta) / (2 * jnp.pi) * wavelength / v_slab
		# Tirso paper
		if paper_mode == TIRSO:
			t_kh = jnp.sqrt(Delta) * wavelength / v_slab
		if paper_mode == ROEDIGER:
			t_kh = jnp.sqrt(Delta) / (2 * jnp.pi) * wavelength / v_slab
		
		print(f"Kelvin-Helmholtz time (inviscid): {t_kh:.3f}")

		if paper_mode == ROEDIGER:
			# e.g. 20 * t_kh
			simulation_time = 20.0 * t_kh

		if paper_mode == TIRSO:
			# Tirso
			# simulation_time = 2.0 * t_kh
			simulation_time = 1.0 * t_kh

		print(f"Adapting simulation time to {simulation_time:.3f} to capture KHI growth.")

	setup = KHISetup(
		num_cells = num_cells,
		simulation_time = simulation_time,
		perturbation_type = perturbation_type,
		perturbation_setup = PressurePerturbationSetup(
			amplitude = amplitude,
			wavelength = wavelength,
			gaussian_width = 5 * smoothing_length,
		),
		diffusion = diffusion,
		viscosity = kinematic_viscosity,
		mach_number = mach_number,
		density_contrast = density_contrast,
		slab_radius = slab_radius,
		smoothing_length = smoothing_length,
	)

	result, helper_data, registered_variables = simulate_khi(setup, return_snapshots = return_snapshots)

	return result, registered_variables, helper_data, Re_crit, M_crit

def side_by_side_comparison():

	# very close
	# density_contrast_A = 10.0
	# reynolds_number_A = 100.0 # float("inf")
	# mach_number_A = 0.5

	# density_contrast_B = 10.0
	# reynolds_number_B = 300.0
	# mach_number_B = 0.5

	density_contrast_A = 10.0
	reynolds_number_A = float("inf")
	mach_number_A = 0.5

	density_contrast_B = 10.0
	reynolds_number_B = 600.0
	mach_number_B = 0.5

	print(f"👨‍🔧 Running setup A: χ={density_contrast_A}, Re={reynolds_number_A}, M={mach_number_A}")
	result_A, registered_variables_A, helper_data_A, Re_crit_A, M_crit_A = example_setup_run(
		density_contrast = density_contrast_A,
		Re_or_nu = reynolds_number_A,
		mach_number = mach_number_A,
		adapt_simulation_time = True,
	)

	print(f"👨‍🔧 Running setup B: χ={density_contrast_B}, Re={reynolds_number_B}, M={mach_number_B}")
	result_B, registered_variables_B, helper_data_B, Re_crit_B, M_crit_B = example_setup_run(
		density_contrast = density_contrast_B,
		Re_or_nu = reynolds_number_B,
		mach_number = mach_number_B,
		adapt_simulation_time = True,
	)

	fig, axs = plt.subplots(2, 3, figsize=(15, 10))
	extent = [0, box_size, 0, box_size]

	num_snapshots = len(result_A.states)

	# Initialize images for animation
	im_A_rho = axs[0, 0].imshow(result_A.states[0][registered_variables_A.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
	im_A_vy = axs[0, 1].imshow(result_A.states[0][registered_variables_A.velocity_index.y].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	im_A_P = axs[0, 2].imshow(result_A.states[0][registered_variables_A.pressure_index].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)

	im_B_rho = axs[1, 0].imshow(result_B.states[0][registered_variables_B.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
	im_B_vy = axs[1, 1].imshow(result_B.states[0][registered_variables_B.velocity_index.y].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	im_B_P = axs[1, 2].imshow(result_B.states[0][registered_variables_B.pressure_index].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)

	# Set labels
	re_A_str = "∞" if reynolds_number_A == float("inf") else f"{reynolds_number_A:.0f}"
	setup_A_str = f"χ={density_contrast_A:.0f}, Re={re_A_str}, M={mach_number_A:.2f}"

	re_B_str = "∞" if reynolds_number_B == float("inf") else f"{reynolds_number_B:.0f}"
	setup_B_str = f"χ={density_contrast_B:.0f}, Re={re_B_str}, M={mach_number_B:.2f}"

	axs[0, 0].set_ylabel(setup_A_str, fontsize=11, fontweight='bold')
	axs[1, 0].set_ylabel(setup_B_str, fontsize=11, fontweight='bold')

	axs[0, 0].set_title('Density')
	axs[0, 1].set_title('Transverse Velocity ($v_y$)')
	axs[0, 2].set_title('Pressure')

	# Set equal aspect ratio for all axes
	for ax in axs.flat:
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])

	for ax, im in [
		(axs[0, 0], im_A_rho),
		(axs[0, 1], im_A_vy),
		(axs[0, 2], im_A_P),
		(axs[1, 0], im_B_rho),
		(axs[1, 1], im_B_vy),
		(axs[1, 2], im_B_P),
	]:
		cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
		fig.colorbar(im, cax=cax)


	suptitle_text = fig.suptitle("", fontsize=16, fontweight='bold')

	def animate(frame):
		time = result_A.time_points[frame]
		suptitle_text.set_text(
			rf"KHI, $t$ = {time:.3f}, $M_\text{{crit}} \approx$ {M_crit_A:.2f}, $Re_\text{{crit}} \approx$ {Re_crit_A:.0f}"
		)
		
		im_A_rho.set_data(result_A.states[frame][registered_variables_A.density_index].T)
		im_A_vy.set_data(result_A.states[frame][registered_variables_A.velocity_index.y].T)
		im_A_P.set_data(result_A.states[frame][registered_variables_A.pressure_index].T)
		
		im_B_rho.set_data(result_B.states[frame][registered_variables_B.density_index].T)
		im_B_vy.set_data(result_B.states[frame][registered_variables_B.velocity_index.y].T)
		im_B_P.set_data(result_B.states[frame][registered_variables_B.pressure_index].T)
		
		return [suptitle_text, im_A_rho, im_A_vy, im_A_P, im_B_rho, im_B_vy, im_B_P]

	print("🎬 Creating animation...")
	anim = animation.FuncAnimation(fig, animate, frames=num_snapshots, interval=50, blit=False, repeat=True)
	plt.tight_layout()

	anim.save('figures/khi_comparison.gif', writer='pillow')

	# also save the final state as a static figure
	fig_final, axs_final = plt.subplots(2, 3, figsize=(15, 10))
	extent = [0, box_size, 0, box_size]
	im_A_rho_final = axs_final[0, 0].imshow(result_A.states[-1][registered_variables_A.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
	im_A_vy_final = axs_final[0, 1].imshow(result_A.states[-1][registered_variables_A.velocity_index.y].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	im_A_P_final = axs_final[0, 2].imshow(result_A.states[-1][registered_variables_A.pressure_index].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	im_B_rho_final = axs_final[1, 0].imshow(result_B.states[-1][registered_variables_B.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
	im_B_vy_final = axs_final[1, 1].imshow(result_B.states[-1][registered_variables_B.velocity_index.y].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	im_B_P_final = axs_final[1, 2].imshow(result_B.states[-1][registered_variables_B.pressure_index].T, cmap='RdBu_r', aspect='auto', origin='lower', extent=extent)
	axs_final[0, 0].set_ylabel(setup_A_str, fontsize=11, fontweight='bold')
	axs_final[1, 0].set_ylabel(setup_B_str, fontsize=11, fontweight='bold')
	axs_final[0, 0].set_title('Density')
	axs_final[0, 1].set_title('Transverse Velocity ($v_y$)')
	axs_final[0, 2].set_title('Pressure')
	for ax in axs_final.flat:
		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])
	for ax, im in [
		(axs_final[0, 0], im_A_rho_final),
		(axs_final[0, 1], im_A_vy_final),
		(axs_final[0, 2], im_A_P_final),
		(axs_final[1, 0], im_B_rho_final),
		(axs_final[1, 1], im_B_vy_final),
		(axs_final[1, 2], im_B_P_final),
	]:
		cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
		fig_final.colorbar(im, cax=cax)
	plt.tight_layout()
	fig_final.suptitle(f"KHI final state, $t$ = {result_A.time_points[-1]:.3f}", fontsize=16, fontweight='bold')
	plt.savefig('figures/khi_comparison_final.png')

# def parameter_sweep():

# 	"""
# 	We are interested in the following sweeps:
	
# 	- at a fixed contrast of χ = 10, vary the Reynolds number and Mach number across the critical values (should include 1.0)
# 	- at a fixed sub-critical Mach number, vary the density contrast and Reynolds number across the critical values
# 	- at a fixed non-critical Reynolds number, vary the density contrast and Mach number across the critical values
# 	"""

# 	pass

def khi_growth_over_time():
	"""
	Reproduces Fig. 3 from https://arxiv.org/pdf/2504.15345.

	Plots ln(v_y / v_y0) vs t / τ_KH for χ = 10 and varying Mach numbers
	at infinite Reynolds number (inviscid). Solid lines indicate unstable
	KHI (M < M_crit), dash-dotted lines indicate suppressed instability
	(M > M_crit).
	"""

	density_contrast = 10.0
	# reynolds_number = float("inf")
	# reynolds_number = 4000
	nu = 0.0138 * 0.3

	# Mach numbers from 0.1 to 1.8 in steps of 0.1 (matching the paper's sweep)
	mach_numbers = jnp.arange(0.1, 1.8, 0.1)
	# mach_numbers = [0.5, 2.5]

	# Critical Mach number for χ = 10 (Eq. 27 in Mandelker et al. 2016)
	# M_crit = (1 + density_contrast**(-1/3))**(3/2)
	M_crit = 1.65
	print(f"Critical Mach number for χ={density_contrast}: {M_crit:.3f}")

	# Precompute shared quantities for τ_KH calculation
	slab_density = density_contrast * background_density
	c_background = float(jnp.sqrt(gamma * pressure / background_density))
	Delta = float((slab_density + background_density)**2 / (slab_density * background_density))

	# The perturbation wavelength used in example_setup_run
	# wavelength = box_size / 5
	# Tirso paper
	wavelength = box_size / 2

	# Colormap setup: rainbow from blue (low M) to red (high M)
	cmap = plt.cm.jet
	norm = mcolors.Normalize(vmin=0.0, vmax=1.8)

	fig, ax = plt.subplots(figsize=(8, 6))

	for i, M in enumerate(mach_numbers):
		print(f"Running M = {M:.1f} ...")
		print(f"Simulation {i + 1}/{len(mach_numbers)}")

		v_slab = M * c_background
		# t_kh = jnp.sqrt(Delta) / (2 * jnp.pi) * wavelength / v_slab
		t_kh = jnp.sqrt(Delta) * wavelength / v_slab
		t_kh = float(t_kh)
		print(f"  τ_KH = {t_kh:.4f}")

		result, registered_variables, helper_data, Re_crit, M_crit_val = example_setup_run(
			density_contrast = density_contrast,
			Re_or_nu = nu, # reynolds_number,
			mach_number = M,
			adapt_simulation_time = True,
			return_snapshots = True,
			nu_specified = True,
		)

		num_snapshots = len(result.states)
		times = jnp.array([float(result.time_points[j]) for j in range(num_snapshots)])

		# Prepare McNally's discrete convolution variables
		k = 2 * jnp.pi / wavelength
		X = helper_data.geometric_centers[:, :, 0]
		Y = helper_data.geometric_centers[:, :, 1]

		sin_kx = jnp.sin(k * X)
		cos_kx = jnp.cos(k * X)

		# This is Eq 8 (d_i) from McNally: The exponential KHI envelope
		# (Assuming the single interface is at y_center = 0.5)
		W = jnp.exp(-k * jnp.abs(Y - y_center)) 

		sum_W = jnp.sum(W) # Denominator for Eq 9

		mode_amplitudes = []
		for j in range(num_snapshots):
			vy_field = result.states[j][registered_variables.velocity_index.y]
			
			# This evaluates Eq 6 (s_i) and Eq 7 (c_i) integrated over the domain
			sum_s = jnp.sum(vy_field * sin_kx * W)
			sum_c = jnp.sum(vy_field * cos_kx * W)
			
			# This evaluates Eq 9 (M) from McNally
			mode_amp = 2 * jnp.sqrt((sum_s / sum_W)**2 + (sum_c / sum_W)**2)
			
			mode_amplitudes.append(float(mode_amp))

		mode_amplitudes = jnp.array(mode_amplitudes)

		# Normalize time by τ_KH
		t_normalized = times / t_kh

		# Normalize by initial mode amplitude and take log
		amp0 = mode_amplitudes[0]
		if amp0 > 0:
			ln_ratio = jnp.log(mode_amplitudes / amp0)
		else:
			ln_ratio = jnp.zeros_like(mode_amplitudes)

		# Only plot up to ~2 τ_KH
		mask = t_normalized <= 2.0
		t_plot = t_normalized[mask]
		ln_plot = ln_ratio[mask]
		last_mask_index = jnp.sum(mask) - 1

		# Line style: solid if M < M_crit, dash-dotted if M >= M_crit
		linestyle = '-' if M < M_crit else '-.'
		linewidth = 2.0 if M >= M_crit else 1.5
		color = cmap(norm(M))

		ax.plot(t_plot, ln_plot, linestyle=linestyle, color=color, linewidth=linewidth)
		print(f"  Done. Final ln(A/A0) = {float(ln_plot[-1]):.2f}")

		# also plot the final density snapshot
		fig_snap, ax_snap = plt.subplots(figsize=(6, 5))
		extent = [0, box_size, 0, box_size]
		im = ax_snap.imshow(result.states[last_mask_index][registered_variables.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
		ax_snap.set_title(f"KHI density snapshot, M={M:.1f}, t={times[last_mask_index]:.3f}")
		cbar = fig_snap.colorbar(im, ax=ax_snap)
		cbar.set_label('Density')
		fig_snap.tight_layout()
		fig_snap.savefig(f'figures/khi_density_snapshot_M{M:.1f}.png', dpi=150)
		print(f"Saved figures/khi_density_snapshot_M{M:.1f}.png")

	# # Vertical dashed line at t/τ_KH = 1
	# ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)

	# Colorbar
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax)
	cbar.set_label(r'$\mathcal{M}_\mathrm{h}$', fontsize=14)

	ax.set_xlabel(r'$t\;/\;\tau_\mathrm{KH}$', fontsize=14)
	ax.set_ylabel(r'$\ln\!\left(\frac{v_y}{v_{y0}}\right)$', fontsize=14)
	# ax.set_xlim(0, 2.0)
	# ax.set_ylim(-4.5, 2.5)
	ax.set_title(rf'Growth of KHI with time for $\chi = {density_contrast:.0f}$', fontsize=14)

	fig.tight_layout()
	fig.savefig('figures/khi_growth_over_time.png', dpi=150)
	print("Saved figures/khi_growth_over_time.png")


def parameter_sweep():
	
	density_contrast = 10.0

	num_nu = 100
	num_Ma = 100

	nu_ref = 1809 # "Spitzer viscosity"
	# nus = jnp.linspace(0.0, 0.6, num_nu) * nu_ref
	Res = jnp.geomspace(30, 3000, num_nu)
	mach_numbers = jnp.linspace(0.5, 2.5, num_Ma)

	# num_nu = 1
	# num_Ma = 1

	# Res = [30]
	# mach_numbers = [1.4]

	# Critical Mach number for χ = 10 (Eq. 27 in Mandelker et al. 2016)
	M_crit = (1 + density_contrast**(-1/3))**(3/2)
	print(f"Critical Mach number for χ={density_contrast}: {M_crit:.3f}")

	# Precompute shared quantities for τ_KH calculation
	slab_density = density_contrast * background_density
	c_background = float(jnp.sqrt(gamma * pressure / background_density))
	Delta = float((slab_density + background_density)**2 / (slab_density * background_density))

	wavelength = box_size / 2

	heat_map = jnp.zeros((num_nu, num_Ma))

	for nu_index in range(num_nu):
		for mach_index in range(num_Ma):

			M = mach_numbers[mach_index]
			# nu = nus[nu_index]
			Re = Res[nu_index]

			# print(f"Running M = {M:.1f}, ν = {nu:.2e} ...")
			print(f"Running M = {M:.1f}, Re = {Re:.2e} ...")
			print(f"Simulation {nu_index * num_Ma + mach_index + 1}/{num_nu * num_Ma}")

			v_slab = M * c_background
			# t_kh = jnp.sqrt(Delta) / (2 * jnp.pi) * wavelength / v_slab
			t_kh = jnp.sqrt(Delta) * wavelength / v_slab
			t_kh = float(t_kh)
			print(f"  τ_KH = {t_kh:.4f}")

			result, registered_variables, helper_data, Re_crit, M_crit_val = example_setup_run(
				density_contrast = density_contrast,
				Re_or_nu = Re,
				mach_number = M,
				adapt_simulation_time = True,
				return_snapshots = True,
				nu_specified = False,
			)

			num_snapshots = len(result.states)
			times = jnp.array([float(result.time_points[j]) for j in range(num_snapshots)])

			X = helper_data.geometric_centers[:, :, 0]
			k = 2 * jnp.pi / wavelength
			sin_kx = jnp.sin(k * X)  # shape (nx, ny)
			cos_kx = jnp.cos(k * X)  # shape (nx, ny)

			mode_amplitudes = []
			for j in range(num_snapshots):
				vy_field = result.states[j][registered_variables.velocity_index.y]
				
				# Project onto both components, to remove phase dependence
				proj_sin = jnp.mean(vy_field * sin_kx, axis=0) * 2  # shape (ny,)
				proj_cos = jnp.mean(vy_field * cos_kx, axis=0) * 2  # shape (ny,)
				
				# The true amplitude is the vector magnitude of the Fourier coefficients
				amp_y = jnp.sqrt(proj_sin**2 + proj_cos**2)
				
				# Max over y to get the envelope peak
				mode_amp = jnp.max(amp_y)
				mode_amplitudes.append(float(mode_amp))

			mode_amplitudes = jnp.array(mode_amplitudes)

			# Normalize time by τ_KH
			t_normalized = times / t_kh

			# Normalize by initial mode amplitude and take log
			amp0 = mode_amplitudes[0]
			if amp0 > 0:
				ln_ratio = jnp.log(mode_amplitudes / amp0)
			else:
				ln_ratio = jnp.zeros_like(mode_amplitudes)

			# Only plot up to ~2 τ_KH
			mask = t_normalized <= 2.0
			t_plot = t_normalized[mask]
			ln_plot = ln_ratio[mask]
			last_mask_index = jnp.sum(mask) - 1

			result_value = ln_ratio[last_mask_index]
			heat_map = heat_map.at[nu_index, mach_index].set(result_value)
			print(f"  Done. Final ln(A/A0) = {result_value:.2f}")

			# also plot the final density snapshot
			fig_snap, ax_snap = plt.subplots(figsize=(6, 5))
			extent = [0, box_size, 0, box_size]
			im = ax_snap.imshow(result.states[last_mask_index][registered_variables.density_index].T, cmap='viridis', aspect='auto', origin='lower', extent=extent, norm=LogNorm())
			ax_snap.set_title(f"KHI density snapshot, M={M:.1f}, Re={Re:.2e}, t={times[last_mask_index]:.3f}")
			cbar = fig_snap.colorbar(im, ax=ax_snap)
			cbar.set_label('Density')
			fig_snap.tight_layout()
			fig_snap.savefig(f'figures/sweep/khi_density_snapshot_M{M:.1f}_Re{Re:.2e}.png', dpi=150)
			print(f"Saved figures/sweep/khi_density_snapshot_M{M:.1f}_Re{Re:.2e}.png")

	print(heat_map)
	# save the heatmap for later data analysis
	jnp.save(f'khi_growth_heatmap_chi{density_contrast:.0f}.npy', heat_map)
	print(f"Saved khi_growth_heatmap_chi{density_contrast:.0f}.npy")

	# plot the results as a heatmap
	fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
	im = ax_heat.imshow(heat_map, origin='lower', aspect='auto', extent=[mach_numbers[0], mach_numbers[-1], Res[0], Res[-1]], cmap='viridis')
	ax_heat.set_xlabel('Mach number')
	ax_heat.set_yscale('log')
	ax_heat.set_ylabel('Reynolds number')
	ax_heat.set_title(f'KHI growth (ln(A/A0) at t ~ 2 τ_KH) for χ={density_contrast:.0f}')
	cbar = fig_heat.colorbar(im, ax=ax_heat)
	cbar.set_label(r'$\ln\!\left(\frac{A}{A_0}\right)$', fontsize=14)
	fig_heat.tight_layout()
	fig_heat.savefig(f'figures/khi_growth_heatmap_chi{density_contrast:.0f}.png', dpi=150)
	print(f"Saved figures/khi_growth_heatmap_chi{density_contrast:.0f}.png")


# -------------------------------------------------------------
# ===================== ↓ Eigenmode test ↓ ====================
# -------------------------------------------------------------

def eigenmode_growth_test():
	"""
	Demonstrates that the JAX-Arnoldi computed eigenmode initiates pure 
	exponential growth immediately, free of the initial "sound wave" 
	transients present in ad-hoc velocity perturbations.
	"""
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
	result_eig, _, _ = simulate_khi(setup_eig, return_snapshots=True, plot_spectra_comparison=True)

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
	ax_line.plot(t_vel / t_kh, ln_vel, label="Ad-Hoc Velocity Perturbation", linestyle="--", color="red", linewidth=2)
	ax_line.plot(t_eig / t_kh, ln_eig, label="Numerical Eigenmode (Arnoldi)", linestyle="-", color="blue", linewidth=2)
	
	ax_line.set_xlabel(r"Time ($t / \tau_{KH}$)", fontsize=14)
	ax_line.set_ylabel(r"$\ln(A(t) / A_0)$", fontsize=14)
	ax_line.set_title("KHI Growth: Elimination of Initialization Transients", fontsize=14, fontweight="bold")
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

	axs[0, 0].set_ylabel("Ad-hoc Velocity", fontsize=12, fontweight='bold')
	axs[1, 0].set_ylabel("Arnoldi Eigenmode", fontsize=12, fontweight='bold')
	
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
	# side_by_side_comparison()
	# parameter_sweep()
	# khi_growth_over_time()
	eigenmode_growth_test()