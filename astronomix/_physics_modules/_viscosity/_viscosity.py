"""
Viscosity in Fluids

Intuition of the 1D case
------------------------

It might be intuitive that in a setting with viscosity,
a velocity gradient \partial_z v_x of v_x leads to 
a diffusive momentum flux -\mu \partial_z v_x = -\tau_xz
(the assumption of a linear relation here implies a 
Newtonian fluid) which gradient then enters the Euler 
equations, leading to a source term 
\partial_z \tau_xz = \mu \partial_z^2 v_x.

3D generalization
-----------------

Given the velocity gradient tensor G_{ij} = ∂v_i/∂x_j,
most generally, the viscous stress tensor (which encodes 
momentum fluxes in all directions) for a Newtonian fluid
is given by 

\tau_{ij} = K_{ij}^{kl} G_{kl}

where K is a 4th order tensor with 3^4 = 81 components.

Our fluid is isotropic, which
means that the components of K must be invariant under any 
rotation of the coordinate system R \in SO(3). The only
2nd order isotropic tensor is \delta_{ij}, so naturally,

K_{ij}^{kl} = \lambda \delta_{ij} \delta^{kl} + \mu \delta_i^k \delta_j^l + \nu \delta_i^l \delta_j^k

so we are down to 3 parameters. Additionally, \tau_{ij}
must be symmetric (otherwise we would create a net torque
and violate angular momentum conservation, imagine a small 
fluid element), implying K_{ij}^{kl} = K_{ji}^{kl},
which implies \nu = \mu (we are down to two parameters), such that

K_{ij}^{kl} = \lambda \delta_{ij} \delta^{kl} + \mu (δ_i^k δ_j^l + δ_i^l δ_j^k)

Inserting this into the definition of \tau_{ij} gives

\tau_{ij} = \lambda \delta_{ij} G_{kk} + \mu (G_{ij} + G_{ji})

We can split the velocity gradient tensor into a symmetric and
antisymmetric part

G_{ij} = ∂v_i/∂x_j = 1/2 * (∂v_i/∂x_j + ∂v_j/∂x_i) + 1/2 * (∂v_i/∂x_j - ∂v_j/∂x_i)
                     |-------- symmetric --------|  |------ antisymmetric -------| 

where the antisymmetric part corresponds to the vorticity and drops out by
the symmetry of \tau_{ij}, leaving only the symmetric part.

Therefore (also use G_{kk} = ∇·v)

\tau_{ij} = \lambda \delta_{ij} ∇·v + \mu * (∂v_i/∂x_j + ∂v_j/∂x_i)

As a next step, we split \tau_{ij} into a 
(isotropic) hydrostatic part

h_{ij} = 1/3 * \tau_{kk} \delta_{ij} = (λ + 2/3 μ) ∇·v δ_{ij}
        |- mean trace -|

acting like a pressure and a deviatoric part 

s_{ij} = \tau_{ij} - h_{ij} = \mu * (∂v_i/∂x_j + ∂v_j/∂x_i - 2/3 δ_{ij} ∇·v)

and we impose Stoke's hypothesis (λ + 2/3 μ = 0), eliminating the bulk isotropic viscosity,
leaving only the deviatoric part, such that

\tau_{ij} = s_{ij} = \mu * (∂v_i/∂x_j + ∂v_j/∂x_i - 2/3 δ_{ij} ∇·v)

resulting in a momentum source term of ∇·τ and an energy 
source term of ∇·(v·τ).

# Video explanation: https://www.youtube.com/watch?v=YPDaFQUqVE4

# TODO: we might also support a variant without Stoke's hypothesis.
"""


from functools import partial

import jax
import jax.numpy as jnp   
from astronomix._fluid_equations._equations import conserved_state_from_primitive, primitive_state_from_conserved
from astronomix._stencil_operations._stencil_operations import _shift, _stencil_add
from astronomix.option_classes.simulation_config import DYNAMIC_VISCOSITY, KINEMATIC_VISCOSITY

# the fv one is preliminary, in the future I want an all source
# term scheme
@partial(jax.jit, static_argnames=('config', 'registered_variables'))
def fv_viscosity_update(primitive_state, params, config, registered_variables, dt):

    if config.viscosity_type == DYNAMIC_VISCOSITY:
        mu = params.viscosity
    elif config.viscosity_type == KINEMATIC_VISCOSITY:
        mu = params.viscosity * primitive_state[registered_variables.density_index]
    
    dx = config.grid_spacing
    ndim = config.dimensionality

    rho = primitive_state[registered_variables.density_index]
    v = primitive_state[1:ndim + 1]                            # (ndim, *spatial)

    # Cell-center velocity gradient (2nd-order centered, for tangential terms)
    grad_v_cc = jnp.stack([
        (_shift(v, -1, axis=j + 1) - _shift(v, 1, axis=j + 1)) / (2.0 * dx)
        for j in range(ndim)
    ], axis=1)                                                 # (ndim_i, ndim_j, *spatial)

    div_v_cc = jnp.trace(grad_v_cc, axis1=0, axis2=1)         # (*spatial)

    mom_src = jnp.zeros_like(v)                                # (ndim, *spatial)
    energy_src = jnp.zeros_like(rho)                           # (*spatial)

    for j in range(ndim):
        ax = j + 1  # array axis (0 = component, 1..3 = spatial)

        # ── right face i+1/2 along direction j ────────────────────

        # normal derivative: compact, 2nd-order exact at face
        dv_dxj = (_shift(v, -1, axis=ax) - v) / dx            # (ndim, *spatial)

        # tangential derivatives: average cell-center values to face
        # grad_v_cc[j][i] = ∂v_j/∂x_i  →  need this for all i
        dvj_dxi = 0.5 * (grad_v_cc[j] + _shift(grad_v_cc[j], -1, axis=ax))

        # ∇·v at face
        div_v = 0.5 * (div_v_cc + _shift(div_v_cc, -1, axis=ax))

        # δ_{ij} with broadcasting shape (ndim, 1, 1, ...)
        d_ij = jnp.zeros(ndim).at[j].set(1.0)
        d_ij = d_ij.reshape((-1,) + (1,) * rho.ndim)

        # τ_{ij} at face for all i:
        # τ_{ij} = μ (∂v_i/∂x_j + ∂v_j/∂x_i − ⅔ δ_{ij} ∇·v)
        if config.viscosity_type == DYNAMIC_VISCOSITY:
            mu_face = mu
        elif config.viscosity_type == KINEMATIC_VISCOSITY:
            mu_face = 0.5 * (mu + _shift(mu, -1, axis=ax))
        
        tau_face = mu_face * (dv_dxj + dvj_dxi - (2.0 / 3.0) * d_ij * div_v)

        # velocity at face (for energy flux)
        v_face = 0.5 * (v + _shift(v, -1, axis=ax))

        # viscous energy flux:  Σ_i v_i τ_{ij}
        e_flux = jnp.sum(v_face * tau_face, axis=0)           # (*spatial)

        # ── conservative divergence: (F_{i+1/2} − F_{i-1/2}) / dx ──
        mom_src    += (tau_face - _shift(tau_face, 1, axis=ax)) / dx
        energy_src += (e_flux   - _shift(e_flux,   1, axis=ax)) / dx

    S_visc = jnp.zeros_like(primitive_state)
    S_visc = S_visc.at[1:ndim + 1].set(mom_src)
    S_visc = S_visc.at[registered_variables.energy_index].set(energy_src)

    # add the source term to the conserved state with an Euler step
    # (this is a bit hacky and I would prefer a more proper solution in 
    # the future)

    conserved_state = conserved_state_from_primitive(primitive_state, params.gamma, config, registered_variables)
    conserved_state += S_visc * dt
    primitive_state = primitive_state_from_conserved(conserved_state, params.gamma, config, registered_variables)
    return primitive_state

@partial(jax.jit, static_argnames=('config', 'registered_variables'))
def fd_viscosity_source(primitive_state, params, config, registered_variables):
    
    if config.viscosity_type == DYNAMIC_VISCOSITY:
        mu = params.viscosity # the dynamic viscosity
    elif config.viscosity_type == KINEMATIC_VISCOSITY:
        mu = params.viscosity * primitive_state[registered_variables.density_index]
    
    dx = config.grid_spacing
    ndim = config.dimensionality

    rho = primitive_state[registered_variables.density_index]
    v = primitive_state[1:ndim + 1]                            # (ndim, *spatial)

    def _d1(field, ax):
        return _stencil_add(
            field, indices=(3, 2, 1, -1, -2, -3),
            factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0), axis=ax,
        ) / (60.0 * dx)

    # velocity gradient tensor  G_{ij} = ∂v_i/∂x_j   (ndim, ndim, *spatial)
    grad_v = jnp.stack([_d1(v, j + 1) for j in range(ndim)], axis=1)

    # stress tensor  τ_{ij} = μ (G_{ij} + G_{ji} − ⅔ δ_{ij} ∇·v)
    div_v = jnp.trace(grad_v, axis1=0, axis2=1)               # (*spatial)
    delta = jnp.eye(ndim)[(slice(None), slice(None)) + (None,) * rho.ndim]
    tau = mu * (grad_v + grad_v.swapaxes(0, 1)
                - (2.0 / 3.0) * delta * div_v)                 # (ndim, ndim, *spatial)

    # momentum source  (∇·τ)_i = Σ_j ∂τ_{ij}/∂x_j            (ndim, *spatial)
    div_tau = sum(_d1(tau[:, j], j + 1) for j in range(ndim))

    # energy source  Σ_j ∂/∂x_j (Σ_i v_i τ_{ij})
    v_dot_tau = jnp.einsum('i...,ij...->j...', v, tau)        # (ndim, *spatial)
    energy_src = sum(_d1(v_dot_tau[j], j) for j in range(ndim))

    S_visc = jnp.zeros_like(primitive_state)
    S_visc = S_visc.at[1:ndim + 1].set(div_tau)
    S_visc = S_visc.at[registered_variables.energy_index].set(energy_src)

    return S_visc