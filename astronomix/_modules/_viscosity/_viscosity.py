r"""
Newtonian viscosity source terms.

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

Video explanation: https://www.youtube.com/watch?v=YPDaFQUqVE4

TODO: we might also support a variant without Stoke's hypothesis.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    DYNAMIC_VISCOSITY,
    KINEMATIC_VISCOSITY,
)

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _stencil_add


@partial(jax.jit, static_argnames=("config", "registered_variables"))
def fv_viscosity_update(primitive_state, params, config, registered_variables, dt):
    """Finite-volume viscosity update.

    NOTE: not yet implemented. The finite-volume viscosity is intended to be
    folded into a unified all-source-term scheme in the future; until then it
    raises ``NotImplementedError``.
    """
    raise NotImplementedError("Not implemented yet.")


@partial(jax.jit, static_argnames=("config", "registered_variables"))
def fd_viscosity_source(primitive_state, params, config, registered_variables):
    """Finite-difference Newtonian viscosity source term.

    Builds the deviatoric viscous stress tensor under Stokes' hypothesis and
    returns its divergence as a momentum source plus the divergence of v·τ as an
    energy source (see the module docstring for the derivation).

    Args:
        primitive_state: The primitive state array.
        params: The simulation parameters (providing the viscosity).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The viscous source term in the layout of the (primitive) state array.
    """

    # Resolve the dynamic viscosity mu; for a kinematic-viscosity setting it is
    # the kinematic viscosity scaled by the local density.
    if config.viscosity_type == DYNAMIC_VISCOSITY:
        mu = params.viscosity
    elif config.viscosity_type == KINEMATIC_VISCOSITY:
        mu = params.viscosity * primitive_state[registered_variables.density_index]

    dx = config.grid_spacing
    ndim = config.dimensionality

    rho = primitive_state[registered_variables.density_index]
    velocity = primitive_state[1:ndim + 1]  # shape (ndim, *spatial)

    # Sixth-order central first derivative along the given array axis.
    def central_first_derivative(field, axis):
        return _stencil_add(
            field,
            indices=(3, 2, 1, -1, -2, -3),
            factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
            axis=axis,
        ) / (60.0 * dx)

    # Velocity gradient tensor G_{ij} = ∂v_i/∂x_j, shape (ndim, ndim, *spatial).
    grad_v = jnp.stack(
        [central_first_derivative(velocity, j + 1) for j in range(ndim)],
        axis=1,
    )

    # Deviatoric viscous stress tensor τ_{ij} = μ (G_{ij} + G_{ji} − ⅔ δ_{ij} ∇·v).
    div_v = jnp.trace(grad_v, axis1=0, axis2=1)  # shape (*spatial)
    delta = jnp.eye(ndim)[(slice(None), slice(None)) + (None,) * rho.ndim]
    tau = mu * (
        grad_v + grad_v.swapaxes(0, 1) - (2.0 / 3.0) * delta * div_v
    )  # shape (ndim, ndim, *spatial)

    # Momentum source (∇·τ)_i = Σ_j ∂τ_{ij}/∂x_j, shape (ndim, *spatial).
    div_tau = sum(
        central_first_derivative(tau[:, j], j + 1) for j in range(ndim)
    )

    # Energy source Σ_j ∂/∂x_j (Σ_i v_i τ_{ij}).
    v_dot_tau = jnp.einsum('i...,ij...->j...', velocity, tau)  # shape (ndim, *spatial)
    energy_src = sum(
        central_first_derivative(v_dot_tau[j], j) for j in range(ndim)
    )

    source_term = jnp.zeros_like(primitive_state)
    source_term = source_term.at[1:ndim + 1].set(div_tau)
    source_term = source_term.at[registered_variables.energy_index].set(energy_src)

    return source_term
