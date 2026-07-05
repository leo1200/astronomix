"""
Cosmic-ray injection at shock fronts (diffusive shock acceleration).

Locates the strongest shock in the domain, estimates its Mach number from the
jump conditions, and converts a configurable fraction of the dissipated energy
into cosmic-ray pressure. The injected energy is distributed across the
broadened numerical shock layer. The scheme follows Pfrommer et al. (2017) and
Dubois et al. (2019); see ``inject_crs_at_strongest_shock`` for the references.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import SPHERICAL, STATE_TYPE

# astronomix containers
from astronomix._modules._cosmic_rays.cosmic_ray_options import CosmicRayParams
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix.shock_finder.shock_finder import find_shock_zone

# NOTE: this routine currently only supports 1D setups; generalising it to
# 2D / 3D is still outstanding.


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def inject_crs_at_strongest_shock(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    helper_data: HelperData,
    cosmic_ray_params: CosmicRayParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    dt: Union[float, Float[Array, ""]],
) -> STATE_TYPE:
    """
    Cosmic ray injection at shock fronts.
    Currently only at the strongest shock in the domain.

    The implementation generally follows

    Pfrommer, Christoph, et al. "Simulating cosmic ray physics on a moving mesh."
    Monthly Notices of the Royal Astronomical Society 465.4 (2017): 4500-4529.
    https://arxiv.org/abs/1604.07399

    and

    Dubois, Yohan, et al. "Shock-accelerated cosmic rays and streaming instability
    in the adaptive mesh refinement code Ramses."
    Astronomy & Astrophysics 631 (2019): A121.
    https://arxiv.org/abs/1907.04300

    Args:
        primitive_state: The primitive state array.
        gamma: The adiabatic index.
        helper_data: The helper data.
        cosmic_ray_params: The cosmic ray parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        dt: The time step.

    Returns:
        The primitive state array with injected cosmic rays.

    """

    num_cells = primitive_state.shape[1]

    # The injection efficiency (fraction of dissipated energy that goes into
    # cosmic rays) is supplied by the user. In future this could be replaced by
    # a Mach-dependent model, e.g. the ones in
    # https://github.com/LudwigBoess/DiffusiveShockAccelerationModels.jl/tree/main/src/mach_models
    injection_efficiency = cosmic_ray_params.diffusive_shock_acceleration_efficiency

    # The adiabatic indices for the two fluids are currently hard-coded; the gas
    # index follows the user-supplied ``gamma``.
    gamma_cr = 4 / 3
    gamma_gas = gamma

    # -------------------------------------------------------------
    # ============= ↓ Locate the strongest shock ↓ ===============
    # -------------------------------------------------------------

    max_shock_idx, left_idx, right_idx = find_shock_zone(
        primitive_state,
        config,
        registered_variables,
        helper_data,
    )

    # NOTE: shifting ``left_idx`` outward by +2 smooths the transition of the
    # different pressure components across the shock, but causes problems at
    # lower resolutions. There is also the subtlety that cosmic-ray pressure
    # injected into the broadened shock layer experiences P_CR * div(u) forces
    # (Dubois et al. 2019), which can lead to effective over-injection.

    # We only consider a shock moving from left to right, so the pre-shock state
    # is upstream and the post-shock state is downstream in the shock frame.
    pre_shock_idx = right_idx + 1
    post_shock_idx = left_idx - 1

    # -------------------------------------------------------------
    # ======== ↓ Pre- and post-shock fluid quantities ↓ =========
    # -------------------------------------------------------------

    # Pre-shock (upstream) state: density, total/CR/gas pressures and energies.
    rho1 = primitive_state[registered_variables.density_index, pre_shock_idx]
    P1 = primitive_state[registered_variables.pressure_index, pre_shock_idx]
    P1_CRs = (
        primitive_state[registered_variables.cosmic_ray_n_index, pre_shock_idx]
        ** gamma_cr
    )
    P1_gas = P1 - P1_CRs
    e1_gas = P1_gas / (gamma_gas - 1)  # gas energy density
    e1_crs = P1_CRs / (gamma_cr - 1)  # cosmic-ray energy density
    e1 = e1_gas + e1_crs  # total energy density

    # Post-shock (downstream) state.
    rho2 = primitive_state[registered_variables.density_index, post_shock_idx]
    P2 = primitive_state[registered_variables.pressure_index, post_shock_idx]
    P2_CRs = (
        primitive_state[registered_variables.cosmic_ray_n_index, post_shock_idx]
        ** gamma_cr
    )
    P2_gas = P2 - P2_CRs
    e2_gas = P2_gas / (gamma_gas - 1)
    e2_crs = P2_CRs / (gamma_cr - 1)
    e2 = e2_gas + e2_crs

    # -------------------------------------------------------------
    # ============ ↓ Mach number and dissipated flux ↓ ===========
    # -------------------------------------------------------------

    # Effective adiabatic index of the upstream mixture, and the corresponding
    # upstream sound speed.
    gamma_eff1 = (gamma_cr * P1_CRs + gamma_gas * P1_gas) / P1
    c1 = jnp.sqrt(gamma_eff1 * P1 / rho1)

    # Compression ratio across the shock.
    x_s = rho2 / rho1

    # Effective adiabatic indices on both sides of the shock.
    gamma_eff1 = (gamma_cr * P1_CRs + gamma_gas * P1_gas) / P1
    gamma_eff2 = (gamma_cr * P2_CRs + gamma_gas * P2_gas) / P2

    # Pressure-weighted "energy" adiabatic indices used by the Mach-number
    # estimate below.
    gamma1 = P1 / e1 + 1
    gamma2 = P2 / e2 + 1

    gammat = P2 / P1
    C = ((gamma2 + 1) * gammat + gamma2 - 1) * (gamma1 - 1)

    # Squared pre-shock Mach number following Eq. 16 of Dubois et al. (2019).
    # This differs slightly from the expression in Pfrommer et al. (2017), where
    # the simpler formula
    #     M_1_sq = (P2 / P1 - 1) * x_s / (gamma_eff1 * (x_s - 1))
    # is used for the injection itself (it is only a lower bound there). In our
    # experience that simpler form led to more crashes in spherical-geometry
    # setups, so the Dubois form is preferred here.
    M_1_sq = (
        1
        / gamma_eff2
        * (gammat - 1)
        * C
        / (C - ((gamma1 + 1) + (gamma1 - 1) * gammat) * (gamma2 - 1))
    )

    # Dissipated energy density and the corresponding dissipated energy flux
    # through the shock surface.
    e_diss = e2_gas - e1_gas * x_s**gamma_gas + e2_crs - e1_crs * x_s**gamma_cr
    f_diss = e_diss * jnp.sqrt(M_1_sq) * c1 / x_s

    # -------------------------------------------------------------
    # ============ ↓ Distribute the injected energy ↓ ============
    # -------------------------------------------------------------

    # Shock surface area: a sphere in spherical geometry, otherwise the
    # transverse cell area implied by the grid spacing and dimensionality.
    if config.geometry == SPHERICAL:
        shock_radius = helper_data.geometric_centers[max_shock_idx]
        shock_surface = 4 * jnp.pi * shock_radius**2
    else:
        shock_surface = config.grid_spacing ** (config.dimensionality - 1)

    # Total energy to be injected as cosmic-ray pressure over this time step.
    DeltaE_CR = f_diss * shock_surface * dt * injection_efficiency

    # Build a mask for the broadened shock zone over which the energy is spread.
    # NOTE: Pfrommer et al. (2017) use ``post_shock_idx`` here instead of
    # ``left_idx`` as the lower bound.
    indices = jnp.arange(num_cells)
    shock_zone_mask = (indices >= left_idx) & (indices <= max_shock_idx)

    # Distribute the injected energy across the shock zone in proportion to each
    # cell's total energy excess relative to the upstream reference cell.
    cosmic_ray_pressure = (
        primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    )
    gas_pressure = (
        primitive_state[registered_variables.pressure_index] - cosmic_ray_pressure
    )
    e_th = gas_pressure / (gamma_gas - 1)
    e_cr = cosmic_ray_pressure / (gamma_cr - 1)
    E_tot = (e_th + e_cr) * helper_data.cell_volumes
    DeltaEtot = jnp.sum(jnp.where(shock_zone_mask, E_tot - E_tot[right_idx], 0))
    DeltaE_CR_split = DeltaE_CR * (E_tot - E_tot[right_idx]) / DeltaEtot
    DeltaE_CR_split = jnp.where(shock_zone_mask, DeltaE_CR_split, 0)

    # -------------------------------------------------------------
    # ============ ↓ Apply the cosmic-ray injection ↓ ============
    # -------------------------------------------------------------

    # Existing cosmic-ray pressure, then the updated pressure after injection.
    p_cr_injection = (
        primitive_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    )
    p_cr_injection_new = p_cr_injection + DeltaE_CR_split / helper_data.cell_volumes * (
        gamma_cr - 1
    )
    # The cosmic rays are tracked through n_cr = P_CR ** (1 / gamma_cr), so the
    # updated pressure is converted back to the advected scalar before storing.
    n_cr_injection_new = p_cr_injection_new ** (1 / gamma_cr)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_n_index].set(
        n_cr_injection_new
    )

    # We want energy (not pressure) conservation, so removing thermal energy and
    # converting it into cosmic-ray energy requires adapting the stored total
    # pressure accordingly.
    delta_p_gas = DeltaE_CR_split / helper_data.cell_volumes * (gamma_gas - 1)
    p_gas_new = (
        primitive_state[registered_variables.pressure_index]
        - p_cr_injection
        - delta_p_gas
    )
    total_pressure_new = p_gas_new + p_cr_injection_new

    primitive_state = primitive_state.at[registered_variables.pressure_index].set(
        total_pressure_new
    )

    return primitive_state
