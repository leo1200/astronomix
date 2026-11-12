"""
Face states, fluxes and shared helpers of the Riemann solvers of the finite-volume stage.

The solvers of this directory are transcriptions of the interface solvers of
AthenaPK (and hence Athena++): HLLD (Miyoshi & Kusano 2005; ``_hlld.py``), HLLE
(``_hlle.py``) and local Lax-Friedrichs (``_llf.py``) for GLM-MHD, and HLLC
(``_hllc.py``), HLLE and local Lax-Friedrichs for adiabatic hydrodynamics. The
GLM-MHD solvers first solve the decoupled ``(B_normal, psi)`` subsystem exactly
(Mignone & Tzeferacos 2010, eq. 24) and use the resulting interface normal
field in the MHD solver. The expressions follow AthenaPK's formulation,
including its special cases (the HLLD degeneracy check, the wave-speed clamps),
and the scheme agrees with AthenaPK to round-off
(``pytests/mhd/vl2_athenapk_regression.py``). These solvers serve the VL2
scheme only; the classic finite-volume schemes use ``_riemann_solver/hll.py``,
``_riemann_solver/_lax_friedrichs.py`` and the dispatch in
``_riemann_solver/_riemann_solver.py``.

Every function is purely elementwise on its operands — it never indexes,
reduces or reshapes — so the same code is evaluated on whole arrays by the
native JAX path and on register tiles inside the Pallas kernels.

All states are given in the frame of the interface: ``velocity_normal`` and
``field_normal`` are the components along the flux direction and the two
transverse components follow in AthenaPK's cyclic order (for an x-interface:
y then z; for a y-interface: z then x; for a z-interface: x then y).

This file holds the containers of the face states and fluxes, the wave speeds
and the helpers that more than one solver uses.
"""

# typing
from typing import (
    Any,
    NamedTuple,
)

# jax
import jax.numpy as jnp


# -------------------------------------------------------------
# ======================= ↓ Containers ↓ ======================
# -------------------------------------------------------------


class MHDFaceState(NamedTuple):
    """Primitive GLM-MHD state on one side of an interface (interface frame)."""

    density: Any
    velocity_normal: Any
    velocity_transverse_1: Any
    velocity_transverse_2: Any
    pressure: Any
    field_normal: Any
    field_transverse_1: Any
    field_transverse_2: Any
    psi: Any


class MHDFlux(NamedTuple):
    """
    Interface flux of the GLM-MHD conserved variables (interface frame).

    The fields are ordered like :class:`MHDFaceState`; the integrator scatters
    them back to state order by position.
    """

    mass: Any
    momentum_normal: Any
    momentum_transverse_1: Any
    momentum_transverse_2: Any
    energy: Any
    field_normal: Any
    field_transverse_1: Any
    field_transverse_2: Any
    psi: Any


class HydroFaceState(NamedTuple):
    """Primitive hydrodynamic state on one side of an interface (interface frame)."""

    density: Any
    velocity_normal: Any
    velocity_transverse_1: Any
    velocity_transverse_2: Any
    pressure: Any


class HydroFlux(NamedTuple):
    """
    Interface flux of the hydrodynamic conserved variables (interface frame).

    The fields are ordered like :class:`HydroFaceState`; the integrator
    scatters them back to state order by position.
    """

    mass: Any
    momentum_normal: Any
    momentum_transverse_1: Any
    momentum_transverse_2: Any
    energy: Any


# -------------------------------------------------------------
# ======================= ↑ Containers ↑ ======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ==================== ↓ Shared helpers ↓ =====================
# -------------------------------------------------------------


#: Parthenon's ``TINY_NUMBER``: the outer wave speeds of the hydro HLLE / HLLC
#: solvers are clamped to at least this magnitude.
_TINY_NUMBER = 1.0e-20


def _square(value):
    """Return ``value * value``."""
    return value * value


def fast_magnetosonic_speed(
    gamma,
    density,
    pressure,
    field_normal,
    field_transverse_1,
    field_transverse_2,
):
    """
    Fast magnetosonic speed along the ``field_normal`` direction (AthenaPK:
    ``AdiabaticGLMMHDEOS::FastMagnetosonicSpeed``).

    Args:
        gamma: The adiabatic index.
        density: The density.
        pressure: The gas pressure.
        field_normal: The field component along the propagation direction.
        field_transverse_1: The first transverse field component.
        field_transverse_2: The second transverse field component.

    Returns:
        The fast magnetosonic speed.
    """
    gamma_pressure = gamma * pressure
    transverse_field_squared = (
        field_transverse_1 * field_transverse_1 + field_transverse_2 * field_transverse_2
    )
    sum_of_squares = field_normal * field_normal + transverse_field_squared + gamma_pressure
    difference_of_squares = field_normal * field_normal + transverse_field_squared - gamma_pressure
    discriminant_root = jnp.sqrt(
        difference_of_squares * difference_of_squares
        + 4.0 * gamma_pressure * transverse_field_squared
    )
    return jnp.sqrt(0.5 * (sum_of_squares + discriminant_root) / density)


def adiabatic_sound_speed(gamma, density, pressure):
    """Return the adiabatic sound speed ``sqrt(gamma * p / rho)``."""
    return jnp.sqrt(gamma * pressure / density)


def _glm_interface_field_and_psi(left: MHDFaceState, right: MHDFaceState, cleaning_speed):
    """
    Solve the decoupled ``(B_normal, psi)`` GLM subsystem exactly
    (Mignone & Tzeferacos 2010, eq. 24).

    Args:
        left: The state left of the interface.
        right: The state right of the interface.
        cleaning_speed: The hyperbolic divergence-cleaning speed ``c_h``.

    Returns:
        The interface normal field and the interface psi.
    """
    interface_field_normal = 0.5 * (left.field_normal + right.field_normal) - (
        0.5 / cleaning_speed * (right.psi - left.psi)
    )
    interface_psi = 0.5 * (left.psi + right.psi) - (
        0.5 * cleaning_speed * (right.field_normal - left.field_normal)
    )
    return interface_field_normal, interface_psi


class _MHDConserved(NamedTuple):
    """
    The seven conserved quantities HLLD carries across its wave fan (the
    normal field and psi are handled by the GLM subsystem).
    """

    density: Any
    momentum_normal: Any
    momentum_transverse_1: Any
    momentum_transverse_2: Any
    energy: Any
    field_transverse_1: Any
    field_transverse_2: Any


def _hydro_energy(state: HydroFaceState, inverse_gamma_minus_one):
    """Return the total energy density ``p / (gamma - 1) + rho v^2 / 2``."""
    return state.pressure * inverse_gamma_minus_one + 0.5 * state.density * (
        _square(state.velocity_normal)
        + _square(state.velocity_transverse_1)
        + _square(state.velocity_transverse_2)
    )


def _hydro_flux_minus_speed_times_state(
    state: HydroFaceState,
    relative_velocity,
    energy,
) -> HydroFlux:
    """
    Return ``F - S U`` of one outer state, the flux through a surface moving
    with the wave speed ``S``.

    Args:
        state: The primitive outer state.
        relative_velocity: ``v_normal - S``.
        energy: The total energy density of the outer state.

    Returns:
        ``F - S U`` (interface frame).
    """
    return HydroFlux(
        mass=state.density * relative_velocity,
        momentum_normal=state.density * state.velocity_normal * relative_velocity + state.pressure,
        momentum_transverse_1=state.density * state.velocity_transverse_1 * relative_velocity,
        momentum_transverse_2=state.density * state.velocity_transverse_2 * relative_velocity,
        energy=energy * relative_velocity + state.pressure * state.velocity_normal,
    )


# -------------------------------------------------------------
# ==================== ↑ Shared helpers ↑ =====================
# -------------------------------------------------------------
