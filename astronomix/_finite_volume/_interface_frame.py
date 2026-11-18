"""
The state in the frame of an interface.

The Riemann solvers (``_riemann/``) take the variables of a face in the frame
of the interface: the normal component of velocity and field first, then the
two transverse components in AthenaPK's cyclic order. This file holds the
layout between that frame and the state (which state rows are the normal and
the transverse components, and the way back from frame order to state order)
and the call of the configured solver on values in frame order, with the
pressure and the sound speed of a barotropic law where the state has no
pressure slot (``BarotropicLaw``). It imports nothing of the stage, so the stage (``_stage.py``), the fused kernel
(``_stage_pallas.py``), the first-order flux correction
(``_flux_correction.py``) and the constrained transport
(``_divergence/_constrained_transport.py``) share it.
"""

# typing
from typing import (
    Callable,
    NamedTuple,
)

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    HLL,
    HLLC,
    HLLD,
    ISOTHERMAL,
    LAX_FRIEDRICHS,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._equation_of_state import (
    barotropic_pressure,
    is_barotropic,
    sound_speed_squared,
)
from astronomix._finite_volume._riemann._face_state import (
    HydroFaceState,
    MHDFaceState,
)
from astronomix._finite_volume._riemann._hllc import hllc_hydro_flux
from astronomix._finite_volume._riemann._hlld import hlld_flux
from astronomix._finite_volume._riemann._hlle import (
    hlle_hydro_flux,
    hlle_mhd_flux,
)
from astronomix._finite_volume._riemann._llf import (
    llf_hydro_flux,
    llf_mhd_flux,
)


# -------------------------------------------------------------
# ===================== ↓ State layout ↓ ======================
# -------------------------------------------------------------

#: Positions of density and pressure in the interface frame (:class:`HydroFaceState`, :class:`MHDFaceState`).
FRAME_DENSITY = 0
FRAME_PRESSURE = 4
#: Positions of the velocity and of the field (normal, transverse 1, transverse 2) in :class:`MHDFaceState`.
FRAME_VELOCITY = slice(1, 4)
FRAME_FIELD = slice(5, 8)


def _velocity_component_indices(registered_variables: RegisteredVariables):
    """
    The state indices of the x, y and z velocity. Components the layout does
    not carry (the dimension-reduced hydro layouts) are ``None``; a missing
    component behaves exactly like a component that is zero everywhere.

    Args:
        registered_variables: The registered variables.

    Returns:
        A tuple of three state indices (``None`` for absent components).
    """
    velocity_index = registered_variables.velocity_index
    if isinstance(velocity_index, int):
        return (velocity_index, None, None)
    return tuple(component if component >= 0 else None for component in tuple(velocity_index))


def _interface_frame_indices(
    axis: int,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The state indices of the variables in the frame of an interface normal to
    ``axis`` (1-based), in AthenaPK's cyclic order: normal, then transverse 1
    and transverse 2 (x: y, z; y: z, x; z: x, y).

    Args:
        axis: The spatial axis (1, 2 or 3) normal to the interface.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        A tuple of state indices (``None`` for components the layout lacks),
        ordered like :class:`MHDFaceState` (MHD) or :class:`HydroFaceState`.
    """
    velocity = _velocity_component_indices(registered_variables)
    normal = axis - 1
    transverse_1 = axis % 3
    transverse_2 = (axis + 1) % 3

    # The layout of a barotropic gas has no pressure slot (and no energy row for the flux).
    pressure_index = registered_variables.pressure_index
    hydro_indices = (
        registered_variables.density_index,
        velocity[normal],
        velocity[transverse_1],
        velocity[transverse_2],
        pressure_index if pressure_index >= 0 else None,
    )
    if not config.mhd:
        return hydro_indices

    field = tuple(registered_variables.magnetic_index)
    psi_index = registered_variables.magnetic_psi_index
    return hydro_indices + (
        field[normal],
        field[transverse_1],
        field[transverse_2],
        psi_index if psi_index >= 0 else None,
    )


def _state_components_from_frame(frame_values, frame_indices, num_vars: int):
    """
    Scatter interface-frame values back to state order.

    Args:
        frame_values: The values in the interface frame (e.g. a flux tuple).
        frame_indices: The state index of every frame value (``None`` for
            components the layout lacks, which are dropped).
        num_vars: The number of state variables.

    Returns:
        A list of the values indexed like the state.
    """
    state_components = [None] * num_vars
    for index, value in zip(frame_indices, frame_values):
        if index is not None:
            state_components[index] = value
    return state_components


# -------------------------------------------------------------
# ===================== ↑ State layout ↑ ======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ==================== ↓ Riemann solver ↓ =====================
# -------------------------------------------------------------


class BarotropicLaw(NamedTuple):
    """
    Pressure and squared sound speed of a barotropic gas as functions of the
    density: the form in which the Riemann solvers and the Hancock predictor
    take the equation of state. ``exponent`` is the exponent of the density in
    the pressure (one for the isothermal gas).
    """

    pressure_of: Callable
    sound_speed_squared_of: Callable
    exponent: float


def _barotropic_law(params: SimulationParams, config: SimulationConfig):
    """
    The barotropic law of the configured equation of state.

    Args:
        params: The simulation parameters.
        config: The simulation configuration.

    Returns:
        The :class:`BarotropicLaw`, or ``None`` for the ideal gas.
    """
    if not is_barotropic(config):
        return None
    return BarotropicLaw(
        pressure_of=lambda density: barotropic_pressure(density, params, config),
        sound_speed_squared_of=lambda density: sound_speed_squared(density, None, params, config),
        exponent=1.0 if config.equation_of_state == ISOTHERMAL else params.polytropic_exponent,
    )


def _with_pressure_of_law(frame_values, barotropic_law: BarotropicLaw):
    """Return the frame values with the pressure of the barotropic law at the pressure position."""
    pressure = barotropic_law.pressure_of(frame_values[FRAME_DENSITY])
    return frame_values[:FRAME_PRESSURE] + (pressure,) + frame_values[FRAME_PRESSURE + 1 :]


def _mhd_face_state(state, frame_indices, barotropic_law: BarotropicLaw = None) -> MHDFaceState:
    """
    The rows of a state array as an MHD face state in the frame of an interface.

    Args:
        state: A primitive state array (values on one side of the interfaces).
        frame_indices: The state indices of the frame (``_interface_frame_indices``).
        barotropic_law: ``None`` for the ideal gas, or the law of a barotropic gas.

    Returns:
        The face state: zeros for the components the layout lacks, the pressure of the law for a barotropic gas.
    """
    zero = jnp.zeros_like(state[frame_indices[FRAME_DENSITY]])
    values = tuple(state[index] if index is not None else zero for index in frame_indices)
    if barotropic_law is not None:
        values = _with_pressure_of_law(values, barotropic_law)
    return MHDFaceState(*values)


def _interface_flux_from_states(
    left_values,
    right_values,
    gamma,
    cleaning_speed,
    riemann_solver: int,
    mhd: bool,
    barotropic_law: BarotropicLaw = None,
):
    """
    Evaluate the configured Riemann solver on interface-frame value tuples.

    Under a barotropic law the state has no pressure slot: the face values get
    the pressure of the law from their density here, and the solver bounds its
    waves with the sound speed of the law. The energy value of the flux then
    has no meaning.

    Args:
        left_values: Tuple of left-state values (interface frame).
        right_values: Tuple of right-state values (interface frame).
        gamma: The adiabatic index (ideal gas only).
        cleaning_speed: The GLM cleaning speed (MHD only).
        riemann_solver: The configured Riemann solver.
        mhd: Whether this is GLM-MHD.
        barotropic_law: ``None`` for the ideal gas, or the law of a barotropic gas.

    Returns:
        The interface flux (:class:`MHDFlux` or :class:`HydroFlux`, interface
        frame).
    """
    sound_speed_squared_of = None
    if barotropic_law is not None:
        left_values = _with_pressure_of_law(left_values, barotropic_law)
        right_values = _with_pressure_of_law(right_values, barotropic_law)
        sound_speed_squared_of = barotropic_law.sound_speed_squared_of
        if riemann_solver not in (HLL, LAX_FRIEDRICHS):
            raise ValueError("A barotropic equation of state needs the HLL or the LAX_FRIEDRICHS Riemann solver.")

    if mhd:
        left = MHDFaceState(*left_values)
        right = MHDFaceState(*right_values)
        if riemann_solver == HLLD:
            return hlld_flux(left, right, gamma, cleaning_speed)
        if riemann_solver == HLL:
            return hlle_mhd_flux(left, right, gamma, cleaning_speed, sound_speed_squared_of)
        if riemann_solver == LAX_FRIEDRICHS:
            return llf_mhd_flux(left, right, gamma, cleaning_speed, sound_speed_squared_of)
        raise ValueError("Unsupported Riemann solver for VL2 MHD.")

    left = HydroFaceState(*left_values)
    right = HydroFaceState(*right_values)
    if riemann_solver == HLLC:
        return hllc_hydro_flux(left, right, gamma)
    if riemann_solver == HLL:
        return hlle_hydro_flux(left, right, gamma, sound_speed_squared_of)
    if riemann_solver == LAX_FRIEDRICHS:
        return llf_hydro_flux(left, right, gamma, sound_speed_squared_of)
    raise ValueError("Unsupported Riemann solver for finite-volume hydrodynamics.")


# -------------------------------------------------------------
# ==================== ↑ Riemann solver ↑ =====================
# -------------------------------------------------------------
