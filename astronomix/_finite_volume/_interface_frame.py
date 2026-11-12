"""
The state in the frame of an interface.

The Riemann solvers (``_riemann/``) take the variables of a face in the frame
of the interface: the normal component of velocity and field first, then the
two transverse components in AthenaPK's cyclic order. This file holds the
layout between that frame and the state (which state rows are the normal and
the transverse components, and the way back from frame order to state order)
and the call of the configured solver on values in frame order. It imports
nothing of the stage, so the stage (``_stage.py``), the fused kernel
(``_stage_pallas.py``), the first-order flux correction
(``_flux_correction.py``) and the constrained transport
(``_divergence/_constrained_transport.py``) share it.
"""

# astronomix constants
from astronomix.option_classes.simulation_config import (
    HLL,
    HLLC,
    HLLD,
    LAX_FRIEDRICHS,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
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

    hydro_indices = (
        registered_variables.density_index,
        velocity[normal],
        velocity[transverse_1],
        velocity[transverse_2],
        registered_variables.pressure_index,
    )
    if not config.mhd:
        return hydro_indices

    field = tuple(registered_variables.magnetic_index)
    return hydro_indices + (
        field[normal],
        field[transverse_1],
        field[transverse_2],
        registered_variables.magnetic_psi_index,
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


def _interface_flux_from_states(
    left_values,
    right_values,
    gamma,
    cleaning_speed,
    riemann_solver: int,
    mhd: bool,
):
    """
    Evaluate the configured Riemann solver on interface-frame value tuples.

    Args:
        left_values: Tuple of left-state values (interface frame).
        right_values: Tuple of right-state values (interface frame).
        gamma: The adiabatic index.
        cleaning_speed: The GLM cleaning speed (MHD only).
        riemann_solver: The configured Riemann solver.
        mhd: Whether this is GLM-MHD.

    Returns:
        The interface flux (:class:`MHDFlux` or :class:`HydroFlux`, interface
        frame).
    """
    if mhd:
        left = MHDFaceState(*left_values)
        right = MHDFaceState(*right_values)
        if riemann_solver == HLLD:
            return hlld_flux(left, right, gamma, cleaning_speed)
        if riemann_solver == HLL:
            return hlle_mhd_flux(left, right, gamma, cleaning_speed)
        if riemann_solver == LAX_FRIEDRICHS:
            return llf_mhd_flux(left, right, gamma, cleaning_speed)
        raise ValueError("Unsupported Riemann solver for VL2 MHD.")

    left = HydroFaceState(*left_values)
    right = HydroFaceState(*right_values)
    if riemann_solver == HLLC:
        return hllc_hydro_flux(left, right, gamma)
    if riemann_solver == HLL:
        return hlle_hydro_flux(left, right, gamma)
    if riemann_solver == LAX_FRIEDRICHS:
        return llf_hydro_flux(left, right, gamma)
    raise ValueError("Unsupported Riemann solver for finite-volume hydrodynamics.")


# -------------------------------------------------------------
# ==================== ↑ Riemann solver ↑ =====================
# -------------------------------------------------------------
