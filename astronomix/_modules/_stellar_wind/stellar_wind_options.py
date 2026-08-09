"""
Configuration and parameter containers for stellar-wind injection.

``WindConfig`` holds the static choices (which injection scheme, how many cells
to inject into) and ``WindParams`` the physical wind parameters. The module-level
integer constants name the available injection schemes from
https://arxiv.org/abs/2107.14673.
"""

# typing
from typing import NamedTuple, Tuple, Union

# jax
import jax.numpy as jnp

# Wind injection schemes (see https://arxiv.org/abs/2107.14673).
MEO = 0  # momentum and energy overwrite
EI = 1  # thermal energy injection
MEI = 2  # momentum and energy injection


class WindConfig(NamedTuple):
    stellar_wind: bool = False
    num_injection_cells: int = 10
    wind_injection_scheme: int = EI
    trace_wind_density: bool = False

    #: Use tabulated, time-dependent stellar-evolution wind tracks
    #: (``WindParams.real_params``, see ``stellar_wind_functions.get_wind_parameters``)
    #: instead of the static ``wind_mass_loss_rate(s)`` / ``wind_final_velocity(ies)``.
    #: Only supported by the 3D EI scheme (``_wind_ei3D`` / ``_wind_ei3D_source``).
    real_wind_params: bool = False


class WindParams(NamedTuple):
    # Single-source parameters, used by the 1D injection schemes
    # (MEO / MEI / EI).
    wind_mass_loss_rate: float = 0.0
    wind_final_velocity: float = 0.0

    # Only required for the MEO injection scheme.
    pressure_floor: float = 100000.0

    #: Per-source mass-loss rate / terminal velocity and injection positions
    #: for the 3D EI scheme (``_wind_ei3D`` / ``_wind_ei3D_source``), shapes
    #: (n_sources,) / (n_sources,) / (n_sources, 3). Positions are in the same
    #: box-centered coordinates as the N-body state (box center at the
    #: origin; see ``astronomix._modules._nbody._nbody``). Ignored — and
    #: overridden by the current N-body positions — when
    #: ``config.nbody_config.nbody`` is also enabled.
    wind_mass_loss_rates: jnp.ndarray = jnp.array([0.0])
    wind_final_velocities: jnp.ndarray = jnp.array([0.0])
    wind_injection_positions: jnp.ndarray = jnp.array([[0.0, 0.0, 0.0]])

    #: Tabulated (time, log-mass-loss-rate, wind-velocity) tracks, one row of
    #: tracks per source: a 3-tuple ``(time, mass_loss_rate, wind_velocity)``,
    #: each of shape (n_sources, n_track_points), already converted to code
    #: units by the caller (see ``stellar_wind_functions.get_wind_parameters``,
    #: which returns the raw Ekstrom+2012 track values and does not itself do
    #: any unit conversion). Interpolated at the current simulation time by
    #: ``stellar_wind_functions.get_current_wind_params`` in place of
    #: ``wind_mass_loss_rates`` / ``wind_final_velocities`` when
    #: ``config.wind_config.real_wind_params``.
    real_params: Union[Tuple, None] = None

    #: Set internally, once per step, by the time-integration loop (see
    #: astronomix.time_stepping.time_integration) to the current simulation
    #: time. Only needed to make "now" available to the finite-difference
    #: source-term path (which runs per RK stage, without direct access to
    #: the absolute time) for the ``real_params`` time interpolation; not
    #: meant to be set directly by the user.
    current_time: float = 0.0
