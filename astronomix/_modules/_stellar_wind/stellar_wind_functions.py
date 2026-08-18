"""
Real (tabulated) stellar-wind parameters from stellar-evolution tracks.

Extracts wind mass-loss rate and terminal-velocity tracks from the Ekstrom+2012
grid of rotating/non-rotating stellar-evolution models (the wind-velocity
prescription follows Gatto+2016, https://arxiv.org/pdf/1606.05346.pdf; the
evolutionary-phase split follows Georgy+2012,
https://arxiv.org/pdf/1203.5243.pdf), and interpolates them to the current
simulation time. Used by ``astronomix._modules._stellar_wind.stellar_wind``
when ``config.wind_config.real_wind_params`` is enabled.

``get_wind_parameters`` returns the raw track values (time in yr, log10 mass
loss rate, wind velocity in km/s) — it does not perform any unit conversion,
so the caller is responsible for converting them to code units before storing
them in ``WindParams.real_params``.
"""

# general
from functools import partial
from pathlib import Path

# jax
import jax
import jax.numpy as jnp
import numpy as np

# fits table I/O
from astropy.io import fits

#: The Ekstrom+2012 grid of rotating/non-rotating stellar-evolution models,
#: bundled alongside this module.
_DEFAULT_TRACK_TABLE = Path(__file__).resolve().parent / "ekstroem+2012.fit"


def get_mass_entry(data, mass, rotation):
    """
    Return the table rows for a given initial mass, considering whether the
    rotating or non-rotating model should be returned.
    """
    idx = np.where(data["Mini"] == mass)
    if rotation:
        rot = np.where(data[idx]["Rot"] == "r")
    else:
        rot = np.where(data[idx]["Rot"] == "n")
    return data[idx][rot]


# functions to split the model tracks into different evolutionary phases;
# see Georgy+2012 (https://arxiv.org/pdf/1203.5243.pdf) for more details

def is_wr_type(teff, x):
    idx = (teff > 1e4) * (x < 0.3)
    return np.where(idx == True)


def is_o_type(teff, x):
    idx = (teff > (10 ** 4.5)) * (x >= 0.3)
    return np.where(idx == True)


def is_wc_type(teff, x, c12, c13, n):
    idx = (teff > 1e4) * (x == 0.0) * ((c12 + c13) > n)
    return np.where(idx == True)


# wind velocities for different evolutionary phases; see Gatto+2016
# (https://arxiv.org/pdf/1606.05346.pdf) for more details

def wc_wind(teff):
    """Linear interpolation: 700 km/s at T_eff=2e4 K to 2800 km/s at T_eff=8e4 K."""
    m = (2800 - 700) / (8e4 - 2e4)
    x = teff - 2e4
    b = 700
    return m * x + b


def wnl_wind(teff):
    """Linear interpolation: 700 km/s at T_eff=2e4 K to 2100 km/s at T_eff=5e4 K."""
    m = (2100 - 700) / (5e4 - 2e4)
    x = teff - 2e4
    b = 700
    return m * x + b


def wind_pulse(teff, vesc):
    """O-type-star wind velocity as a multiple of the escape velocity."""
    low_teff = np.where(teff < 1.8e4)
    high_teff = np.where(teff > 2.3e4)
    interp = ((2.45 - 1.3) / (2.3e4 - 1.8e4)) * (teff - 1.84e4) + 1.3
    interp[low_teff] = 1.3
    interp[high_teff] = 2.45

    return interp * vesc


def supergiant_wind(L):
    return 10 * (L / 3e4) ** 0.25


def get_wind_velocity(teff, mass, L, mini, x, c12, c13, n):
    """Combine the phase-dependent wind-velocity prescriptions above into one
    complete model track."""
    # L = 4*pi*R^2*sigma_boltz*T^4
    sigma_b = 5.67e-8
    Lsun = 3.8e26  # Watts
    r = np.sqrt(L * Lsun / (4 * np.pi * sigma_b * teff ** 4))  # in meter

    # vesc = sqrt(2GM/r)
    G = 6.674e-11  # m^3 kg^-1 s^-2
    vesc = np.sqrt(2 * G * mass * 2e30 / r) / 1000  # in km/s

    v_wind = wind_pulse(teff, vesc)

    wr = is_wr_type(teff, x)
    v_wind[wr] = wnl_wind(teff)[wr]

    wc = is_wc_type(teff, x, c12, c13, n)
    v_wind[wc] = wc_wind(teff)[wc]

    return v_wind


def wind_parameters(data, mass, rotation):
    """
    Given a stellar-evolution track table, return the wind parameters (wind
    velocity and mass-loss rate) for a model of initial mass ``mass``, for
    the rotating or non-rotating model.

    Returns:
        [time in yr, mass loss rate in log10(Msun/yr), wind velocity in km/s]
    """
    model = get_mass_entry(data, mass, rotation)
    wind = get_wind_velocity(
        10 ** model["logTe"], model["Mass"], 10 ** model["logL"],
        model["Mini"], model["X"], model["C12"], model["C13"], model["N14"],
    )

    return [model["Time"], model["logdM_dt"], wind]


def get_wind_parameters(particle_masses, rotation=True, track_table=_DEFAULT_TRACK_TABLE):
    """
    Build the raw (time, log10 mass-loss-rate, wind-velocity) tracks for a set
    of stars from the Ekstrom+2012 grid.

    Args:
        particle_masses: Initial masses (in solar masses, matching the grid's
            ``Mini`` column) of the stars to look up, one track per star.
        rotation: Whether to use the rotating or non-rotating model track.
        track_table: Path to the FITS track table (defaults to the
            Ekstrom+2012 grid bundled alongside this module).

    Returns:
        ``(t_yr, log_mass_rates, vel_scales_kms)``, each of shape
        (n_stars, n_track_points) — the raw track values, NOT yet converted to
        code units. Convert before building ``WindParams.real_params``.
    """
    n = len(particle_masses)
    hdul = fits.open(track_table)
    t_yr, mass_rates, vel_scales = [], [], []
    for idx in range(n):
        t, m, v = wind_parameters(hdul[1].data, particle_masses[idx], rotation)
        t_yr.append(t)
        mass_rates.append(m)
        vel_scales.append(v)

    t_yr = jnp.array(t_yr)
    log_mass_rates = jnp.array(mass_rates)
    vel_scales_kms = jnp.array(vel_scales)
    return t_yr, log_mass_rates, vel_scales_kms


def _piecewise_linear(t, x, y):
    """
    Piecewise-linear interpolation of ``(x, y)`` at time ``t``, with linear
    extrapolation before the first track point and clamping to the last value
    after it.

    Args:
        t: The current time.
        x: The track's time coordinates.
        y: The track's values (f(x)).

    Returns:
        The interpolated value at ``t``.
    """
    idx = jnp.searchsorted(x, t, side="right")
    n = x.shape[0]
    left_idx = jnp.clip(idx - 1, 0, n - 2)
    right_idx = left_idx + 1

    x_left = x[left_idx]
    x_right = x[right_idx]
    y_left = y[left_idx]
    y_right = y[right_idx]

    denom = x_right - x_left
    frac = jnp.where(denom == 0, 0.0, (t - x_left) / denom)
    interp = y_left + frac * (y_right - y_left)

    # left extrapolation: use the slope of the first segment
    first_denom = x[1] - x[0]
    first_slope = jnp.where(first_denom == 0, 0.0, (y[1] - y[0]) / first_denom)
    left_extrap = y[0] + first_slope * (t - x[0])

    right_clamp = y[-1]

    is_left = t < x[0]
    is_right = t > x[-1]
    out = jnp.where(is_left, left_extrap, interp)
    out = jnp.where(is_right, right_clamp, out)

    return out


@jax.jit
def get_current_wind_params(mass_rates_value, vel_scales_value, current_time, time_value):
    """
    Interpolate the per-source (mass-loss rate, wind velocity) tracks to the
    current simulation time.

    Args:
        mass_rates_value: Mass-loss-rate tracks, shape (n_sources, n_track_points).
        vel_scales_value: Wind-velocity tracks, shape (n_sources, n_track_points).
        current_time: The current simulation time (code units, matching
            ``time_value``).
        time_value: Per-source time coordinates, shape (n_sources, n_track_points).

    Returns:
        ``(mass_rates, vel_scales)``, each of shape (n_sources,) — the tracks
        interpolated to ``current_time``.
    """
    interp_fn = lambda x_i, y_i: _piecewise_linear(current_time, x_i, y_i)
    mass_rates = jax.vmap(interp_fn)(time_value, mass_rates_value)
    vel_scales = jax.vmap(interp_fn)(time_value, vel_scales_value)

    return mass_rates, vel_scales
