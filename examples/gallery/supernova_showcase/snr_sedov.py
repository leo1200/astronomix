"""
Sedov-Taylor supernova-remnant blast, following the Athena++ ``snr.athinput``.

A high-order finite-difference (WENO, RK4-SSP) rendering of the supernova-remnant
problem of Guo, Kim & Stone (2025, ApJ 990, 49), distributed as
``cassiopax/athena_inputs/snr.athinput``:

  * a triply-periodic 64 pc box (Athena nx = 256, [-32, 32]^3), ideal gas
    gamma = 5/3, CFL 0.3;
  * a uniform ambient medium (damb = 0.862, code temperature tamb = 93.13);
  * a supernova remnant deposited in a small central sphere: a thermal energy
    etot_snr ~ 3.47e9 (= 1e51 erg = 1 bethe) and a mass mass_snr, on top of the
    ambient -- a Sedov-Taylor thermal bomb;
  * optional radiative cooling (the athinput's ``ism_cooling``), here the
    Schure et al. (2009) ISM cooling curve applied with the unconditionally
    stable implicit method.

Where the Athena setup uses ``ppm4 + hllc`` finite-volume reconstruction with a
first-order flux correction (``fofc``), this uses astronomix's high-order WENO
finite-difference solver instead -- the "consistent high-order scheme" -- kept
stable on the Sedov-strength bomb by the positivity-preserving flux limiter (no
finite-volume path, no first-order fallback). See ``_common.py`` for the
solver-configuration rationale and ``README.md`` for the full athinput mapping.

Notes / deviations from the athinput, none of which change the Sedov dynamics:
  * the injection sphere is a well-resolved, tanh-tapered region (default 3 pc,
    renormalised so the deposited mass/energy are exact) rather than the sharp
    1 pc top-hat, which a high-order scheme cannot resolve;
  * no thermal conduction (the athinput sets conductivity = 0 anyway) and no
    photoelectric heating term (astronomix has none) -- the temperature floor
    stands in for the heating balance;
  * default t_end (0.01 Myr) keeps the blast well inside the periodic box; the
    athinput's tlim = 1 Myr would overrun a 64 pc periodic domain.

Default resolution (128^3) finishes in a few minutes on one GPU. Writes
``figures/snr_sedov.png``.
"""

# ==== GPU selection ====
# Under a scheduler (pq) CUDA_VISIBLE_DEVICES is already pinned to the
# assigned GPU -- only fall back to autocvd for interactive runs.
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# units and constants
from astropy import units as u

# astronomix containers / functions
from astronomix import (
    SimulationParams,
    SnapshotSettings,
    get_registered_variables,
    get_helper_data,
    finalize_config,
    construct_primitive_state,
    time_integration,
)
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
    PIECEWISE_POWER_LAW,
    IMPLICIT_COOLING,
)
from astronomix._modules._cooling._cooling_tables import schure_cooling

# shared showcase helpers
from _common import (
    GAMMA,
    FIGURES_DIR,
    athena_code_units,
    make_fd_config,
    centered_radius,
    tapered_sphere_weight,
    temperature_K,
    radial_velocity_field,
    snr_figure,
)


# ---------------------------------------------------------------------------
# snr.athinput <problem> / <units> values (Athena code units)
# ---------------------------------------------------------------------------
BOX_SIZE = 64.0                 # pc, [-32, 32]^3
DAMB = 0.8620                   # ambient density
TAMB = 93.125130                # ambient (Athena) code temperature T = p / rho
MASS_SNR = 197.85148            # deposited SNR mass
ETOT_SNR = 3.4691068e9          # deposited SNR thermal energy (~1e51 erg)
T_COLD = 2.589                  # cold-gas temperature -> cooling floor
DFLOOR, PFLOOR = 1e-4, 1e-2     # athinput dfloor / pfloor
INJECTION_RADIUS = 3.0          # pc (resolved tapered sphere; see module docstring)
# cooling composition (mu = 0.618 as in the athinput <units> block)
HYDROGEN_MASS_FRACTION = 0.698
METAL_MASS_FRACTION = 0.02
MU = 0.618


def build(num_cells, t_end, cooling=True, mhd=False, num_snapshots=5):
    code_units = athena_code_units()

    cooling_config = None
    cooling_params = None
    if cooling:
        cooling_config = CoolingConfig(
            cooling=True,
            cooling_method=IMPLICIT_COOLING,
            cooling_curve_config=CoolingCurveConfig(cooling_curve_type=PIECEWISE_POWER_LAW),
        )
        cooling_params = CoolingParams(
            hydrogen_mass_fraction=HYDROGEN_MASS_FRACTION,
            metal_mass_fraction=METAL_MASS_FRACTION,
            floor_temperature=MU * T_COLD,          # astronomix rescaled T~ = mu * T
            cooling_curve_params=schure_cooling(code_units),
        )

    snaps = SnapshotSettings(
        return_states=False,
        return_final_state=True,
        return_total_mass=True,
        return_total_energy=True,
        return_internal_energy=True,
        return_kinetic_energy=True,
    )
    config = make_fd_config(BOX_SIZE, num_cells, mhd=mhd,
                            cooling_config=cooling_config,
                            snapshot_settings=snaps, num_snapshots=num_snapshots)
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    # -------------------------------------------------------------
    # ============ ↓ Initial condition (thermal bomb) ↓ ===========
    # -------------------------------------------------------------
    r, _, _, _ = centered_radius(helper_data, BOX_SIZE, num_cells)
    dx = BOX_SIZE / num_cells
    cell_vol = dx ** 3

    # tapered spherical injection weight, renormalised to unit sum so the total
    # deposited mass (MASS_SNR) and thermal energy (ETOT_SNR) are exact.
    w = tapered_sphere_weight(r, INJECTION_RADIUS, dx, taper_cells=3.0)
    w = w / jnp.sum(w)

    p_amb = DAMB * TAMB                                # p = rho * T (Athena convention)
    rho = DAMB + MASS_SNR * w / cell_vol
    # thermal-energy deposit: e = p/(gamma-1) -> dp = (gamma-1) dE / V
    p = p_amb + (GAMMA - 1.0) * ETOT_SNR * w / cell_vol

    z = jnp.zeros((num_cells, num_cells, num_cells))
    fields = dict(density=rho, velocity_x=z, velocity_y=z, velocity_z=z, gas_pressure=p)
    if config.mhd:
        fields.update(magnetic_field_x=z, magnetic_field_y=z, magnetic_field_z=z)

    initial_state = construct_primitive_state(
        config=config, registered_variables=registered_variables, **fields
    )
    config = finalize_config(config, initial_state.shape)

    params = SimulationParams(
        gamma=GAMMA, C_cfl=0.3, t_end=t_end,
        minimum_density=DFLOOR, minimum_pressure=PFLOOR,
        cooling_params=cooling_params,
    )
    # -------------------------------------------------------------
    # ============ ↑ Initial condition (thermal bomb) ↑ ===========
    # -------------------------------------------------------------

    age_yr = float((t_end * code_units.code_time).to(u.yr).value)
    print(f"[snr] N={num_cells} {'mhd' if mhd else 'hydro'} cooling={cooling} "
          f"dx={dx:.3f}pc (R_inj={INJECTION_RADIUS/dx:.1f} cells) "
          f"E_dep~{ETOT_SNR:.3e} evolved to ~{age_yr:.0f} yr")
    return initial_state, config, params, registered_variables, code_units, age_yr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="cells per axis")
    ap.add_argument("--t-end", type=float, default=0.01, help="end time (Myr)")
    ap.add_argument("--no-cooling", action="store_true", help="disable radiative cooling")
    ap.add_argument("--mhd", action="store_true", help="run the FD backend in MHD mode (B=0)")
    args = ap.parse_args()

    print(f"[snr] devices: {jax.devices()}")
    state, config, params, rv, cu, age_yr = build(
        args.n, args.t_end, cooling=not args.no_cooling, mhd=args.mhd
    )
    snaps = time_integration(state, config, params, rv)
    jax.block_until_ready(snaps)

    fs = np.asarray(snaps.final_state)
    rho = fs[rv.density_index]
    p = fs[rv.pressure_index]
    T = temperature_K(rho, p, cu)
    helper_data = get_helper_data(config)
    r, vr_kms = radial_velocity_field(fs, rv, helper_data, BOX_SIZE, args.n, cu)

    te = np.asarray(snaps.total_energy)
    print(f"[snr] final rho[{rho.min():.3e},{rho.max():.3e}] "
          f"T[{np.nanmin(T):.1e},{np.nanmax(T):.1e}]K  "
          f"total_energy {te[0]:.3e} -> {te[-1]:.3e} (cooling radiates it away)")

    cool_tag = "adiabatic" if args.no_cooling else "with cooling"
    out = snr_figure(
        rho, T, r, vr_kms, BOX_SIZE,
        title=f"Sedov-Taylor SNR (snr.athinput, high-order FD, {cool_tag}, ~{age_yr:.0f} yr)",
        out_path=FIGURES_DIR / "snr_sedov.png",
    )
    print(f"[snr] saved {out}")


if __name__ == "__main__":
    main()
