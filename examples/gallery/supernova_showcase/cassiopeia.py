"""
Cassiopeia A: a young, ejecta-driven supernova remnant in a stellar wind.

A high-order finite-difference (WENO, RK4-SSP) model of the Cas A remnant,
following the setup of Orlando et al. (2021, A&A 645, A66):

  * cold, freely-expanding (homologous) supernova ejecta with an explosion
    (kinetic) energy of 1.5 x 10^51 erg (= 1.5 bethe) and an ejecta mass of
    3.3 M_sun -- the values Orlando et al. adopt for Cas A;
  * expanding into the r^-2 wind of the progenitor star,
        n(r) = n_w (r_fs / r)^2 + n_c ,
    with n_w = 0.8 cm^-3 at r_fs = 2.5 pc (the estimated current shock radius)
    flattening to a uniform n_c = 0.1 cm^-3 at large radii.

The supersonic ejecta drives a forward shock into the wind while the wind drives
a reverse shock back into the ejecta -- the double-shock structure that gives
Cas A its bright main shell (forward-shocked wind + reverse-shocked ejecta)
around a cold, un-shocked freely-expanding interior.

This is a deliberately simplified stand-in for the Orlando et al. model: their
ejecta comes from a full 3D neutrino-driven core-collapse simulation (with
large-scale asymmetries, radioactive decay heating and, in the MHD runs, a
magnetic field), whereas here the ejecta is the standard analytic freely-
expanding profile (a flat inner core joined to a steep rho ∝ r^-9 envelope),
spherically symmetric and normalised to the same mass and energy. What it does
reproduce is the essential Cas A gas dynamics: the forward/reverse-shock shell
and the cold expanding interior, evolved with a consistent high-order scheme.

Runs in single precision (float32): the positivity-preserving flux limiter in
``_common.fd_positivity`` keeps this cold, high-Mach blast stable and
energy-conserving without dropping to double precision or a low-order fallback.

Default resolution (128^3) finishes in a few minutes on one GPU; raise ``--n``
for sharper shells. Writes ``figures/cassiopeia.png``.
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
import astropy.constants as const

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

# shared showcase helpers
from _common import (
    GAMMA,
    FIGURES_DIR,
    snr_code_units,
    make_fd_config,
    centered_radius,
    freely_expanding_ejecta,
    temperature_K,
    radial_velocity_field,
    snr_figure,
)


# ---------------------------------------------------------------------------
# problem parameters (Orlando et al. 2021, Table 1)
# ---------------------------------------------------------------------------
BOX_SIZE = 7.0                 # pc, box [-3.5, 3.5]^3
EXPLOSION_ENERGY = 1.5e51       # erg (= 1.5 bethe)
EJECTA_MASS = 3.3               # Msun
EJECTA_RADIUS = 1.5             # pc, outer ejecta radius at t = 0
# wind: n(r) = N_W (R_FS / r)^2 + N_C   [cm^-3]
N_W, R_FS, N_C = 0.8, 2.5, 0.1
WIND_TEMPERATURE = 1e4 * u.K
MASS_PER_NUCLEUS = 1.4          # mean gas mass per H nucleus, in m_p


def build(num_cells, t_end, mhd=False, num_snapshots=5):
    code_units = snr_code_units()
    snaps = SnapshotSettings(
        return_states=False,
        return_final_state=True,
        return_total_mass=True,
        return_total_energy=True,
        return_internal_energy=True,
        return_kinetic_energy=True,
    )
    config = make_fd_config(BOX_SIZE, num_cells, mhd=mhd,
                            snapshot_settings=snaps, num_snapshots=num_snapshots)
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    # -------------------------------------------------------------
    # ============ ↓ Initial condition (ejecta + wind) ↓ ==========
    # -------------------------------------------------------------
    r, _, _, _ = centered_radius(helper_data, BOX_SIZE, num_cells)
    dx = BOX_SIZE / num_cells
    r_safe = jnp.maximum(r, 0.5 * dx)

    # r^-2 progenitor wind, flattening to a uniform floor density. Cap the
    # cusp inside the ejecta: uncapped, the innermost cells get an immobile
    # ~10^4 cm^-3 wind knot (the ejecta is ADDED onto the ambient) that
    # survives as a spurious bright dot at the box centre.
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(code_units.code_density).value)
    p_per_n_wind = float((const.k_B * WIND_TEMPERATURE / u.cm ** 3).to(code_units.code_pressure).value)
    r_wind = jnp.maximum(r_safe, 0.5 * EJECTA_RADIUS)
    n_wind = N_W * (R_FS / r_wind) ** 2 + N_C
    rho_wind = n_wind * rho_per_n
    p_wind = n_wind * p_per_n_wind

    fields, info = freely_expanding_ejecta(
        helper_data, code_units, BOX_SIZE, num_cells,
        explosion_energy_erg=EXPLOSION_ENERGY,
        ejecta_mass_msun=EJECTA_MASS,
        ejecta_radius=EJECTA_RADIUS,
        rho_ambient=rho_wind, p_ambient=p_wind,
        mass_per_nucleus=MASS_PER_NUCLEUS,
    )
    if config.mhd:
        z = jnp.zeros((num_cells, num_cells, num_cells))
        fields.update(magnetic_field_x=z, magnetic_field_y=z, magnetic_field_z=z)

    initial_state = construct_primitive_state(
        config=config, registered_variables=registered_variables, **fields
    )
    config = finalize_config(config, initial_state.shape)

    # code-unit floors from the un-shocked wind floor (n_c)
    dfloor = float(N_C * rho_per_n) * 1e-3
    pfloor = float(N_C * p_per_n_wind) * 1e-4
    params = SimulationParams(
        gamma=GAMMA, C_cfl=0.3, t_end=t_end,
        minimum_density=dfloor, minimum_pressure=pfloor,
    )
    # -------------------------------------------------------------
    # ============ ↑ Initial condition (ejecta + wind) ↑ ==========
    # -------------------------------------------------------------

    # the free-expansion age implied by R_ej / v_max, and the evolved age
    t0_yr = float((EJECTA_RADIUS * code_units.code_length / (info["v_max_kms"] * u.km / u.s)).to(u.yr).value)
    age_yr = t0_yr + float((t_end * code_units.code_time).to(u.yr).value)
    print(f"[casa] N={num_cells} {'mhd' if mhd else 'hydro'} dx={info['dx']:.3f}pc "
          f"(ejecta {info['cells_across_ejecta']:.0f} cells) v_max={info['v_max_kms']:.0f} km/s")
    print(f"[casa] E={info['KE_achieved']:.4e}/{info['E_target']:.4e}  "
          f"M_ej={info['M_ej_achieved']:.3f}/{info['M_ej_target']:.3f}  "
          f"initial age~{t0_yr:.0f} yr, evolved to ~{age_yr:.0f} yr")
    return initial_state, config, params, registered_variables, code_units, age_yr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="cells per axis")
    ap.add_argument("--t-end", type=float, default=0.15,
                    help="evolution time (code units ~978 yr); default lands the "
                         "remnant near Cas A's ~350 yr age / ~2.5 pc shock radius")
    ap.add_argument("--mhd", action="store_true", help="run the FD backend in MHD mode (B=0)")
    args = ap.parse_args()

    print(f"[casa] devices: {jax.devices()}")
    state, config, params, rv, cu, age_yr = build(args.n, args.t_end, mhd=args.mhd)
    snaps = time_integration(state, config, params, rv)
    jax.block_until_ready(snaps)

    fs = np.asarray(snaps.final_state)
    rho = fs[rv.density_index]
    p = fs[rv.pressure_index]
    T = temperature_K(rho, p, cu)
    helper_data = get_helper_data(config)
    r, vr_kms = radial_velocity_field(fs, rv, helper_data, BOX_SIZE, args.n, cu)

    print(f"[casa] final rho[{rho.min():.3e},{rho.max():.3e}] "
          f"T[{np.nanmin(T):.1e},{np.nanmax(T):.1e}]K  "
          f"total_energy={np.asarray(snaps.total_energy)[-1]:.4e}")

    out = snr_figure(
        rho, T, r, vr_kms, BOX_SIZE,
        title=f"Cassiopeia A: ejecta-driven SNR in an $r^{{-2}}$ wind "
              f"(high-order FD, ~{age_yr:.0f} yr)",
        out_path=FIGURES_DIR / "cassiopeia.png",
    )
    print(f"[casa] saved {out}")


if __name__ == "__main__":
    main()
