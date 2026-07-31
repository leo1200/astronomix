"""
A young, ejecta-driven supernova remnant expanding into a uniform ISM.

The companion to ``cassiopeia.py``: the *same* cold, freely-expanding ejecta
(1.5 x 10^51 erg, 3.3 M_sun; flat core + rho ∝ r^-9 envelope), but launched into
a uniform interstellar medium (n = 1 cm^-3) instead of the r^-2 progenitor wind.
The contrast is instructive:

  * into a wind (``cassiopeia.py``): the ambient density falls off as r^-2, the
    forward shock accelerates outward and the swept-up mass grows slowly;
  * into a uniform ISM (here): the forward shock sweeps up mass ∝ r^3, decelerates
    much sooner and the remnant is more compact and more Sedov-Taylor-like, with
    a thinner, denser shell.

Same high-order finite-difference (WENO, RK4-SSP) solver and single-precision
positivity-preserving flux limiter as the rest of the showcase.

Default resolution (128^3) finishes in a few minutes on one GPU. Writes
``figures/young_snr_ism.png``.
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
    freely_expanding_ejecta,
    temperature_K,
    radial_velocity_field,
    snr_figure,
)


# ---------------------------------------------------------------------------
# problem parameters
# ---------------------------------------------------------------------------
BOX_SIZE = 7.0                 # pc, box [-3.5, 3.5]^3
EXPLOSION_ENERGY = 1.5e51       # erg
EJECTA_MASS = 3.3               # Msun
EJECTA_RADIUS = 1.5             # pc, outer ejecta radius at t = 0
N_ISM = 1.0                     # cm^-3, uniform ambient number density
ISM_TEMPERATURE = 1e4 * u.K
MASS_PER_NUCLEUS = 1.4


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
    # ============ ↓ Initial condition (ejecta + ISM) ↓ ==========
    # -------------------------------------------------------------
    shape = (num_cells, num_cells, num_cells)
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(code_units.code_density).value)
    p_per_n = float((const.k_B * ISM_TEMPERATURE / u.cm ** 3).to(code_units.code_pressure).value)
    rho_ism = jnp.ones(shape) * (N_ISM * rho_per_n)
    p_ism = jnp.ones(shape) * (N_ISM * p_per_n)

    fields, info = freely_expanding_ejecta(
        helper_data, code_units, BOX_SIZE, num_cells,
        explosion_energy_erg=EXPLOSION_ENERGY,
        ejecta_mass_msun=EJECTA_MASS,
        ejecta_radius=EJECTA_RADIUS,
        rho_ambient=rho_ism, p_ambient=p_ism,
        mass_per_nucleus=MASS_PER_NUCLEUS,
    )
    if config.mhd:
        z = jnp.zeros(shape)
        fields.update(magnetic_field_x=z, magnetic_field_y=z, magnetic_field_z=z)

    initial_state = construct_primitive_state(
        config=config, registered_variables=registered_variables, **fields
    )
    config = finalize_config(config, initial_state.shape)

    dfloor = float(N_ISM * rho_per_n) * 1e-3
    pfloor = float(N_ISM * p_per_n) * 1e-4
    params = SimulationParams(
        gamma=GAMMA, C_cfl=0.3, t_end=t_end,
        minimum_density=dfloor, minimum_pressure=pfloor,
    )
    # -------------------------------------------------------------
    # ============ ↑ Initial condition (ejecta + ISM) ↑ ==========
    # -------------------------------------------------------------

    t0_yr = float((EJECTA_RADIUS * code_units.code_length / (info["v_max_kms"] * u.km / u.s)).to(u.yr).value)
    age_yr = t0_yr + float((t_end * code_units.code_time).to(u.yr).value)
    print(f"[ism] N={num_cells} {'mhd' if mhd else 'hydro'} dx={info['dx']:.3f}pc "
          f"(ejecta {info['cells_across_ejecta']:.0f} cells) v_max={info['v_max_kms']:.0f} km/s")
    print(f"[ism] E={info['KE_achieved']:.4e}/{info['E_target']:.4e}  "
          f"M_ej={info['M_ej_achieved']:.3f}/{info['M_ej_target']:.3f}  "
          f"n_ISM={N_ISM} cm^-3  evolved to ~{age_yr:.0f} yr")
    return initial_state, config, params, registered_variables, code_units, age_yr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="cells per axis")
    ap.add_argument("--t-end", type=float, default=0.15,
                    help="evolution time (code units ~978 yr); same age as cassiopeia.py "
                         "for a like-for-like contrast (~350 yr)")
    ap.add_argument("--mhd", action="store_true", help="run the FD backend in MHD mode (B=0)")
    args = ap.parse_args()

    print(f"[ism] devices: {jax.devices()}")
    state, config, params, rv, cu, age_yr = build(args.n, args.t_end, mhd=args.mhd)
    snaps = time_integration(state, config, params, rv)
    jax.block_until_ready(snaps)

    fs = np.asarray(snaps.final_state)
    rho = fs[rv.density_index]
    p = fs[rv.pressure_index]
    T = temperature_K(rho, p, cu)
    helper_data = get_helper_data(config)
    r, vr_kms = radial_velocity_field(fs, rv, helper_data, BOX_SIZE, args.n, cu)

    print(f"[ism] final rho[{rho.min():.3e},{rho.max():.3e}] "
          f"T[{np.nanmin(T):.1e},{np.nanmax(T):.1e}]K  "
          f"total_energy={np.asarray(snaps.total_energy)[-1]:.4e}")

    out = snr_figure(
        rho, T, r, vr_kms, BOX_SIZE,
        title=f"Young SNR in a uniform ISM ($n = {N_ISM:.0f}$ cm$^{{-3}}$) "
              f"(high-order FD, ~{age_yr:.0f} yr)",
        out_path=FIGURES_DIR / "young_snr_ism.png",
    )
    print(f"[ism] saved {out}")


if __name__ == "__main__":
    main()
