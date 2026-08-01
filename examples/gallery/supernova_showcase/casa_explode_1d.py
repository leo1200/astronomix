"""
Explode the real progenitor: shock propagation from the mass cut to breakout.

:mod:`casa_progenitor` prepares a KEPLER presupernova star -- strips it to a
Type IIb, excises the iron core -- and this drives a thermal bomb through what
is left. The output is an ejecta profile that was *computed* from a real stellar
structure rather than fitted to the answer, which is the whole point: the inner
slope, outer slope and core radius that ``casa_calibrate_1d.py`` currently takes
as free parameters are all consequences of the shock crossing this star.

**BLOCKED, and the reason is physical rather than numerical.** A presupernova
core is held up by ELECTRON DEGENERACY, and this solver has an ideal-gas
equation of state. Measured on s16.0, the fraction of KEPLER's pressure that
ideal gas plus radiation can account for is:

    m = 0.002 Msun   2.7 %       m = 1.40 Msun   22 %
    m = 0.70  Msun   6.0 %       m = 2.32 Msun   53 %
    m = 3.98  Msun   80 %

so in the iron core 97 % of the support is degeneracy. Handing those (rho, p)
to a gamma = 5/3 solver describes a different star: the implied ideal-gas
temperature is 2.7e11 K against KEPLER's 7.3e9 K, a factor 37, and the model is
nowhere near hydrostatic equilibrium under the EOS it is being integrated with.
It disassembles on the first steps whatever the bomb does, which is exactly the
NaN this script produced. The bomb is not the problem and no amount of
resolution, bomb mass or bomb geometry fixes it.

The gas only becomes ideal-dominated outside ~3 Msun, so the options are: add a
degenerate-electron EOS (the correct fix, a solver project); or start the
calculation above the degenerate region and inject an already-established shock
there, accepting that the innermost ~2 Msun of Si/O/Fe ejecta is imposed rather
than computed. Everything below is written and verified and waits on that
decision.

Three further things about this setup are worth stating up front, because each
one is a place where a plausible-looking wrong answer is easy to produce.

**The bomb energy is not the explosion energy.** The material above the mass cut
is bound by 5.3e50 erg, a quarter of Cas A's calibrated 2.09e51 erg kinetic
energy. The bomb therefore has to deposit the sum, and ``--energy`` is the
KINETIC energy wanted at infinity, with the binding energy added internally and
reported.

**Gravity is an external monopole, held static.** In spherical symmetry gravity
is exactly ``-G M(<r) / r``, so no Poisson solve is needed (and the solver's
self-gravity is Cartesian-only anyway). It is frozen at the initial enclosed
mass, which is good while the shock crosses -- the crossing is fast compared
with the free-fall time of the layers ahead of it -- and irrelevant afterwards,
because the ejecta are unbound. It would NOT be good for a failed explosion.

**The grid has no inner boundary.** The solver's 1D spherical grid starts at
r = 0, and the collapsed core is not on it: the region inside the mass cut is
filled with floor material and the core's mass appears only in the potential.
That inner hole evacuates, exactly as the origin cell does in
``casa_calibrate_1d``, and is handled the same way -- with the per-step hard
floor. The mass it injects is reported, because a silent floor there would
quietly add ejecta.

Usage (GPU, minutes)::

    ./run.sh casa_progenitor.py --model s16.0 --strip 0.1 --save casa_prog_s16_IIb.npz
    ./run.sh casa_explode_1d.py casa_prog_s16_IIb.npz --energy-51 2.09 --save casa_ej_s16.npz
"""

# ==== GPU selection ====
import os
import sys
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

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
    construct_primitive_state,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix import SPHERICAL, FINITE_VOLUME
from astronomix.option_classes.simulation_config import (
    BoundarySettings1D,
    GravityConfig,
    PositivityConfig,
    POSITIVITY_HARD_FLOOR,
    OPEN_BOUNDARY,
    REFLECTIVE_BOUNDARY,
)

# shared showcase helpers
from _common import GAMMA, snr_code_units

MSUN = float((1.0 * u.Msun).to(u.g).value)
G_CGS = float(const.G.cgs.value)


# =============================================================================
# ============ ↓ Mapping the star onto the grid ↓ =============================
# =============================================================================
def map_star(star, r_grid, code_units, floor_density_cgs):
    """Interpolate a KEPLER zone structure onto the uniform radial grid.

    KEPLER zones are Lagrangian and enormously non-uniform -- the innermost is
    1e8 cm across while the outermost spans 1e12 -- so the mapping is done in
    LOG radius on log density and log pressure, which is the only way the steep
    core does not turn into a staircase. Outside the star, and inside the mass
    cut, the grid is filled with floor material.
    """
    r_star, rho_star, p_star = star["radius"], star["density"], star["pressure"]

    lr = np.log(np.maximum(r_grid, 1e-30))
    lrs = np.log(r_star)
    rho = np.exp(np.interp(lr, lrs, np.log(rho_star)))
    p = np.exp(np.interp(lr, lrs, np.log(p_star)))

    # Outside the star: floor material standing in for the wind. The blast
    # breaks out into it, which is what it is there for.
    outside = r_grid > r_star[-1]
    rho = np.where(outside, floor_density_cgs, rho)
    p = np.where(outside, float(p_star[-1]) * 1e-6, p)

    # Inside the mass cut: CONTINUE the innermost stellar zone rather than
    # filling with floor. The cavity is tiny (~1e-4 Msun at these densities) so
    # the added mass is irrelevant, but the alternative is a vacuum sitting
    # directly against the bomb -- a pressure ratio of ~1e30 across one
    # interface, which NaNs the Riemann solve on the first step. A smooth
    # continuation costs nothing and removes the discontinuity entirely.
    inside = r_grid < r_star[0]
    rho = np.where(inside, float(rho_star[0]), rho)
    p = np.where(inside, float(p_star[0]), p)
    return rho, p


def enclosed_mass(r_grid, rho_cgs, mass_cut_g=0.0):
    """M(<r) from the fluid on the grid.

    ``mass_cut_g`` is kept at zero: the whole star is on the grid, so its own
    mass is the whole potential. Adding the cut mass on top would double-count
    the core and leave the star out of the hydrostatic equilibrium KEPLER
    handed us.
    """
    dr = r_grid[1] - r_grid[0]
    shell = 4.0 * np.pi * r_grid ** 2 * rho_cgs * dr
    return mass_cut_g + np.cumsum(shell)


def monopole_potential(r_grid, m_enc, code_units):
    """``phi(r) = -G M(<r) / r`` in code units, softened at the origin.

    Exact in spherical symmetry, so the Cartesian Poisson solver is not needed
    (and could not be used here anyway). Softening below the first cell keeps
    the r -> 0 divergence off the grid; nothing physical lives there, because
    the collapsed core is represented by its mass, not by fluid.
    """
    r_soft = np.maximum(r_grid, r_grid[1])
    phi_cgs = -G_CGS * m_enc / r_soft
    per_code = float((1.0 * code_units.code_velocity ** 2).to(u.erg / u.g).value)
    return phi_cgs / per_code


# =============================================================================
# ============ ↑ Mapping the star onto the grid ↑ =============================
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("progenitor", help="casa_progenitor.py --save npz")
    ap.add_argument("--n", type=int, default=200000, help="radial cells")
    ap.add_argument("--r-max", type=float, default=None, metavar="CM",
                    help="outer radius (default: 1.5x the stellar radius)")
    ap.add_argument("--energy-51", type=float, default=2.09,
                    help="KINETIC energy at infinity in 1e51 erg (Cas A's "
                         "calibrated value). The binding energy is added to the "
                         "bomb on top of this")
    ap.add_argument("--bomb-mass", type=float, default=0.2, metavar="MSUN",
                    help="mass of the shell the bomb heats, just above the mass cut")
    ap.add_argument("--t-end", type=float, default=None, metavar="S",
                    help="stop time (default: 3x the shock crossing estimate)")
    ap.add_argument("--nsnap", type=int, default=21, help="snapshots")
    ap.add_argument("--save", default=None, help="write the ejecta profile npz")
    args = ap.parse_args()

    code_units = snr_code_units()
    d = np.load(args.progenitor)
    star = {k: d[k] for k in ("mass", "radius", "velocity", "density",
                              "pressure", "specific_energy", "dm")}
    mass_cut_g = float(d["mass_cut"]) * MSUN
    r_star = float(star["radius"][-1])
    m_ej = (star["mass"][-1] - mass_cut_g) / MSUN
    print(f"[boom] {str(d['model'])}: {m_ej:.3f} Msun of ejecta above a "
          f"{mass_cut_g / MSUN:.3f} Msun mass cut, R = {r_star:.3e} cm")

    # ---- the energy budget -------------------------------------------------
    e_bind = float(np.sum((G_CGS * star["mass"] / star["radius"]
                           - star["specific_energy"]) * star["dm"]))
    e_kin_target = args.energy_51 * 1e51
    e_bomb = e_kin_target + e_bind
    print(f"[boom] energy: {e_kin_target:.3e} erg wanted as kinetic at infinity "
          f"+ {e_bind:.3e} erg to unbind = {e_bomb:.3e} erg deposited "
          f"({e_bomb / e_kin_target:.2f}x)")

    # ---- the grid ----------------------------------------------------------
    r_max = args.r_max if args.r_max else 1.5 * r_star
    box = float((r_max * u.cm).to(code_units.code_length).value)
    dr_cgs = r_max / args.n
    print(f"[boom] grid: {args.n} cells over {r_max:.3e} cm, dr = {dr_cgs:.3e} cm "
          f"({r_star / dr_cgs:.0f} cells across the star, "
          f"{max(float(star['radius'][0]) / dr_cgs, 0.0):.1f} inside the mass cut)")

    snaps = SnapshotSettings(return_states=False, return_final_state=True,
                             return_total_mass=True, return_total_energy=True,
                             return_kinetic_energy=True, return_internal_energy=True)
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        geometry=SPHERICAL,
        dimensionality=1,
        box_size=box,
        num_cells=args.n,
        # r = 0 is reflecting because it is the centre of a sphere, not a wall;
        # the outer edge is open so the blast can leave without reflecting back
        # into the ejecta once it breaks out.
        boundary_settings=BoundarySettings1D(REFLECTIVE_BOUNDARY, OPEN_BOUNDARY),
        # same two requirements as casa_calibrate_1d: the steep envelope breaks
        # plain MUSCL reconstruction, and the evacuating origin needs the
        # per-step floor
        first_order_fallback=True,
        positivity_config=PositivityConfig(
            per_step_mode=POSITIVITY_HARD_FLOOR, nan_safe=True, vacuum_rest=True),
        gravity_config=GravityConfig(external_potential=True),
        return_snapshots=True, snapshot_settings=snaps,
        num_snapshots=args.nsnap, progress_bar=True,
    )
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    r_code = np.asarray(helper_data.geometric_centers)
    r_cgs = r_code * float((1.0 * code_units.code_length).to(u.cm).value)

    # ---- map the star ------------------------------------------------------
    rho_floor_cgs = float(star["density"][-1]) * 1e-3
    rho_cgs, p_cgs = map_star(star, r_cgs, code_units, rho_floor_cgs)

    m_enc = enclosed_mass(r_cgs, rho_cgs)
    phi = monopole_potential(r_cgs, m_enc, code_units)
    print(f"[boom] mapped mass on the grid: "
          f"{(m_enc[-1] - mass_cut_g) / MSUN:.3f} Msun "
          f"(star has {m_ej:.3f}); potential from {phi.min():.3e} to {phi.max():.3e}")

    # ---- the bomb: thermal energy in a shell just above the mass cut --------
    dm_grid = 4.0 * np.pi * r_cgs ** 2 * rho_cgs * dr_cgs
    # Take cells outward from the mass cut until the requested shell mass is
    # reached, and never fewer than one. Selecting instead by "cumulative mass
    # from r = 0 below the target" silently returns an EMPTY shell whenever a
    # single cell already outweighs it, which is the normal case at the base of
    # a presupernova core -- the density there is 1e7 g/cm^3.
    r_cut = float(np.interp(mass_cut_g, star["mass"], star["radius"]))
    above = np.where(r_cgs >= r_cut)[0]
    if above.size == 0:
        raise SystemExit("no grid cell lies outside the mass cut: raise --n")
    n_take = max(1, int(np.searchsorted(np.cumsum(dm_grid[above]),
                                        args.bomb_mass * MSUN)))
    bomb = np.zeros_like(r_cgs, dtype=bool)
    bomb[above[:n_take]] = True
    if n_take < 8:
        print(f"[boom] WARNING: the bomb spans only {n_take} cell(s). A thermal "
              f"bomb concentrated into one cell is a delta function in pressure "
              f"against its neighbour and NaNs the first Riemann solve; raise "
              f"--n (the cells are Eulerian, so resolution is what spreads it).")
    vol = 4.0 * np.pi * r_cgs ** 2 * dr_cgs
    p_bomb = (GAMMA - 1.0) * e_bomb / float(np.sum(vol[bomb]))
    p_cgs = np.where(bomb, p_cgs + p_bomb, p_cgs)
    print(f"[boom] bomb: {np.sum(bomb)} cells, "
          f"{np.sum(dm_grid[bomb]) / MSUN:.4f} Msun, "
          f"r = {r_cut:.3e} to {r_cgs[bomb][-1]:.3e} cm, p = {p_bomb:.3e} erg/cm^3")

    # ---- to code units and go ---------------------------------------------
    rho = jnp.asarray(rho_cgs / float((1.0 * code_units.code_density).to(u.g / u.cm ** 3).value))
    p = jnp.asarray(p_cgs / float((1.0 * code_units.code_pressure).to(u.erg / u.cm ** 3).value))
    v = jnp.zeros_like(rho)

    state = construct_primitive_state(
        config=config, registered_variables=rv,
        density=rho, velocity_x=v, gas_pressure=p, gamma=GAMMA)

    # shock crossing estimate: the bomb's sound speed across the star
    v_est = np.sqrt(2.0 * e_bomb / (m_ej * MSUN))
    t_end_s = args.t_end if args.t_end else 3.0 * r_star / v_est
    t_end = float((t_end_s * u.s).to(code_units.code_time).value)
    print(f"[boom] v_est = {v_est / 1e5:.0f} km/s, running to "
          f"t = {t_end_s:.3e} s ({t_end:.4e} code)")

    params = SimulationParams(
        t_end=t_end, gamma=GAMMA,
        gravitational_constant=float(
            (const.G.cgs.value * u.cm ** 3 / u.g / u.s ** 2)
            .to(code_units.code_length ** 3 / code_units.code_mass
                / code_units.code_time ** 2).value),
        gravitational_potential=jnp.asarray(phi),
    )

    result = time_integration(state, config, params, rv)
    jax.block_until_ready(result)

    tm = np.asarray(result.total_mass)
    te = np.asarray(result.total_energy)
    ke = np.asarray(result.kinetic_energy)
    print(f"[boom] total_mass:    {np.array2string(tm, precision=5, max_line_width=200)}")
    print(f"[boom] kinetic_energy:{np.array2string(ke, precision=5, max_line_width=200)}")
    good = tm[np.isfinite(tm) & (tm > 0)]
    if good.size and good.max() > 1.01 * good[0]:
        print(f"[boom] MASS NOT CONSERVED: {good[0]:.4e} -> {good.max():.4e}; the "
              f"floor in the evacuated core is injecting ejecta and nothing "
              f"downstream is trustworthy")

    fs = np.asarray(result.final_state)
    rho_f = fs[rv.density_index]
    v_f = fs[rv.velocity_index]
    p_f = fs[rv.pressure_index]
    if not np.all(np.isfinite(rho_f)) or not np.all(np.isfinite(p_f)):
        print("[boom] REFUSING to report: the final state carries non-finite cells")
        return

    ke_cgs = float(ke[-1]) * float((1.0 * code_units.code_energy).to(u.erg).value)
    print(f"[boom] final kinetic energy {ke_cgs:.3e} erg "
          f"({ke_cgs / e_kin_target:.3f}x the {e_kin_target:.2e} target); "
          f"v_max = {float(np.max(v_f)) * float((1.0 * code_units.code_velocity).to(u.km / u.s).value):.0f} km/s")

    if args.save:
        np.savez_compressed(args.save, r=r_cgs, rho=rho_f, v=v_f, p=p_f,
                            t_end_s=t_end_s, mass_cut=mass_cut_g / MSUN,
                            e_bomb=e_bomb, e_bind=e_bind, model=str(d["model"]))
        print(f"[boom] saved {args.save}")


if __name__ == "__main__":
    main()
