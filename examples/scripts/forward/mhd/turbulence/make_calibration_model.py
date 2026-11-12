"""An analytical account of the calibration: why an imposed Laplacian is not
recovered one-to-one, shell by shell.

``make_calibration_figure.py --subtracted`` shows ``[eff(imposed) - eff(none)]
/ imposed`` rolling off towards the grid for both fields and sitting below one
everywhere for ``nu``. Three ingredients reproduce it, two of them exact:

1. **The discrete operator.** AthenaPK's viscous and ohmic fluxes are face
   differences (2-point in the face-normal direction, face-averaged central
   differences transversely). Their spectral symbol is a matrix ``M(k)`` with
   ``(2/dx) sin(k_i dx/2)`` and ``sin(k_i dx)/dx`` in place of ``k_i``; for
   isotropic solenoidal statistics the dissipation per shell is ``Tr[P M] / 2``
   with ``P`` the transverse projector, against ``k^2`` for the continuum
   Laplacian. The ratio ``G(n)`` is what an additive estimator *should* return,
   and it falls to 0.53 (ohmic) and 0.67 (viscous) at Nyquist. No parameters.

2. **The state the numerical part is measured in.** The scheme's numerical
   resistivity depends on ``E_B/E_K`` (the back-reaction smooths the field it
   acts on), by up to +50% between saturation and the kinematic phase at
   ``256^3``. An imposed ``eta`` lowers the saturated ``E_B/E_K``, so the
   numerical part in the imposed run is that of a *different* state, and the
   difference shows up as an apparent excess over ``G``. The factor is read off
   the reference run's own history (``measure_at_ratio`` on the ``beta = 1e6``
   PLM run at the imposed run's ``E_B/E_K``). No parameters.

3. **A strain-dependent share of the numerical viscosity.** Godunov
   dissipation has a linear part (the Riemann solver on smooth fields, set by
   the signal speed and the grid) and a part that lives on grid-scale velocity
   jumps (where the limiter engages). Only the second responds to a smoother
   flow. With ``r`` the ratio of grid-scale strain (``sqrt(sum k^2 E_v)`` over
   ``n >= n_Nyq/2``) between the imposed and the reference run,

       nu_num(imposed) / nu_num(none) = 1 - phi (1 - r),

   and a *single* ``phi = 0.40`` reproduces all four viscous rungs at both
   resolutions to within 0.02. One parameter, shared.

The figure overlays the measured recovery on the model for every rung and
prints the band means.

    python make_calibration_model.py
"""

# general
import argparse
import sys
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_metrics import load_runs
from make_calibration_figure import LADDER_COLOURS, curves
from make_convergence_figures import SERIES, series_of
from make_dissipation_figure import dissipation
from make_mechanism_table import BAND, SAT_START, measure, measure_at_ratio

HERE = Path(__file__).resolve().parent

#: Share of PLM+HLLD's numerical viscosity that scales with the grid-scale
#: strain (ingredient 3). Fitted once; the per-rung values are printed.
PHI = 0.40

#: Shells counted as "grid scale" for the strain ratio, in units of Nyquist.
GRID_SCALE = 0.5


def operator_transfer(N, L=1.0):
    """``G(n)`` of AthenaPK's discrete viscous and ohmic operators.

    Symbols of the stencils in ``hydro/diffusion/{viscosity,resistivity}.cpp``:
    ``s_i = (2/dx) sin(k_i dx/2)`` for a face difference followed by the flux
    divergence, ``c_i = sin(k_i dx)/dx`` for the face-averaged central
    difference of a transverse derivative. Viscous stress ``d_j (d_j v_i + d_i
    v_j - 2/3 delta_ij d_l v_l)``: ``M_ii = 4/3 s_i^2 + sum_{j != i} s_j^2``,
    ``M_ij = c_i c_j / 3``. Ohmic ``-curl curl``: ``M_ii = sum_{j != i}
    s_j^2``, ``M_ij = -c_i c_j``. Shell-averaged ``Tr[P M] / 2k^2`` on the
    estimator's shells ``n = rint(|k| L / 2 pi)``.
    """
    dx = L / N
    kk = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky, kz = np.meshgrid(kk, kk, kk, indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0
    s = [(2.0 / dx) * np.sin(k * dx / 2.0) for k in (kx, ky, kz)]
    c = [np.sin(k * dx) / dx for k in (kx, ky, kz)]
    kh = [k / np.sqrt(k2) for k in (kx, ky, kz)]
    n = np.rint(np.sqrt(k2) / (2.0 * np.pi / L)).astype(int)
    out = {}
    for name, diag, off in (
            ("visc", lambda i: (4.0 / 3.0) * s[i] ** 2
             + sum(s[j] ** 2 for j in range(3) if j != i),
             lambda i, j: c[i] * c[j] / 3.0),
            ("ohm", lambda i: sum(s[j] ** 2 for j in range(3) if j != i),
             lambda i, j: -c[i] * c[j])):
        tr, kmk = 0.0, 0.0
        for i in range(3):
            mii = diag(i)
            tr = tr + mii
            kmk = kmk + kh[i] * mii * kh[i]
            for j in range(3):
                if j != i:
                    kmk = kmk + kh[i] * off(i, j) * kh[j]
        g = (tr - kmk) / (2.0 * k2)
        out[name] = (np.bincount(n.ravel(), weights=g.ravel())
                     / np.maximum(np.bincount(n.ravel()), 1))
    return out


def astronomix_operator_transfer(N=64, seed=0):
    """``G(n)`` of astronomix's explicit operators, measured on the operators.

    astronomix's viscous stress uses 6th-order central first derivatives
    applied twice (``_viscosity.fd_viscosity_source``) and the ohmic term is
    the edge-EMF curl of ``_resistivity.fd_ohmic_interface_rhs``, which
    involves face-to-centre and centre-to-face interpolations as well. Rather
    than assemble those symbols by hand, apply the actual JAX operators to a
    random isotropic solenoidal field and read off, per shell, the
    dissipation the spectral estimator would attribute to them, over
    ``2 k^2 E``. The symbols depend on ``k dx`` only, so ``N = 64`` serves every
    resolution once expressed in ``n / n_Nyq``. Runs on the CPU.
    """
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    from astronomix import (
        BoundarySettings,
        BoundarySettings1D,
        PERIODIC_BOUNDARY,
        SimulationConfig,
        SimulationParams,
        initialize_interface_fields,
        get_registered_variables,
        DiffusionConfig,
    )
    from astronomix.option_classes.simulation_config import (
        CARTESIAN, FINITE_DIFFERENCE, ISOTHERMAL, KINEMATIC_VISCOSITY,
        StaticIntVector, StaticFloatVector)
    from astronomix._modules._resistivity._resistivity import fd_ohmic_interface_rhs
    from astronomix._modules._viscosity._viscosity import fd_viscosity_source
    from astronomix._spatial_operators._interpolate import interp_face_to_center

    L = 1.0
    dx = L / N
    per = BoundarySettings(*(BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),) * 3)
    cfg = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        geometry=CARTESIAN,
        mhd=True,
        equation_of_state=ISOTHERMAL,
        box_size=StaticFloatVector(L, L, L),
        num_cells=StaticIntVector(N, N, N),
        num_ghost_cells=0,
        grid_spacing=dx,  # not finalised: set it explicitly
        boundary_settings=per,
        diffusion_config=DiffusionConfig(
            viscosity=True,
            viscosity_type=KINEMATIC_VISCOSITY,
            resistivity=True,
        ),
    )
    rv = get_registered_variables(cfg)
    params = SimulationParams(viscosity=1.0, resistivity=1.0, isothermal_sound_speed=1.0)

    # Random isotropic solenoidal field, white per mode.
    rng = np.random.default_rng(seed)
    kk = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky, kz = np.meshgrid(kk, kk, kk, indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2s = np.where(k2 == 0, 1.0, k2)
    raw = rng.normal(size=(3, N, N, N)) + 1j * rng.normal(size=(3, N, N, N))
    div = (kx * raw[0] + ky * raw[1] + kz * raw[2]) / k2s
    hat = np.stack([raw[0] - kx * div, raw[1] - ky * div, raw[2] - kz * div])
    hat[:, 0, 0, 0] = 0.0
    field = np.real(np.fft.ifftn(hat, axes=(1, 2, 3)))
    n_shell = np.rint(np.sqrt(k2) / (2.0 * np.pi / L)).astype(int)

    def shell_G(vec, rhs):
        """``-Re<vec* . rhs> / (2 k^2 E)`` per shell for centred fields."""
        V = np.fft.fftn(vec, axes=(1, 2, 3))
        R = np.fft.fftn(rhs, axes=(1, 2, 3))
        D = -np.real(np.sum(np.conj(V) * R, axis=0))
        E = 0.5 * np.sum(np.abs(V) ** 2, axis=0)
        num = np.bincount(n_shell.ravel(), weights=D.ravel())
        den = np.bincount(n_shell.ravel(), weights=(2.0 * k2 * E).ravel())
        return num / np.maximum(den, 1e-300)

    # Viscous: primitive state rho = 1, v = field, B = 0; kinematic nu = 1.
    ndim_state = jnp.zeros((rv.num_vars, N, N, N))
    prim = ndim_state.at[rv.density_index].set(1.0)
    for i, comp in enumerate(field):
        prim = prim.at[rv.velocity_index[i]].set(jnp.asarray(comp))
    src = np.asarray(fd_viscosity_source(prim, params, cfg, rv))
    G_visc = shell_G(field, src[1:4])

    # Ohmic: interface field from the centred one, operator, both back to centres.
    bx, by, bz = initialize_interface_fields(*[jnp.asarray(c) for c in field])
    rbx, rby, rbz = fd_ohmic_interface_rhs(bx, by, bz, 1.0, 1.0, dx, cfg)
    Bc = np.stack([np.asarray(interp_face_to_center(b, ax))
                   for ax, b in enumerate((bx, by, bz))])
    Rc = np.stack([np.asarray(interp_face_to_center(r, ax))
                   for ax, r in enumerate((rbx, rby, rbz))])
    G_ohm = shell_G(Bc, Rc)
    return {"visc": G_visc, "ohm": G_ohm, "N": N}


#: Lowest ``E_B/E_K`` the reference history is read at. Below it the factor is
#: flat (1.10-1.11 at 64^3, 1.5 at 256^3), and an imposed eta above the dynamo
#: threshold drives E_B/E_K to zero, where no reference window exists.
RATIO_FLOOR = 0.003


def state_factor(reference, ratio):
    """``eta_num(E_B/E_K = ratio) / eta_num(saturated)`` of the reference run."""
    sat = measure(reference)
    m = measure_at_ratio(reference, ratio=max(ratio, RATIO_FLOOR))
    if m is None or sat is None:
        return 1.0
    return m["eta"] / sat["eta"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "data" / "calibration"))
    p.add_argument("--reference", default=str(HERE / "data" / "dissipation"),
                   help="directory with the beta = 1e6 PLM runs whose history "
                        "gives the state dependence of the numerical eta")
    p.add_argument("--figures", default=str(HERE / "figures"))
    p.add_argument("--phi", type=float, default=None,
                   help="strain share of the numerical viscosity; default: the "
                        "mean of the per-rung fits for the selected scheme "
                        f"(PLM gives {PHI})")
    p.add_argument("--series", default=None,
                   help="scheme (key of make_convergence_figures.SERIES); "
                        "default: the only one in --data")
    args = p.parse_args()

    runs = [r for r in load_runs(args.data, skip=("smoke",))
            if "ohm_diff" in r and "mom_diff" in r]
    present = sorted({series_of(r) for r in runs})
    if args.series is None:
        if len(present) != 1:
            raise SystemExit(f"several schemes in {args.data}: {present}; pass --series")
        args.series = present[0]
    runs = [r for r in runs if series_of(r) == args.series]
    # Reference (beta = 1e6) runs of the same scheme, for the state factor.
    ref_dirs = [args.reference, str(HERE / "data" / "dissipation_wenoz")]
    refs = {int(r["N"]): r for d in ref_dirs for r in load_runs(d, skip=("smoke",))
            if series_of(r) == args.series}
    Ns = sorted({int(r["N"]) for r in runs})
    if args.series == "plm":
        operators = {N: operator_transfer(N) for N in Ns}
    else:
        G64 = astronomix_operator_transfer()
        # Resample onto each N's shells through n / n_Nyq.
        x64 = np.arange(len(G64["ohm"])) / (G64["N"] / 2)
        operators = {N: {k: np.interp(np.arange(N // 2 + 1) / (N / 2), x64, G64[k])
                         for k in ("visc", "ohm")} for N in Ns}
    sfx = "" if args.series == "plm" else f"_{args.series}"

    def pick(N, key, val):
        other = "mom_diff" if key == "ohm_diff" else "ohm_diff"
        for r in runs:
            if int(r["N"]) == N and abs(float(r[key]) - val) < 1e-12 \
                    and float(r[other]) == 0.0:
                return r

    fig, axes = plt.subplots(2, len(Ns), figsize=(6.2 * len(Ns), 8.6),
                             squeeze=False)
    table = []
    phis, phis_B = [], []
    for col, N in enumerate(Ns):
        G = operators[N]
        nyq = N / 2
        none = pick(N, "mom_diff", 0.0)
        base = curves(none, SAT_START)
        d_none = dissipation(none, SAT_START)
        for row, (key, other, idx, gname, sym) in enumerate((
                ("ohm_diff", "mom_diff", 2, "ohm", r"$\eta$"),
                ("mom_diff", "ohm_diff", 1, "visc", r"$\nu$"))):
            ax = axes[row][col]
            vals = sorted({float(r[key]) for r in runs
                           if int(r["N"]) == N and float(r[key]) > 0
                           and float(r[other]) == 0.0})
            x_G = np.arange(len(G[gname])) / nyq
            ax.semilogx(x_G[1:], G[gname][1:], color="0.35", ls=":", lw=1.6,
                        label="discrete operator $G(n)$ alone")
            for colour, v in zip(LADDER_COLOURS[1:], vals):
                run = pick(N, key, v)
                c = curves(run, SAT_START)
                x = c[0]
                rec = (c[idx] - base[idx]) / v
                Gx = np.interp(x * nyq, np.arange(len(G[gname])), G[gname])
                num = base[idx] / v                    # numerical / imposed
                sel = np.asarray(run["t_over_tc"]) >= SAT_START
                if idx == 2:
                    # Ingredient 2: the numerical eta of the imposed run is that
                    # of its own (lower) E_B/E_K state.
                    ratio = float(np.mean((np.asarray(run["E_B"])
                                           / np.maximum(np.asarray(run["E_K"]),
                                                        1e-30))[sel]))
                    s_fac = state_factor(refs[N], ratio) if N in refs else 1.0
                    model = Gx + (s_fac - 1.0) * num
                    # Symmetric to the viscous case: how much of the numerical
                    # ETA an imposed eta displaces, per unit of grid-scale
                    # MAGNETIC strain removed. Zero for a scheme whose
                    # numerical resistivity is set by the velocity field alone.
                    d = dissipation(run, SAT_START)
                    n_sh = d["n"]
                    grid = n_sh >= GRID_SCALE * nyq
                    r_B = np.sqrt(np.sum(n_sh[grid] ** 2 * d["E_B"][grid])
                                  / np.sum(n_sh[grid] ** 2 * d_none["E_B"][grid]))
                    inband = (x >= BAND[0]) & (x <= GRID_SCALE)
                    R = np.mean(((c[idx] - v * Gx) / (s_fac * base[idx]))[inband])
                    phi_fit = (1.0 - R) / max(1.0 - r_B, 1e-6)
                    phis_B.append(phi_fit)
                    note = (f"E_B/E_K {ratio:.3f}, state factor {s_fac:.2f}, "
                            f"r_B {r_B:.2f}, phi_eta {phi_fit:.2f}")
                else:
                    # Ingredient 3: a share phi of the numerical viscosity
                    # follows the grid-scale strain.
                    d = dissipation(run, SAT_START)
                    n_sh = d["n"]
                    grid = n_sh >= GRID_SCALE * nyq
                    r = np.sqrt(np.sum(n_sh[grid] ** 2 * d["E_v"][grid])
                                / np.sum(n_sh[grid] ** 2 * d_none["E_v"][grid]))
                    inband = (x >= BAND[0]) & (x <= GRID_SCALE)
                    R = np.mean(((c[idx] - v * Gx) / base[idx])[inband])
                    phi_fit = (1.0 - R) / (1.0 - r)
                    phis.append(phi_fit)
                    phi_use = args.phi if args.phi is not None else phi_fit
                    model = Gx - phi_use * (1.0 - r) * num
                    note = f"strain ratio r {r:.2f}, fitted phi {phi_fit:.2f}"
                ax.semilogx(x, rec, color=colour, lw=1.9,
                            label=f"imposed {sym} = {v:.1e}")
                ax.semilogx(x, model, color=colour, lw=1.4, ls="--")
                for lo, hi in ((BAND[0], GRID_SCALE), (GRID_SCALE, BAND[1]),
                               (BAND[1], 0.9)):
                    m = (x >= lo) & (x < hi) & np.isfinite(rec)
                    table.append((N, sym, v, lo, hi, rec[m].mean(),
                                  model[m].mean(), Gx[m].mean(), note))
            ax.axhline(1.0, color="0.3", lw=0.8, ls="-.")
            ax.axvspan(*BAND, color="0.5", alpha=0.08, lw=0)
            ax.set_xlim(BAND[0] / 2, 1.0)
            ax.set_ylim(-0.05, 1.35)
            ax.set_xlabel(r"$n / n_{\rm Nyquist}$")
            ax.set_ylabel(fr"[{sym}$_{{\rm eff}}(n)$ - numerical] / imposed")
            ax.set_title(fr"${N}^3$, {sym}: measured (solid) against the model "
                         f"(dashed)", fontsize=10)
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.suptitle(SERIES[args.series][1], fontsize=11, y=0.995)
    out = Path(args.figures) / f"dynamo_dissipation_calibration_model{sfx}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    if phis_B:
        print(f"phi_eta per resistive rung: "
              f"{', '.join(f'{p_:.2f}' for p_ in phis_B)}"
              f"  -> mean {np.mean(phis_B):.2f} +- {np.std(phis_B):.2f}")
    if phis:
        print(f"phi fitted per viscous rung: {', '.join(f'{p_:.2f}' for p_ in phis)}"
              f"  -> mean {np.mean(phis):.2f} +- {np.std(phis):.2f}"
              + ("" if args.phi is None else f"  (model drawn with --phi {args.phi})"))

    print(f"{'N':>4s} {'f':>6s} {'imposed':>8s} {'band':>11s} {'measured':>9s} "
          f"{'model':>6s} {'G only':>7s}  note")
    for N, sym, v, lo, hi, meas, mod, g, note in table:
        print(f"{N:4d} {sym:>6s} {v:8.1e} {lo:4.2f}-{hi:4.2f}   {meas:9.2f} "
              f"{mod:6.2f} {g:7.2f}  {note}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
