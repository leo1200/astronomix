"""Non-equilibrium ionization: ion fractions from ``(kT_e, n_e t)``.

Cas A's shocked plasma has ``n_e t ~ 1e11 cm^-3 s``, an order of magnitude short
of the ``~1e12`` an ion needs to reach collisional equilibrium at these
temperatures. It is therefore UNDER-ionized: at ``kT_e = 3 keV`` equilibrium
would have oxygen fully stripped and iron in the He-like state, while the real
plasma still holds O VII/VIII and Fe XVII-XX. That is not a subtle correction to
the spectrum -- it moves emission from the continuum back into lines, softens
the whole band, and it is why a CIE model of this remnant comes out too hard.

The measurement that motivated this module: the synthetic spectrum produced with
CIE emissivities, compared with Chandra in the same aperture through the same
response, is 0.30x at 0.5-1.5 keV and 2.9x at Fe-K. Under-ionized plasma is
exactly that signature.

Nothing here integrates an ionization network. The simulation already carries
the ionization age per parcel (``density_time``, converted by
``_plasma.ionization_age``), and for a parcel shocked once the ion populations
are a function of ``(kT_e, n_e t)`` alone. So this tabulates that function with
AtomDB's eigenvector solution, once, and interpolates it per cell.

Two assumptions are worth stating:

* **single-shock history.** The tabulated solution starts from neutral gas and
  ionizes at a constant electron temperature. A parcel that was shocked, cooled
  and re-shocked is not described by its ``n_e t`` alone. For the forward and
  reverse shocks of a 350 yr remnant this is the standard approximation.
* **T_e constant since shocking.** The parcel's CURRENT ``T_e`` is used for the
  whole ionization history. Since ``T_e`` rises with time behind the shock (the
  electrons are still equilibrating), this slightly OVER-estimates the
  ionization -- in the same direction as the CIE error, so it is conservative.

Requires ``pyatomdb`` and the AtomDB data files, but only to BUILD the table;
the table is cached and the forward model reads it without pyatomdb.
"""

# general
import os
from pathlib import Path

# numerics
import numpy as np

#: where the cached table lives (next to the other big data, not in the repo)
TABLE_PATH = Path(os.environ.get(
    "CASA_NEI_TABLE",
    "/export/data/lstorcks/supernova_showcase/nei_ion_fractions.npz"))

#: the elements the forward model emits, by atomic number
ELEMENTS = {"He": 2, "O": 8, "Ne": 10, "Mg": 12, "Si": 14, "S": 16, "Ar": 18,
            "Ca": 20, "Fe": 26}

#: table grid. kT_e spans everything that emits in the X-ray band; n_e t spans
#: from "just shocked" to well past equilibrium, so the table's upper edge IS
#: the CIE limit and can be checked against it.
KT_GRID = np.logspace(np.log10(0.05), np.log10(30.0), 48)      # keV
NET_GRID = np.logspace(7.0, 14.0, 57)                          # cm^-3 s


#: AtomDB's eigenvector files are tabulated on this temperature grid (K).
EIGEN_TE_GRID = np.logspace(4.0, 9.0, 1251)
KBOLTZ_KEV_PER_K = 8.617385e-8


def _eigen_solution(Z, kT_keV, tau, init_pop):
    """Ion fractions from AtomDB's eigenvector decomposition of the NEI matrix.

    The ionization balance obeys ``dn/d(n_e t) = A(T_e) n``, and AtomDB ships
    the eigenvalues and left/right eigenvectors of ``A`` per temperature, so the
    solution is exact and explicit:

        ``n(tau) = n_eq + VR^T [ exp(lambda tau) * (VL (n_0 - n_eq)) ]``

    This is done here rather than through ``pyatomdb.apec.return_ionbal``
    because that routine's NEI branch builds a ``numpy.matrix`` and then assigns
    ``fspectmp[i]`` (a 1x1 matrix) into a float array, which raises under
    numpy >= 2. The equilibrium branch is unaffected, and :func:`_self_check`
    checks this implementation against it.

    Args:
        Z: Atomic number.
        kT_keV: Electron temperature (keV), scalar.
        tau: Ionization ages (cm^-3 s), array.
        init_pop: Ion populations before the shock, shape ``(Z + 1,)``.

    Returns:
        Array of shape ``(len(tau), Z + 1)``.
    """
    import pyatomdb.atomdb as atomdb

    d = atomdb.get_data(Z, False, "eigen")
    kT_list = EIGEN_TE_GRID * KBOLTZ_KEV_PER_K
    # 1251 points over five decades is 0.004 dex, so snapping to the nearest is
    # far below every other error here
    k = int(np.argmin(np.abs(kT_list - kT_keV)))
    feqb = np.asarray(d[1].data["FEQB"][k], dtype=np.float64)
    eig = np.asarray(d[1].data["EIG"][k], dtype=np.float64)
    vr = np.asarray(d[1].data["VR"][k], dtype=np.float64).reshape(Z, Z)
    vl = np.asarray(d[1].data["VL"][k], dtype=np.float64).reshape(Z, Z)

    work = np.asarray(init_pop, dtype=np.float64)[1:] - feqb[1:]
    fspec = vl @ work                                   # (Z,)
    # exp(lambda tau) can overflow for the fast-decaying modes at large tau;
    # those modes have already died, so clipping the exponent is exact
    decay = np.exp(np.clip(np.outer(np.asarray(tau, dtype=np.float64), eig),
                           -700.0, 700.0))              # (n_tau, Z)
    frac = np.zeros((len(np.atleast_1d(tau)), Z + 1))
    frac[:, 1:] = (fspec * decay) @ vr + feqb[1:]

    # the eigenvector reconstruction can leave small negatives and a sum
    # slightly off one; neutral takes up the remainder, as in AtomDB's own code
    np.clip(frac, 0.0, None, out=frac)
    over = frac[:, 1:].sum(axis=1) > 1.0
    if np.any(over):
        s = frac[over, 1:].sum(axis=1, keepdims=True)
        frac[over, 1:] /= s
    frac[:, 0] = np.clip(1.0 - frac[:, 1:].sum(axis=1), 0.0, None)
    return frac


def build_table(path=TABLE_PATH, *, elements=ELEMENTS, kt=KT_GRID, net=NET_GRID):
    """Tabulate ion fractions on the ``(kT_e, n_e t)`` grid.

    Starts from neutral gas, which is the right initial condition for material
    crossing a shock into cold ejecta or cold circumstellar wind. Writes an npz
    of ``f[element]`` with shape ``(n_kT, n_net, Z + 1)``.
    """
    out = {}
    for el, Z in elements.items():
        init = np.zeros(Z + 1)
        init[0] = 1.0                       # neutral before the shock
        f = np.zeros((len(kt), len(net), Z + 1))
        for i, kT in enumerate(kt):
            f[i] = _eigen_solution(Z, float(kT), net, init)
        out[el] = f
        print(f"[nei] {el:2s} (Z = {Z:2d}) tabulated")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, kt=kt, net=net,
                        **{f"f_{el}": v for el, v in out.items()})
    print(f"[nei] wrote {path}")
    return out


def load_table(path=TABLE_PATH):
    """Read the cached table, building it if it is not there yet."""
    if not Path(path).exists():
        print(f"[nei] no table at {path} -- building it (needs pyatomdb)")
        build_table(Path(path))
    d = np.load(path)
    return (np.asarray(d["kt"]), np.asarray(d["net"]),
            {k[2:]: np.asarray(d[k]) for k in d.files if k.startswith("f_")})


def interpolate_fractions(f, kt_grid, net_grid, kT_e, net):
    """Bilinear interpolation of the ion fractions in ``log kT_e`` and ``log n_e t``.

    Clipped at both ends of both axes: below the grid the plasma is barely
    ionized, above it the table already holds the equilibrium solution, and in
    both cases the clip is the physically correct limit rather than an
    extrapolation.

    Returns an array of shape ``(Z + 1,) + kT_e.shape``.
    """
    lx = np.clip(np.log(np.asarray(kT_e, dtype=np.float64)),
                 np.log(kt_grid[0]), np.log(kt_grid[-1]))
    ly = np.clip(np.log(np.maximum(np.asarray(net, dtype=np.float64), 1e-30)),
                 np.log(net_grid[0]), np.log(net_grid[-1]))
    gx, gy = np.log(kt_grid), np.log(net_grid)

    i = np.clip(np.searchsorted(gx, lx) - 1, 0, len(gx) - 2)
    j = np.clip(np.searchsorted(gy, ly) - 1, 0, len(gy) - 2)
    tx = ((lx - gx[i]) / (gx[i + 1] - gx[i]))[None, ...]
    ty = ((ly - gy[j]) / (gy[j + 1] - gy[j]))[None, ...]

    # move the ion axis to the front so the weights broadcast over the cells
    f00 = np.moveaxis(f[i, j], -1, 0)
    f10 = np.moveaxis(f[i + 1, j], -1, 0)
    f01 = np.moveaxis(f[i, j + 1], -1, 0)
    f11 = np.moveaxis(f[i + 1, j + 1], -1, 0)
    return ((1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10
            + (1 - tx) * ty * f01 + tx * ty * f11)


def mean_charge(f_ion):
    """Mean charge per nucleus from an ion-fraction array (ion axis first)."""
    z = np.arange(f_ion.shape[0]).reshape((-1,) + (1,) * (f_ion.ndim - 1))
    return (f_ion * z).sum(axis=0)


def significant_ions(f, kt_grid, net_grid, *, kt_range, net_range, threshold):
    """Which ions ever matter, over the part of the grid the remnant occupies.

    Every ion carried costs a full 3D field in the forward model, so this keeps
    only those reaching ``threshold`` of the element somewhere in the region of
    ``(kT_e, n_e t)`` the simulation actually populates. The bare nucleus is
    kept whenever it appears: it has no lines but it does radiate continuum.
    """
    ik = (kt_grid >= kt_range[0]) & (kt_grid <= kt_range[1])
    jn = (net_grid >= net_range[0]) & (net_grid <= net_range[1])
    peak = f[np.ix_(ik, jn)].reshape(-1, f.shape[-1]).max(axis=0)
    return np.where(peak >= threshold)[0], peak


def _self_check():
    """The table must reproduce collisional equilibrium at large ionization age."""
    import pyatomdb.apec as apec

    kt, net, tables = load_table()
    print("[nei] mean charge at kT_e = 2 keV:")
    for el in ("O", "Si", "Fe"):
        Z = ELEMENTS[el]
        row = []
        for target in (1e10, 1e11, 1e12, 1e13):
            f = interpolate_fractions(tables[el], kt, net,
                                      np.array([2.0]), np.array([target]))
            row.append(f"n_e t={target:.0e}: <Z>={mean_charge(f).item():5.2f}")
        cie = np.ravel(apec.return_ionbal(Z, 2.0, teunit="keV"))
        row.append(f"CIE(direct): <Z>={float((cie * np.arange(Z + 1)).sum()):5.2f}")
        print(f"    {el:2s} (Z={Z:2d})  " + "   ".join(row))
        # the top of the ionization-age axis must BE equilibrium
        f_eq = interpolate_fractions(tables[el], kt, net,
                                     np.array([2.0]), np.array([net[-1]]))
        err = float(np.max(np.abs(f_eq[:, 0] - cie)))
        assert err < 0.02, (el, err)
    print("[nei] self-check passed (the large-n_e t limit reproduces CIE)")


if __name__ == "__main__":
    _self_check()
