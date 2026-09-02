"""How much of a GLM scheme's measured ``eta_eff`` is the divergence cleaning?

The budget in ``make_dissipation_figure.py`` forms the magnetic dissipation as
``D_B = T_ideal - dE_B/dt`` with ``T_ideal`` the shell-summed projection of
``curl(v x B)``. A GLM code does not evolve that equation: it evolves

    dB/dt = curl(v x B) - grad psi

so the residual ``D_B`` also contains the work of the cleaning coupling. That is
defensible only if the coupling is small, and since AthenaPK dumps ``psi`` the
question can be settled by measuring it rather than argued:

    T_psi(n) = sum_shell Re[ B-hat*(k) . (-i k psi-hat(k)) ]

which is the exact shell-wise contribution the ideal transfer leaves out. With
``dE_B/dt = T_ideal + T_psi - D_num``, the measured residual is
``D_B = D_num - T_psi``, so ``T_psi`` is precisely the error in reading ``D_B``
as the scheme's own magnetic dissipation.

Needs a run whose ``.phdf`` dumps were kept:

    python athenapk_turb.py --n 64 --scheme plm --seed-field sin --beta 1e6 \\
           --tcross 40 --nsnap 20 --transfer --keep-snapshots \\
           --tag psiprobe_plm_N64 --outdir data/psi
    python measure_glm_psi_term.py --dumps /export/data/lstorcks/mhd_dynamo/athenapk_psiprobe_plm_N64
"""

# general
import argparse
import sys
from pathlib import Path

# numerics
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)      # before any jnp array is made
import jax.numpy as jnp                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_spectral import (_fft_vec, _shell_sum, _wavevectors,  # noqa: E402
                           shell_spectrum)
from athenapk_turb import BOX_SIZE, read_phdf_fields  # noqa: E402
from make_mechanism_table import BAND, SAT_START  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dumps", required=True,
                   help="directory of kept .phdf dumps from a GLM run")
    p.add_argument("--t-cross", type=float, default=0.5)
    p.add_argument("--sat-start", type=float, default=SAT_START)
    p.add_argument("--eta-total", type=float, default=2.03e-3,
                   help="the eta_eff the budget reports for this run, to "
                        "compare the omitted term against")
    args = p.parse_args()

    files = sorted(Path(args.dumps).glob("*.prim.*.phdf"))
    if not files:
        raise SystemExit(f"no .phdf dumps in {args.dumps} (--keep-snapshots)")

    times, T_psi, E_B, psi2, b2 = [], [], [], [], []
    for path in files:
        t, f = read_phdf_fields(path)
        if "magnetic_psi" not in f:
            raise SystemExit(f"{path.name} has no psi field; not a GLM run?")
        n_grid = f["magnetic_psi"].shape[0]
        kx, ky, kz = _wavevectors(n_grid, BOX_SIZE)
        bx, by, bz = (jnp.asarray(f[f"magnetic_field_{i}"], dtype=jnp.float64)
                      for i in (1, 2, 3))
        psi = jnp.asarray(f["magnetic_psi"], dtype=jnp.float64)
        b_hat, psi_hat = _fft_vec(bx, by, bz), jnp.fft.fftn(psi)
        grad_psi = jnp.stack([-1j * kx * psi_hat, -1j * ky * psi_hat,
                              -1j * kz * psi_hat])
        times.append(t)
        T_psi.append(np.asarray(_shell_sum(
            jnp.real(jnp.sum(jnp.conj(b_hat) * grad_psi, axis=0))
            / float(n_grid) ** 6, n_grid)))
        E_B.append(np.asarray(shell_spectrum(bx, by, bz)))
        psi2.append(float(jnp.mean(psi ** 2)))
        b2.append(float(jnp.mean(bx ** 2 + by ** 2 + bz ** 2)))

    order = np.argsort(times)
    tc = np.asarray(times)[order] / args.t_cross
    T_psi, E_B = np.asarray(T_psi)[order], np.asarray(E_B)[order]
    sat = tc >= args.sat_start
    if sat.sum() < 2:
        raise SystemExit(f"only {sat.sum()} dumps past t/t_cross = {args.sat_start}")

    n = np.arange(n_grid // 2 + 1, dtype=float)
    k = 2.0 * np.pi * n
    band = (n / (n_grid / 2) >= BAND[0]) & (n / (n_grid / 2) <= BAND[1])
    eta_psi = (T_psi[sat].mean(0)[band]
               / (2.0 * k[band] ** 2 * E_B[sat].mean(0)[band]))
    mean_eta_psi = float(np.mean(eta_psi))

    print(f"\n{len(files)} dumps, {sat.sum()} in the saturated window, "
          f"band n/n_Nyq = {BAND[0]}-{BAND[1]}")
    print(f"<psi^2> / <B^2> in the saturated state: "
          f"{np.mean(np.asarray(psi2)[order][sat]) / np.mean(np.asarray(b2)[order][sat]):.3e}")
    print(f"\neta implied by the omitted -grad(psi) term alone: {mean_eta_psi:+.3e}")
    print(f"eta_eff the budget reports for this run:            {args.eta_total:+.3e}")
    print(f"ratio:                                              "
          f"{mean_eta_psi / args.eta_total:+.4f}")
    # T_psi enters dE_B/dt with a + sign, so T_psi < 0 is a SINK of magnetic
    # energy. The measured residual is D = D_num - T_psi, so a sink makes the
    # budget *overstate* the scheme's own dissipation by |T_psi|.
    kind = "a sink" if mean_eta_psi < 0 else "a source"
    direction = "over" if mean_eta_psi < 0 else "under"
    print(f"\nThe cleaning coupling is {kind} of magnetic energy, of magnitude "
          f"{abs(mean_eta_psi):.3e} in eta units.\nBecause the budget attributes "
          f"it to the scheme, eta_eff {direction}states the scheme's own "
          f"numerical\nresistivity by "
          f"{abs(100 * mean_eta_psi / args.eta_total):.2f}%.")


if __name__ == "__main__":
    main()
