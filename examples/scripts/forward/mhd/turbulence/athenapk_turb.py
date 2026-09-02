"""Driven subsonic MHD turbulence in AthenaPK: 2nd- and 3rd-order GLM-MHD.

The AthenaPK half of the dynamo convergence study; see ``dynamo_convergence.py``
for the astronomix half and for the physical setup both codes share. This script
writes a Parthenon input file for AthenaPK's built-in ``turbulence`` problem
generator, runs the pre-built A100 binary, and reduces the resulting ``.phdf``
dumps through the *same* estimator the astronomix side uses
(``_mhd_spectral.snapshot_spectra``).

Two schemes are exposed, both at the CFL number AthenaPK's own convergence tests
use for them (``pytests/mhd/data/athenapk/``):

    --scheme plm   PLM + VL2  + HLLD, nghost 2, cfl 0.3   (2nd order)
    --scheme ppm   PPM + RK3  + HLLD, nghost 3, cfl 0.4   (3rd order)

Notes on how the configuration is matched to astronomix:

    - AthenaPK has no isothermal EOS, so the isothermal box is emulated with
      ``gamma = 1.0001`` and ``p0 = rho0 a^2 / gamma`` -- the standard AthenaPK
      turbulence setup (``inputs/turbulence.in``). The sound speed is then ``a``
      and the ~10x thermalisation of the injected energy over 30 crossing times
      moves the temperature (and hence the Mach number) by < 0.1%, where an
      ideal-gas box at gamma = 5/3 would heat until the Mach number fell 3x.
    - ``sol_weight = 1`` (purely solenoidal) and ``corr_time`` equal to
      astronomix's OU correlation time. The driving band is AthenaPK's
      ``kpeak = 2`` few-modes set (30 modes, ``1 <= |n| <= 3``); astronomix drives
      the smooth ``k^6 exp(-8k/kpk)`` spectrum peaked at ``n = 1.5``. The two
      envelopes are not identical -- there is no way to make them so -- so the
      codes are matched on the *achieved flow* instead: ``calibrate_athenapk.py``
      tunes ``accel_rms`` until AthenaPK sits at astronomix's stationary
      ``v_rms``, exactly as the hydrodynamic study matches Mach numbers.
    - Single meshblock covering the whole grid, one GPU, so the timing is the
      solver's and not the block-boundary machinery's.

    python athenapk_turb.py --n 128 --scheme plm --accel-rms 3.5
"""

# general
import argparse
import re
import shutil
import subprocess
import sys
import time as walltime
from pathlib import Path

# numerics
import numpy as np
import h5py
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_spectral import SCALAR_NAMES, SPECTRUM_NAMES, snapshot_spectra, shell_numbers

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RUN_ROOT = Path("/export/data/lstorcks/mhd_dynamo")

#: Double-precision A100 build. AthenaPK's near-isothermal trick puts the whole
#: thermal energy budget at p/(gamma-1) ~ 4e4 while the kinetic energy is ~0.5,
#: so the pressure has to be recovered from a difference of numbers that differ
#: by five orders of magnitude -- single precision has no headroom for that.
ATHENAPK_BIN = Path("/export/home/lstorcks/athena/athenapk/build-a100/bin/athenaPK")
ATHENAPK_BIN_SP = Path("/export/home/lstorcks/athena/athenapk/build-a100-sp/bin/athenaPK")

#: Shared physical setup — must stay in sync with dynamo_convergence.py.
BOX_SIZE = 1.0
RHO0 = 1.0
L_INJ = 0.5
VRMS_TARGET = 1.0

#: (reconstruction, integrator, nghost, cfl) per scheme label.
SCHEMES = {
    "plm": ("plm", "vl2", 2, 0.3, "AthenaPK PLM+VL2 (2nd)"),
    "ppm": ("ppm", "rk3", 3, 0.4, "AthenaPK PPM+RK3 (3rd)"),
    # Controls: the 3rd-order limiter and the 5th-order reconstruction, both on
    # RK3, to separate "reconstruction order" from "this particular limiter".
    "limo3": ("limo3", "rk3", 3, 0.4, "AthenaPK LimO3+RK3 (3rd)"),
    "wenoz": ("wenoz", "rk3", 4, 0.4, "AthenaPK WENO-Z+RK3 (5th)"),
}

#: The 30 driving modes of AthenaPK's own turbulence input deck (kpeak = 2,
#: 1 <= |n| <= 3), inlined so a run is reproducible from this file alone.
MODES = """\
k_1_0	= +2
k_1_1	= -1
k_1_2	= +0
k_2_0	= +1
k_2_1	= +0
k_2_2	= +2
k_3_0	= +1
k_3_1	= +1
k_3_2	= -1
k_4_0	= +2
k_4_1	= +0
k_4_2	= +1
k_5_0	= +0
k_5_1	= +0
k_5_2	= -1
k_6_0	= +1
k_6_1	= -1
k_6_2	= -2
k_7_0	= +0
k_7_1	= +0
k_7_2	= -2
k_8_0	= +1
k_8_1	= +0
k_8_2	= -1
k_9_0	= +0
k_9_1	= +2
k_9_2	= +1
k_10_0	= +0
k_10_1	= -1
k_10_2	= +2
k_11_0	= +0
k_11_1	= +0
k_11_2	= +2
k_12_0	= +0
k_12_1	= +2
k_12_2	= -1
k_13_0	= +2
k_13_1	= +1
k_13_2	= +1
k_14_0	= +1
k_14_1	= -1
k_14_2	= +0
k_15_0	= +0
k_15_1	= -1
k_15_2	= -1
k_16_0	= +1
k_16_1	= +0
k_16_2	= +1
k_17_0	= +0
k_17_1	= -1
k_17_2	= +1
k_18_0	= +0
k_18_1	= +1
k_18_2	= +0
k_19_0	= +1
k_19_1	= -1
k_19_2	= +1
k_20_0	= +2
k_20_1	= +1
k_20_2	= -1
k_21_0	= +0
k_21_1	= -1
k_21_2	= -2
k_22_0	= +2
k_22_1	= -1
k_22_2	= +1
k_23_0	= +0
k_23_1	= +1
k_23_2	= +1
k_24_0	= +1
k_24_1	= -2
k_24_2	= +1
k_25_0	= +1
k_25_1	= -2
k_25_2	= +0
k_26_0	= +1
k_26_1	= +2
k_26_2	= +0
k_27_0	= +1
k_27_1	= -2
k_27_2	= -1
k_28_0	= +2
k_28_1	= -1
k_28_2	= -1
k_29_0	= +1
k_29_1	= +2
k_29_2	= -1
k_30_0	= +1
k_30_1	= +1
k_30_2	= +2
"""

OUTPUT2 = """\
<parthenon/output2>
file_type  = hdf5
variables  = prim
dt         = {dt_snap}
id         = prim
single_precision_output = true
"""

#: Explicit Laplacian diffusion, used only by the calibration runs that check the
#: measured ``nu_eff`` / ``eta_eff`` against a coefficient that is known exactly.
DIFFUSION = """\
<diffusion>
integrator        = unsplit
{viscosity}{resistivity}
"""

ATHINPUT = """\
# Driven subsonic MHD turbulence (small-scale dynamo) -- astronomix/AthenaPK
# convergence study. Generated by athenapk_turb.py, do not edit.

<comment>
problem   = subsonic driven MHD turbulence, seed field at beta = {beta:g}

<job>
problem_id = turbulence

<parthenon/output1>
file_type  = hst
dt         = {dt_hst}

{output2}
<parthenon/time>
cfl        = {cfl}
nlim       = -1
tlim       = {tlim}
integrator = {integrator}
ncycle_out = 1000
ncycle_out_mesh = -100000

<parthenon/mesh>
nghost     = {nghost}
nx1        = {n}
x1min      = 0.0
x1max      = {box}
ix1_bc     = periodic
ox1_bc     = periodic

nx2        = {n}
x2min      = 0.0
x2max      = {box}
ix2_bc     = periodic
ox2_bc     = periodic

nx3        = {n}
x3min      = 0.0
x3max      = {box}
ix3_bc     = periodic
ox3_bc     = periodic

packs_per_rank = 1

<parthenon/meshblock>
nx1        = {mb}
nx2        = {mb}
nx3        = {mb}

<hydro>
fluid          = glmmhd
eos            = adiabatic
riemann        = hlld
reconstruction = {reconstruction}
gamma          = {gamma}
glmmhd_alpha   = {glm_alpha}

{diffusion}
<problem/turbulence>
rho0         = {rho0}
p0           = {p0}
b0           = {b0}
b_config     = {b_config}
kpeak        = 2.0
corr_time    = {corr_time}
rseed        = {rseed}
sol_weight   = 1.0        # purely solenoidal driving
accel_rms    = {accel_rms}
num_modes    = {num_modes}

<modes>
{modes}
"""


# -------------------------------------------------------------
# ============== ↓ Parthenon output readers ↓ ==================
# -------------------------------------------------------------
def read_hst(path):
    """Parse a Parthenon ``.hst`` history file into ``{column: array}``."""
    header = None
    with open(path) as fh:
        lines = fh.readlines()
    for line in lines:
        if line.startswith("#") and "[1]=" in line:
            header = re.findall(r"\[\d+\]=(\S+)", line)
            break
    if header is None:
        raise RuntimeError(f"no column header in {path}")
    rows = np.array([[float(x) for x in ln.split()]
                     for ln in lines if not ln.startswith("#") and ln.strip()])
    # A restart or an overwritten file can repeat times; keep the last of each.
    _, keep = np.unique(rows[:, 0][::-1], return_index=True)
    rows = rows[::-1][keep]
    return {name: rows[:, i] for i, name in enumerate(header)}


def read_phdf_fields(path):
    """Read a Parthenon ``.phdf`` and return ``(time, {field: (N,N,N) array})``.

    Meshblocks are reassembled onto the global grid from their logical
    locations, and every field is transposed from Parthenon's ``(k, j, i)``
    storage into the ``(x, y, z)`` order the spectral estimator expects.
    """
    with h5py.File(path, "r") as f:
        info = f["Info"].attrs
        time = float(info["Time"])
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in info["ComponentNames"]]
        mb = np.asarray(info["MeshBlockSize"])          # (nx1, nx2, nx3)
        root = np.asarray(info["RootGridSize"])[:3]     # (nx1, nx2, nx3)
        prim = f["prim"][...]                           # (nblk, nvar, k, j, i)
        loc = np.asarray(f["LogicalLocations"])         # (nblk, 3) in (x, y, z)
        levels = np.asarray(f["Levels"])
    if np.any(levels != levels[0]):
        raise RuntimeError(f"{path} is refined; this study runs on a uniform grid")

    nx, ny, nz = (int(v) for v in root)
    out = {}
    for ivar, name in enumerate(names):
        key = name.replace("prim_", "")
        glob = np.empty((nx, ny, nz), dtype=prim.dtype)
        for b in range(prim.shape[0]):
            i0, j0, k0 = (int(loc[b, d]) * int(mb[d]) for d in range(3))
            block = prim[b, ivar].transpose(2, 1, 0)    # (k,j,i) -> (i,j,k)
            glob[i0:i0 + mb[0], j0:j0 + mb[1], k0:k0 + mb[2]] = block
        out[key] = glob
    return time, out


def reduce_run(run_dir, sound_speed, grid_spacing, deconvolve=True,
               transfer=False, gamma=None, dealias=False,
               slice_series=False):
    """Reduce every ``prim`` dump in ``run_dir`` through the shared estimator."""
    files = sorted(p for p in Path(run_dir).glob("*.prim.*.phdf")
                   if "final" not in p.name)
    files += sorted(Path(run_dir).glob("*.prim.final.phdf"))
    times, scalars, spectra_raw, spectra_dec, eb_slices = [], [], [], [], []
    for path in files:
        t, fields = read_phdf_fields(path)
        args = [jnp.asarray(fields[k]) for k in (
            "density", "velocity_1", "velocity_2", "velocity_3",
            "magnetic_field_1", "magnetic_field_2", "magnetic_field_3")]
        sc, sp = snapshot_spectra(*args, sound_speed, grid_spacing,
                                  dealias=dealias,
                                  deconvolve_cell_average=False,
                                  transfer=transfer, gamma=gamma,
                                  pressure=(jnp.asarray(fields["pressure"])
                                            if transfer else None))
        times.append(t)
        scalars.append(np.asarray(sc))
        spectra_raw.append(np.asarray(sp))
        if slice_series:
            z_mid = fields["density"].shape[2] // 2
            eb_slices.append(np.float32(0.5) * sum(
                fields[f"magnetic_field_{i}"][:, :, z_mid].astype(np.float32) ** 2
                for i in (1, 2, 3)))
        if deconvolve:
            _, spd = snapshot_spectra(*args, sound_speed, grid_spacing,
                                      dealias=dealias,
                                      deconvolve_cell_average=True,
                                      transfer=transfer, gamma=gamma,
                                      pressure=(jnp.asarray(fields["pressure"])
                                                if transfer else None))
            spectra_dec.append(np.asarray(spd))
        print(f"    reduced {path.name}  t={t:.4f}", flush=True)
    order = np.argsort(times)
    result = dict(times=np.asarray(times)[order],
                  scalars=np.asarray(scalars)[order],
                  spectra=np.asarray(spectra_raw)[order])
    if deconvolve:
        result["spectra_deconv"] = np.asarray(spectra_dec)[order]
    if slice_series:
        result["EB_slice_series"] = np.asarray(eb_slices)[order]
    # Mid-plane slices of the last dump, for the qualitative figure (the
    # astronomix side stores the same three under --save-slices).
    _, last = read_phdf_fields(files[int(np.argmax(times))])
    z = last["density"].shape[2] // 2
    result["rho_slice"] = last["density"][:, :, z]
    result["EB_slice"] = 0.5 * sum(
        last[f"magnetic_field_{i}"][:, :, z] ** 2 for i in (1, 2, 3))
    result["EK_slice"] = 0.5 * last["density"][:, :, z] * sum(
        last[f"velocity_{i}"][:, :, z] ** 2 for i in (1, 2, 3))
    return result
# -------------------------------------------------------------
# ============== ↑ Parthenon output readers ↑ ==================
# -------------------------------------------------------------


#: Largest meshblock AthenaPK runs here. A single 256^3 block overflows the
#: Kokkos team-scratch limit ("could not find a valid execution configuration"),
#: so 256^3 is decomposed into eight 128^3 blocks -- which is how AthenaPK is
#: normally run anyway, and the block-boundary cost that carries is a real part
#: of what the code costs at that size.
MAX_MESHBLOCK = 128


def meshblock_size(args):
    """Block size for a run: the whole grid where that fits, else MAX_MESHBLOCK."""
    if args.meshblock > 0:
        return args.meshblock
    return min(args.n, MAX_MESHBLOCK)


def write_athinput(path, args, a, p0, b0, tlim, dt_snap, dt_hst):
    reconstruction, integrator, nghost, cfl, _ = SCHEMES[args.scheme]
    path.write_text(ATHINPUT.format(
        n=args.n, mb=meshblock_size(args),
        box=BOX_SIZE, rho0=RHO0, p0=p0, b0=b0, beta=args.beta,
        gamma=args.gamma, cfl=args.cfl if args.cfl > 0 else cfl,
        integrator=integrator, reconstruction=reconstruction, nghost=nghost,
        tlim=tlim, dt_snap=dt_snap, dt_hst=dt_hst,
        corr_time=args.tau, accel_rms=args.accel_rms, rseed=args.rseed,
        b_config={"uniform": 0, "sin": 2}[args.seed_field],
        num_modes=args.num_modes,
        modes="\n".join(MODES.splitlines()[:3 * args.num_modes]) + "\n",
        output2=(OUTPUT2.format(dt_snap=dt_snap) if args.nsnap > 0 else ""),
        diffusion=diffusion_block(args), glm_alpha=args.glm_alpha,
    ))


def diffusion_block(args):
    """The ``<diffusion>`` stanza, or nothing when both coefficients are zero."""
    if args.mom_diff <= 0.0 and args.ohm_diff <= 0.0:
        return ""
    visc = res = ""
    if args.mom_diff > 0.0:
        visc = ("viscosity         = isotropic\n"
                "viscosity_coeff   = fixed\n"
                f"mom_diff_coeff_code = {args.mom_diff:.8g}\n")
    if args.ohm_diff > 0.0:
        res = ("resistivity       = ohmic\n"
               "resistivity_coeff = fixed\n"
               f"ohm_diff_coeff_code = {args.ohm_diff:.8g}\n")
    return DIFFUSION.format(viscosity=visc, resistivity=res)


def run_athenapk(binary, input_file, run_dir):
    """Run AthenaPK in ``run_dir``; return ``(stdout, wall_s, cycles, throughput)``."""
    t0 = walltime.time()
    proc = subprocess.run([str(binary), "-i", str(input_file)],
                          cwd=run_dir, capture_output=True, text=True)
    wall = walltime.time() - t0
    (Path(run_dir) / "athenapk.log").write_text(proc.stdout + "\n=== stderr ===\n"
                                                + proc.stderr)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"athenaPK exited {proc.returncode}")
    cycles = _grep_float(proc.stdout, r"cycle=(\d+)", last=True)
    thr = _grep_float(proc.stdout, r"zone-cycles/wallsecond\s*=\s*([\d.eE+\-]+)")
    solver_wall = _grep_float(proc.stdout, r"walltime used\s*=\s*([\d.eE+\-]+)")
    return proc.stdout, wall, cycles, thr, solver_wall


def _grep_float(text, pattern, last=False):
    hits = re.findall(pattern, text)
    if not hits:
        return float("nan")
    return float(hits[-1] if last else hits[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=128, help="cells per dimension")
    p.add_argument("--scheme", choices=list(SCHEMES), default="plm")
    p.add_argument("--mturb", type=float, default=0.5, help="target turbulent Mach number")
    p.add_argument("--beta", type=float, default=1e6, help="initial plasma beta")
    p.add_argument("--seed-field", choices=("uniform", "sin"), default="uniform",
                   help="uniform: b_config = 0, the net-flux seed. sin: "
                        "b_config = 2, B_x ~ sin(2 pi z / L) at the same "
                        "magnetic energy and zero net flux (see the README)")
    p.add_argument("--tcross", type=float, default=40.0, help="run length in crossing times")
    p.add_argument("--accel-rms", type=float, default=3.5,
                   help="driving amplitude; calibrate_athenapk.py tunes this")
    p.add_argument("--tau", type=float, default=0.5, help="OU correlation time")
    p.add_argument("--gamma", type=float, default=1.0001, help="near-isothermal gamma")
    p.add_argument("--cfl", type=float, default=-1.0, help="override the scheme's CFL")
    p.add_argument("--nsnap", type=int, default=40,
                   help="hdf5 dumps over the run; 0 disables hdf5 output "
                        "entirely (for clean timing runs -- no spectra are then "
                        "produced, only the .hst time series)")
    p.add_argument("--num-modes", type=int, default=30,
                   help="driving modes. 30 is AthenaPK's own deck; a timing "
                        "control uses 1 to isolate what the explicit 30-mode "
                        "inverse transform costs per cycle")
    p.add_argument("--nhst", type=int, default=800, help="history rows over the run")
    p.add_argument("--meshblock", type=int, default=-1,
                   help="meshblock size; <=0 auto-selects the largest block that "
                        "runs (see MAX_MESHBLOCK)")
    p.add_argument("--mom-diff", type=float, default=0.0,
                   help="explicit isotropic kinematic viscosity in code units "
                        "(calibration runs only; 0 disables)")
    p.add_argument("--ohm-diff", type=float, default=0.0,
                   help="explicit ohmic resistivity in code units "
                        "(calibration runs only; 0 disables)")
    p.add_argument("--rseed", type=int, default=20190729)
    p.add_argument("--single-precision", action="store_true",
                   help="use the single-precision build (see ATHENAPK_BIN_SP)")
    p.add_argument("--glm-alpha", type=float, default=0.1,
                   help="Dedner divergence-cleaning damping strength "
                        "(AthenaPK default 0.1). Control for how much of the "
                        "measured eta_eff is the cleaning sink.")
    p.add_argument("--slice-series", action="store_true",
                   help="keep the mid-plane magnetic-energy slice from every "
                        "dump, not just the last (for the animation)")
    p.add_argument("--dealias", action="store_true",
                   help="form the transfer spectra on a 3/2-refined grid "
                        "(Orszag), the control for the aliasing systematic")
    p.add_argument("--transfer", action="store_true",
                   help="also record the ideal spectral transfer (see "
                        "dynamo_convergence.py --transfer)")
    p.add_argument("--keep-snapshots", action="store_true",
                   help="keep the .phdf dumps after reducing them")
    p.add_argument("--tag", type=str, default="", help="suffix for the output file name")
    p.add_argument("--outdir", type=str, default=str(DATA_DIR))
    args = p.parse_args()

    binary = ATHENAPK_BIN_SP if args.single_precision else ATHENAPK_BIN
    if not binary.is_file():
        raise FileNotFoundError(binary)

    a = 1.0 / args.mturb                        # sound speed, as in astronomix
    p0 = RHO0 * a ** 2 / args.gamma             # so sqrt(gamma p0 / rho0) == a
    b0 = float(np.sqrt(2.0 * (a ** 2 * RHO0) / args.beta))
    t_cross = L_INJ / VRMS_TARGET
    tlim = args.tcross * t_cross

    tag = args.tag or f"{args.scheme}_N{args.n}"
    run_dir = RUN_ROOT / f"athenapk_{tag}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    input_file = run_dir / "turb_dynamo.in"
    write_athinput(input_file, args, a, p0, b0, tlim,
                   dt_snap=tlim / max(args.nsnap, 1), dt_hst=tlim / args.nhst)

    reconstruction, integrator, nghost, cfl_default, label = SCHEMES[args.scheme]
    cfl = args.cfl if args.cfl > 0 else cfl_default
    print(f"[AthenaPK N={args.n} {args.scheme}] {reconstruction}+{integrator}+hlld "
          f"cfl={cfl} meshblock={meshblock_size(args)}^3 "
          f"gamma={args.gamma} a={a} p0={p0:.6g} b0={b0:.4g} "
          f"accel_rms={args.accel_rms} tlim={tlim} ({args.tcross} t_cross)", flush=True)

    stdout, wall, cycles, thr, solver_wall = run_athenapk(binary, input_file, run_dir)
    print(f"[AthenaPK N={args.n} {args.scheme}] wall={wall:.1f} s "
          f"(parthenon walltime={solver_wall:.1f} s) cycles={cycles:.0f} "
          f"zone-cycles/s={thr:.3e}", flush=True)

    hst = read_hst(next(run_dir.glob("*.hst")))
    if args.nsnap <= 0:
        # No hdf5 dumps, so no spectra and no per-snapshot scalars -- but the
        # .hst series is the point of such a run (a clean wall clock, or a
        # high-cadence E_B(t) for the growth-rate fit), so it is still written.
        print(f"[AthenaPK N={args.n} {args.scheme}] hdf5 output disabled "
              f"(--nsnap 0): .hst series only, no spectra", flush=True)
        red = dict(times=np.zeros(0), scalars=np.zeros((0, len(SCALAR_NAMES))),
                   spectra=np.zeros((0, 5, args.n // 2 + 1)),
                   spectra_deconv=np.zeros((0, 5, args.n // 2 + 1)),
                   rho_slice=np.zeros((0, 0)), EB_slice=np.zeros((0, 0)),
                   EK_slice=np.zeros((0, 0)))
    else:
        red = reduce_run(run_dir, a, BOX_SIZE / args.n,
                         transfer=args.transfer, gamma=args.gamma,
                         dealias=args.dealias, slice_series=args.slice_series)
    scalars = red["scalars"]

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"athenapk_{tag}.npz"
    payload = dict(
        code="athenapk", label=label,
        scheme=f"{reconstruction}+{integrator}+hlld", scheme_key=args.scheme,
        N=args.n, tag=tag, finite_volume=True,
        t_wall=solver_wall if np.isfinite(solver_wall) else wall,
        t_wall_subprocess=wall, t_compile=0.0,
        n_steps=cycles, n_steps_estimated=cycles,
        zone_updates_per_s=thr, cfl=cfl,
        times=red["times"], t_over_tc=red["times"] / t_cross, t_cross=t_cross,
        n_shell=shell_numbers(args.n),
        spectra=red["spectra"], spectra_deconv=red["spectra_deconv"],
        rho_slice=red["rho_slice"], EB_slice=red["EB_slice"],
        **({"EB_slice_series": red["EB_slice_series"]}
           if "EB_slice_series" in red else {}),
        EK_slice=red["EK_slice"],
        a=a, B0=b0, p0=p0, gamma=args.gamma, beta0=args.beta, mturb=args.mturb,
        rho0=RHO0, accel_rms=args.accel_rms, tau=args.tau, rseed=args.rseed,
        mom_diff=args.mom_diff, ohm_diff=args.ohm_diff,
        glm_alpha=args.glm_alpha,
        seed_field=args.seed_field,
        meshblock=meshblock_size(args),
        scalar_names=np.array(SCALAR_NAMES), spectrum_names=np.array(SPECTRUM_NAMES),
        # The .hst history: E_K, E_B and the Mach numbers at ~600 points, far
        # finer in time than the hdf5 dumps the spectra come from.
        **{f"hst_{k}": v for k, v in hst.items()},
        **{name: scalars[:, i] for i, name in enumerate(SCALAR_NAMES)},
    )
    np.savez_compressed(path, **payload)
    print(f"[AthenaPK N={args.n} {args.scheme}] wrote {path}", flush=True)

    if not args.keep_snapshots and args.nsnap > 0:
        for f in run_dir.glob("*.phdf"):
            f.unlink()
        for f in run_dir.glob("*.xdmf"):
            f.unlink()
        print(f"[AthenaPK N={args.n} {args.scheme}] removed .phdf dumps from {run_dir} "
              f"(--keep-snapshots to keep them)", flush=True)


if __name__ == "__main__":
    main()
