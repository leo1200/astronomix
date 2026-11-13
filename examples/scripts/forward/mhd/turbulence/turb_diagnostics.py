"""
Offline diagnostics for a ``turbulence_dist.py`` checkpoint sequence.

Walks every checkpoint step in ``--ckpt``, computes the scalar time series
(v_rms, M_turb, density contrast, kinetic and magnetic energies) and --
optionally, host-side via scipy on the Grace CPU -- the kinetic / magnetic /
density power spectra of selected steps.  Results are written to a single
NPZ, re-plottable like the ``paper_turbulence.py`` output.

Scales to hero-size grids by loading each checkpoint sharded across all
launched processes (scalar reductions are collective); spectra gather one
field at a time to host memory (a 2048^3 fp32 field is 34 GB -- fine in the
480 GB LPDDR of a GH200 node).

Launch (single GPU is fine up to 1024^3; shard for bigger)::

    srun --ntasks=<G> --ntasks-per-node=4 --gpu-bind=none \
        python examples/scripts/forward/mhd/turbulence/turb_diagnostics.py \
            --ckpt /e/scratch/astronomix/turb2048 --mturb 0.5 --N 2048 \
            --spectra-every 10 --out diag_icm2048.npz
"""

# --- bootstrap multi-process mode BEFORE importing astronomix ---
import argparse  # noqa: E402
import os  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--N", type=int, required=True, help="cells per dimension")
parser.add_argument("--mturb", type=float, default=0.5,
                    help="target Mach number (for a = 1/M_turb and t_cross)")
parser.add_argument("--spectra-every", type=int, default=0,
                    help="compute spectra for every k-th checkpoint "
                         "(0 = scalars only; the last step always included)")
parser.add_argument("--out", type=str, required=True, help="output NPZ path")
args = parser.parse_args()

_multi = "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1

import jax  # noqa: E402

if _multi:
    _cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    _localid = int(os.environ.get("SLURM_LOCALID", "0"))
    jax.distributed.initialize(
        local_device_ids=[_localid] if len(_cvd) > 1 else [0]
    )

jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import AxisType, PartitionSpec  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402

from astronomix.option_classes.simulation_config import (  # noqa: E402
    ISOTHERMAL,
    SimulationConfig,
    StaticIntVector,
)
from astronomix.variable_registry.registered_variables import (  # noqa: E402
    get_registered_variables,
)
from astronomix._snapshotting._orbax_storage import (  # noqa: E402
    latest_step,
    load_loop_checkpoint,
)

G = jax.device_count()
rank = jax.process_index()

mesh = jax.make_mesh((1, G, 1, 1), (0, 1, 2, 3), axis_types=(AxisType.Auto,) * 4)
sharding = jax.NamedSharding(mesh, PartitionSpec(0, 1, 2, 3))

# Variable indices come from a minimal config mirroring the run (isothermal
# MHD; the index layout depends only on these switches).
_cfg = SimulationConfig(
    equation_of_state=ISOTHERMAL,
    dimensionality=3,
    num_cells=StaticIntVector(args.N, args.N, args.N),
    mhd=True,
)
rv = get_registered_variables(_cfg)
sound_speed = 1.0 / args.mturb
t_cross = 0.5 / 1.0  # L_inj / v_rms_target, as in turbulence_dist.py


@jax.jit
def _scalar_diagnostics(state):
    rho = state[rv.density_index]
    v2 = (
        state[rv.velocity_index.x] ** 2
        + state[rv.velocity_index.y] ** 2
        + state[rv.velocity_index.z] ** 2
    )
    b2 = (
        state[rv.magnetic_index.x] ** 2
        + state[rv.magnetic_index.y] ** 2
        + state[rv.magnetic_index.z] ** 2
    )
    return jnp.stack([
        jnp.sqrt(jnp.mean(v2)),                  # v_rms
        jnp.sqrt(jnp.mean((rho - 1.0) ** 2)),    # density contrast rms
        jnp.mean(0.5 * rho * v2),                # E_K
        jnp.mean(0.5 * b2),                      # E_B
        jnp.min(rho),
        jnp.any(~jnp.isfinite(state)).astype(jnp.float32),
    ])


def _radial_spectrum(field_np):
    """Radially binned power spectrum of a real field, host-side (scipy)."""
    from scipy import fft as sfft

    n = field_np.shape[0]
    fk = sfft.rfftn(field_np, workers=64) / field_np.size
    power = np.abs(fk) ** 2
    # Hermitian double-count correction for the interior rfft plane.
    power[:, :, 1:-1] *= 2.0
    freqs = np.fft.fftfreq(n) * n
    rfreqs = np.arange(power.shape[2])
    kmag = np.sqrt(
        freqs[:, None, None] ** 2
        + freqs[None, :, None] ** 2
        + rfreqs[None, None, :] ** 2
    )
    bins = np.arange(0.5, n // 2)
    digitized = np.digitize(kmag.ravel(), bins)
    sums = np.bincount(digitized, weights=power.ravel(), minlength=len(bins) + 1)
    return sums


def _gather_field(state, index):
    """One field of the sharded state as a host numpy array (all hosts)."""
    return np.asarray(
        multihost_utils.process_allgather(state[index], tiled=True)
    )


def main():
    last = latest_step(args.ckpt)
    if last is None:
        raise SystemExit(f"no checkpoints in {args.ckpt}")
    steps = sorted(
        int(child) for child in os.listdir(args.ckpt) if child.isdigit()
    )

    times, scalars = [], []
    spectra = {}
    for step in steps:
        ckpt = load_loop_checkpoint(
            args.ckpt, step, sharding=sharding
        )
        stats = np.asarray(_scalar_diagnostics(ckpt.primitive_state))
        times.append(float(ckpt.time))
        scalars.append(stats)
        if rank == 0:
            v_rms, drho, e_k, e_b, rho_min, bad = stats
            print(
                f"[diag] step {step:4d} t={float(ckpt.time):.4f} "
                f"t/tc={float(ckpt.time) / t_cross:.3f} v_rms={v_rms:.3f} "
                f"M={v_rms / sound_speed:.3f} drho={drho:.4f} EK={e_k:.4f} "
                f"EB={e_b:.3e} min_rho={rho_min:.3f} NaN={int(bad)}",
                flush=True,
            )

        want_spectrum = args.spectra_every > 0 and (
            step == steps[-1]
            or (step - steps[0]) % args.spectra_every == 0
        )
        if want_spectrum:
            rho = _gather_field(ckpt.primitive_state, rv.density_index)
            spec_rho = _radial_spectrum(rho - rho.mean()) if rank == 0 else None
            del rho
            vx = _gather_field(ckpt.primitive_state, rv.velocity_index.x)
            vy = _gather_field(ckpt.primitive_state, rv.velocity_index.y)
            vz = _gather_field(ckpt.primitive_state, rv.velocity_index.z)
            if rank == 0:
                spec_v = (
                    _radial_spectrum(vx)
                    + _radial_spectrum(vy)
                    + _radial_spectrum(vz)
                )
            del vx, vy, vz
            bx = _gather_field(ckpt.primitive_state, rv.magnetic_index.x)
            by = _gather_field(ckpt.primitive_state, rv.magnetic_index.y)
            bz = _gather_field(ckpt.primitive_state, rv.magnetic_index.z)
            if rank == 0:
                spec_b = (
                    _radial_spectrum(bx)
                    + _radial_spectrum(by)
                    + _radial_spectrum(bz)
                )
                spectra[step] = (spec_rho, spec_v, spec_b)
            del bx, by, bz

    if rank == 0:
        scalars_arr = np.asarray(scalars)
        out = dict(
            steps=np.asarray(steps),
            times=np.asarray(times),
            t_over_tc=np.asarray(times) / t_cross,
            v_rms=scalars_arr[:, 0],
            mach=scalars_arr[:, 0] / sound_speed,
            drho_rms=scalars_arr[:, 1],
            E_K=scalars_arr[:, 2],
            E_B=scalars_arr[:, 3],
            rho_min=scalars_arr[:, 4],
            nan_flag=scalars_arr[:, 5],
        )
        for step, (spec_rho, spec_v, spec_b) in spectra.items():
            out[f"spec_rho_{step}"] = spec_rho
            out[f"spec_EK_{step}"] = spec_v
            out[f"spec_EB_{step}"] = spec_b
        np.savez(args.out, **out)
        print(f"[diag] wrote {args.out}", flush=True)

    if _multi:
        multihost_utils.sync_global_devices("diag_done")


if __name__ == "__main__":
    main()
