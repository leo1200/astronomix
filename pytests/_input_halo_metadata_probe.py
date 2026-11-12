"""Correctness probe for per-input Pallas halo metadata.

Run under Slurm with one process per GPU, for example:

    srun --ntasks=4 --ntasks-per-node=4 --gpu-bind=none \
        python pytests/_input_halo_metadata_probe.py
"""

import os

import jax

_cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
_localid = int(os.environ.get("SLURM_LOCALID", "0"))
jax.distributed.initialize(
    local_device_ids=[_localid] if len(_cvd) > 1 else [0]
)

jax.config.update("jax_use_shardy_partitioner", False)
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.sharding import AxisType, PartitionSpec  # noqa: E402

from astronomix._finite_difference._interface_fluxes._weno_pallas import (  # noqa: E402
    _weno_flux_mhd_pallas_local,
)
from astronomix._finite_difference._magnetic_update._constrained_transport_pallas import (  # noqa: E402
    _ct_curl_pallas_local,
    _ct_edge_emf_pallas_local,
    _ct_modified_flux_pallas_local,
    _ct_update_cell_center_fields_pallas_local,
)
from astronomix._finite_difference._time_integrators._ssprk_pallas import (  # noqa: E402
    _hydro_flux_div_axis_pallas_local,
)
from astronomix._pallas_helpers import (  # noqa: E402
    _pallas_call_sharded,
    _pallas_compiler_params,
    pallas_mesh_context,
)
from astronomix.option_classes.simulation_config import (  # noqa: E402
    FINITE_DIFFERENCE,
    PALLAS,
    BackendConfig,
    FDSolverConfig,
    SimulationConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams  # noqa: E402
from astronomix.variable_registry.registered_variables import (  # noqa: E402
    get_registered_variables,
)


SHAPE = (64, 16, 16)
BLOCK = (4, 4, 8)


def _config():
    return SimulationConfig(
        backend_config=BackendConfig(
            backend=PALLAS,
            pallas_block_shape=BLOCK,
            pallas_use_triton=True,
            pallas_interpret=False,
            pallas_ct=True,
        ),
        solver_mode=FINITE_DIFFERENCE,
        solver_config=FDSolverConfig(),
        mhd=True,
        dimensionality=3,
        donate_state=False,
        progress_bar=False,
    )


def _state(config, sharding):
    rv = get_registered_variables(config)
    gamma = 5.0 / 3.0
    nx, ny, nz = SHAPE
    x, y, z = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        np.arange(nz, dtype=np.float32),
        indexing="ij",
    )
    rho = 1.0 + 0.01 * np.sin(0.1 * x)
    vx = 0.02 * np.cos(0.2 * y)
    vy = 0.015 * np.sin(0.15 * z)
    vz = 0.01 * np.cos(0.05 * x)
    bx = 0.1 + 0.005 * np.sin(0.13 * x + 0.07 * y)
    by = 0.07 + 0.004 * np.cos(0.11 * y + 0.03 * z)
    bz = 0.05 + 0.003 * np.sin(0.09 * z + 0.02 * x)
    pressure = 1.0 + 0.02 * np.cos(0.04 * x)
    energy = (
        pressure / (gamma - 1.0)
        + 0.5 * rho * (vx * vx + vy * vy + vz * vz)
        + 0.5 * (bx * bx + by * by + bz * bz)
    )

    q = np.zeros((8, nx, ny, nz), dtype=np.float32)
    q[int(rv.density_index)] = rho
    q[int(rv.momentum_index.x)] = rho * vx
    q[int(rv.momentum_index.y)] = rho * vy
    q[int(rv.momentum_index.z)] = rho * vz
    q[int(rv.magnetic_index.x)] = bx
    q[int(rv.magnetic_index.y)] = by
    q[int(rv.magnetic_index.z)] = bz
    q[int(rv.pressure_index)] = energy
    return jax.device_put(jnp.asarray(q), sharding)


def _stacked_fields(sharding, channels):
    nx, ny, nz = SHAPE
    x, y, z = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
        np.arange(nz, dtype=np.float32),
        indexing="ij",
    )
    out = []
    for c in range(channels):
        out.append(
            0.1
            + 0.01 * (c + 1) * np.sin(0.05 * x + 0.03 * y + 0.02 * z + c)
        )
    return jax.device_put(jnp.asarray(np.stack(out).astype(np.float32)), sharding)


def _helper_kernel(b_local, a_local, config):
    nx, ny, nz = (int(x) for x in a_local.shape[1:])
    bx, by, bz = BLOCK
    grid = (nx // bx, ny // by, nz // bz)
    field_spec = pl.BlockSpec(a_local.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    out_spec = pl.BlockSpec((1, bx, by, bz), lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(b_ref, a_ref, out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        ii = (bi * bx + jnp.arange(bx)[:, None, None]) % nx
        jj = (bj * by + jnp.arange(by)[None, :, None]) % ny
        kk = (bk * bz + jnp.arange(bz)[None, None, :]) % nz
        out_ref[0, ...] = (
            b_ref[0, ii, jj, kk]
            + a_ref[0, ii, jj, kk]
            - a_ref[0, (ii - 1) % nx, jj, kk]
        )

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(a_local.shape, a_local.dtype),
        grid=grid,
        in_specs=[field_spec, field_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="input_halo_metadata_probe",
        **kwargs,
    )(b_local, a_local)


def _max_diff(a, b):
    diff = jnp.max(jnp.abs(a - b))
    finite = jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b))
    diff_all = multihost_utils.process_allgather(diff, tiled=True)
    finite_all = multihost_utils.process_allgather(finite, tiled=True)
    return float(np.max(np.asarray(diff_all))), bool(np.all(np.asarray(finite_all)))


def _check(name, a, b, atol):
    diff, finite = _max_diff(a, b)
    if jax.process_index() == 0:
        print(f"{name}: max_diff={diff:.3e} finite={finite}", flush=True)
    if (not finite) or diff > atol:
        raise RuntimeError(f"{name} failed: max_diff={diff}")


def main():
    rank = jax.process_index()
    gpus = jax.device_count()
    mesh = jax.make_mesh(
        (1, gpus, 1, 1), (0, 1, 2, 3), axis_types=(AxisType.Auto,) * 4
    )
    sharding = jax.NamedSharding(mesh, PartitionSpec(0, 1, 2, 3))
    config = _config()
    params = SimulationParams(C_cfl=1.5)
    rv = get_registered_variables(config)
    q = _state(config, sharding)
    a = _stacked_fields(sharding, 1)
    b = _stacked_fields(sharding, 1) * 0.5
    flux = _stacked_fields(sharding, 8)
    rhs = _stacked_fields(sharding, 8) * 0.25
    b_interfaces = _stacked_fields(sharding, 3)
    flux_slices = _stacked_fields(sharding, 6)

    @jax.jit
    def helper_old(b_in, a_in):
        return _pallas_call_sharded(
            lambda b_local, a_local: _helper_kernel(b_local, a_local, config),
            state_inputs=(b_in, a_in),
            halo=(1, 0, 0),
            block_shape=BLOCK,
        )

    @jax.jit
    def helper_new(b_in, a_in):
        return _pallas_call_sharded(
            lambda b_local, a_local: _helper_kernel(b_local, a_local, config),
            state_inputs=(b_in, a_in),
            halo=(1, 0, 0),
            input_halos=((0, 0, 0), (1, 0, 0)),
            block_shape=BLOCK,
        )

    @jax.jit
    def weno_old(q_in):
        return _pallas_call_sharded(
            lambda q_local: _weno_flux_mhd_pallas_local(
                q_local, params, config, rv, axis=0
            ),
            state_inputs=(q_in,),
            halo=(3, 0, 0),
            block_shape=BLOCK,
        )

    @jax.jit
    def weno_new(q_in):
        return _pallas_call_sharded(
            lambda q_local: _weno_flux_mhd_pallas_local(
                q_local, params, config, rv, axis=0
            ),
            state_inputs=(q_in,),
            halo=(3, 0, 0),
            input_halos=((3, 0, 0),),
            block_shape=BLOCK,
        )

    @jax.jit
    def div_old(rhs_in, flux_in):
        return _pallas_call_sharded(
            lambda r, d: _hydro_flux_div_axis_pallas_local(
                d, 0.1, config, axis=0, rhs_accumulator=r, scale_in=0.7
            ),
            state_inputs=(rhs_in, flux_in),
            halo=(1, 0, 0),
            block_shape=BLOCK,
        )

    @jax.jit
    def div_new(rhs_in, flux_in):
        return _pallas_call_sharded(
            lambda r, d: _hydro_flux_div_axis_pallas_local(
                d, 0.1, config, axis=0, rhs_accumulator=r, scale_in=0.7
            ),
            state_inputs=(rhs_in, flux_in),
            halo=(1, 0, 0),
            input_halos=((0, 0, 0), (1, 0, 0)),
            block_shape=BLOCK,
        )

    @jax.jit
    def center_old(q_in, b_in):
        return _pallas_call_sharded(
            lambda q_local, b_local: _ct_update_cell_center_fields_pallas_local(
                q_local, b_local, config, rv
            ),
            state_inputs=(q_in, b_in),
            halo=(3, 3, 3),
            block_shape=BLOCK,
        )

    @jax.jit
    def center_new(q_in, b_in):
        return _pallas_call_sharded(
            lambda q_local, b_local: _ct_update_cell_center_fields_pallas_local(
                q_local, b_local, config, rv
            ),
            state_inputs=(q_in, b_in),
            halo=(3, 3, 3),
            input_halos=((0, 0, 0), (3, 3, 3)),
            block_shape=BLOCK,
        )

    def ct_rhs(use_metadata, q_in, f_in):
        input_halos_mod = ((2, 2, 2), (0, 0, 0)) if use_metadata else None
        input_halos_edge = ((2, 2, 2),) if use_metadata else None
        input_halos_curl = ((4, 4, 4),) if use_metadata else None
        flux_mod = _pallas_call_sharded(
            lambda q_local, f_local: _ct_modified_flux_pallas_local(
                q_local, f_local, config, rv
            ),
            state_inputs=(q_in, f_in),
            halo=(2, 2, 2),
            input_halos=input_halos_mod,
            block_shape=BLOCK,
        )
        omega = _pallas_call_sharded(
            lambda f_local: _ct_edge_emf_pallas_local(f_local, config),
            state_inputs=(flux_mod,),
            halo=(2, 2, 2),
            input_halos=input_halos_edge,
            block_shape=BLOCK,
        )
        return _pallas_call_sharded(
            lambda om_local, dx, dy, dz: _ct_curl_pallas_local(
                om_local, dx, dy, dz, config
            ),
            state_inputs=(omega,),
            other_args=(jnp.asarray(0.1), jnp.asarray(0.1), jnp.asarray(0.1)),
            halo=(4, 4, 4),
            input_halos=input_halos_curl,
            block_shape=BLOCK,
        )

    @jax.jit
    def ct_old(q_in, f_in):
        return ct_rhs(False, q_in, f_in)

    @jax.jit
    def ct_new(q_in, f_in):
        return ct_rhs(True, q_in, f_in)

    with pallas_mesh_context(mesh):
        _check("helper", helper_old(b, a), helper_new(b, a), 0.0)
        _check("mhd_weno_x", weno_old(q), weno_new(q), 2.0e-5)
        _check("div_acc", div_old(rhs, flux), div_new(rhs, flux), 0.0)
        _check("ct_center", center_old(q, b_interfaces), center_new(q, b_interfaces), 1.0e-6)
        _check("ct_rhs", ct_old(q, flux_slices), ct_new(q, flux_slices), 1.0e-6)

    multihost_utils.sync_global_devices("input_halo_metadata_probe_done")
    if rank == 0:
        print("PROBE PASS", flush=True)


if __name__ == "__main__":
    main()
