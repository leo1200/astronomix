"""
Sink particles on several devices pytest.

Runs the sink clump with the state split along x across 4 simulated CPU
devices and checks that the result matches a single device, that the sink
steps do not copy full-grid arrays to every device, and that a disk restart
reproduces an uninterrupted run.

The number of simulated devices can only be set before JAX starts, so each
check runs this file as a script in a fresh process, with
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` and
``JAX_PLATFORMS=cpu``. The checks use the native JAX backend: the Pallas
kernels do not run on CPU devices, and the default ``OPTIMAL_BACKEND`` would
still pick Pallas on a machine with a GPU, since it decides from
``nvidia-smi``, not from the devices JAX runs on. The pytest process itself
does not use JAX. A check can also be run by hand::

    python pytests/sink_particles/sink_multi_device.py same_result
"""

# general
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# testing
import pytest


NUM_DEVICES = 4
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _run_check(check, *args):
    """Run one check of this file in a fresh process with simulated devices.

    Args:
        check: The name of the check (a key of ``CHECKS``).
        *args: Further command-line arguments of the check.

    Returns:
        The completed process, with its output captured.
    """
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={NUM_DEVICES}"
    ).strip()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(HERE)]
        + [path for path in env.get("PYTHONPATH", "").split(os.pathsep) if path]
    )
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), check, *args],
        env=env,
        capture_output=True,
        text=True,
    )


def _assert_check_passes(check, *args):
    """Run a check and fail with its output if it does not pass.

    Args:
        check: The name of the check (a key of ``CHECKS``).
        *args: Further command-line arguments of the check.
    """
    result = _run_check(check, *args)
    assert result.returncode == 0, (
        f"check '{check}' failed:\n{result.stdout}\n{result.stderr[-4000:]}"
    )


@pytest.mark.parametrize(
    "sinks_passed_in",
    [False, True],
    ids=["no_sinks_passed_in", "far_sink_passed_in"],
)
def test_same_result_as_one_device(sinks_passed_in):
    """A run with sinks on 4 devices gives the same sinks and gas as on one
    device. The clump is centred on the boundary between the second and third
    device (the control volume of the centre cell covers both), so formation
    and accretion straddle two devices.

    Args:
        sinks_passed_in: Whether a far sink is passed in through the
            ``StateStruct``.
    """
    check = "same_result_with_sink_in" if sinks_passed_in else "same_result"
    _assert_check_passes(check)


def test_no_full_grid_all_gather():
    """One formation call and one accretion call on the split state copy no
    array with as many elements as the grid to every device. Periodic
    boundaries, where the Poisson solve is distributed."""
    _assert_check_passes("no_full_grid_all_gather")


def test_disk_restart():
    """On 4 devices, a run restarted from a disk checkpoint ends with exactly
    the same gas and sinks as the uninterrupted run."""
    pytest.importorskip("orbax.checkpoint")
    with tempfile.TemporaryDirectory() as directory:
        _assert_check_passes("disk_restart", directory)


# -------------------------------------------------------------
# ============ ↓ Checks, run in a fresh process ↓ =============
# -------------------------------------------------------------


def _mesh_and_sharding():
    """The 4-device mesh, split along x, and the sharding of the state.

    String axis names are used: with integer names, the distributed Poisson
    solve fails on some JAX versions.

    Returns:
        ``(mesh, sharding)``.
    """
    import jax
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    assert jax.device_count() == NUM_DEVICES, jax.devices()
    mesh = jax.make_mesh(
        (1, NUM_DEVICES, 1, 1),
        ("vars", "x", "y", "z"),
        axis_types=(AxisType.Auto,) * 4,
    )
    return mesh, NamedSharding(mesh, PartitionSpec("vars", "x", "y", "z"))


def _assert_close(name, value, reference, tolerance):
    """Fail if ``max|value − reference|`` exceeds ``tolerance``.

    Args:
        name: The quantity, for the message.
        value: The multi-device result.
        reference: The single-device result.
        tolerance: The largest allowed absolute difference.
    """
    import numpy as np

    difference = float(np.max(np.abs(np.asarray(value) - np.asarray(reference))))
    print(f"{name}: max|4 devices − 1 device| = {difference:.3e} (tolerance {tolerance:.1e})")
    assert difference <= tolerance, name


def _check_same_result(sink_passed_in):
    """Run the clump on 1 and on 4 devices, with in-memory snapshots, and
    compare the gas and the sinks at every snapshot.

    Args:
        sink_passed_in: Whether a far sink is passed in through the
            ``StateStruct``.
    """
    import jax
    import numpy as np

    from astronomix import NATIVE_JAX, SnapshotSettings, finalize_state, time_integration
    from _sink_helpers import _one_sink_at, _setup

    _, sharding = _mesh_and_sharding()
    state, config, params, registered_variables = _setup(backend=NATIVE_JAX)
    # Adaptive steps up to t = 0.03; the sink forms in the first step.
    config = config._replace(
        fixed_timestep=False,
        return_snapshots=True,
        num_snapshots=4,
        snapshot_settings=SnapshotSettings(return_states=True),
    )
    params = params._replace(t_end=0.03)
    if sink_passed_in:
        far_sink = _one_sink_at((0.1, 0.1, 0.1), config, state.primitive_state.dtype)
        state = finalize_state(config, state.primitive_state, far_sink)

    single = time_integration(state, config, params, registered_variables)
    split_state = state._replace(
        primitive_state=jax.device_put(state.primitive_state, sharding)
    )
    split = time_integration(
        split_state, config, params, registered_variables, sharding=sharding
    )

    # Same sinks at every snapshot. Float32 rounding differs between the two
    # runs, because the reductions run in a different order across devices.
    num_sinks_single = np.asarray((single.sink_particles.mass > 0.0).sum(axis=1))
    num_sinks_split = np.asarray((split.sink_particles.mass > 0.0).sum(axis=1))
    print("sinks per snapshot:", num_sinks_single, num_sinks_split)
    assert np.array_equal(num_sinks_single, num_sinks_split)
    assert num_sinks_single[-1] == 1 + int(sink_passed_in)

    sinks_single, sinks_split = single.sink_particles, split.sink_particles
    _assert_close(
        "sink mass",
        sinks_split.mass,
        sinks_single.mass,
        1e-5 * float(np.max(sinks_single.mass)),
    )
    _assert_close("sink position", sinks_split.position, sinks_single.position, 1e-5)
    # The sink velocities are close to zero (the clump is at rest), so they
    # are compared with the sound speed, c_s = 1.
    _assert_close("sink velocity", sinks_split.velocity, sinks_single.velocity, 1e-5)
    # Gas: the criterion of pytests/_sharded_correctness_probe.py.
    _assert_close(
        "gas state",
        split.states,
        single.states,
        1e-5 * float(np.max(np.abs(np.asarray(single.states)))),
    )


def _check_no_full_grid_all_gather():
    """Compile one formation call and one accretion call on the split state
    and fail if any all-gather produces an array with as many elements as the
    grid."""
    import re

    import jax
    import numpy as np

    from astronomix._modules._sink_particles._sink_particle_accretion import (
        _accrete_gas,
    )
    from astronomix._modules._sink_particles._sink_particle_formation import (
        _empty_sink_particles,
        _form_sink_particles,
    )
    from astronomix import NATIVE_JAX
    from astronomix._pallas_helpers import pallas_mesh_context
    from _sink_helpers import _setup

    mesh, sharding = _mesh_and_sharding()
    state, config, params, registered_variables = _setup(backend=NATIVE_JAX)
    primitive_state = jax.device_put(state.primitive_state, sharding)
    sink_particles = _empty_sink_particles(config, primitive_state.dtype)
    num_cells = int(np.prod(primitive_state.shape[1:]))

    static = ["config", "registered_variables"]
    form = jax.jit(_form_sink_particles, static_argnames=static)
    accrete = jax.jit(_accrete_gas, static_argnames=static)
    # The same mesh contexts as time_integration, so the Poisson solve takes
    # its distributed path.
    with mesh, pallas_mesh_context(mesh):
        programs = {
            "formation": form.lower(
                primitive_state, sink_particles, config, params, registered_variables
            ).compile(),
        }
        new_sinks, num_active_sinks = form(
            primitive_state, sink_particles, config, params, registered_variables
        )
        programs["accretion"] = accrete.lower(
            primitive_state,
            new_sinks,
            num_active_sinks,
            config,
            params,
            registered_variables,
        ).compile()

    # An all-gather line of the compiled program reads
    #   %name = <type>[<dims>]{<layout>} all-gather(...)
    all_gather = re.compile(r"= (\w+)\[([\d,]*)\][^ ]* all-gather(?:-start)?\(")
    full_grid_gathers = []
    for step, program in programs.items():
        for line in program.as_text().splitlines():
            match = all_gather.search(line)
            if match is None:
                continue
            dims = [int(dim) for dim in match.group(2).split(",") if dim]
            num_elements = int(np.prod(dims)) if dims else 1
            print(f"{step}: all-gather {match.group(1)}[{match.group(2)}]")
            if num_elements >= num_cells:
                full_grid_gathers.append(f"{step}: {match.group(1)}[{match.group(2)}]")
    assert not full_grid_gathers, (
        f"full-grid all-gathers ({num_cells} cells): {full_grid_gathers}"
    )


def _check_disk_restart(directory):
    """On 4 devices: run the clump with disk snapshots in two segments,
    restart from the checkpoint of the first segment and compare the end of
    both runs.

    Args:
        directory: A temporary directory for the checkpoints.
    """
    import jax
    import numpy as np

    from astronomix import (
        NATIVE_JAX,
        TO_DISK,
        finalize_state,
        restart_from_latest_checkpoint,
        time_integration,
    )
    from _sink_helpers import _num_sinks, _setup

    _, sharding = _mesh_and_sharding()
    state, config, params, registered_variables = _setup(backend=NATIVE_JAX)
    uninterrupted_path = str(Path(directory) / "uninterrupted")
    config = config._replace(
        fixed_timestep=False,
        snapshot_storage_mode=TO_DISK,
        snapshot_storage_path=uninterrupted_path,
        num_snapshots=2,
    )
    split_state = state._replace(
        primitive_state=jax.device_put(state.primitive_state, sharding)
    )
    final_state = time_integration(
        split_state, config, params, registered_variables, sharding=sharding
    )
    assert _num_sinks(final_state.sink_particles) == 1

    primitive_state, restart_params, restart_state = restart_from_latest_checkpoint(
        uninterrupted_path,
        params,
        step=1,
        sharding=sharding,
    )
    assert _num_sinks(restart_state.sink_particles) == 1
    restart_config = config._replace(
        snapshot_storage_path=str(Path(directory) / "restarted"),
        num_snapshots=1,
    )
    restarted_state = time_integration(
        finalize_state(restart_config, primitive_state),
        restart_config,
        restart_params,
        registered_variables,
        sharding=sharding,
        restart_state=restart_state,
    )

    difference = float(
        np.max(
            np.abs(
                np.asarray(restarted_state.primitive_state)
                - np.asarray(final_state.primitive_state)
            )
        )
    )
    print(f"gas: max|restarted − uninterrupted| = {difference:.3e}")
    assert np.array_equal(
        np.asarray(restarted_state.primitive_state),
        np.asarray(final_state.primitive_state),
    )
    for name, restarted_field, final_field in zip(
        ("mass", "position", "velocity"),
        restarted_state.sink_particles,
        final_state.sink_particles,
    ):
        print(f"sink {name} equal:", bool(np.array_equal(restarted_field, final_field)))
        assert np.array_equal(np.asarray(restarted_field), np.asarray(final_field))


CHECKS = {
    "same_result": lambda: _check_same_result(sink_passed_in=False),
    "same_result_with_sink_in": lambda: _check_same_result(sink_passed_in=True),
    "no_full_grid_all_gather": _check_no_full_grid_all_gather,
    "disk_restart": lambda directory: _check_disk_restart(directory),
}


if __name__ == "__main__":
    # Run by hand: set the simulated devices before JAX starts.
    if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={NUM_DEVICES}"
        ).strip()
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path[:0] = [str(REPO_ROOT), str(HERE)]
    CHECKS[sys.argv[1]](*sys.argv[2:])
    print(f"check '{sys.argv[1]}' passed")
