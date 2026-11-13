"""
Pytest configuration of the astronomix test suite: device selection, the precision of a test module and the markers.

One pytest process runs the whole suite (``pytest`` from the repository root; ``pyproject.toml`` names the test
directory and deselects the slow tests, ``pytest -m ""`` runs every test). Nothing here changes the global JAX state
of the process:

* device selection happens once, before the first test module is imported: with ``JAX_PLATFORMS=cpu`` the CPU is
  split into four devices, so that the tests of sharded runs take part; otherwise, without ``CUDA_VISIBLE_DEVICES``,
  ``autocvd`` (if installed) picks one free GPU of a shared node;
* a module marked ``pytest.mark.x64`` runs its tests and its fixtures with 64-bit floats (``jax.enable_x64`` around
  the module), every other module with the default 32-bit floats;
* ``pytest.mark.gpu`` and ``pytest.mark.devices(n)`` skip a test without a GPU or with fewer than ``n`` devices.
"""

# general
import os

# testing
import pytest

# jax
import jax

#: Test files whose unrolled steps XLA:CPU compiles in seconds only at optimisation level 0 (minutes and tens of GB
#: otherwise; the numerics differ at round-off). A CPU session started on these files alone uses that level.
FAST_COMPILE_FILES = ("test_ad_tangent_safety.py", "test_fd_reverse_mode.py")
FAST_COMPILE_FLAGS = ("--xla_backend_optimization_level=0", "--xla_llvm_disable_expensive_passes=true")

#: The number of devices the CPU is split into.
NUM_CPU_DEVICES = 4


def pytest_addoption(parser):
    parser.addoption(
        "--reproduce-paper",
        action="store_true",
        default=False,
        help="regenerate every methods-paper figure from the examples/ "
             "generators (see pytests/test_reproduce_paper.py; slow, needs a GPU).",
    )


def _started_on_fast_compile_files_only(config) -> bool:
    """Whether every path given on the command line is one of the files of ``FAST_COMPILE_FILES``."""
    paths = [os.path.basename(argument.split("::")[0]) for argument in config.args]
    return bool(paths) and all(path in FAST_COMPILE_FILES for path in paths)


def pytest_configure(config):
    """Select the devices before any test module starts JAX."""
    if os.environ.get("JAX_PLATFORMS") == "cpu":
        flags = [f"--xla_force_host_platform_device_count={NUM_CPU_DEVICES}"]
        if _started_on_fast_compile_files_only(config):
            flags += FAST_COMPILE_FLAGS
        os.environ["XLA_FLAGS"] = " ".join(flags + [os.environ.get("XLA_FLAGS", "")]).strip()
    elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        try:
            from autocvd import autocvd
        except ImportError:
            return
        autocvd(num_gpus=1)


@pytest.fixture(autouse=True, scope="module")
def _module_precision(request):
    """64-bit floats for the tests and fixtures of a module marked ``x64``."""
    if request.node.get_closest_marker("x64") is None:
        yield
        return
    with jax.enable_x64(True):
        yield


def pytest_runtest_setup(item):
    """Skip a test whose device requirement (``gpu``, ``devices(n)``) is not met."""
    if item.get_closest_marker("gpu") is not None and jax.default_backend() != "gpu":
        pytest.skip("needs a GPU (the compiled kernels)")
    devices = item.get_closest_marker("devices")
    if devices is not None and len(jax.devices()) < devices.args[0]:
        pytest.skip(f"needs {devices.args[0]} devices")
