"""
The distributed FFT and the shell spectra on a device mesh against their single-device counterparts.

On four devices (four CPU devices with ``JAX_PLATFORMS=cpu``) in both decompositions of the state mesh, (4, 1, 1)
and (2, 2, 1): the transform against ``jnp.fft.fftn``, the periodic Poisson solve against the replicated one, the
shells against those of the single-device diagnostics (exactly), and the spectra against ``energy_spectrum`` and the
dynamo study's estimator (``examples/scripts/forward/mhd/turbulence/_mhd_spectral.py``) to float32 rounding.
"""

# general
import sys
from pathlib import Path

# testing
import pytest

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec

# astronomix
import astronomix
from astronomix import PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D, GravityConfig, SimulationConfig
from astronomix._modules._gravity._poisson_solver import _compute_gravitational_potential
from astronomix._pallas_helpers import pallas_mesh_context
from astronomix.option_classes.simulation_config import StaticIntVector
from astronomix.analysis_helpers.energy_spectrum import (
    get_kinetic_energy_spectrum,
    get_magnetic_energy_spectrum,
    _wavenumber_bins,
)
from astronomix.analysis_helpers.sharded_spectra import (
    ANGULAR_SHELLS,
    INTEGER_SHELLS,
    local_shells,
    shell_centres,
    sharded_energy_spectrum,
)
from astronomix.parallel.distributed_fft import (
    FIELD_SPEC,
    SPECTRUM_SPEC,
    distributed_fft,
    distributed_ifft,
    spectral_mesh,
)

STUDY_DIRECTORY = Path(astronomix.__file__).resolve().parents[1] / "examples/scripts/forward/mhd/turbulence"

#: The decompositions (px, py, 1) of the state mesh.
LAYOUTS = [(4, 1, 1), (2, 2, 1)]

#: Largest difference to the single-device result, relative to the largest value (float32 rounding).
ROUND_OFF = 2e-6

pytestmark = pytest.mark.devices(4)


@pytest.fixture(autouse=True, scope="module")
def without_shardy():
    """The GSPMD partitioner for the module (the state mesh has integer axis names, which Shardy does not take)."""
    previous = jax.config.jax_use_shardy_partitioner
    jax.config.update("jax_use_shardy_partitioner", False)
    yield
    jax.config.update("jax_use_shardy_partitioner", previous)


def state_mesh(layout):
    return jax.make_mesh((1, *layout), (0, 1, 2, 3), axis_types=(AxisType.Auto,) * 4,
                         devices=jax.devices()[:int(np.prod(layout))])


def split(field, mesh):
    """A field split like a field of the state."""
    return jax.device_put(field, NamedSharding(mesh, PartitionSpec(1, 2, 3)))


def random_fields(shape, count, seed=1):
    """Smooth random fields (spectrum ~ k^-2) with unit standard deviation."""
    generator = np.random.default_rng(seed)
    modes = np.meshgrid(*[np.fft.fftfreq(n, d=1.0 / n) for n in shape], indexing="ij")
    magnitude = np.sqrt(sum(m**2 for m in modes))
    amplitude = np.where(magnitude > 0, np.maximum(magnitude, 1.0) ** -2.0, 0.0)
    fields = []
    for _ in range(count):
        field = np.real(np.fft.ifftn(np.fft.fftn(generator.standard_normal(shape)) * amplitude))
        fields.append(jnp.asarray(field / np.std(field), jnp.float32))
    return fields


def assert_round_off(result, reference):
    result, reference = np.asarray(result), np.asarray(reference)
    assert result.shape == reference.shape
    np.testing.assert_allclose(result, reference, rtol=0, atol=ROUND_OFF * float(np.max(np.abs(reference))))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", [(32, 16, 8), (16, 16, 16)])
def test_transform_equals_fftn(layout, shape):
    mesh = state_mesh(layout)
    (field,) = random_fields(shape, 1)
    spectrum = jax.jit(lambda f: distributed_fft(f, mesh))(split(field, mesh))
    assert spectrum.sharding.is_equivalent_to(NamedSharding(spectral_mesh(mesh), SPECTRUM_SPEC), 3)
    assert_round_off(spectrum, jnp.fft.fftn(field))
    assert_round_off(jax.jit(lambda s: distributed_ifft(s, mesh))(spectrum), field)
    if layout[1] == 1:
        with pytest.raises(ValueError, match="divisible"):
            distributed_fft(jnp.zeros((8, 6, 8)), mesh)


def test_one_device_without_a_mesh():
    (field,) = random_fields((8, 8, 8), 1)
    assert_round_off(distributed_fft(field), jnp.fft.fftn(field))
    assert_round_off(distributed_ifft(distributed_fft(field)), field)


@pytest.mark.parametrize("discrete_laplacian", [False, True])
def test_sharded_poisson_solve_equals_replicated(discrete_laplacian):
    """The self-gravity solve on a (4, 1, 1) mesh, built on the distributed transform."""
    mesh = state_mesh((4, 1, 1))
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        dimensionality=3, num_cells=StaticIntVector(16, 16, 16),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        gravity_config=GravityConfig(poisson_discrete_laplacian=discrete_laplacian),
    )
    (density,) = random_fields((16, 16, 16), 1)
    density = 1.0 + 0.1 * density
    jax.clear_caches()  # the mesh context is read when the solver is traced
    with pallas_mesh_context(mesh, PartitionSpec(0, 1, 2, 3)):
        solve = jax.jit(lambda rho: _compute_gravitational_potential(rho, 1.0 / 16, config)).lower(split(density, mesh))
    assert "all_to_all" in solve.as_text()
    jax.clear_caches()
    assert_round_off(solve.compile()(split(density, mesh)), _compute_gravitational_potential(density, 1.0 / 16, config))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_shells_equal_the_single_device_shells(layout):
    mesh = state_mesh(layout)
    num_cells = (16, 16, 16)
    field = split(jnp.zeros(num_cells), mesh)

    def shells(binning):
        return jax.shard_map(lambda block: local_shells(num_cells, binning)[0], mesh=spectral_mesh(mesh),
                             in_specs=FIELD_SPEC, out_specs=SPECTRUM_SPEC)(field)

    frequencies = np.fft.fftfreq(16, d=1.0 / 16)
    magnitude = np.sqrt(frequencies[:, None, None] ** 2 + frequencies[None, :, None] ** 2
                        + frequencies[None, None, :] ** 2)
    np.testing.assert_array_equal(shells(INTEGER_SHELLS), np.rint(magnitude).astype(np.int32))
    angular, number_of_bins, _ = _wavenumber_bins(*num_cells)
    np.testing.assert_array_equal(shells(ANGULAR_SHELLS).ravel(), angular)
    assert len(shell_centres(num_cells, ANGULAR_SHELLS)) == number_of_bins


@pytest.mark.parametrize("layout", LAYOUTS)
def test_energy_spectra_equal_energy_spectrum(layout):
    mesh = state_mesh(layout)
    density, *fields = random_fields((32, 32, 32), 7)
    density = jnp.exp(0.5 * density)
    velocity, magnetic = fields[:3], fields[3:]
    weighted = [jnp.sqrt(density) * v for v in velocity]
    centres, kinetic = sharded_energy_spectrum(*[split(f, mesh) for f in weighted], mesh)
    reference_centres, reference = get_kinetic_energy_spectrum(*velocity, density)
    np.testing.assert_array_equal(centres, reference_centres)
    assert_round_off(kinetic, reference)
    _, magnetic_spectrum = sharded_energy_spectrum(*[split(f, mesh) for f in magnetic], mesh)
    assert_round_off(magnetic_spectrum, get_magnetic_energy_spectrum(*magnetic)[1])


@pytest.fixture(scope="module")
def estimator():
    if not (STUDY_DIRECTORY / "_mhd_spectral.py").exists():
        pytest.skip("the dynamo study's estimator is not in this tree")
    sys.path.insert(0, str(STUDY_DIRECTORY))
    import _mhd_spectral

    return _mhd_spectral


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("finite_volume", [False, True], ids=["isothermal", "glm_ideal_gas"])
def test_estimator_spectra_equal_the_single_device_reduction(estimator, layout, finite_volume):
    """mesh_snapshot_spectra against the reduction of the Mach 4 run script on one device."""
    mesh = state_mesh(layout)
    cells = 32
    grid_spacing = 1.0 / cells
    log_density, *fields = random_fields((cells,) * 3, 9)
    density = jnp.exp(log_density)
    velocity = [4.0 * f for f in fields[:3]]
    magnetic = [0.3 * f for f in fields[3:6]]
    pressure = density * jnp.exp(0.05 * fields[6]) if finite_volume else None
    psi = 1e-3 * fields[7] if finite_volume else None

    _, rows = estimator.snapshot_spectra(density, *velocity, *magnetic, 1.0, grid_spacing, transfer=True,
                                         pressure=pressure)
    if finite_volume:
        rows = jnp.concatenate([rows, estimator.glm_cleaning_transfer(*magnetic, psi, grid_spacing)[None]])
    centres, kinetic = get_kinetic_energy_spectrum(*velocity, density)

    def split_or_none(field):
        return None if field is None else split(field, mesh)

    result = jax.jit(lambda *f: estimator.mesh_snapshot_spectra(
        *f[:7], 1.0, grid_spacing, mesh, pressure=f[7], psi=f[8], deconvolve_cell_average=finite_volume
    ))(*[split(f, mesh) for f in [density, *velocity, *magnetic]], split_or_none(pressure), split_or_none(psi))

    assert result.shells.shape == rows.shape
    for row, reference in zip(result.shells, rows):
        assert_round_off(row, reference)
    np.testing.assert_array_equal(result.angular_wavenumbers, centres)
    assert_round_off(result.angular[0], kinetic)
    assert_round_off(result.angular[1], get_magnetic_energy_spectrum(*magnetic)[1])
    if finite_volume:
        _, deconvolved = estimator.snapshot_spectra(density, *velocity, *magnetic, 1.0, grid_spacing,
                                                    deconvolve_cell_average=True)
        for row, reference in zip(result.shells_deconvolved, deconvolved):
            assert_round_off(row, reference)
    else:
        assert result.shells_deconvolved is None
