"""
Pallas kernel of the CT half of the Poynting-consistent energy flux
(``_energy_flux.ct_energy_flux_rhs``; ``FDSolverConfig.scheme = RIEMANN_FLUX``,
ideal gas).

One per-cell kernel, ``_ct_energy_flux_rhs_pallas_local``: from the stacked edge
EMFs (Omega_z, Omega_x, Omega_y, the channel order of the CT kernels) and the
cell-centred field of the conserved state, the energy increment

    -dt/h sum_a [G_a(i + 1/2) - G_a(i - 1/2)],
    G_a = sum_{c != a} B_f,a[c] eps_{cae} I_fc^c(W^a[E_e]),

with the two-point face field B_f, the edge combination W (whose difference is
the staggered sixth-order derivative) and the face-to-centre interpolation
I_fc. Reach: three cells for the EMFs (W two edges along a around both faces,
I_fc -3 ... +2 along c), one for the field. The fluid half of the correction
runs inside the flux-divergence kernel (``_ssprk_pallas.py``).

The kernel runs where the constrained-transport kernels of
``_flux_form_ct_pallas.py`` run (one predicate, asked in ``_flux_form_ct.py``,
which also puts the call behind ``diffable_pallas_call_n``: the derivative is
the array form's). The call goes through ``_pallas_call_sharded``; the
``*_local`` build derives every shape from its own arguments. Following the
Pallas convention of this codebase
(``agent_guides/pallas_backend_implementation_guide.md``), the kernel is a
translation of the array module and is regenerated from it when it changes.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix._pallas_helpers import _pallas_call_sharded, _shard_cells, pl
from astronomix._finite_difference._magnetic_update._constrained_transport_pallas import (
    _block_cell_indices,
    _channel_reader,
    _compiler_kwargs,
    _ct_block_and_grid,
    _face_to_center,
)

#: The staggered sixth-order derivative coefficients (``_energy_flux._C1`` ...;
#: not imported: a kernel module does not load its array module).
_C1, _C2, _C3 = 75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0

#: (field component c, flux axis a) -> (EMF component e, sign eps_{c a e})
#: (``_energy_flux._LEVI_CIVITA``).
_LEVI_CIVITA = {
    (0, 1): (2, 1.0),
    (1, 0): (2, -1.0),
    (1, 2): (0, 1.0),
    (2, 1): (0, -1.0),
    (2, 0): (1, 1.0),
    (0, 2): (1, -1.0),
}

#: Channel of E_e in the stacked EMFs (Omega_z, Omega_x, Omega_y).
_EMF_CHANNEL = {2: 0, 0: 1, 1: 2}

#: Per-axis reach of the EMF and of the field reads.
_EMF_HALO = 3
_FIELD_HALO = 1


def _ct_energy_flux_rhs_pallas(conserved_state, emfs, dt_over_dx, config: SimulationConfig, magnetic_channels):
    """
    The CT half of the energy-flux correction (Pallas).

    Args:
        conserved_state: The conserved state (its cell-centred field is read).
        emfs: The stacked ``(3, nx, ny, nz)`` EMFs Omega_z, Omega_x, Omega_y.
        dt_over_dx: The (stage) time step over the cell size.
        config: The simulation configuration.
        magnetic_channels: The state channels of ``(B_x, B_y, B_z)``.

    Returns:
        The ``(1, nx, ny, nz)`` energy increment.
    """
    _, block_shape, _ = _ct_block_and_grid(conserved_state.shape[1:], config)

    def build(state_local, emfs_local, time_step_over_spacing):
        return _ct_energy_flux_rhs_pallas_local(
            state_local, emfs_local, time_step_over_spacing, config=config, magnetic_channels=magnetic_channels,
        )

    return _pallas_call_sharded(
        build,
        state_inputs=(conserved_state, emfs),
        other_args=(jnp.asarray(dt_over_dx, dtype=emfs.dtype),),
        halo=(_EMF_HALO,) * 3,
        input_halos=((_FIELD_HALO,) * 3, (_EMF_HALO,) * 3),
        block_shape=block_shape,
        local_output=True,
    )


def _ct_energy_flux_rhs_pallas_local(state, emfs, dt_over_dx, *, config: SimulationConfig, magnetic_channels):
    """
    Single-shard build of the CT half of the energy-flux correction.

    Args:
        state: The conserved state (possibly halo-padded).
        emfs: The stacked EMFs Omega_z, Omega_x, Omega_y (possibly halo-padded).
        dt_over_dx: The (stage) time step over the cell size (scalar array).
        config: The simulation configuration.
        magnetic_channels: The state channels of ``(B_x, B_y, B_z)``.

    Returns:
        The ``(1, nx, ny, nz)`` energy increment.
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(state.shape[1:], config)
    emf_shape = tuple(int(extent) for extent in emfs.shape[1:])

    state_spec = pl.BlockSpec(state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    emf_spec = pl.BlockSpec(emfs.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())
    out_spec = pl.BlockSpec((1,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(state_ref, emf_ref, dt_over_dx_ref, out_ref):
        ii, jj, kk = _block_cell_indices(block_shape, spatial_shape)
        same_cells = emf_shape == tuple(spatial_shape)
        ei, ej, ek = (ii, jj, kk) if same_cells else _block_cell_indices(block_shape, emf_shape)
        emf_readers = {
            component: _channel_reader(emf_ref, _EMF_CHANNEL[component], ei, ej, ek, emf_shape)
            for component in range(3)
        }
        field_readers = {
            component: _channel_reader(state_ref, magnetic_channels[component], ii, jj, kk, spatial_shape)
            for component in range(3)
        }
        # Each EMF value enters four face fluxes; load it once.
        loaded = {}

        def load(readers, kind, component, offset):
            key = (kind, component, offset)
            if key not in loaded:
                loaded[key] = readers[component](offset)
            return loaded[key]

        def displaced(axis, axis_offset, other_axis=None, other_offset=0):
            offset = [0, 0, 0]
            offset[axis] += axis_offset
            if other_axis is not None:
                offset[other_axis] += other_offset
            return tuple(offset)

        def ct_flux(component, axis, face):
            """F^CT_{component, axis} at the face ``face`` (0: i + 1/2, -1: i - 1/2)."""
            emf_component, sign = _LEVI_CIVITA[(component, axis)]

            def wide_at(component_offset):
                def at(edge_offset):
                    return load(
                        emf_readers, "emf", emf_component,
                        displaced(axis, face + edge_offset, component, component_offset),
                    )

                three = at(1) + at(0) + at(-1)
                five = three + at(2) + at(-2)
                return _C1 * at(0) + _C2 * three + _C3 * five

            return sign * _face_to_center(wide_at)

        def face_field(component, axis, face):
            return 0.5 * (
                load(field_readers, "field", component, displaced(axis, face))
                + load(field_readers, "field", component, displaced(axis, face + 1))
            )

        def energy_flux(axis, face):
            flux = 0.0
            for component in range(3):
                if component == axis:
                    continue
                flux = flux + face_field(component, axis, face) * ct_flux(component, axis, face)
            return flux

        time_step_over_spacing = dt_over_dx_ref[()]
        increment = 0.0
        for axis in range(3):
            increment = increment - time_step_over_spacing * (energy_flux(axis, 0) - energy_flux(axis, -1))
        out_ref[0, ...] = increment

    out_shape = jax.ShapeDtypeStruct((1,) + _shard_cells(spatial_shape)[1], emfs.dtype)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[state_spec, emf_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_energy_flux",
        **_compiler_kwargs(config),
    )(state, emfs, dt_over_dx)
