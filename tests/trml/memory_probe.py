"""Compile-only memory probe for the TRML setup at N=64.

Reuses the configuration / initial-condition machinery from ``trml.py``
but only runs ``time_integration_jit.lower(...).compile()`` so the
``memory_analysis`` printout fires without paying for the full
simulation. The script optionally toggles whether frame tracking
pulls the new 1D ``cell_centers_z`` field or the original 3D meshgrid
slice, so we can read off the saving from the X/Y/Z split directly.

Usage::

    python memory_probe.py            # X/Y/Z split (the new default)
    python memory_probe.py --legacy   # force frame_tracking to need the
                                      # full geometric_centers meshgrid
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import argparse
import os
import sys

import jax

# Import the TRML setup as a module. This runs everything at the top
# level (config, params, registered_variables, helper_data, initial_state)
# but stops short of the ``__main__`` block that executes the actual
# simulation.
sys.path.insert(0, os.path.dirname(__file__))
import trml  # noqa: E402

from astronomix.data_classes.simulation_helper_data import (  # noqa: E402
    HelperDataRequirements,
    _helper_data_requirements,
    get_helper_data,
)
from astronomix.option_classes.simulation_config import (  # noqa: E402
    GHOST_CELLS,
    PERIODIC_ROLL,
)
from astronomix.time_stepping.time_integration import _time_integration  # noqa: E402


def _human(n_bytes: int) -> str:
    return f"{n_bytes / (1024**2):.2f} MB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy",
        action="store_true",
        help=(
            "Force the frame-tracking dispatch to ask for the full "
            "geometric_centers meshgrid instead of the new 1D "
            "cell_centers_z field, so we can compare the two."
        ),
    )
    args = parser.parse_args()

    config = trml.config
    params = trml.params
    initial_state = trml.initial_state
    registered_variables = trml.registered_variables

    print("=== Config summary ===")
    print(f"  dimensionality: {config.dimensionality}")
    print(
        "  num_cells: "
        f"({config.num_cells.x}, {config.num_cells.y}, {config.num_cells.z})"
    )
    print(f"  num_ghost_cells: {config.num_ghost_cells}")
    print(f"  boundary_handling: {config.boundary_handling}")
    print(f"  solver_mode: {config.solver_mode}")
    print(f"  frame_tracking: {config.frame_tracking}")

    requirements = _helper_data_requirements(config)
    if args.legacy:
        # Mimic the pre-split behaviour: frame tracking forces the full
        # geometric_centers meshgrid. cell_centers_z is also kept so the
        # current frame-tracking code path (which now reads it) still
        # compiles; the meshgrid still shows up in argument_size and
        # reflects the memory that would have been live without the
        # split.
        requirements = requirements._replace(
            needs_geometric_centers=True,
        )
    print("=== Helper-data requirements ===")
    for name, value in requirements._asdict().items():
        print(f"  {name}: {value}")

    helper_data_pad = get_helper_data(
        config,
        sharding=None,
        padded=config.boundary_handling != PERIODIC_ROLL,
        requirements=requirements,
    )
    print("=== Helper-data field footprint ===")
    total_helper_bytes = 0
    for name, value in helper_data_pad._asdict().items():
        if value is None:
            continue
        print(
            f"  {name}: shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, bytes={value.nbytes}"
        )
        total_helper_bytes += value.nbytes
    print(f"  total helper_data: {_human(total_helper_bytes)}")

    # Mirror ``time_integration``'s jit construction.
    time_integration_jit = jax.jit(
        _time_integration,
        static_argnames=["config", "registered_variables", "snapshot_callable"],
    )

    print("=== Compiling _time_integration (memory_analysis only) ===")
    compiled = time_integration_jit.lower(
        initial_state,
        config,
        params,
        registered_variables,
        helper_data_pad,
        None,
    ).compile()

    stats = compiled.memory_analysis()
    if stats is None:
        print("memory_analysis returned None for this backend.")
        return

    total = (
        stats.temp_size_in_bytes
        + stats.argument_size_in_bytes
        + stats.output_size_in_bytes
        - stats.alias_size_in_bytes
    )
    print("=== Compiled memory usage PER DEVICE ===")
    print(f"  Temp size:     {_human(stats.temp_size_in_bytes)}")
    print(f"  Argument size: {_human(stats.argument_size_in_bytes)}")
    print(f"  Output size:   {_human(stats.output_size_in_bytes)}")
    print(f"  Alias size:    {_human(stats.alias_size_in_bytes)}")
    print(f"  Net total:     {_human(total)}")


if __name__ == "__main__":
    main()
