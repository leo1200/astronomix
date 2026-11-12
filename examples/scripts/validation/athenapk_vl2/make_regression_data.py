"""
Store AthenaPK's results for the small regression cases.

Runs AthenaPK on every case of ``REGRESSION_CASES`` and writes the initial and
final primitive states and the cycle count to
``pytests/mhd/data/athenapk_vl2/<case>.npz``, so that the regression test can
compare astronomix against AthenaPK without an AthenaPK installation.

The CPU build (``ATHENAPK_BIN``) is the reference, except for the cases with
the first-order flux correction: AthenaPK's CPU build applies the correction in
an order-dependent way (see README.md), while its GPU build
(``ATHENAPK_GPU_BIN``) uses the order-independent form astronomix implements.

Usage:
    ATHENAPK_BIN=... ATHENAPK_GPU_BIN=... python make_regression_data.py
"""

# general
import os
import sys
import tempfile

# numerics
import numpy as np

# jax
import jax

jax.config.update("jax_enable_x64", True)

# validation helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from athenapk_runner import final_output, initial_output, run_case  # noqa: E402
from cases import REGRESSION_CASES, initial_primitive_state, astronomix_setup  # noqa: E402
from compare_to_athenapk import athenapk_layout_to_astronomix, astronomix_layout_to_athenapk  # noqa: E402

# astronomix
from astronomix.option_classes.simulation_config import NATIVE_JAX  # noqa: E402
from astronomix._finite_volume._stage import _conserved_from_primitive_vl2  # noqa: E402

OUTPUT_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "pytests", "mhd", "data", "athenapk_vl2"
)


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    for case in REGRESSION_CASES:
        own_initial_state = initial_primitive_state(case)
        initial_conserved = None
        if own_initial_state is not None:
            config, _, registered_variables = astronomix_setup(case, own_initial_state.shape, NATIVE_JAX)
            initial_conserved = astronomix_layout_to_athenapk(
                np.asarray(_conserved_from_primitive_vl2(own_initial_state, case.gamma, config, registered_variables)),
                case,
            )
        with tempfile.TemporaryDirectory() as run_directory:
            run_case(case, run_directory, initial_conserved, gpu=case.first_order_flux_correction)
            final_state, _, cycles = final_output(run_directory, "prim")
            if own_initial_state is None:
                initial_state = athenapk_layout_to_astronomix(initial_output(run_directory, "prim")[0], case)
            else:
                initial_state = np.asarray(own_initial_state)
        np.savez_compressed(
            os.path.join(OUTPUT_DIRECTORY, f"{case.name}.npz"),
            initial_primitive_state=np.asarray(initial_state),
            final_primitive_state=athenapk_layout_to_astronomix(final_state, case),
            cycles=cycles,
        )
        print(f"{case.name}: {cycles} cycles", flush=True)


if __name__ == "__main__":
    main()
