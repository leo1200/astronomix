"""
Tests of our implementation of the HOW-MHD scheme (Seo & Ryu 2023).
"""

multi_gpu = False
double_precision = False

if multi_gpu:
    # ==== GPU selection ====
    from autocvd import autocvd
    autocvd(num_gpus = 4)
    # =======================
else:
    # ==== GPU selection ====
    from autocvd import autocvd
    autocvd(num_gpus = 1)
    # =======================

import jax

# double precision
jax.config.update("jax_enable_x64", double_precision)

from astronomix import SimulationConfig, SimulationParams
from astronomix.option_classes.simulation_config import DOUBLE_PRECISION, FINITE_DIFFERENCE, FINITE_VOLUME, HLL, IMPLICIT_EULER, LAX_FRIEDRICHS, PERIODIC_ROLL, SINGLE_PRECISION

# tests
from arena_tests.scaling.scaling import scaling_test
from arena_tests.mhd.blast_test1 import mhd_blast_test1
from arena_tests.scaling.memory_scaling import memory_scaling

# test name
test_name = "fv_mhd_hll_eul"

# setting up a baseline config
base_config = SimulationConfig(
    solver_mode = FINITE_VOLUME,
    numerical_precision = DOUBLE_PRECISION if double_precision else SINGLE_PRECISION,
    riemann_solver = HLL,
    fv_magnetic_integrator = IMPLICIT_EULER,
    boundary_handling = PERIODIC_ROLL,
    mhd = True,
    dimensionality = 3,
    progress_bar = True,
)

# setting up baseline params
base_params = SimulationParams(
    C_cfl = 0.8,
)

# running the tests

# blast test 1
# mhd_blast_test1(
#     config = base_config._replace(num_cells=32),
#     params = base_params,
#     configuration_name = test_name,
# )

# mhd_blast_test1(
#     config = base_config._replace(num_cells=64),
#     params = base_params,
#     configuration_name = test_name,
# )

# mhd_blast_test1(
#     config = base_config._replace(num_cells=128),
#     params = base_params,
#     configuration_name = test_name,
# )

mhd_blast_test1(
    config = base_config._replace(num_cells=400),
    params = base_params,
    configuration_name = test_name,
)