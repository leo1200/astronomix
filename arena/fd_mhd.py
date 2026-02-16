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
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE, PERIODIC_ROLL

# tests
from arena_tests.scaling.scaling import scaling_test
from arena_tests.mhd.blast_test1 import mhd_blast_test1
from arena_tests.scaling.memory_scaling import memory_scaling
from arena_tests.turbulent_forcing.turbulent_forcing import turbulent_forcing_test

# test name
test_name = "fd_mhd"

# setting up a baseline config
base_config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    donate_state = False,
    boundary_handling = PERIODIC_ROLL,
    mhd = True,
    dimensionality = 3,
    progress_bar = True,
)

# setting up baseline params
base_params = SimulationParams(
    C_cfl = 1.5,
    minimum_density = 1e-3,
    minimum_pressure = 1e-3,
)

# running the tests

# blast test 1
mhd_blast_test1(
    config = base_config._replace(num_cells=142),
    params = base_params,
    configuration_name = test_name,
)

# mhd_blast_test1(
#     config = base_config._replace(num_cells=64),
#     params = base_params,
#     configuration_name = test_name,
# )

# mhd_blast_test1(
#     config = base_config._replace(num_cells=200),
#     params = base_params,
#     configuration_name = test_name,
# )

# mhd_blast_test1(
#     config = base_config._replace(num_cells=400),
#     params = base_params,
#     configuration_name = test_name,
# )

# memory_scaling(
#     config = base_config,
#     params = base_params,
#     resolutions = [50, 100, 200, 400],
#     configuration_name = test_name,
# )

# scaling_test(
#     config = base_config,
#     params = base_params,
#     resolutions = [400],
#     configuration_name = test_name,
#     multi_gpu = multi_gpu,
# )