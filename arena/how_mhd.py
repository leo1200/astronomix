"""
Tests of our implementation of the HOW-MHD scheme (Seo & Ryu 2023).
"""

# ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)
# =======================

from jf1uids import SimulationConfig, SimulationParams
from jf1uids.option_classes.simulation_config import FINITE_DIFFERENCE

# tests
from arena_tests.mhd.blast_test1 import mhd_blast_test1

# test name
test_name = "how_mhd"

# setting up a baseline config
base_config = SimulationConfig(
    solver_mode=FINITE_DIFFERENCE,
    mhd=True,
    dimensionality=3,
    progress_bar=True,
)

# setting up baseline params
base_params = SimulationParams(
    C_cfl=1.5,
)

# running the tests
mhd_blast_test1(
    config=base_config._replace(num_cells=128),
    params=base_params,
    configuration_name=test_name,
)
