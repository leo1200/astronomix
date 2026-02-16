import jax.numpy as jnp

# constants
from astronomix import CARTESIAN

# astronomix option structures
from astronomix import SimulationConfig
from astronomix import SimulationParams

# simulation setup
from astronomix import get_helper_data
from astronomix import finalize_config
from astronomix import get_registered_variables
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state

# time integration, core function
from astronomix import time_integration

# / ... imports ... /

num_cells = 100

# configuration with which the
# simulation is compiled
config = SimulationConfig(
    geometry = CARTESIAN,
    box_size = 1.0,
    num_cells = num_cells,
    dimensionality = 1,
    # further options like choice of
    # riemann solver, limiter, boundary
    # conditions, ...
)

# params with which the 
# simulation is run
params = SimulationParams(
    t_end = 0.2,
)

# helper data like cell centers
helper_data = get_helper_data(config)

# variable registry, specifying where in the state
# vector which fluid variable is stored
registered_variables = get_registered_variables(config)

# shock tube setup
x = helper_data.geometric_centers
rho = jnp.where(x < 0.5, 1.0, 0.125)
u = jnp.zeros_like(x)
p = jnp.where(x < 0.5, 1.0, 0.1)

# construct the initial state array
# number of fluid variables x num_cells
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = u,
    gas_pressure = p,
)

# finalize and check the configuration, set the 
# grid spacing, ...
config = finalize_config(config, initial_state.shape)

# run the simulation
final_state = time_integration(
    initial_state, config, params, 
    registered_variables
)