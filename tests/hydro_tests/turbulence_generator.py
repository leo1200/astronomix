#!/usr/bin/env python
# coding: utf-8

# # Simple Turbulence Tests

# ## Imports

# In[ ]:


# IMPORTANT: check gpustat --watch before 
# running scripts on the cluster
# BETTER NOT USE NOTEBOOKS AT ALL,
# THEY BLOCK THE GPU MEMORY IF NOT
# RESET PROPERLY
import os
# set the correct GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# you may also use
# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================
# in regular python scripts

# numerics
import jax
import jax.numpy as jnp

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# astronomix data structures
from astronomix import SimulationConfig
from astronomix import SimulationParams
from astronomix.option_classes import WindConfig
from astronomix.option_classes.simulation_config import BACKWARDS, HLL, HLLC, MINMOD, OSHER, FORWARDS, SPLIT

# astronomix setup functions
from astronomix import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.option_classes.simulation_config import PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D


# turbulent ic setup
from astronomix.initial_condition_generation.turbulent_ic_generator import create_turb_field

# main simulation function
from astronomix import time_integration

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c


# ## Initiating the stellar wind simulation

# In[2]:


print("👷 Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
box_size = 1.0

# resolution
num_cells = 256

# turbulence
turbulence = False
wanted_rms = 50 * u.km / u.s

fixed_timestep = False
dt_max = 0.1

# setup simulation config
config = SimulationConfig(
    runtime_debugging = False,
    first_order_fallback = False,
    progress_bar = True,
    dimensionality = 3,
    num_ghost_cells = 2,
    box_size = box_size,
    # split = SPLIT,
    riemann_solver = HLLC,
    limiter = MINMOD,
    num_cells = num_cells,
    fixed_timestep = fixed_timestep,
    differentiation_mode = FORWARDS,
    return_snapshots = True,
    num_snapshots = 80,
    boundary_settings = BoundarySettings(
        BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY),
        BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY),
        BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY)
    )
)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)


# ## Setting the simulation parameters and initial state

# In[3]:

def generate_turbulent_initial_state(
    num_cells=128,
    turbulence_slope=-2,
    kmin=2,
    kmax=64,
    wanted_rms=50 * u.km / u.s,
    seed_offset=0,
):
    # unit system
    code_length = 3 * u.parsec
    code_mass = 1 * u.M_sun
    code_velocity = 100 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)

    rho_0 = 2 * c.m_p / u.cm**3
    p_0 = 3e4 * u.K / u.cm**3 * c.k_B
    rho = jnp.ones((num_cells, num_cells, num_cells)) * rho_0.to(code_units.code_density).value
    p = jnp.ones((num_cells, num_cells, num_cells)) * p_0.to(code_units.code_pressure).value

    u_x = create_turb_field(num_cells, 1, turbulence_slope, kmin, kmax, seed=1 + seed_offset)
    u_y = create_turb_field(num_cells, 1, turbulence_slope, kmin, kmax, seed=2 + seed_offset)
    u_z = create_turb_field(num_cells, 1, turbulence_slope, kmin, kmax, seed=3 + seed_offset)

    rms_vel = jnp.sqrt(jnp.mean(u_x**2 + u_y**2 + u_z**2))
    scale = wanted_rms.to(code_units.code_velocity).value / rms_vel
    u_x *= scale
    u_y *= scale
    u_z *= scale

    return rho, p, u_x, u_y, u_z


# unit system
code_length = 3 * u.parsec
code_mass = 1 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# time domain
C_CFL = 0.1

# set the final time of the simulation
t_final = 1.0 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value

# set the simulation parameters
params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = dt_max,
    gamma = gamma,
    t_end = t_end,
)

from itertools import product

slopes = [-1.7, -2.0, -2.3]
kmins = 2
kmaxs =  64
rms_values = [20, 40, 50] * u.km / u.s


for i, (slope, rms) in enumerate(product(slopes, rms_values)):
    print(f"Generating field {i} with slope={slope}, rms={rms}")
    rho, p, u_x, u_y, u_z = generate_turbulent_initial_state(
        turbulence_slope=slope,
        kmin=kmins,
        kmax=kmaxs,
        wanted_rms=rms,
        seed_offset=i,

    )

    # construct primitive state
    initial_state = construct_primitive_state(
        config = config,
        registered_variables=registered_variables,
        density = rho,
        velocity_x = u_x,
        velocity_y = u_y,
        velocity_z = u_z,
        gas_pressure = p
    )

    config = finalize_config(config, initial_state.shape)


    # ## Simulation and Gradient

    # In[4]:


    result = time_integration(initial_state, config, params, registered_variables)
    final_state = result.states[-1]

    safe_rms = str(rms.value).replace('.', 'p')  # e.g., 20.0 → 20p0
    safe_filename = f"./turbulent_fields/{i}_s{slope}_rms{safe_rms}.npy"
    jnp.save(safe_filename, final_state[0, :, :, :])
    

print("Done ;)")

