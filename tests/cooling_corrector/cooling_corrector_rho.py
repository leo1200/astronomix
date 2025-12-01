from autocvd import autocvd

from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
autocvd(num_gpus = 1)

from astronomix._physics_modules._cooling._cooling import get_pressure_from_temperature, get_temperature_from_pressure
from astronomix._physics_modules._cooling._cooling_tables import schure_cooling

from astronomix._physics_modules._stellar_wind.stellar_wind_options import EI, MEI, MEO

import jax.numpy as jnp

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from astronomix import WindParams
from astronomix import SimulationConfig
from astronomix import SimulationParams
from astronomix.option_classes import WindConfig
from astronomix._physics_modules._cooling.cooling_options import NEURAL_NET_COOLING, NEURAL_NET_COOLING_WITH_DENSITY, PIECEWISE_POWER_LAW, SIMPLE_POWER_LAW, CoolingConfig, CoolingCurveConfig, CoolingNetConfig, CoolingNetParams, CoolingParams, PiecewisePowerLawParams, SimplePowerLawParams

from astronomix import get_helper_data
from astronomix._fluid_equations._equations import conserved_state_from_primitive, primitive_state_from_conserved
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import BACKWARDS, finalize_config

import pickle

from astronomix import time_integration

# astronomix constants
from astronomix.option_classes.simulation_config import OPEN_BOUNDARY, REFLECTIVE_BOUNDARY, SPHERICAL

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# wind-specific
from astronomix._physics_modules._stellar_wind.weaver import Weaver


import equinox as eqx
import jax
import optax
from matplotlib.gridspec import GridSpec

# options
plot_problem_setting = False
run_simulation = False
train_model = False
run_high_res_for_error_plot = False
low_res = 500
high_res_s = [100000]
num_epochs_corr = [500]
domain_size = 1.0
r_inj = 0.01 * domain_size

def get_num_injection_cells(r_inj, num_cells):
    ngc = 0
    grid_spacing = domain_size / num_cells
    r = jnp.linspace(grid_spacing / 2 - ngc * grid_spacing, domain_size + grid_spacing / 2 + ngc * grid_spacing, num_cells + 2 * ngc, endpoint = False)
    outer_cell_boundaries = r + grid_spacing / 2
    r_inj = r_inj * domain_size
    return int(jnp.argmin(jnp.abs(outer_cell_boundaries - r_inj)))

# code units
code_length = 10e18 * u.cm
code_mass = 1e-3 * u.M_sun
code_velocity = 1 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# low resolution simulation
print("LOW RES = ", low_res)
print("HIGH RES = ", high_res_s)

# beginning index of loss
beginning_index = int(low_res // 4)

# general cooling params
hydrogen_mass_fraction = 0.76
metal_mass_fraction = 0.02
# without a floor temperature, the simulations crash
floor_temperature = (1e2 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value


class CoolingNet(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        hidden_features: int,
        *,
        key
    ):
        self.mlp = eqx.nn.MLP(
            in_size=2,       # input: log10_T, log10_rho
            out_size=1,      # output: log10_Lambda
            width_size=hidden_features,
            depth=5,
            key=key,
        )
        
    def __call__(self, x):
        return self.mlp(x)
        

# initialize cooling corrector network
key = jax.random.PRNGKey(42)
cooling_corrector_network = CoolingNet(
    hidden_features = 1024,
    key = key
)
cooling_corrector_params, cooling_corrector_static = eqx.partition(cooling_corrector_network, eqx.is_array)

# get final density from high res simulation with cooling
states = jnp.load(f"data/reference_states{high_res_s[-1]}.npy", allow_pickle=True)
densities = jnp.log10(states[:, 0, :].flatten())

# pre-train the network to log10_Lambda_table(log10_T_table)
# from the schure cooling table
schure_cooling_params = schure_cooling(code_units)
log10_T_table = schure_cooling_params.log10_T_table
log10_Lambda_table = schure_cooling_params.log10_Lambda_table
# log10_T_table = log10_T_table[:, None]
log10_Lambda_table = log10_Lambda_table[:, None]

# print(log10_Lambda_table)
# print(log10_T_table)

# simple training loop
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(cooling_corrector_params)

@eqx.filter_jit
def loss_fn_pre_train(network_params_arrays, key):
    """Calculates the difference between the final state and the target."""
    network = eqx.combine(network_params_arrays, cooling_corrector_static)

    # Randomly sample densities from the precomputed array
    key, subkey = jax.random.split(key)
    sampled_densities = jax.random.choice(subkey, densities, shape=log10_T_table.shape)

    # Evaluate model and compute loss
    log10_Lambda_pred = jax.vmap(network)(jnp.stack([log10_T_table, sampled_densities], axis=-1))
    loss_value = jnp.mean((log10_Lambda_pred - log10_Lambda_table) ** 2)
    return loss_value, key


@eqx.filter_jit
def train_step_pre_train(network_params_arrays, opt_state, key):
    """Performs one step of gradient descent."""
    (loss_value, key), grads = eqx.filter_value_and_grad(loss_fn_pre_train, has_aux=True)(
        network_params_arrays, key
    )
    updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
    network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
    return network_params_arrays, opt_state, key, loss_value


# Training loop
num_epochs = 10000
key = jax.random.PRNGKey(42)

for epoch in range(num_epochs):
    cooling_corrector_params, opt_state, key, loss_value = train_step_pre_train(
        cooling_corrector_params, opt_state, key
    )
    if epoch % 500 == 0 or epoch == num_epochs - 1:
        print(f"Pre-training Epoch {epoch+1}/{num_epochs}, Loss: {loss_value:.6f}")

cooling_corrector_network = eqx.combine(cooling_corrector_params, cooling_corrector_static)

cooling_curve_paramsC = CoolingNetParams(
    network_params = cooling_corrector_params
)



def setup_simulation(num_cells, cooling_curve_type, cooling_curve_params, return_snapshots, num_snapshots, cooling = True, num_injection_cells = 20, use_specific_snapshot_timepoints = True, t_final = 1.25e12 * u.s):
    print("👷 Setting up simulation...")

    # simulation settings
    gamma = 5/3

    # spatial domain
    geometry = SPHERICAL
    box_size = domain_size

    # activate stellar wind
    stellar_wind = True

    # time stepping
    C_CFL = 0.8
    t_end = t_final.to(code_units.code_time).value
    dt_max = 0.1 * t_end

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging = False,
        progress_bar = False,
        geometry = geometry,
        box_size = box_size, 
        num_cells = num_cells,
        wind_config = WindConfig(
            stellar_wind = stellar_wind,
            num_injection_cells = num_injection_cells,
            trace_wind_density = False,
            wind_injection_scheme = EI
        ),
        cooling_config = CoolingConfig(
            cooling = cooling,
            cooling_curve_config = CoolingCurveConfig(
                cooling_curve_type = cooling_curve_type,
                cooling_net_config = CoolingNetConfig(
                    network_static = cooling_corrector_static
                )
            )
        ),
        return_snapshots = return_snapshots,
        num_snapshots = num_snapshots,
        use_specific_snapshot_timepoints = use_specific_snapshot_timepoints,
        differentiation_mode = BACKWARDS,
        num_checkpoints = 100
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # wind parameters
    wind_final_velocity = 1500 * u.km / u.s
    wind_mass_loss_rate = 1e-6 * u.M_sun / u.yr
    wind_params = WindParams(
        wind_mass_loss_rate = wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
        wind_final_velocity = wind_final_velocity.to(code_units.code_velocity).value
    )

    # simulation params
    params = SimulationParams(
        C_cfl = C_CFL,
        dt_max = dt_max,
        gamma = gamma,
        t_end = t_end,
        wind_params = wind_params,
        cooling_params = CoolingParams(
            hydrogen_mass_fraction = hydrogen_mass_fraction,
            metal_mass_fraction = metal_mass_fraction,
            floor_temperature = floor_temperature,
            cooling_curve_params = cooling_curve_params
        ),
        snapshot_timepoints = jnp.linspace(0, t_end, num_snapshots + 2)[2:]
    )

    # homogeneous initial state
    rho_0 = 10 ** (-22.5) * u.g / u.cm**3
    T_0 = 100 * u.K * c.k_B / c.m_p
    p_0 = get_pressure_from_temperature(rho_0, T_0, hydrogen_mass_fraction, metal_mass_fraction)
    print(p_0)

    rho_init = jnp.ones(num_cells) * rho_0.to(code_units.code_density).value
    u_init = jnp.zeros(num_cells)
    p_init = jnp.ones(num_cells) * p_0.to(code_units.code_pressure).value

    # get initial state
    initial_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho_init,
        velocity_x = u_init,
        gas_pressure = p_init
    )

    config = finalize_config(config, initial_state.shape)

    return initial_state, config, params, helper_data, registered_variables

def plot_density(ax, state, registered_variables, helper_data, code_units, label):
    rho = state[registered_variables.density_index]
    rho = rho * code_units.code_density
    r = helper_data.geometric_centers * code_units.code_length
    ax.plot(r.to(u.cm), (rho).to(u.g * u.cm**-3), label=label)
    ax.set_yscale("log")
    ax.set_title("density")
    ax.set_ylabel(r"$\rho$ in g cm$^{-3}$")
    ax.set_ylim(1e-27, 1e-21)
    ax.set_xlim(0, 1e19)
    ax.set_xlabel("r in cm")


# compare with weaver solution
def plot_profiles(axs, final_state, registered_variables, helper_data, code_units, label = "astronomix", left_gray = False, start_index = 0, color = "blue"):
    print("👷 generating plots")

    rho = final_state[registered_variables.density_index]
    vel = final_state[registered_variables.velocity_index]
    p = final_state[registered_variables.pressure_index]

    rho = rho * code_units.code_density
    vel = vel * code_units.code_velocity
    p = p * code_units.code_pressure

    r = helper_data.geometric_centers * code_units.code_length

    # temperature

    T = get_temperature_from_pressure(rho, p * c.m_p / c.k_B, hydrogen_mass_fraction, metal_mass_fraction)
    T = T.to(u.K)

    axs[0].set_yscale("log")
    axs[0].plot(r.to(u.cm), (rho).to(u.g * u.cm**-3), label=label, color = color)
    axs[0].set_title("density")
    axs[0].set_ylabel(r"$\rho$ in m$_p$ cm$^{-3}$")
    axs[0].set_ylim(1e-27, 1e-21)
    axs[0].set_xlim(0, 1e19)
    axs[0].legend(loc="lower right")
    axs[0].set_xlabel("r in cm")

    axs[1].set_yscale("log")
    axs[1].plot(r.to(u.cm), (p / c.k_B).to(u.K / u.cm**3), label=label, color = color)
    axs[1].set_title("pressure")
    axs[1].set_ylabel(r"$p$/k$_b$ in K cm$^{-3}$")
    axs[1].set_xlim(0, 1e19)
    axs[1].legend(loc="upper left")
    axs[1].set_xlabel("r in cm")

    axs[2].set_yscale("log")
    axs[2].plot(r.to(u.cm), vel.to(u.km / u.s), label=label, color = color)
    axs[2].set_title("velocity")
    axs[2].set_ylim(1, 1e4)
    axs[2].set_xlim(0, 1e19)
    axs[2].set_ylabel("v in km/s")
    axs[2].legend(loc="upper right")
    axs[2].set_xlabel("r in cm")

    axs[3].set_yscale("log")
    axs[3].plot(r.to(u.cm), T.to(u.K), label=label, color = color)
    axs[3].set_title("temperature")
    axs[3].set_ylim(10, 1e9)
    axs[3].set_xlim(0, 1e19)
    axs[3].set_ylabel("T in K")
    axs[3].legend(loc="upper right")
    axs[3].set_xlabel("r in cm")

    if left_gray:
        # add gray background to r < r[start_index]
        axs[0].axvspan(0, r[start_index].to(u.cm).value, color="gray", alpha=0.3)
        axs[1].axvspan(0, r[start_index].to(u.cm).value, color="gray", alpha=0.3)
        axs[2].axvspan(0, r[start_index].to(u.cm).value, color="gray", alpha=0.3)
        axs[3].axvspan(0, r[start_index].to(u.cm).value, color="gray", alpha=0.3)

# problem setting
if plot_problem_setting:

    # one ax with cooling, one without
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for high_res in high_res_s:

        num_cells = high_res
        
        # setup simulation without cooling
        initial_state, config, params, helper_data, registered_variables = setup_simulation(
            num_cells = num_cells,
            cooling_curve_type = PIECEWISE_POWER_LAW,
            cooling_curve_params = schure_cooling_params,
            return_snapshots = False,
            num_snapshots = 0,
            cooling = False,
        )
        # run simulation without cooling
        # final_state = time_integration(initial_state, config, params, registered_variables)
        # load from file
        final_state = jnp.load(f"data/reference_states_no_cooling{high_res}.npy", allow_pickle=True)[-1]

        plot_density(axs[0], final_state, registered_variables, helper_data, code_units, label = f"{num_cells} cells") # , {str(num_injection_cells) + ' inj. cells' if not fixed_r else 'R_inj = ' + str(r_inj) + ' L'}

        # setup simulation with cooling
        initial_state, config, params, helper_data, registered_variables = setup_simulation(
            num_cells = num_cells,
            cooling_curve_type = PIECEWISE_POWER_LAW,
            cooling_curve_params = schure_cooling_params,
            return_snapshots = False,
            num_snapshots = 0,
            cooling = True,
        )
        # run simulation with cooling
        # final_state = time_integration(initial_state, config, params, registered_variables)
        final_state = jnp.load(f"data/reference_states{high_res}.npy", allow_pickle=True)[-1]
        plot_density(axs[1], final_state, registered_variables, helper_data, code_units, label = f"{num_cells} cells") # , {str(num_injection_cells) + ' inj. cells' if not fixed_r else 'R_inj = ' + str(r_inj) + ' L'}
    
    axs[0].set_title("without cooling")
    axs[1].set_title("with cooling")
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.savefig("figures/problem_setting.svg")

reference_state_collection = []

for high_res in high_res_s:
    # get reference simulation
    initial_state, config, params, helper_data, registered_variables = setup_simulation(
        num_cells = high_res,
        cooling_curve_type = PIECEWISE_POWER_LAW,
        cooling_curve_params = schure_cooling_params,
        return_snapshots = True,
        num_snapshots = 5,
        num_injection_cells = get_num_injection_cells(r_inj, high_res)
    )

    if run_simulation:
        result = time_integration(initial_state, config, params, registered_variables)
        reference_states = result.states
        # save reference states as numpy array
        jnp.save(f"data/reference_states{high_res}.npy", jnp.array(reference_states))
        
        # also run without cooling for comparison
        config_no_cooling = config._replace(
            cooling_config = CoolingConfig(
                cooling = False,
            )
        )
        result_no_cooling = time_integration(initial_state, config_no_cooling, params, registered_variables)
        reference_states_no_cooling = result_no_cooling.states
        jnp.save(f"data/reference_states_no_cooling{high_res}.npy", jnp.array(reference_states_no_cooling))
    else:
        reference_states = jnp.load(f"data/reference_states{high_res}.npy", allow_pickle=True)

    reference_state_collection.append(reference_states)

def downsample(states):
    return jnp.mean(jnp.reshape(states, (states.shape[0], states.shape[1], low_res, -1)), axis = -1)

downsampled_collection = [downsample(states) for states in reference_state_collection]


initial_state, config, params, helper_data_low_res, registered_variables = setup_simulation(
    num_cells = low_res,
    cooling_curve_type = NEURAL_NET_COOLING_WITH_DENSITY,
    cooling_curve_params = cooling_curve_paramsC,
    return_snapshots = True,
    num_snapshots = 5,
    num_injection_cells = get_num_injection_cells(r_inj, low_res)
)

learning_reference_timepoints = (params.snapshot_timepoints * code_units.code_time).to(u.yr).value


# training loop
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(cooling_corrector_params)

best_loss = jnp.inf
best_params = cooling_corrector_params

if train_model:

    # time training
    start_time = timer()

    for training_phase_index in range(len(high_res_s)):

        print(f"--- Training phase {training_phase_index + 1}/{len(high_res_s)} ---")

        reference_states_downsampled = downsampled_collection[training_phase_index]
        best_loss = jnp.inf

        if training_phase_index > 0:
            # start from best params of previous phase
            cooling_corrector_params = best_params
            opt_state = optimizer.init(cooling_corrector_params)

        # loss function
        @eqx.filter_jit
        def loss_fn_train(network_params):
            """Calculates the difference between the final state and the target."""
            params_new = params._replace(
                cooling_params = params.cooling_params._replace(
                    cooling_curve_params = CoolingNetParams(
                        network_params = network_params
                    )
                )
            )
            result = time_integration(initial_state, config, params_new, registered_variables)
            return jnp.mean(((result.states[:, :, beginning_index:] - reference_states_downsampled[:, :, beginning_index:]) / jnp.max(reference_states_downsampled[:, :, beginning_index:], axis = (0, 2))[None, :, None]) ** 2)

        @eqx.filter_jit
        def train_step_train(network_params_arrays, opt_state):
            """Performs one step of gradient descent."""
            loss_value, grads = eqx.filter_value_and_grad(loss_fn_train)(network_params_arrays)
            updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
            network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
            return network_params_arrays, opt_state, loss_value

        for epoch in range(num_epochs_corr[training_phase_index]):

            cooling_corrector_params, opt_state, loss_value = train_step_train(cooling_corrector_params, opt_state)

            if loss_value < best_loss:
                best_loss = loss_value
                best_params = cooling_corrector_params

            if epoch % 1 == 0:
                print(f"Training Epoch {epoch+1}/{num_epochs_corr[training_phase_index]}, Loss: {loss_value:.6f}")

    end_time = timer()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # use best params
    cooling_corrector_params = best_params

    # save the trained network parameters with pickle
    with open("models/cooling_corrector_rho" + str(high_res) + "_" + str(low_res) +  ".pkl", "wb") as f:
        pickle.dump(cooling_corrector_params, f)
else:
    # load the trained network parameters with pickle
    # import pickle
    with open("models/cooling_corrector_rho" + str(high_res) + "_" + str(low_res) +  ".pkl", "rb") as f:
        cooling_corrector_params = pickle.load(f)


# plot low res simulation result after training
params = params._replace(
    cooling_params = params.cooling_params._replace(
        cooling_curve_params = CoolingNetParams(
            network_params = cooling_corrector_params
        )
    )
)
result = time_integration(initial_state, config, params, registered_variables)
low_res_states_corrected = result.states
final_state = low_res_states_corrected[-1]
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
plot_profiles(axs, final_state, registered_variables, helper_data_low_res, code_units)
plt.tight_layout()
plt.savefig("figures/low_res_output_corrected.svg")

high_res = high_res_s[-1]

if run_high_res_for_error_plot:
    initial_state, config, params, helper_data_low_res, registered_variables = setup_simulation(
        num_cells = high_res,
        cooling_curve_type = PIECEWISE_POWER_LAW,
        cooling_curve_params = schure_cooling_params,
        return_snapshots = True,
        num_snapshots = 200,
        t_final = 1.3e12 * u.s,
        num_injection_cells = get_num_injection_cells(r_inj, high_res)
    )
    err_timepoints = (params.snapshot_timepoints * code_units.code_time).to(u.yr).value
    result = time_integration(initial_state, config, params, registered_variables)
    reference_states = result.states

    # save err_timepoints and reference_states as numpy array
    jnp.save(f"data/timepoints_error_plot{high_res}.npy", jnp.array(err_timepoints))
    jnp.save(f"data/reference_states_error_plot{high_res}.npy", jnp.array(reference_states))
else:
    err_timepoints = jnp.load(f"data/timepoints_error_plot{high_res}.npy", allow_pickle=True)
    reference_states = jnp.load(f"data/reference_states_error_plot{high_res}.npy", allow_pickle=True)

reference_states_downwampled = jnp.mean(jnp.reshape(reference_states, (reference_states.shape[0], reference_states.shape[1], low_res, -1)), axis = -1)

# uncorrected low res
initial_state, config, params, helper_data_low_res, registered_variables = setup_simulation(
    num_cells = low_res,
    cooling_curve_type = PIECEWISE_POWER_LAW,
    cooling_curve_params = schure_cooling_params,
    return_snapshots = True,
    num_snapshots = 200,
    t_final = 1.3e12 * u.s,
    num_injection_cells = get_num_injection_cells(r_inj, low_res)
)
result = time_integration(initial_state, config, params, registered_variables)
low_res_states = result.states
error_uncorrected = jnp.mean(((low_res_states[:, :, beginning_index:] - reference_states_downwampled[:, :, beginning_index:]) / jnp.max(reference_states_downwampled[:, :, beginning_index:], axis = (0, 2))[None, :, None]) ** 2, axis = (1, 2))

# corrected low res
initial_state, config, params, helper_data_low_res, registered_variables = setup_simulation(
    num_cells = low_res,
    cooling_curve_type = NEURAL_NET_COOLING_WITH_DENSITY,
    cooling_curve_params = CoolingNetParams(
        network_params = cooling_corrector_params
    ),
    return_snapshots = True,
    num_snapshots = 200,
    t_final = 1.3e12 * u.s,
    num_injection_cells = get_num_injection_cells(r_inj, low_res)
)
result = time_integration(initial_state, config, params, registered_variables)
low_res_states_corrected = result.states
error_corrected = jnp.mean(((low_res_states_corrected[:, :, beginning_index:] - reference_states_downwampled[:, :, beginning_index:]) / jnp.max(reference_states_downwampled[:, :, beginning_index:], axis = (0, 2))[None, :, None]) ** 2, axis = (1, 2))

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 4, height_ratios=[2, 1])

# --- Top row: profiles ---
axs_profiles = [fig.add_subplot(gs[0, i]) for i in range(4)]

plot_profiles(
    axs_profiles, reference_states_downwampled[-1],
    registered_variables, helper_data_low_res, code_units,
    label=f"reference ({high_res} cells)", color="black"
)
plot_profiles(
    axs_profiles, low_res_states[-1],
    registered_variables, helper_data_low_res, code_units,
    label=f"uncorrected ({low_res} cells)", color="red"
)
plot_profiles(
    axs_profiles, low_res_states_corrected[-1],
    registered_variables, helper_data_low_res, code_units,
    label=f"low res corrected ({low_res} cells)",
    start_index=beginning_index, left_gray=True, color="green"
)

# Remove legends from individual profile axes (if any)
for ax in axs_profiles:
    leg = ax.get_legend()
    if leg:
        leg.remove()

# --- Bottom row: error plot ---
ax_error = fig.add_subplot(gs[1, :])
ax_error.plot(err_timepoints, error_uncorrected, color="red")
ax_error.plot(err_timepoints, error_corrected, color="green")
for i, t in enumerate(learning_reference_timepoints):
    if i == 0:
        ax_error.axvline(t, color="gray", linestyle="--", alpha=0.5, label="reference timepoints")
    else:
        ax_error.axvline(t, color="gray", linestyle="--", alpha=0.5)
ax_error.set_xlabel("time in years")
ax_error.set_ylabel("mean squared error")
ax_error.set_title("error over time")
ax_error.legend(loc="upper left")  # keep its own legend

# --- Combined legend for profiles ---
handles = [
    plt.Line2D([], [], color="black", label=f"reference ({high_res} cells)"),
    plt.Line2D([], [], color="red", label=f"uncorrected low resolution ({low_res} cells)"),
    plt.Line2D([], [], color="green", label=f"corrected low resolution ({low_res} cells)")
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.00),   # closer to the bottom subplot
    frameon=False
)

plt.tight_layout(rect=[0, 0.03, 1, 1])  # leave a little space for legend
plt.savefig("figures/final_comparison_rho" + str(high_res) + "_" + str(low_res) +  ".svg")