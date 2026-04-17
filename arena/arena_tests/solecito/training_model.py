from autocvd import autocvd
import os

from jax.sharding import PartitionSpec

# autocvd(num_gpus=1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
import logging
from typing import Tuple, Callable, Union
from jaxtyping import Array, PyTree, Scalar
from jax import transfer_guard_host_to_device, vmap
import optax
import jax.numpy as jnp
import equinox as eqx
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.data_classes.simulation_snapshot_data import SnapshotData
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    BoundarySettings,
    BoundarySettings1D,
    PERIODIC_BOUNDARY,
    BACKWARDS,
)

from timeit import default_timer as timer
import jax
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
    ScalarFieldCorrectorCNN,
    VectorField,
    VectorFieldCorrectorCNN,
)
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import (
    RegisteredVariables,
    StaticIntVector,
)
from astronomix import (
    get_helper_data,
    finalize_config,
    get_registered_variables,
    initialize_interface_fields,
    construct_primitive_state,
    SimulationParams,
    SimulationConfig,
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from functools import partial
from pathlib import Path
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int
    model_name: str = "default"
    learning_rate: float = 7e-5
    peak_lr: float = 2.5e-4
    end_lr: float = 7e-5
    warmup_steps_fraction: float = 0.4
    hidden_channels: int = 5
    hidden_layers: int = 3
    model_initialization_scale: float = 0.05
    noise_level: float = 0.05
    gradient_clip: float = 1.0
    num_cells_high_res: int = 32
    downaverage_factor: int = 2
    c_cfl: float = 1.0
    limiter: int = 4  # VAN ALBADA safe at r=0
    t_end: float = 0.2
    num_timesteps: int = 2000
    use_early_stopper: bool = True
    patience: int = 35

    def __str__(self) -> str:
        """Pretty print the configuration."""
        lines = ["TrainingConfig:"]
        for key, value in asdict(self).items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def save(self, base_path: str = "arena/arena_tests/solecito/models") -> Path:
        """
        Save the configuration to a folder named after the experiment.

        Args:
            base_path: Base directory where experiment folders are created

        Returns:
            Path to the created experiment folder
        """
        # Create the experiment folder path
        experiment_path = Path(base_path) / self.model_name
        experiment_path.mkdir(parents=True, exist_ok=True)

        # Save configuration as JSON
        config_file = experiment_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(asdict(self), f, indent=2)

        # Also save as a human-readable text file
        config_txt = experiment_path / "config.txt"
        with open(config_txt, "w") as f:
            f.write(str(self))

        print(f"Configuration saved to: {experiment_path}")
        return experiment_path

    @classmethod
    def load(
        cls, experiment_name: str, base_path: str = "arena/arena_tests/solecito/models"
    ):
        """
        Load a configuration from an experiment folder.

        Args:
            experiment_name: Name of the experiment (folder name)
            base_path: Base directory where experiment folders are located

        Returns:
            TrainingConfig instance loaded from the saved configuration
        """
        config_file = Path(base_path) / experiment_name / "config.json"

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                f"Make sure the experiment '{experiment_name}' exists."
            )

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        return cls(**config_dict)


@dataclass
class FullCNNConfig(TrainingConfig):
    def create_model(self):
        return CorrectorCNN(
            in_channels=11,
            hidden_channels=self.hidden_channels,
            hidden_layers=self.hidden_layers,
            key=jax.random.PRNGKey(100),
            scale=self.model_initialization_scale,
        )


@dataclass
class ScalarFieldCNNConfig(TrainingConfig):
    channel_index: int = 0

    def create_model(self):
        return ScalarFieldCorrectorCNN(
            in_channels=11,
            hidden_channels=self.hidden_channels,
            hidden_layers=self.hidden_layers,
            channel_index_output=self.channel_index,
            key=jax.random.PRNGKey(100),
            scale=self.model_initialization_scale,
        )


@dataclass
class VectorFieldCNNConfig(TrainingConfig):
    channel_index: int = 0
    vector_field: str = "velocity"

    def create_model(self):
        # Convert string to VectorField enum when creating model
        vector_field_enum = VectorField(self.vector_field)
        return VectorFieldCorrectorCNN(
            in_channels=11,
            hidden_channels=self.hidden_channels,
            hidden_layers=self.hidden_layers,
            vector_field_output=vector_field_enum,
            key=jax.random.PRNGKey(100),
            scale=self.model_initialization_scale,
        )


def downaverage(state: Array, downaverage_factor: int) -> Array:
    """Downaverage spatial (and depth) dimensions by non-overlapping block averaging.

    This function accepts either:
      - unbatched input of shape (NUM_VARS, H, W, D)
      - batched input of shape (N, NUM_VARS, H, W, D)

    The downaverage_factor is an integer factor by which each spatial/depth
    dimension (H, W, D) is reduced:
        h_out = H // downaverage_factor
        w_out = W // downaverage_factor
        d_out = D // downaverage_factor

    Args:
        state: JAX ndarray with shape (NUM_VARS, H, W, D) or (N, NUM_VARS, H, W, D).
        downaverage_factor: integer factor > 0 that divides H, W and D.

    Returns:
        downaveraged array with shape:
            - (NUM_VARS, h_out, w_out, d_out) for unbatched input
            - (N, NUM_VARS, h_out, w_out, d_out) for batched input

    Raises:
        ValueError: if input ndim is not 4 or 5, or if spatial/depth dims are not divisible
                    by downaverage_factor.

    """
    downaverage_factor = int(downaverage_factor)
    if downaverage_factor <= 0:
        raise ValueError("downaverage_factor must be a positive integer")

    if state.ndim == 4:
        # (NUM_VARS, H, W, D)
        num_vars, H, W, D = state.shape
        if (
            (H % downaverage_factor) != 0
            or (W % downaverage_factor) != 0
            or (D % downaverage_factor) != 0
        ):
            raise ValueError(
                f"Spatial/depth dims {(H, W, D)} must be divisible by downaverage_factor={downaverage_factor}"
            )
        h_out = H // downaverage_factor
        w_out = W // downaverage_factor
        d_out = D // downaverage_factor

        # reshape into blocks and mean over block axes
        reshaped = state.reshape(
            num_vars,
            h_out,
            downaverage_factor,
            w_out,
            downaverage_factor,
            d_out,
            downaverage_factor,
        )
        # mean over the block axes (2, 4, 6)
        downaveraged = reshaped.mean(axis=(2, 4, 6))
        return downaveraged

    elif state.ndim == 5:
        # (N, NUM_VARS, H, W, D)
        N, num_vars, H, W, D = state.shape
        if (
            (H % downaverage_factor) != 0
            or (W % downaverage_factor) != 0
            or (D % downaverage_factor) != 0
        ):
            raise ValueError(
                f"Spatial/depth dims {(H, W, D)} must be divisible by downaverage_factor={downaverage_factor}"
            )
        h_out = H // downaverage_factor
        w_out = W // downaverage_factor
        d_out = D // downaverage_factor

        reshaped = state.reshape(
            N,
            num_vars,
            h_out,
            downaverage_factor,
            w_out,
            downaverage_factor,
            d_out,
            downaverage_factor,
        )
        # mean over the block axes (3, 5, 7)
        downaveraged = reshaped.mean(axis=(3, 5, 7))
        return downaveraged

    else:
        raise ValueError(
            f"Unsupported input ndim {state.ndim}. Expected 4 (NUM_VARS,H,W,D) or "
            f"5 (N,NUM_VARS,H,W,D)."
        )


def initialize_training_data(
    training_config: TrainingConfig,
) -> Tuple[
    Tuple[Array, SimulationConfig, SimulationParams, HelperData, RegisteredVariables],
    Tuple[Array, SimulationConfig, SimulationParams, HelperData, RegisteredVariables],
]:
    simulation_config_high_res = SimulationConfig(
        num_cells=training_config.num_cells_high_res,
        solver_mode=FINITE_DIFFERENCE,
        differentiation_mode=BACKWARDS,
        mhd=True,
        dimensionality=3,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ),
        # TODO: Try without exact end time to see if compilation time improves
        exact_end_time=True,
        num_checkpoints=50,
        limiter=training_config.limiter,
        fixed_timestep=False,
        # NOTE: Couldnt manage to get fixed timestep to work (some problems with the checkpointed graph computation)
        # num_timesteps=training_config.num_timesteps,
    )

    simulation_params = SimulationParams(
        C_cfl=training_config.c_cfl, t_end=training_config.t_end
    )
    helper_data_high_res = get_helper_data(simulation_config_high_res)
    registered_variables = get_registered_variables(simulation_config_high_res)

    r = helper_data_high_res.r
    r0 = 0.125
    r1 = 1.1 * r0

    rho = jnp.ones_like(r)
    P = jnp.ones_like(r) * 1.0
    P = jnp.where(r <= r0, 100.0, P)
    P = jnp.where((r > r0) & (r <= r1), 1.0 + 99.0 * (r1 - r) / (r1 - r0), P)

    V_x = jnp.zeros_like(r)
    V_y = jnp.zeros_like(r)
    V_z = jnp.zeros_like(r)

    B0 = 10

    B_x = B0 / jnp.sqrt(2)
    B_y = B0 / jnp.sqrt(2)
    B_z = 0.0

    B_x = jnp.ones_like(r) * B_x
    B_y = jnp.ones_like(r) * B_y
    B_z = jnp.ones_like(r) * B_z

    bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

    initial_state_high_res = construct_primitive_state(
        config=simulation_config_high_res,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=V_x,
        velocity_y=V_y,
        velocity_z=V_z,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
        gas_pressure=P,
    )

    simulation_config_high_res = finalize_config(
        simulation_config_high_res, initial_state_high_res.shape
    )
    initial_state_low_res = downaverage(
        state=initial_state_high_res,
        downaverage_factor=training_config.downaverage_factor,
    )
    simulation_config_low_res = simulation_config_high_res._replace(
        num_cells=simulation_config_high_res.num_cells
        // training_config.downaverage_factor
    )
    simulation_config_low_res = finalize_config(
        simulation_config_low_res, initial_state_low_res.shape
    )
    helper_data_low_res = get_helper_data(simulation_config_low_res)
    return (
        (
            initial_state_high_res,
            simulation_config_high_res,
            simulation_params,
            helper_data_high_res,
            registered_variables,
        ),
        (
            initial_state_low_res,
            simulation_config_low_res,
            simulation_params,
            helper_data_low_res,
            registered_variables,
        ),
    )


def initialize_target_data(
    initial_state: Array,
    simulation_config: SimulationConfig,
    simulation_params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    training_config: TrainingConfig,
) -> Array:
    filepath = "arena/data/solecito_target.npy"
    if not os.path.exists(filepath):
        final_state_high_res = time_integration(
            primitive_state=initial_state,
            config=simulation_config._replace(progress_bar=True),
            params=simulation_params,
            registered_variables=registered_variables,
        )
        assert isinstance(final_state_high_res, Array)
        final_state_low_res = downaverage(
            state=final_state_high_res,
            downaverage_factor=training_config.downaverage_factor,
        )
        jnp.save(filepath, final_state_low_res)
    else:
        final_state_low_res = jnp.load(filepath)
    return final_state_low_res


def perturb_state(key: Array, state: jnp.ndarray, noise_level: float = 0.01):
    mask = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])[
        :, None, None, None
    ]
    noise = jax.random.normal(key, shape=state.shape) * noise_level * mask
    perturbed_state = state + noise
    perturbed_state = perturbed_state.at[0].set(jnp.maximum(perturbed_state[0], 0.0))
    perturbed_state = perturbed_state.at[4].set(jnp.maximum(perturbed_state[4], 0.0))
    return perturbed_state


def create_train_step(
    loss_fn: Callable[[Array, Array], Array],
    optimizer: optax.GradientTransformation,
    simulation_config: SimulationConfig,
    initial_state: Array,
    target_state: Array,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    simulation_params: SimulationParams,
    noise_level: float,
):
    perturb_state_partial = partial(
        perturb_state, state=initial_state, noise_level=noise_level
    )

    def train_step_core(network_params_arrays, opt_state, key):
        initial_state = perturb_state_partial(key)

        def loss_fn_sol(network_params_arrays):
            results_low_res = time_integration(
                initial_state,
                simulation_config,
                simulation_params._replace(
                    cnn_mhd_corrector_params=simulation_params.cnn_mhd_corrector_params._replace(
                        network_params=network_params_arrays
                    )
                ),
                registered_variables,
            )
            # assert isinstance(results_low_res, Array), "results is not a snapshot data"
            loss = loss_fn(
                results_low_res,
                target_state,
            )

            return loss

        loss_value, grads = eqx.filter_value_and_grad(loss_fn_sol)(
            network_params_arrays
        )
        gradients_modulus = jnp.sqrt(
            sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads))
        )
        updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
        network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
        return (
            network_params_arrays,
            opt_state,
            loss_value,
            gradients_modulus,
        )

    train_step = jax.jit(train_step_core)

    return train_step


def initialize_optimizer(
    training_config: TrainingConfig, neural_net_params: PyTree
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    warmup_steps = int(training_config.epochs * training_config.warmup_steps_fraction)
    decay_steps = int(training_config.epochs - warmup_steps)
    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=training_config.learning_rate,
        peak_value=training_config.peak_lr,
        end_value=training_config.end_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )

    # Initialize_optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(training_config.gradient_clip),
        optax.adamw(learning_rate=lr_scheduler),
    )
    opt_state = optimizer.init(neural_net_params)

    return (optimizer, opt_state)


def initialize_loss_fn(target_state: Array):
    channel_normalizers = jnp.maximum(jnp.std(target_state, axis=(1, 2, 3)), 1e-8)

    def normalized_mse_loss(
        pred_state: jnp.ndarray,
        target_state: jnp.ndarray,
    ):
        normalized_error = (pred_state - target_state) / channel_normalizers[
            :, None, None, None
        ]
        return jnp.mean(normalized_error[:8] ** 2)

    return normalized_mse_loss


def plot_training(
    neural_net_params: PyTree,
    times_eval: jnp.ndarray,
    training_config: Union[ScalarFieldCNNConfig, VectorFieldCNNConfig, FullCNNConfig],
    losses: list[float],
    image_folder: Path,
):
    """
    Args:
        times_eval: times at which to evaluate the loss
        snapshot_timepoints_train: times used for training the model
    """

    if training_config.t_end not in times_eval:
        times_eval = jnp.sort(
            jnp.concatenate([times_eval, jnp.array([training_config.t_end])])
        )

    # Get the index of the trained times
    t_end_index = int(jnp.argmax(times_eval == training_config.t_end))

    num_cells_low_res = (
        training_config.num_cells_high_res // training_config.downaverage_factor
    )

    (
        (
            initial_state_high_res,
            simulation_config_high_res,
            simulation_params,
            helper_data_high_res,
            registered_variables,
        ),
        (
            initial_state_low_res,
            simulation_config_low_res,
            simulation_params,
            helper_data_low_res,
            registered_variables,
        ),
    ) = initialize_training_data(training_config=training_config)
    simulation_config_high_res = simulation_config_high_res._replace(
        return_snapshots=True,
        num_snapshots=len(times_eval),
        use_specific_snapshot_timepoints=True,
    )
    simulation_config_low_res = simulation_config_low_res._replace(
        return_snapshots=True,
        num_snapshots=len(times_eval),
        use_specific_snapshot_timepoints=True,
    )
    simulation_params = simulation_params._replace(
        snapshot_timepoints=jnp.array(times_eval)
    )

    result_high_res = time_integration(
        primitive_state=initial_state_high_res,
        config=simulation_config_high_res,
        params=simulation_params,
        registered_variables=registered_variables,
    )
    assert isinstance(result_high_res, SnapshotData)
    states_target_low_res = downaverage(
        result_high_res.states, downaverage_factor=training_config.downaverage_factor
    )

    states_low_res_uncorrected = time_integration(
        primitive_state=initial_state_low_res,
        config=simulation_config_low_res,
        params=simulation_params,
        helper_data=helper_data_low_res,
        registered_variables=registered_variables,
    ).states

    logger.info("Initializing model")
    cnn_mhd_corrector_params, cnn_mhd_corrector_config = initialize_model(
        registered_variables=registered_variables, training_config=training_config
    )

    # NOTE: sol stands for solver in the loop
    simulation_config_low_res_sol = simulation_config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    simulation_params_sol = simulation_params._replace(
        cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
            network_params=neural_net_params
        )
    )

    states_low_res = time_integration(
        primitive_state=initial_state_low_res,
        config=simulation_config_low_res_sol,
        params=simulation_params_sol,
        helper_data=helper_data_low_res,
        registered_variables=registered_variables,
    ).states

    final_state_target_low_res = states_target_low_res[t_end_index]
    final_state_low_res_uncorrected = states_low_res_uncorrected[t_end_index]
    final_state_low_res = states_low_res[t_end_index]

    # --- Create figure and layout ---
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 3, height_ratios=[1, 1, 1, 1])

    # First row: density images
    axs_density = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axs_magnetic = [fig.add_subplot(gs[1, i]) for i in range(3)]

    titles_density = [
        "Target State (Density)",
        "Final State before Training (Density)",
        "Final State after Training (Density)",
    ]
    titles_magnetic = [
        "Target State (|B|^2 along diagonal)",
        "Final State before Training (|B|^2 along diagonal)",
        "Final State after Training (|B|^2 along diagonal)",
    ]

    states = [
        final_state_target_low_res,
        final_state_low_res_uncorrected,
        final_state_low_res,
    ]

    loss_fn = initialize_loss_fn(target_state=final_state_target_low_res)
    l2_error_initial = float(
        loss_fn(final_state_low_res_uncorrected, final_state_target_low_res)
    )

    v_loss = vmap(loss_fn, in_axes=(0, 0))

    l2_errors_corrected = v_loss(states_low_res, states_target_low_res)
    l2_errors_uncorrected = v_loss(states_low_res_uncorrected, states_target_low_res)

    # Shared color scale
    vmin = float(
        min(
            jnp.min(s[registered_variables.density_index]).astype(float) for s in states
        )
    )
    vmax = float(
        max(
            jnp.max(s[registered_variables.density_index]).astype(float) for s in states
        )
    )

    for ax_density, ax_magnetic, state, title_density, title_magnetic in zip(
        axs_density, axs_magnetic, states, titles_density, titles_magnetic, strict=True
    ):
        im = ax_density.imshow(
            state[registered_variables.density_index, :, :, num_cells_low_res // 2],
            extent=(
                0,
                simulation_config_low_res.box_size,
                0,
                simulation_config_low_res.box_size,
            ),
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax_density.set_title(title_density)
        ax_density.set_aspect("equal", adjustable="box")
        divider = make_axes_locatable(ax_density)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, label="Density")

        # Second row: magnetic field across the diagonal
        diag_indices = jnp.arange(0, num_cells_low_res)

        assert isinstance(registered_variables.magnetic_index, StaticIntVector)
        b_squared = (
            state[registered_variables.magnetic_index.x] ** 2
            + state[registered_variables.magnetic_index.y] ** 2
            + state[registered_variables.magnetic_index.z] ** 2
        )
        B_diag = b_squared[diag_indices, diag_indices, num_cells_low_res // 2]
        r_diag = jnp.sqrt((diag_indices) ** 2 + (diag_indices) ** 2) * (
            simulation_config_low_res.box_size / num_cells_low_res
        )
        ax_magnetic.plot(r_diag, B_diag)
        ax_magnetic.set_ylabel("|B|^2")
        ax_magnetic.set_xlabel("diagonal")
        ax_magnetic.set_title(title_magnetic)

    # Third row: loss curve
    ax_loss = fig.add_subplot(gs[2, :])
    ax_loss.plot(losses, label="Training Loss")
    ax_loss.axhline(
        y=l2_error_initial,
        color="r",
        linestyle="--",
        label="Initial L2 Error (uncorrected)",
    )

    ax_loss.set_xlabel("Training Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss During Training")
    ax_loss.legend()

    # Fourth row: L2 error over time
    ax_errors = fig.add_subplot(gs[3, :])
    ax_errors.plot(
        times_eval, l2_errors_corrected, label="Corrected Integration", color="tab:blue"
    )
    ax_errors.plot(
        times_eval,
        l2_errors_uncorrected,
        label="Uncorrected Integration",
        color="tab:orange",
        linestyle="--",
    )

    ax_errors.set_xlabel("Time")
    ax_errors.set_ylabel("L2 Error")
    ax_errors.set_yscale("log")
    ax_errors.set_title("Mean Squared Error Over Time")
    ax_errors.legend()

    plt.tight_layout()
    plt.savefig(image_folder / "summary.png", dpi=400)


class EarlyStopper:
    max_patience: int = 10
    best_loss: float = math.inf
    patience: int = 0
    early_stopped: bool = False

    def __init__(self, max_patience: int, use_early_stopper: bool = False):
        self.max_patience = max_patience
        self.use_early_stopper = use_early_stopper

    def new_epoch(self, loss: float) -> bool:
        if self.use_early_stopper:
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience = 0
            else:
                self.patience += 1
                if self.patience == self.max_patience:
                    self.early_stopped = True
                    return True
            return False
        else:
            return False

    def reset_patience(self):
        self.patience = 0
        self.best_loss = math.inf


def initialize_model(
    registered_variables: RegisteredVariables,
    training_config: Union[ScalarFieldCNNConfig, VectorFieldCNNConfig, FullCNNConfig],
) -> Tuple[CNNMHDParams, CNNMHDconfig]:
    model = training_config.create_model()
    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=neural_net_static,
        correct_from_beggining=True,
    )

    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)
    return cnn_mhd_corrector_params, cnn_mhd_corrector_config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    training_config = VectorFieldCNNConfig(
        model_name="best_model",
        vector_field="magnetic",
        c_cfl=1.2,
        hidden_layers=2,
        hidden_channels=24,
        learning_rate=8.5e-5,
        peak_lr=0.004,
        end_lr=3e-5,
        warmup_steps_fraction=0.25,
        model_initialization_scale=0.018,
        noise_level=0.018,
        epochs=300,
    )
    print(training_config)
    model_folder = training_config.save()
    (
        (
            initial_state_high_res,
            simulation_config_high_res,
            simulation_params,
            helper_data_high_res,
            registered_variables,
        ),
        (
            initial_state_low_res,
            simulation_config_low_res,
            simulation_params,
            helper_data_low_res,
            registered_variables,
        ),
    ) = initialize_training_data(training_config=training_config)

    logger.info("Initializing model")
    cnn_mhd_corrector_params, cnn_mhd_corrector_config = initialize_model(
        registered_variables=registered_variables, training_config=training_config
    )

    logging.info("initializing target data")
    target_state = initialize_target_data(
        initial_state=initial_state_high_res,
        simulation_config=simulation_config_high_res,
        simulation_params=simulation_params,
        helper_data=helper_data_high_res,
        registered_variables=registered_variables,
        training_config=training_config,
    )

    logging.info("initializing optimizer")
    optimizer, opt_state = initialize_optimizer(
        training_config=training_config,
        neural_net_params=cnn_mhd_corrector_params.network_params,
    )
    # NOTE: sol stands for solver in the loop
    simulation_config_low_res_sol = simulation_config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    simulation_params_low_res_sol = simulation_params._replace(
        cnn_mhd_corrector_params=cnn_mhd_corrector_params
    )

    losses = []
    best_loss = float("inf")
    trained_params = cnn_mhd_corrector_params.network_params
    best_params = cnn_mhd_corrector_params.network_params
    key = jax.random.PRNGKey(100)
    early_stopper = EarlyStopper(
        max_patience=training_config.patience,
        use_early_stopper=training_config.use_early_stopper,
    )
    logger.info("Starting training...")

    logging.info("initializing loss")
    loss_fn = initialize_loss_fn(target_state=target_state)

    logging.info("creating train step")
    train_step = create_train_step(
        loss_fn=loss_fn,
        optimizer=optimizer,
        simulation_config=simulation_config_low_res_sol,
        initial_state=initial_state_low_res,
        target_state=target_state,
        helper_data=helper_data_low_res,
        registered_variables=registered_variables,
        simulation_params=simulation_params_low_res_sol,
        noise_level=training_config.noise_level,
    )

    logging.info("Starting training")
    for step in range(training_config.epochs):
        start_time_epoch = timer()
        key, subkey = jax.random.split(key)
        trained_params, opt_state, loss, gradients_mod = train_step(
            network_params_arrays=trained_params, opt_state=opt_state, key=subkey
        )

        if math.isnan(loss):
            raise ValueError("Nan in loss")

        if early_stopper.new_epoch(loss):
            logger.info("Finished training due to early stopper")
            break

        losses.append(float(loss))

        if loss < best_loss:
            best_loss, best_params = float(loss), trained_params

        logger.info(
            f"Step {step + 1}/{training_config.epochs} | Loss: {loss:.6f} | "
            f"Time: {(timer() - start_time_epoch):.3f}s | Grads: {gradients_mod:.3f}"
        )

    plot_training(
        neural_net_params=best_params,
        times_eval=jnp.linspace(0.0, 0.3, 30, endpoint=True),
        training_config=training_config,
        losses=losses,
        image_folder=model_folder,
    )
    pass
