from autocvd import autocvd

autocvd(num_gpus=1)

import argparse
import h5py
import json
import os
import numpy as _np
import jax.numpy as jnp
from tqdm import tqdm

from arena.arena_tests.solver_in_the_loop.multiproblem.problems.ot_vortex_3d import (
    OtVortex,
)
from arena.arena_tests.solver_in_the_loop.model_manager import TrainingConfig
from arena.arena_tests.solver_in_the_loop.multiproblem.problem_manager import (
    _build_hr_config_and_params,
)
from arena.arena_tests.solver_in_the_loop.utils import downaverage
from astronomix import time_integration


def _namedtuple_to_dict(obj):
    """Recursively convert NamedTuples to JSON-serializable dicts."""
    if hasattr(obj, "_asdict"):
        return {k: _namedtuple_to_dict(v) for k, v in obj._asdict().items()}
    if isinstance(obj, (jnp.ndarray, _np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _namedtuple_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_namedtuple_to_dict(x) for x in obj]
    return obj


parser = argparse.ArgumentParser()
parser.add_argument(
    "--save-path",
    type=str,
    default="/export/data/jalegria/solver_in_the_loop",
    help="Directory where the validation H5 will be written.",
)
parser.add_argument(
    "--output-name",
    type=str,
    default="validation_dataset_ot_vortex_epsilon.h5",
    help="Output H5 file name.",
)
args = parser.parse_args()

save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

training_config = TrainingConfig(epochs_per_time=[], snapshot_timepoints_train=[])
config, params = _build_hr_config_and_params(training_config=training_config)
downaverage_factor = training_config.downaverage_factor

# Match plot_ot_vortex_parameter_scan ranges in plot_training_multiproblem.py
vortex_axes = ["x", "y", "z"]
parities = [False, True]
num_tests = 20
epsilon_ps = jnp.linspace(0.1, 2.1, num=num_tests).tolist()

hyperparams_dicts = []
for axis in vortex_axes:
    for parity in parities:
        for epsilon_p in epsilon_ps:
            hyperparams_dicts.append(
                {
                    "vortex_axis": axis,
                    "parity": parity,
                    "epsilon_p": epsilon_p,
                }
            )

num_sims = len(hyperparams_dicts)
lr_cells = config.num_cells // downaverage_factor
lr_shape = (num_sims, 11, lr_cells, lr_cells, lr_cells)
chunk_shape = (1, 11, lr_cells, lr_cells, lr_cells)

h5_path = os.path.join(save_path, args.output_name)
h5f = h5py.File(h5_path, "w")
vortex_group = h5f.create_group("ot_vortex")

final_state_dataset = vortex_group.create_dataset(
    name="final_state",
    shape=lr_shape,
    dtype="float32",
    chunks=chunk_shape,
    compression="gzip",
)
initial_state_dataset = vortex_group.create_dataset(
    name="initial_state",
    shape=lr_shape,
    dtype="float32",
    chunks=chunk_shape,
    compression="gzip",
)

axis_map = {"x": 0, "y": 1, "z": 2}
vortex_axis_dataset = vortex_group.create_dataset(
    "vortex_axis", shape=(num_sims,), dtype="int8"
)
parity_dataset = vortex_group.create_dataset("parity", shape=(num_sims,), dtype="bool")
epsilon_p_dataset = vortex_group.create_dataset(
    "epsilon_p", shape=(num_sims,), dtype="float32"
)

config = config._replace(progress_bar=False)
for i, problem_hyperparams_dict in tqdm(enumerate(hyperparams_dicts), total=num_sims):
    ot_vortex = OtVortex(**problem_hyperparams_dict)
    (initial_state, config, params, registered_variables) = ot_vortex.generate_initial_state(
        config=config, params=params
    )

    if i == 0:
        vortex_group.attrs["config"] = json.dumps(_namedtuple_to_dict(config))
        vortex_group.attrs["params"] = json.dumps(_namedtuple_to_dict(params))

    vortex_axis_dataset[i] = axis_map[problem_hyperparams_dict["vortex_axis"]]
    parity_dataset[i] = problem_hyperparams_dict["parity"]
    epsilon_p_dataset[i] = problem_hyperparams_dict["epsilon_p"]

    final_state = time_integration(
        primitive_state=initial_state,
        config=config,
        params=params,
        registered_variables=registered_variables,
    )
    initial_state_dataset[i] = downaverage(initial_state, downaverage_factor)
    final_state_dataset[i] = downaverage(final_state, downaverage_factor)

h5f.close()
