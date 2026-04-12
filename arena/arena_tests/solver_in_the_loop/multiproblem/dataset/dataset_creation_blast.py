from autocvd import autocvd

autocvd(num_gpus=1)
import argparse
import h5py
import json
import os
import numpy as _np
import jax.numpy as jnp
from tqdm import tqdm
from arena.arena_tests.solver_in_the_loop.multiproblem.problems.mhd_blast import (
    MhdBlast,
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
parser.add_argument("--split", type=int, required=True, help="Split index (0-based)")
parser.add_argument(
    "--num-splits", type=int, required=True, help="Total number of splits"
)
args = parser.parse_args()

save_path = "/export/data/jalegria/solver_in_the_loop"
os.makedirs(save_path, exist_ok=True)

training_config = TrainingConfig(epochs_per_time=[], snapshot_timepoints_train=[])
config, params = _build_hr_config_and_params(training_config=training_config)
downaverage_factor = training_config.downaverage_factor

sims_b = 12
sims_phi = 20
sims_theta = 20
min_b0 = 0.5
max_b0 = 20.5
min_theta = 0.2
max_theta = jnp.pi - 0.2
min_phi = 0.2
max_phi = 2 * jnp.pi - 0.2

B_0s = jnp.linspace(start=min_b0, stop=max_b0, num=sims_b).tolist()

# Split B0 range across files
chunk_size = len(B_0s) // args.num_splits
remainder = len(B_0s) % args.num_splits
start = args.split * chunk_size + min(args.split, remainder)
end = start + chunk_size + (1 if args.split < remainder else 0)
B_0s_split = B_0s[start:end]

r = 1.0
phi_fixed = float(jnp.pi / 4)
thetas = jnp.linspace(start=min_theta, stop=max_theta, num=sims_theta).tolist()

theta_fixed = float(jnp.pi / 2)
phis = jnp.linspace(start=min_phi, stop=max_phi, num=sims_phi).tolist()

hyperparams_dicts = []
for B0 in B_0s_split:
    for theta in thetas:
        for phi in phis:
            hyperparams_dicts.append(
                {
                    "B0": B0,
                    "B_direction": jnp.array(
                        [
                            r * jnp.sin(theta) * jnp.cos(phi),
                            r * jnp.sin(theta) * jnp.sin(phi),
                            r * jnp.cos(theta),
                        ]
                    ),
                    "r0": 0.125,
                }
            )

num_sims = len(hyperparams_dicts)
lr_cells = config.num_cells // downaverage_factor
lr_shape = (num_sims, 11, lr_cells, lr_cells, lr_cells)
chunk_shape = (1, 11, lr_cells, lr_cells, lr_cells)

h5_path = os.path.join(save_path, f"training_blast_split{args.split}.h5")
h5f = h5py.File(h5_path, "w")
blast_group = h5f.create_group("mhd_blast")

final_state_dataset = blast_group.create_dataset(
    name="final_state",
    shape=lr_shape,
    dtype="float32",
    chunks=chunk_shape,
    compression="gzip",
)
initial_state_dataset = blast_group.create_dataset(
    name="initial_state",
    shape=lr_shape,
    dtype="float32",
    chunks=chunk_shape,
    compression="gzip",
)

B0_dataset = blast_group.create_dataset("B0", shape=(num_sims,), dtype="float32")
B_direction_dataset = blast_group.create_dataset(
    "B_direction", shape=(num_sims, 3), dtype="float32"
)
r0_dataset = blast_group.create_dataset("r0", shape=(num_sims,), dtype="float32")

config = config._replace(progress_bar=False)
for i, hyperparams_dict in tqdm(enumerate(hyperparams_dicts), total=num_sims):
    mhd_blast = MhdBlast(**hyperparams_dict)
    (initial_state, config, params, registered_variables) = (
        mhd_blast.generate_initial_state(config=config, params=params)
    )

    if i == 0:
        blast_group.attrs["config"] = json.dumps(_namedtuple_to_dict(config))
        blast_group.attrs["params"] = json.dumps(_namedtuple_to_dict(params))

    B0_dataset[i] = hyperparams_dict["B0"]
    B_direction_dataset[i] = hyperparams_dict["B_direction"]
    r0_dataset[i] = hyperparams_dict["r0"]

    final_state = time_integration(
        primitive_state=initial_state,
        config=config,
        params=params,
        registered_variables=registered_variables,
    )
    initial_state_dataset[i] = downaverage(initial_state, downaverage_factor)
    final_state_dataset[i] = downaverage(final_state, downaverage_factor)

h5f.close()
