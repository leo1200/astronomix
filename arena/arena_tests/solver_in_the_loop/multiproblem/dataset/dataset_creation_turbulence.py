from autocvd import autocvd

autocvd(num_gpus=1)
import argparse
import h5py
import json
import os
import numpy as _np
import jax.numpy as jnp
from tqdm import tqdm
from arena.arena_tests.solver_in_the_loop.multiproblem.problems.turbulence import (
    Turbulence,
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

# Parameter grid
sims_b = 15
sims_phi = 10
sims_theta = 10
min_b0 = 0.1
max_b0 = 5.0

B_0s = jnp.linspace(start=min_b0, stop=max_b0, num=sims_b).tolist()

r = 1.0
thetas = jnp.linspace(start=0.2, stop=jnp.pi - 0.2, num=sims_theta).tolist()
phis = jnp.linspace(start=0.2, stop=2 * jnp.pi - 0.2, num=sims_phi).tolist()

i = 0
hyperparams_dicts = []
for B0 in B_0s:
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
                    # a different seed per simulations
                    "rng_seed": i,
                }
            )
            i += 1

# Split across processes
total = len(hyperparams_dicts)
chunk_size = total // args.num_splits
remainder_count = total % args.num_splits
start = args.split * chunk_size + min(args.split, remainder_count)
end = start + chunk_size + (1 if args.split < remainder_count else 0)
hyperparams_split = hyperparams_dicts[start:end]

num_sims = len(hyperparams_split)
lr_cells = config.num_cells // downaverage_factor
lr_shape = (num_sims, 11, lr_cells, lr_cells, lr_cells)
chunk_shape = (1, 11, lr_cells, lr_cells, lr_cells)

h5_path = os.path.join(save_path, f"training_turbulence_split{args.split}.h5")
h5f = h5py.File(h5_path, "w")
grp = h5f.create_group("turbulence")

final_state_dataset = grp.create_dataset(
    name="final_state",
    shape=lr_shape,
    dtype="float32",
    chunks=chunk_shape,
    compression="gzip",
)
initial_state_dataset = grp.create_dataset(
    name="initial_state",
    shape=lr_shape,
    dtype="float32",
    chunks=chunk_shape,
    compression="gzip",
)

B0_dataset = grp.create_dataset("B0", shape=(num_sims,), dtype="float32")
B_direction_dataset = grp.create_dataset(
    "B_direction", shape=(num_sims, 3), dtype="float32"
)
rng_seed_dataset = grp.create_dataset("rng_seed", shape=(num_sims,), dtype="int32")

config = config._replace(progress_bar=False)
for i, hp in tqdm(enumerate(hyperparams_split), total=num_sims):
    turb = Turbulence(**hp)
    (initial_state, config, params, registered_variables) = turb.generate_initial_state(
        config=config, params=params
    )

    if i == 0:
        grp.attrs["config"] = json.dumps(_namedtuple_to_dict(config))
        grp.attrs["params"] = json.dumps(_namedtuple_to_dict(params))

    B0_dataset[i] = hp["B0"]
    B_direction_dataset[i] = hp["B_direction"]
    rng_seed_dataset[i] = hp["rng_seed"]

    final_state = time_integration(
        primitive_state=initial_state,
        config=config,
        params=params,
        registered_variables=registered_variables,
    )
    initial_state_dataset[i] = downaverage(initial_state, downaverage_factor)
    final_state_dataset[i] = downaverage(final_state, downaverage_factor)

h5f.close()
