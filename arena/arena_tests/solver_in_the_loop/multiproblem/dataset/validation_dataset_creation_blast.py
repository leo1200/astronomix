from autocvd import autocvd
import os

# autocvd(num_gpus=1)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse
import h5py
import json
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

save_path = "/export/data/jalegria/solver_in_the_loop"
os.makedirs(save_path, exist_ok=True)

training_config = TrainingConfig(epochs_per_time=[], snapshot_timepoints_train=[])
config, params = _build_hr_config_and_params(training_config=training_config)
downaverage_factor = training_config.downaverage_factor

num_tests = 30
B_0s = jnp.linspace(start=0.5, stop=20.5, num=num_tests).tolist()
B_0_fixed = 10.0

r = 1.0
phi_fixed = float(jnp.pi / 4)
thetas = jnp.linspace(start=0.2, stop=jnp.pi - 0.2, num=num_tests).tolist()

theta_fixed = float(jnp.pi / 2)
phis = jnp.linspace(start=0.2, stop=2 * jnp.pi - 0.2, num=num_tests).tolist()

hyperparams_dicts_theta = []
for theta in thetas:
    hyperparams_dicts_theta.append(
        {
            "B0": B_0_fixed,
            "B_direction": jnp.array(
                [
                    r * jnp.sin(theta) * jnp.cos(phi_fixed),
                    r * jnp.sin(theta) * jnp.sin(phi_fixed),
                    r * jnp.cos(theta),
                ]
            ),
            "r0": 0.125,
        }
    )

hyperparams_dicts_phi = []
for phi in phis:
    hyperparams_dicts_phi.append(
        {
            "B0": B_0_fixed,
            "B_direction": jnp.array(
                [
                    r * jnp.sin(theta_fixed) * jnp.cos(phi),
                    r * jnp.sin(theta_fixed) * jnp.sin(phi),
                    r * jnp.cos(theta_fixed),
                ]
            ),
            "r0": 0.125,
        }
    )

hyperparams_dicts_b = []
for B_0 in B_0s:
    hyperparams_dicts_b.append(
        {
            "B0": B_0,
            "B_direction": jnp.array(
                [
                    r * jnp.sin(theta_fixed) * jnp.cos(phi_fixed),
                    r * jnp.sin(theta_fixed) * jnp.sin(phi_fixed),
                    r * jnp.cos(theta_fixed),
                ]
            ),
            "r0": 0.125,
        }
    )

hyperparams_dicts: dict[str, list[dict]] = {
    "phi": hyperparams_dicts_phi,
    "theta": hyperparams_dicts_theta,
    "b0": hyperparams_dicts_b,
}
for key, hyperparams_dict in hyperparams_dicts.items():
    num_sims = len(hyperparams_dict)
    lr_cells = config.num_cells // downaverage_factor
    lr_shape = (num_sims, 11, lr_cells, lr_cells, lr_cells)
    chunk_shape = (1, 11, lr_cells, lr_cells, lr_cells)

    h5_path = os.path.join(save_path, f"validation_dataset_blast_{key}.h5")
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
    for i, problem_hyperparams_dict in tqdm(
        enumerate(hyperparams_dict), total=num_sims
    ):
        mhd_blast = MhdBlast(**problem_hyperparams_dict)
        (initial_state, config, params, registered_variables) = (
            mhd_blast.generate_initial_state(config=config, params=params)
        )

        if i == 0:
            blast_group.attrs["config"] = json.dumps(_namedtuple_to_dict(config))
            blast_group.attrs["params"] = json.dumps(_namedtuple_to_dict(params))

        B0_dataset[i] = problem_hyperparams_dict["B0"]
        B_direction_dataset[i] = problem_hyperparams_dict["B_direction"]
        r0_dataset[i] = problem_hyperparams_dict["r0"]

        final_state = time_integration(
            primitive_state=initial_state,
            config=config,
            params=params,
            registered_variables=registered_variables,
        )
        initial_state_dataset[i] = downaverage(initial_state, downaverage_factor)
        final_state_dataset[i] = downaverage(final_state, downaverage_factor)

    h5f.close()
