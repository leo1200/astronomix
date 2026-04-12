import argparse
import glob
import h5py
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-dir",
    type=str,
    default="/export/data/jalegria/solver_in_the_loop",
    help="Directory containing the split files",
)
parser.add_argument(
    "--problem",
    type=str,
    required=True,
    choices=["blast", "ot_vortex", "turbulence"],
    help="Which problem's splits to merge",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output file path (defaults to <input-dir>/training_<problem>.h5)",
)
args = parser.parse_args()

problem_to_group = {
    "blast": "mhd_blast",
    "ot_vortex": "ot_vortex",
    "turbulence": "turbulence",
}
group_name = problem_to_group[args.problem]

split_paths = sorted(
    glob.glob(os.path.join(args.input_dir, f"training_{args.problem}_split*.h5"))
)
print(split_paths)
if not split_paths:
    raise FileNotFoundError(
        f"No split files found in {args.input_dir} for problem '{args.problem}'"
    )

print(f"Found {len(split_paths)} split files for '{args.problem}'")

# Compute total number of simulations across all splits
total_sims = 0
for path in split_paths:
    with h5py.File(path, "r") as f:
        total_sims += f[group_name]["initial_state"].shape[0]

print(f"Total simulations: {total_sims}")

# Read shape info, attributes, and dataset names from the first split
with h5py.File(split_paths[0], "r") as f:
    grp = f[group_name]
    sample_shape = grp["final_state"].shape[1:]
    config_attr = grp.attrs["config"]
    params_attr = grp.attrs["params"]
    # Discover companion datasets (everything that isn't initial/final state)
    companion_names = [
        name for name in grp.keys() if name not in ("initial_state", "final_state")
    ]
    companion_info = {}
    for name in companion_names:
        ds = grp[name]
        companion_info[name] = {
            "shape_tail": ds.shape[1:],
            "dtype": ds.dtype,
        }

output_path = args.output or os.path.join(args.input_dir, f"training_{args.problem}.h5")
chunk_shape = (1,) + sample_shape

with h5py.File(output_path, "w") as out:
    grp = out.create_group(group_name)
    grp.attrs["config"] = config_attr
    grp.attrs["params"] = params_attr

    final_ds = grp.create_dataset(
        "final_state",
        shape=(total_sims,) + sample_shape,
        dtype="float32",
        chunks=chunk_shape,
        compression="gzip",
    )
    initial_ds = grp.create_dataset(
        "initial_state",
        shape=(total_sims,) + sample_shape,
        dtype="float32",
        chunks=chunk_shape,
        compression="gzip",
    )

    companion_ds = {}
    for name, info in companion_info.items():
        companion_ds[name] = grp.create_dataset(
            name,
            shape=(total_sims,) + info["shape_tail"],
            dtype=info["dtype"],
        )

    offset = 0
    for path in split_paths:
        with h5py.File(path, "r") as f:
            src = f[group_name]
            n = src["initial_state"].shape[0]
            print(f"  Merging {path} ({n} sims, offset {offset})")

            final_ds[offset : offset + n] = src["final_state"][:]
            initial_ds[offset : offset + n] = src["initial_state"][:]
            for name in companion_names:
                companion_ds[name][offset : offset + n] = src[name][:]

            offset += n

print(f"Merged into {output_path} ({total_sims} simulations)")
