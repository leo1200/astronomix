"""
Runs the reference driver ``turbgen_ref`` (see ``turbgen_ref.cpp``, ``build.sh``) for a handful of parameter sets
and packs what the tests of the astronomix port need into one compressed file.

usage: python make_reference.py <binary> <work directory> <output .npz>

Per case the file holds (key ``<case>/<name>``):
    parameters      JSON of the TurbGen parameters (keys of TurbGen::init_driving(map)), grid, times, vturb
    scalars         nmodes, ncmp, t_decay, dt, OUvar, sol_weight_norm, kmin, kmax
    modes           (nmodes, ndim) wave vectors;     amplitudes  (nmodes,)
    times           (ntimes, 3): time, pattern updated (0/1), TurbGen step after the call
    deviates        (ndraws, nmodes, ncmp, 2): Gaussian deviates in the order drawn; the first set is the
                    initialisation (OU_noise_init), every further set one OU_noise_update
    ou_phases       (ntimes + 1, nmodes, ncmp, 2): OUphases after initialisation and after every call
    coefficients    (ntimes + 1, nmodes, ncmp, 2): aka, akb of get_decomposition_coeffs
    ampl_factor     (ntimes + 1, ncmp): the internal amplitude factor (user value^1.5, automatic adjustment)
    field           (ntimes, ncmp, nx[, ny[, nz]]): get_turb_vector at the cell centres (i + 1/2) L / n
"""

import json
import os
import subprocess
import sys

import numpy as np

UPDATE_TIMES = "0,0.5,1.2,3.7"  # in update intervals: first pattern, no update, one update, two updates at once
BASE = dict(ndim=3, length="1.0", target_vdisp=1.0, ampl_factor="1.0", ampl_auto_adjust=0, k_driv=2.0, k_min=1.0,
            k_max=3.0, sol_weight=0.5, spect_form=1, random_seed=140281, nsteps_per_t_turb=10, times=UPDATE_TIMES)
CASES = {
    # the driving of Beattie et al. (2025), which is the default parameter file of TurbGen
    "parabola3d": dict(grid="6,8,10"),
    "band3d_solenoidal": dict(spect_form=0, k_max=2.0, sol_weight=1.0, length="2.0", target_vdisp=2.5,
                              ampl_factor="1.3", k_driv=1.5, nsteps_per_t_turb=7, random_seed=7, grid="4,4,4"),
    "band3d_compressive": dict(spect_form=0, k_max=2.0, sol_weight=0.0, random_seed=11, grid="4,4,4"),
    "powerlaw3d": dict(spect_form=2, k_min=2.0, k_max=4.0, power_law_exp=-1.6666666666666667, angles_exp=1.0,
                       sol_weight=0.3, random_seed=2024, grid="4,6,8"),
    "parabola2d": dict(ndim=2, sol_weight=0.3, ampl_factor="0.8,1.2", random_seed=5, grid="8,12,1"),
    "powerlaw2d": dict(ndim=2, spect_form=2, k_max=4.0, power_law_exp=-2.0, angles_exp=1.5, random_seed=99,
                       grid="8,12,1"),
    "band1d": dict(ndim=1, spect_form=0, sol_weight=0.0, random_seed=3, grid="16,1,1"),
    "noncubic": dict(length="1.0,2.0,0.5", k_max=2.0, sol_weight=0.7, random_seed=17, grid="8,12,16",
                     times="0,2.3"),
    "auto_adjust": dict(spect_form=0, k_max=2.0, ampl_auto_adjust=1, ampl_factor="0.8,1.0,1.2", random_seed=23,
                        grid="4,4,4", times="0,1.2,3.7,4.0", vturb="0.4,0.6,0.5"),
}


def read_blocks(path):
    """The blocks ``name count`` + count numbers of the driver output, in order."""
    blocks = []
    with open(path) as file:
        lines = file.read().split("\n")
    position = 0
    while position < len(lines) and lines[position]:
        name, count = lines[position].split()
        values = np.array([float(v) for v in lines[position + 1 : position + 1 + int(count)]])
        blocks.append((name, values))
        position += 1 + int(count)
    return blocks


def run_case(binary, work_directory, name, overrides):
    parameters = {**BASE, **overrides}
    output = os.path.join(work_directory, name + ".txt")
    arguments = [binary, "out=" + output] + [f"{key}={value}" for key, value in parameters.items()]
    with open(os.path.join(work_directory, name + ".log"), "w") as log:
        subprocess.run(arguments, cwd=work_directory, stdout=log, stderr=subprocess.STDOUT, check=True)
    blocks = read_blocks(output)
    scalars = dict(blocks)["scalars"]
    nmodes, ncmp, ndim = int(scalars[0]), int(scalars[1]), int(parameters["ndim"])
    grid = [int(n) for n in parameters["grid"].split(",")]
    per_mode = (nmodes, ncmp, 2)
    collected = {"times": [], "deviates": [], "ou_phases": [], "coefficients": [], "ampl_factor": [], "field": []}
    for block, values in blocks:
        if block == "time":
            if values[0] >= 0:  # the block of the initialisation carries time -1
                collected["times"].append(values)
        elif block == "deviates":
            collected["deviates"].extend(values.reshape(-1, *per_mode))
        elif block in ("ou_phases", "coefficients"):
            collected[block].append(values.reshape(per_mode))
        elif block == "ampl_factor":
            collected[block].append(values)
        elif block == "field":
            field = np.moveaxis(values.reshape(*grid, ncmp), -1, 0)
            collected[block].append(field.reshape(ncmp, *grid[:ndim]))
        elif block == "unigrid_check":
            assert values[0] < 1e-6 * values[1], (name, values)  # TurbGen's single-precision grid routine agrees
    case = {key: np.array(value) for key, value in collected.items()}
    case.update(scalars=scalars, modes=dict(blocks)["modes"].reshape(nmodes, ndim), amplitudes=dict(blocks)["amplitudes"])
    case["parameters"] = np.array(json.dumps(parameters))
    return case


if __name__ == "__main__":
    binary, work_directory, target = sys.argv[1:4]
    os.makedirs(work_directory, exist_ok=True)
    arrays = {}
    for name, overrides in CASES.items():
        for key, value in run_case(binary, work_directory, name, overrides).items():
            arrays[f"{name}/{key}"] = value
        print(name, "modes", arrays[f"{name}/modes"].shape, "draws", arrays[f"{name}/deviates"].shape[0],
              "steps", arrays[f"{name}/times"][:, 2].astype(int).tolist())
    np.savez_compressed(target, **arrays)
    print("wrote", target, os.path.getsize(target), "bytes")
