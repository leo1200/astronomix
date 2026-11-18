"""
Field-level inference of the paper (Fig. ``field_level_inference``): plain Adam
on the full 3 x N^3 initial velocity of a driven MHD turbulence box, so that the
z-projected density after free evolution for t_end = 0.5 l_0 / v_rms forms the
astronomix logo.

The optimisation is resumable: the control, the Adam moments, the best iterate
and the loss history are written to the run directory after every step, and a
new job continues from there (``--wall-budget`` ends a job cleanly before its
time limit). At the end of every job the best initial state is written as
``best_state.npy`` (with ``run_summary.json``) for ``make_panel_snaps.py``.
The run ends after ``--num-steps`` Adam steps or once the best loss has fallen
by ``--target-reduction`` below the initial one, whichever comes first.

The gradient differentiates the WENO flux with its Pallas adjoint
(``--flux-adjoint pallas``, the default) or with the transposed native-JAX
tangent (``native``); both give the same gradient up to round-off.

Usage::

    python optimize_velocity.py --resolution 128 --num-steps 200 \\
        --run-directory <dir> --wall-budget 41400
"""

# general
import argparse
import json
import os
from timeit import default_timer as timer

# gpu selection
from autocvd import autocvd

autocvd(num_gpus=1)

# jax
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# numerics
import numpy as np  # noqa: E402

# field-level inference problem
from _problem import (  # noqa: E402
    COURANT_NUMBER,
    build_state,
    driven_turbulence,
    finalized,
    initial_velocity,
    logo_target,
    make_config,
    make_loss,
    rms_velocity,
)

# astronomix constants
from astronomix import (  # noqa: E402
    NATIVE_JAX,
    PALLAS,
)

DATA_DIRECTORY = "/e/project1/astronomix/storcks1/revision_data/field_level_inference"

# The compiled value-and-grad is cached across the jobs of a run: its first
# compilation takes ~15 minutes at 128^3 with the Pallas adjoint.
COMPILATION_CACHE = "/e/scratch/astronomix/revision/field_level_inference/jax_cache"

# Adam's standard moment decay rates and regulariser.
FIRST_MOMENT_DECAY = 0.9
SECOND_MOMENT_DECAY = 0.999
ADAM_EPSILON = 1e-8


def parse_arguments():
    """
    Parse the command line.

    Returns:
        The argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=200, help="total Adam steps of the run")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--num-checkpoints", type=int, default=16)
    parser.add_argument("--rematerialize-stages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--backend",
        choices=["pallas", "native"],
        default="pallas",
        help="primal backend of the differentiated time loop (the tangent is native either way)",
    )
    parser.add_argument(
        "--flux-adjoint",
        choices=["pallas", "native"],
        default="pallas",
        help="reverse mode of the WENO flux: the Pallas adjoint or the transposed native tangent",
    )
    parser.add_argument(
        "--target-reduction",
        type=float,
        default=0.0,
        help="stop once best loss <= initial loss / this factor; 0 = run all steps",
    )
    parser.add_argument(
        "--wall-budget",
        type=float,
        default=0.0,
        help="stop (resumably) once this many seconds have passed in this job; 0 = no limit",
    )
    parser.add_argument(
        "--compilation-cache",
        type=str,
        default=COMPILATION_CACHE,
        help="persistent JAX compilation cache directory ('' = none)",
    )
    parser.add_argument(
        "--run-directory",
        type=str,
        default="",
        help="optimisation state and results (default: <data>/adam_N<resolution>)",
    )
    return parser.parse_args()


def save_progress(path, progress):
    """
    Write the optimisation state atomically (a job killed while writing must
    not destroy the previous state).

    Args:
        path: The .npz file.
        progress: Dict of arrays and scalars.
    """
    temporary_path = path + ".tmp.npz"
    np.savez(temporary_path, **{key: np.asarray(value) for key, value in progress.items()})
    os.replace(temporary_path, path)


def main():
    """Run (or continue) the Adam optimisation of the initial velocity."""
    args = parse_arguments()
    job_start = timer()
    if args.compilation_cache:
        jax.config.update("jax_compilation_cache_dir", args.compilation_cache)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 60.0)
    resolution = args.resolution
    run_directory = args.run_directory or os.path.join(DATA_DIRECTORY, f"adam_N{resolution}")
    os.makedirs(run_directory, exist_ok=True)
    progress_path = os.path.join(run_directory, "progress.npz")

    # -------------------------------------------------------------
    # =============== ↓ Problem setup ↓ ===========================
    # -------------------------------------------------------------
    turbulence, generation_time = driven_turbulence(
        resolution,
        os.path.join(DATA_DIRECTORY, f"turbulence_N{resolution}.npy"),
    )
    registered_variables, config = finalized(make_config(
        resolution,
        backward=True,
        num_checkpoints=args.num_checkpoints,
        rematerialize_stages=args.rematerialize_stages,
        backend=PALLAS if args.backend == "pallas" else NATIVE_JAX,
        pallas_adjoint=args.flux_adjoint == "pallas",
    ))
    velocity_rms = rms_velocity(turbulence, registered_variables)
    t_end = 0.5 / velocity_rms
    target = logo_target(resolution, float(turbulence[registered_variables.density_index].sum()))
    loss = make_loss(turbulence, config, registered_variables, t_end, target)
    value_and_gradient = jax.jit(jax.value_and_grad(loss))
    print(
        f"N = {resolution}: v_rms = {velocity_rms:.4f} v_0, t_end = {t_end:.4f} t_0 "
        f"(turbulence generated in {generation_time:.0f} s)",
        flush=True,
    )
    # -------------------------------------------------------------
    # =============== ↑ Problem setup ↑ ===========================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =============== ↓ Optimiser state (fresh or resumed) ↓ ======
    # -------------------------------------------------------------
    if os.path.exists(progress_path):
        stored = np.load(progress_path)
        velocity = jnp.asarray(stored["velocity"])
        first_moment = jnp.asarray(stored["first_moment"])
        second_moment = jnp.asarray(stored["second_moment"])
        best_velocity = jnp.asarray(stored["best_velocity"])
        best_loss = float(stored["best_loss"])
        history = [tuple(row) for row in stored["history"]]
        print(f"resumed after step {len(history)} (best loss {best_loss:.4e})", flush=True)
    else:
        velocity = initial_velocity(turbulence, registered_variables)
        first_moment = jnp.zeros_like(velocity)
        second_moment = jnp.zeros_like(velocity)
        best_velocity = velocity
        best_loss = float("inf")
        history = []
    # -------------------------------------------------------------
    # =============== ↑ Optimiser state (fresh or resumed) ↑ ======
    # -------------------------------------------------------------

    # Each row of the history: Adam step, loss of the iterate the gradient was
    # taken at, wall time of the value-and-grad, cumulative optimisation time.
    cumulative_time = history[-1][3] if history else 0.0
    step_times = []
    def target_reached():
        """Whether the best loss is ``--target-reduction`` below the initial one."""
        return bool(history) and args.target_reduction > 0.0 and (
            best_loss * args.target_reduction <= history[0][1]
        )

    while len(history) < args.num_steps and not target_reached():
        step_start = timer()
        value, gradient = jax.block_until_ready(value_and_gradient(velocity))
        gradient_time = timer() - step_start
        value = float(value)
        if not np.isfinite(value):
            print(f"step {len(history) + 1}: non-finite loss, stopping", flush=True)
            break
        if value < best_loss:
            best_loss = value
            best_velocity = velocity

        # Plain Adam with bias correction (the paper's optimiser).
        step = len(history) + 1
        first_moment = FIRST_MOMENT_DECAY * first_moment + (1 - FIRST_MOMENT_DECAY) * gradient
        second_moment = SECOND_MOMENT_DECAY * second_moment + (1 - SECOND_MOMENT_DECAY) * gradient**2
        corrected_first = first_moment / (1 - FIRST_MOMENT_DECAY**step)
        corrected_second = second_moment / (1 - SECOND_MOMENT_DECAY**step)
        velocity = velocity - args.learning_rate * corrected_first / (
            jnp.sqrt(corrected_second) + ADAM_EPSILON
        )

        cumulative_time += gradient_time
        history.append((step, value, gradient_time, cumulative_time))
        step_times.append(timer() - step_start)
        save_progress(progress_path, {
            "velocity": velocity,
            "first_moment": first_moment,
            "second_moment": second_moment,
            "best_velocity": best_velocity,
            "best_loss": best_loss,
            "history": np.array(history),
        })
        print(
            f"step {step:4d}: loss {value:.4e}  (best {best_loss:.4e})  "
            f"value-and-grad {gradient_time:.0f} s",
            flush=True,
        )
        # Stop before the next step would overrun the job's budget.
        elapsed = timer() - job_start
        if args.wall_budget and elapsed + 1.2 * max(step_times) > args.wall_budget:
            print(f"wall budget reached after {elapsed:.0f} s, resume with the same command", flush=True)
            break

    # The best initial state and the numbers the panel renderer needs.
    magnetic_index = registered_variables.magnetic_index
    best_state = build_state(
        config,
        registered_variables,
        turbulence[registered_variables.density_index],
        best_velocity,
        turbulence[registered_variables.pressure_index],
        jnp.stack([
            turbulence[magnetic_index.x],
            turbulence[magnetic_index.y],
            turbulence[magnetic_index.z],
        ]),
    )
    np.save(os.path.join(run_directory, "best_state.npy"), np.asarray(best_state))
    with open(os.path.join(run_directory, "run_summary.json"), "w") as summary_file:
        json.dump({
            "resolution": resolution,
            "velocity_rms": velocity_rms,
            "t_end": t_end,
            "courant_number": COURANT_NUMBER,
            "learning_rate": args.learning_rate,
            "num_checkpoints": args.num_checkpoints,
            "rematerialize_stages": args.rematerialize_stages,
            "backend": args.backend,
            "flux_adjoint": args.flux_adjoint,
            "target_reduction": args.target_reduction,
            "adam_steps": len(history),
            "initial_loss": history[0][1] if history else None,
            "best_loss": best_loss,
            "optimisation_time_s": cumulative_time,
        }, summary_file, indent=2)
    print(f"{len(history)} Adam steps, best loss {best_loss:.4e}, wrote {run_directory}", flush=True)


if __name__ == "__main__":
    main()
