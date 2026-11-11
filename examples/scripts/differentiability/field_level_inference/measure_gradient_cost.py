"""
Cost of one optimisation step of the field-level inference: the memory and
wall time of a value-and-grad of the projected-density loss with respect to the
3 x N^3 initial velocity, for several reverse-mode settings.

For each (stage rematerialisation, num_checkpoints) and each reverse mode of the WENO flux
(the transposed native tangent, ``native``, and the Pallas adjoint, ``pallas``)
the compiled memory of the gradient is read before it runs (so configurations that cannot fit are reported without
running them), then the gradient is timed over a fraction of the loss time.
The forward run over the full loss time counts the time steps, from which the
full-length gradient time is extrapolated.

Usage::

    python measure_gradient_cost.py --resolution 128 --time-fraction 0.1 \\
        --settings stage:16,none:16 --flux-adjoints native,pallas
"""

# general
import argparse
import os
from timeit import default_timer as timer

# gpu selection
from autocvd import autocvd

autocvd(num_gpus=1)

# jax
import jax  # noqa: E402

# field-level inference problem
from _problem import (  # noqa: E402
    driven_turbulence,
    finalized,
    initial_velocity,
    logo_target,
    make_config,
    make_loss,
    make_params,
    rms_velocity,
)

# astronomix constants
from astronomix import (  # noqa: E402
    NATIVE_JAX,
    PALLAS,
)

# astronomix functions
from astronomix import time_integration  # noqa: E402

DATA_DIRECTORY = "/e/project1/astronomix/storcks1/revision_data/field_level_inference"
GIB = 1024.0**3


def parse_arguments():
    """
    Parse the command line.

    Returns:
        The argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument(
        "--time-fraction",
        type=float,
        default=0.1,
        help="fraction of the loss time the timed gradients integrate over",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="stage:16,none:16",
        help="comma-separated remat:num_checkpoints pairs (remat: stage or none)",
    )
    parser.add_argument(
        "--data-directory",
        type=str,
        default=DATA_DIRECTORY,
        help="where the driven turbulent state is cached",
    )
    parser.add_argument(
        "--backward-backend",
        choices=["pallas", "native"],
        default="pallas",
        help="primal backend of the differentiated time loop",
    )
    parser.add_argument(
        "--flux-adjoints",
        type=str,
        default="native,pallas",
        help="comma-separated reverse modes of the WENO flux: native (transposed "
        "native tangent) and/or pallas (Pallas adjoint)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="report the compiled memory without running the gradients",
    )
    return parser.parse_args()


def memory_report(compiled):
    """
    The device memory XLA plans for a compiled function.

    Args:
        compiled: A compiled (AOT) function.

    Returns:
        (temporary, argument, output) sizes in GiB.
    """
    analysis = compiled.memory_analysis()
    return (
        analysis.temp_size_in_bytes / GIB,
        analysis.argument_size_in_bytes / GIB,
        analysis.output_size_in_bytes / GIB,
    )


def main():
    """Measure the forward and gradient cost of the inference problem."""
    args = parse_arguments()
    resolution = args.resolution
    device = jax.devices()[0]
    device_memory = device.memory_stats().get("bytes_limit", 0) / GIB
    print(f"device {device.device_kind}, {device_memory:.1f} GiB usable", flush=True)

    # -------------------------------------------------------------
    # =============== ↓ Background and loss time ↓ ================
    # -------------------------------------------------------------
    cache_path = os.path.join(args.data_directory, f"turbulence_N{resolution}.npy")
    turbulence, generation_time = driven_turbulence(resolution, cache_path)
    registered_variables, forward_config = finalized(make_config(resolution, return_snapshots=True))
    velocity_rms = rms_velocity(turbulence, registered_variables)
    t_end = 0.5 / velocity_rms
    print(
        f"turbulence: v_rms = {velocity_rms:.4f} v_0, t_end = {t_end:.4f} t_0 "
        f"(generated in {generation_time:.0f} s)",
        flush=True,
    )
    target = logo_target(resolution, float(turbulence[registered_variables.density_index].sum()))
    # -------------------------------------------------------------
    # =============== ↑ Background and loss time ↑ ================
    # -------------------------------------------------------------

    # The full-length forward pass counts the steps and times the primal.
    forward = jax.jit(lambda state: time_integration(
        state,
        forward_config,
        make_params(t_end),
        registered_variables,
    ))
    jax.block_until_ready(forward(turbulence))
    start = timer()
    snapshots = jax.block_until_ready(forward(turbulence))
    forward_time = timer() - start
    num_steps = int(snapshots.num_iterations)
    print(
        f"forward over t_end: {num_steps} steps, {forward_time:.1f} s "
        f"({1e3 * forward_time / num_steps:.1f} ms/step)",
        flush=True,
    )

    velocity = initial_velocity(turbulence, registered_variables)
    runs = [
        (setting, flux_adjoint)
        for setting in args.settings.split(",")
        for flux_adjoint in args.flux_adjoints.split(",")
    ]
    gradients = {}
    for setting, flux_adjoint in runs:
        remat, num_checkpoints = setting.split(":")
        num_checkpoints = int(num_checkpoints)
        _, backward_config = finalized(make_config(
            resolution,
            backward=True,
            num_checkpoints=num_checkpoints,
            rematerialize_stages={"stage": True, "none": False}[remat],
            backend=PALLAS if args.backward_backend == "pallas" else NATIVE_JAX,
            pallas_adjoint=flux_adjoint == "pallas",
        ))
        loss = make_loss(
            turbulence,
            backward_config,
            registered_variables,
            args.time_fraction * t_end,
            target,
        )
        label = (
            f"{args.backward_backend} flux adjoint={flux_adjoint:6s} "
            f"remat={remat:5s} checkpoints={num_checkpoints:3d}"
        )
        start = timer()
        compiled = jax.jit(jax.value_and_grad(loss)).lower(velocity).compile()
        compile_time = timer() - start
        temporary, argument, output = memory_report(compiled)
        planned = temporary + argument + output
        print(
            f"{label}: compiled in {compile_time:.0f} s, planned {planned:.1f} GiB "
            f"(temp {temporary:.1f}, args {argument:.2f}, out {output:.2f})",
            flush=True,
        )
        if args.compile_only or planned > 0.97 * device_memory:
            continue
        try:
            start = timer()
            value, gradient = jax.block_until_ready(compiled(velocity))
            gradient_time = timer() - start
        except Exception as error:
            print(f"{label}: failed: {str(error).splitlines()[0][:160]}", flush=True)
            continue
        peak = device.memory_stats().get("peak_bytes_in_use", 0) / GIB
        probe_steps = max(1, round(args.time_fraction * num_steps))
        print(
            f"{label}: gradient over {args.time_fraction:.2f} t_end (~{probe_steps} steps) "
            f"{gradient_time:.1f} s ({gradient_time / probe_steps:.2f} s/step, "
            f"full t_end ~{gradient_time / args.time_fraction / 60:.1f} min), "
            f"peak {peak:.1f} GiB, loss {float(value):.4e}, "
            f"|grad| {float(jax.numpy.linalg.norm(gradient)):.3e}",
            flush=True,
        )
        # The two reverse modes of one setting must give the same gradient.
        if setting in gradients:
            reference = gradients[setting]
            difference = jax.numpy.linalg.norm(gradient - reference) / jax.numpy.linalg.norm(reference)
            print(f"{label}: relative gradient difference to the first flux adjoint {float(difference):.3e}", flush=True)
        else:
            gradients[setting] = gradient


if __name__ == "__main__":
    main()
