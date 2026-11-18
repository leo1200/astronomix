"""
Reverse-mode cost and correctness of the finite-difference WENO step, Pallas vs native.

Evolves a hydro state with positivity-preserving WENO, dual energy, ``SSPRK4``
and the adaptive checkpointed time loop over about ``--steps`` CFL steps and
differentiates
``sum(rho^2)`` at the end with respect to the initial state. Reports, per
backend -- ``native`` (``NATIVE_JAX``), ``pallas`` (the Pallas forward with
the Pallas adjoint, ``backend_config.pallas_adjoint``) and ``pallas_nt`` (the
Pallas forward with the transposed native tangent):

* the wall time of the forward and of the gradient (first call = compile, then
  the mean of the timed repeats) and the peak device memory;
* the gradient against the native backend's: relative L2 difference, cosine,
  and the directional derivative against a central finite difference of the
  Pallas forward.

The state is a ``casa_orlando.py --save-state`` file (fields rho, vx, vy, vz,
press) or, without ``--state``, a random smooth one.

    python examples/scripts/validation/weno_stability/pallas_adjoint_check.py \\
        --state W/fwd/orl_R4bsim_n128_age325.npz --steps 8 --tangent approx
"""

# ==== GPU selection ====
import os
if os.environ.get("JAX_PLATFORMS") != "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
import time

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import BACKWARDS, NATIVE_JAX, PALLAS, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)

#: the linearisations of the WENO reconstruction: exact, or with the nonlinear
#: weights frozen
TANGENTS = {
    "exact": dict(weno_ad_frozen_weights=False),
    "approx": dict(weno_ad_frozen_weights=True),
}


def load_fields(path, num_cells):
    """Primitive fields of a saved state, or a random smooth state."""
    if path:
        data = np.load(path)
        fields = {key: np.asarray(data[key], np.float32) for key in ("rho", "vx", "vy", "vz", "press")}
        return fields, float(data["box"])
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 2.0 * np.pi, num_cells, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = 1.0 + 0.5 * np.sin(X) * np.cos(Y) + 0.1 * rng.standard_normal(X.shape) ** 2
    fields = dict(
        rho=rho.astype(np.float32),
        vx=np.sin(Y).astype(np.float32),
        vy=np.cos(Z).astype(np.float32),
        vz=np.sin(X).astype(np.float32),
        press=(0.6 + 0.3 * np.cos(X + Y)).astype(np.float32),
    )
    return fields, 1.0


def make_setup(fields, box, backend, tangent, num_steps, checkpoints, pallas_adjoint=True):
    """Config, params, registered variables and the initial primitive state."""
    num_cells = fields["rho"].shape[0]
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        num_cells=num_cells,
        box_size=box,
        backend_config=BackendConfig(backend=backend, pallas_adjoint=pallas_adjoint),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        solver_config=FDSolverConfig(weno_positivity_preserving=True, dual_energy=True, **TANGENTS[tangent]),
        differentiation_mode=BACKWARDS,
        num_checkpoints=checkpoints,
    )
    registered_variables = get_registered_variables(config)
    density_scale = float(np.median(fields["rho"]))
    pressure_scale = float(np.median(fields["press"]))
    # about num_steps CFL steps: dt ~ C_cfl dx / max(|v| + c)
    speed = np.sqrt(np.sum([fields[k] ** 2 for k in ("vx", "vy", "vz")], axis=0)) + np.sqrt(
        5.0 / 3.0 * np.abs(fields["press"]) / fields["rho"])
    t_end = num_steps * 0.3 * (box / num_cells) / float(np.max(speed))
    params = SimulationParams(
        C_cfl=0.3,
        t_end=t_end,
        minimum_density=1e-6 * density_scale,
        minimum_pressure=1e-8 * pressure_scale,
    )
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.asarray(fields["rho"]),
        velocity_x=jnp.asarray(fields["vx"]),
        velocity_y=jnp.asarray(fields["vy"]),
        velocity_z=jnp.asarray(fields["vz"]),
        gas_pressure=jnp.asarray(fields["press"]),
    )
    return finalize_config(config, state.shape), params, registered_variables, state


def timed(fn, *args, repeats=2):
    """(result, first-call seconds, mean seconds of the repeats)."""
    start = time.time()
    result = jax.block_until_ready(fn(*args))
    first = time.time() - start
    start = time.time()
    for _ in range(repeats):
        result = jax.block_until_ready(fn(*args))
    return result, first, (time.time() - start) / repeats


def peak_gb():
    stats = jax.devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use", 0) / 2 ** 30


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--state", default=None)
    parser.add_argument("--num-cells", type=int, default=64, help="without --state")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--checkpoints", type=int, default=4)
    parser.add_argument("--tangent", choices=tuple(TANGENTS), default="approx")
    parser.add_argument("--backends", nargs="+", default=["native", "pallas_nt", "pallas"],
                        choices=("native", "pallas_nt", "pallas"))
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    args = parser.parse_args()

    fields, box = load_fields(args.state, args.num_cells)
    density_index = 0
    results = {}
    for name in args.backends:
        backend = NATIVE_JAX if name == "native" else PALLAS
        config, params, registered_variables, state = make_setup(
            fields, box, backend, args.tangent, args.steps, args.checkpoints, pallas_adjoint=name == "pallas")

        def evolve(initial):
            return time_integration(initial, config, params, registered_variables)

        def loss(initial):
            final = evolve(initial)
            return jnp.sum(final[density_index] ** 2)

        forward = jax.jit(loss)
        gradient = jax.jit(jax.grad(loss))
        value, forward_first, forward_time = timed(forward, state)
        grad, grad_first, grad_time = timed(gradient, state)
        results[name] = dict(value=float(value), grad=np.asarray(grad, np.float64),
                             forward=forward_time, gradient=grad_time, peak=peak_gb())
        print(f"[{name}] loss {float(value):.6e}; forward {forward_time:.3f} s (first {forward_first:.1f} s), "
              f"gradient {grad_time:.3f} s (first {grad_first:.1f} s), ratio {grad_time / forward_time:.2f}; "
              f"peak {peak_gb():.2f} GB; |g| {np.linalg.norm(results[name]['grad']):.6e}; "
              f"non-finite {int(np.sum(~np.isfinite(results[name]['grad'])))}", flush=True)

        # directional derivative against a central finite difference of this backend's forward
        direction = jnp.asarray(np.random.default_rng(1).standard_normal(state.shape), state.dtype) * state
        directional = float(np.sum(results[name]["grad"] * np.asarray(direction, np.float64)))
        plus = float(forward(state + args.fd_eps * direction))
        minus = float(forward(state - args.fd_eps * direction))
        central = (plus - minus) / (2.0 * args.fd_eps)
        print(f"[{name}] g.d {directional:+.6e}, central difference {central:+.6e}, "
              f"ratio {directional / central if central else float('nan'):.5f}", flush=True)

    for reference in ("native", "pallas_nt"):
        for name in ("pallas_nt", "pallas"):
            if name == reference or reference not in results or name not in results:
                continue
            ours, theirs = results[name]["grad"], results[reference]["grad"]
            relative = np.linalg.norm(ours - theirs) / np.linalg.norm(theirs)
            cosine = float(np.sum(ours * theirs) / (np.linalg.norm(ours) * np.linalg.norm(theirs)))
            print(f"[compare] gradient {name} vs {reference}: rel L2 {relative:.3e}, cosine {cosine:.9f}; "
                  f"gradient time {reference}/{name} "
                  f"{results[reference]['gradient'] / results[name]['gradient']:.2f}x", flush=True)


if __name__ == "__main__":
    main()
