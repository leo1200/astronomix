"""
Stability of a uniform magnetic field oblique to the grid under the
finite-difference MHD scheme, with and without the upwind dissipation of the
constrained-transport edge EMFs (``SimulationConfig.ct_upwind_dissipation``).

A uniform isothermal state (rho = 1, |B| = 1 along ``--direction``, sound
speed sqrt(beta / 2)) with a random velocity of rms ``--noise`` is an exact
equilibrium whose perturbations are all neutral (no parametric decay without
a finite-amplitude pump), so any growth is a property of the discretisation.
The run prints, per chunk, max |v . b|, max |v_perp|, max |rho - 1| and
max |B - B0|, and at the end the e-folding rate of max |v . b| over the second
half and the dominant in-plane wavevectors of the perturbation; the history
goes to ``data/oblique_<label>.npz``.

    python oblique_field_stability.py --direction 1,1 --N 256                   # production CT
    python oblique_field_stability.py --direction 1,1 --N 256 --ct-upwind 1     # with the upwind EMFs
"""

# ==== GPU selection ====
import os
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
from pathlib import Path

# jax
import jax

# The growth rates are measured from perturbations of 1e-6 on a unit field,
# which float32 resolves only to a few units in the last place.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    ISOTHERMAL,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    PositivityConfig,
    SimulationConfig,
    StaticFloatVector,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

DATA_DIR = Path(__file__).resolve().parent / "data"

#: Cells of the periodic slab along z (the perturbation is three-dimensional
#: but the field lies in or near the x-y plane).
SLAB_CELLS = 8


def parse_arguments():
    """Parse the command line (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", type=str, default="1,1", help="field direction bx,by[,bz]")
    parser.add_argument("--beta", type=float, default=1e-4, help="plasma beta 2 c_s^2 / B^2")
    parser.add_argument("--N", type=int, default=256, help="cells along x and y")
    parser.add_argument("--alfven-times", type=float, default=5.0, help="run time in box Alfven crossing times")
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--noise", type=float, default=1e-6, help="rms of the random seed velocity")
    parser.add_argument("--cfl", type=float, default=0.75)
    parser.add_argument("--ct-upwind", type=float, nargs="?", const=0.03, default=0.0,
                        help="SimulationConfig.ct_upwind_dissipation at strength kappa "
                             "(ct_upwind_strength; the bare flag is the default 0.03, 0 = off)")
    return parser.parse_args()


def build_case(args, field_direction):
    """
    The configuration, parameters, registered variables and the perturbed
    uniform state.

    Args:
        args: The parsed command-line arguments.
        field_direction: The unit vector of the background field.

    Returns:
        (config, params, registered_variables, state)
    """
    grid_spacing = 1.0 / args.N
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        mhd=True,
        dimensionality=3,
        equation_of_state=ISOTHERMAL,
        num_cells=StaticIntVector(args.N, args.N, SLAB_CELLS),
        box_size=StaticFloatVector(1.0, 1.0, SLAB_CELLS * grid_spacing),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        weno_z=True,
        weno_positivity_preserving=True,
        positivity_config=PositivityConfig(),
        ct_upwind_dissipation=args.ct_upwind > 0.0,
        ct_upwind_strength=args.ct_upwind if args.ct_upwind > 0.0 else 0.03,
        progress_bar=False,
    )
    params = SimulationParams(
        C_cfl=args.cfl,
        minimum_density=1e-10,
        isothermal_sound_speed=float(np.sqrt(0.5 * args.beta)),
    )
    registered_variables = get_registered_variables(config)
    shape = (args.N, args.N, SLAB_CELLS)
    keys = jax.random.split(jax.random.key(11), 3)
    velocity = [args.noise * jax.random.normal(key, shape) for key in keys]
    field = [jnp.full(shape, float(component)) for component in field_direction]
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.ones(shape),
        velocity_x=velocity[0],
        velocity_y=velocity[1],
        velocity_z=velocity[2],
        magnetic_field_x=field[0],
        magnetic_field_y=field[1],
        magnetic_field_z=field[2],
        interface_magnetic_field_x=field[0],
        interface_magnetic_field_y=field[1],
        interface_magnetic_field_z=field[2],
    )
    config = finalize_config(config, state.shape)
    return config, params, registered_variables, state


def main():
    """Run one case and report the growth of the perturbation."""
    args = parse_arguments()
    components = [float(value) for value in args.direction.split(",")]
    field_direction = np.asarray((components + [0.0, 0.0])[:3])
    field_direction = field_direction / np.linalg.norm(field_direction)
    config, params, registered_variables, state = build_case(args, field_direction)
    background_field = jnp.asarray(field_direction)[:, None, None, None]

    @jax.jit
    def perturbation_measures(state):
        density = state[registered_variables.density_index]
        velocity = jnp.stack([
            state[registered_variables.velocity_index.x],
            state[registered_variables.velocity_index.y],
            state[registered_variables.velocity_index.z],
        ])
        field = jnp.stack([
            state[registered_variables.magnetic_index.x],
            state[registered_variables.magnetic_index.y],
            state[registered_variables.magnetic_index.z],
        ])
        field_unit = field / jnp.maximum(jnp.sqrt(jnp.sum(field ** 2, axis=0)), 1e-30)
        field_aligned_velocity = jnp.sum(velocity * field_unit, axis=0)
        perpendicular_speed = jnp.sqrt(jnp.maximum(jnp.sum(velocity ** 2, axis=0) - field_aligned_velocity ** 2, 0.0))
        return jnp.stack([
            jnp.max(jnp.abs(field_aligned_velocity)),
            jnp.max(perpendicular_speed),
            jnp.max(jnp.abs(density - 1.0)),
            jnp.max(jnp.sqrt(jnp.sum((field - background_field) ** 2, axis=0))),
        ])

    label = f"d{args.direction.replace(',', '')}_beta{args.beta:g}_N{args.N}_ct{args.ct_upwind:g}"
    times, history = [0.0], [np.asarray(perturbation_measures(state))]
    for chunk in range(args.chunks):
        start = args.alfven_times * chunk / args.chunks
        end = args.alfven_times * (chunk + 1) / args.chunks
        state = time_integration(state, config, params._replace(t_start=start, t_end=end), registered_variables)
        measures = np.asarray(perturbation_measures(state))
        times.append(end)
        history.append(measures)
        print(
            f"[{label}] t={end:.2f} max|v.b|={measures[0]:.3e} max|v_perp|={measures[1]:.3e} "
            f"max|rho-1|={measures[2]:.3e} max|B-B0|={measures[3]:.3e}",
            flush=True,
        )
        if not np.all(np.isfinite(measures)) or measures[2] > 0.9:
            break
    times, history = np.asarray(times), np.asarray(history)
    half = len(times) // 2
    rate = np.log(history[-1, 0] / history[half, 0]) / (times[-1] - times[half])
    print(f"[{label}] RESULT max|v.b| {history[0, 0]:.2e} -> {history[-1, 0]:.2e}, max|rho-1| {history[-1, 2]:.2e}, "
          f"e-folding rate over the second half {rate:+.2f} per Alfven time", flush=True)

    # The dominant in-plane wavevectors of the perpendicular velocity
    # (slab-averaged): a growing mode near the Nyquist wavenumber shows here.
    velocity = np.stack([
        np.asarray(state[registered_variables.velocity_index.x]).mean(axis=2),
        np.asarray(state[registered_variables.velocity_index.y]).mean(axis=2),
    ])
    perpendicular_direction = np.array([-field_direction[1], field_direction[0]])
    if np.linalg.norm(perpendicular_direction) < 1e-6:
        perpendicular_direction = np.array([1.0, 0.0])
    perpendicular_velocity = np.einsum("i...,i->...", velocity, perpendicular_direction / np.linalg.norm(perpendicular_direction))
    power = np.abs(np.fft.fft2(perpendicular_velocity)) ** 2
    wavenumbers = np.fft.fftfreq(args.N, d=1.0 / args.N).astype(int)
    peaks = [np.unravel_index(index, power.shape) for index in np.argsort(power.ravel())[::-1][:4]]
    print(f"[{label}] dominant (kx, ky) of v_perp: " + ", ".join(f"({wavenumbers[i]}, {wavenumbers[j]})" for i, j in peaks),
          flush=True)

    DATA_DIR.mkdir(exist_ok=True)
    np.savez(
        DATA_DIR / f"oblique_{label}.npz",
        times=times,
        history=history,
        names=np.asarray(["max_v_par", "max_v_perp", "max_drho", "max_dB"]),
        rate=rate,
    )


if __name__ == "__main__":
    main()
