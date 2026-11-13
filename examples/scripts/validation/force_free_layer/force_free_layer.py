"""
Static force-free rotational layers, oblique to the grid: a minimal test of
spurious field-aligned forces in low-beta MHD.

Uniform density rho0 = 1, uniform gas pressure p = beta B0^2 / 2, gas at rest,
and a field of constant magnitude B0 = 1 that rotates in the plane normal to
n = (2, 1, 0)/sqrt(5) across two layers,

    B = B0 (cos theta(s) t + sin theta(s) z),   t = (-1, 2, 0)/sqrt(5),
    s = n . x,   theta: 0 -> pi -> 2 pi across layers at phi = 2x + y = 1/4, 3/4,

with tanh profiles of half-width ``--width`` cells (measured along n). The
field is divergence-free with J parallel to B, the magnetic and the gas
pressure are uniform, so the exact solution is static: every velocity is
numerical. The direction n = (2, 1, 0)/sqrt(5) is deliberately not 45 degrees,
where the interpolated cell-centred field has no central divergence by
symmetry. The field is built from a vector potential (discretely
divergence-free faces for the finite-difference CT scheme), on a periodic
N x N x 8 slab (uniform along z).

Diagnostics, reduced on the GPU every chunk: the field-aligned speed
max |v . b| (b = B/|B|), the speed along the layer normal max |v . n| (numerical
diffusion of the rotation drops |B| in the layer and drives a genuine
perpendicular inflow), max |v|, rho_min, the time step the CFL condition sets,
and the relative central divergence dx |div B_cc| / |B| of the cell-centred
field.

Schemes (``--scheme``):
  fd         5th-order FD, WENO-Z + positivity-preserving WENO, SSPRK4, CT,
             C_cfl 0.75 (the production setting of the supersonic turbulence)
  fd_powell  fd plus the Godunov-Powell source -B_cc div B_cc (6th-order central
             divergence) in every stage, added in this script only
  fd_floor   fd plus AthenaPK's density floor at 1e-3 (POSITIVITY_HARD_FLOOR)
  fv         VL2 + PLM + HLLD with GLM cleaning (AthenaPK's scheme), C_cfl 0.3

    python force_free_layer.py --scheme fd --beta 1e-4 --width 0.5 --N 256
    python force_free_layer.py --plot
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
import importlib
import time as walltime
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLLD,
    POSITIVITY_HARD_FLOOR,
    POSITIVITY_NONE,
    VAN_LEER,
    VL2,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    FVSolverConfig,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
)
from astronomix.option_classes.simulation_config import (
    StaticFloatVector,
    StaticIntVector,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)
from astronomix.initial_condition_generation.magnetic_field_from_vector_potential import (
    setup_magnetic_fields_from_vector_potential,
)


#: Script-relative output directories.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURE_DIR = SCRIPT_DIR / "figures"

def layer_directions(normal_integers):
    """The unit layer normal n = (n1, n2, 0)/|.|, the in-plane t = (-n2, n1, 0)/|.| and |(n1, n2)|."""
    n1, n2 = normal_integers
    norm = float(np.hypot(n1, n2))
    return np.array([n1, n2, 0.0]) / norm, np.array([-n2, n1, 0.0]) / norm, norm

#: Cells along z of the slab (the field is uniform along z).
SLAB_CELLS = 8

#: Field strength, density and adiabatic index.
FIELD = 1.0
DENSITY = 1.0
GAMMA = 5.0 / 3.0

#: A run counts as collapsed once its time step falls below this fraction of
#: the initial one.
COLLAPSE_FRACTION = 1e-2

#: The C_cfl of each scheme family.
CFL = {"fd": 0.75, "fv": 0.3}


def parse_arguments():
    """Parse the command line (see the module docstring)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scheme", choices=["fd", "fd_powell", "fd_floor", "fv"], default="fd")
    parser.add_argument("--beta", type=float, default=1e-4, help="plasma beta 2 p / B0^2")
    parser.add_argument("--width", type=float, default=0.5,
                        help="tanh half-width of the layers in cells, along the normal")
    parser.add_argument("--N", type=int, default=256, help="cells along x and y")
    parser.add_argument("--normal", type=str, default="2,1",
                        help="integer layer normal n1,n2 (periodic for integers): 2,1 is oblique; "
                             "1,0 (grid-aligned) and 1,1 (45 degrees) give the cell-centred field no "
                             "central divergence")
    parser.add_argument("--alfven-times", type=float, default=5.0,
                        help="run length in Alfven crossing times of the box (v_A = 1)")
    parser.add_argument("--chunks", type=int, default=50, help="diagnostic snapshots over the run")
    parser.add_argument("--budget-min", type=float, default=20.0, help="wall-time budget per run")
    parser.add_argument("--x64", action="store_true", help="double precision")
    parser.add_argument("--plot", action="store_true", help="make the figure from data/*.npz")
    return parser.parse_args()


# -------------------------------------------------------------
# ================ ↓ Initial condition ↓ ======================
# -------------------------------------------------------------
def layer_potentials(width_phi, norm):
    """
    Periodic tables of the two potentials of the layer field as functions of
    phi = 2x + y = sqrt(5) s (period 1): A_z with dA_z/ds = -B_t and the
    in-plane amplitude G (A = G t) with dG/ds = B_z; phi = n1 x + n2 y = norm s.

    Args:
        width_phi: The tanh half-width of the layers in units of phi.
        norm: |(n1, n2)|, the factor between phi and s.

    Returns:
        ``(phi, a_z, g, mean_field_t, mean_field_z)``; the means (zero up to
        quadrature error) are the uniform background a periodic potential
        cannot carry.
    """
    phi = np.linspace(0.0, 1.0, 2 ** 20 + 1)
    angle = 0.5 * np.pi * (np.tanh((phi - 0.25) / width_phi) + np.tanh((phi - 0.75) / width_phi)) + np.pi
    field_t = FIELD * np.cos(angle)
    field_z = FIELD * np.sin(angle)
    mean_field_t = np.trapezoid(field_t, phi)
    mean_field_z = np.trapezoid(field_z, phi)

    def periodic_integral(values):
        # Integral over s = phi / norm of the zero-mean part, by the
        # cumulative trapezoid rule.
        increments = 0.5 * (values[1:] + values[:-1]) * np.diff(phi) / norm
        return np.concatenate([[0.0], np.cumsum(increments)])

    a_z = -periodic_integral(field_t - mean_field_t)
    g = periodic_integral(field_z - mean_field_z)
    return phi, a_z, g, mean_field_t, mean_field_z


def build_case(args):
    """
    The configuration, parameters, registered variables and initial state.

    Args:
        args: The parsed command-line arguments.

    Returns:
        ``(config, params, registered_variables, state)``.
    """
    finite_volume = args.scheme == "fv"
    dx = 1.0 / args.N
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    if finite_volume:
        config = SimulationConfig(
            solver_mode=FINITE_VOLUME,
            solver_config=FVSolverConfig(
                time_integrator=VL2,
                limiter=VAN_LEER,
                riemann_solver=HLLD,
            ),
        )
    else:
        config = SimulationConfig(
            solver_mode=FINITE_DIFFERENCE,
            solver_config=FDSolverConfig(weno_z=True, weno_positivity_preserving=True),
            positivity_config=PositivityConfig(
                per_step_mode=POSITIVITY_HARD_FLOOR if args.scheme == "fd_floor" else POSITIVITY_NONE,
            ),
        )
    config = config._replace(
        mhd=True,
        dimensionality=3,
        num_cells=StaticIntVector(args.N, args.N, SLAB_CELLS),
        box_size=StaticFloatVector(1.0, 1.0, SLAB_CELLS * dx),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        progress_bar=False,
    )
    params = SimulationParams(
        C_cfl=CFL["fv" if finite_volume else "fd"],
        gamma=GAMMA,
        minimum_density=1e-3 if args.scheme == "fd_floor" else 1e-10,
        minimum_pressure=1e-12,
    )

    normal, tangent, norm = layer_directions(tuple(int(k) for k in args.normal.split(",")))
    n1, n2 = (float(k) for k in args.normal.split(","))
    phi_table, a_z_table, g_table, mean_t, mean_z = layer_potentials(norm * args.width * dx, norm)
    phi_table, a_z_table, g_table = (jnp.asarray(x) for x in (phi_table, a_z_table, g_table))

    def vector_potential(X, Y, Z):
        phi = jnp.mod(n1 * X + n2 * Y, 1.0)
        a_z = jnp.interp(phi, phi_table, a_z_table)
        g = jnp.interp(phi, phi_table, g_table)
        return g * tangent[0], g * tangent[1], a_z

    field_x, field_y, field_z, face_x, face_y, face_z = setup_magnetic_fields_from_vector_potential(
        config,
        vector_potential,
    )
    # The uniform background a periodic potential cannot carry.
    background = mean_t * tangent + mean_z * np.array([0.0, 0.0, 1.0])
    field_x, face_x = field_x + background[0], face_x + background[0]
    field_y, face_y = field_y + background[1], face_y + background[1]
    field_z, face_z = field_z + background[2], face_z + background[2]

    registered_variables = get_registered_variables(config)
    shape = field_x.shape
    zero = jnp.zeros(shape)
    fields = dict(
        config=config,
        registered_variables=registered_variables,
        density=jnp.full(shape, DENSITY),
        velocity_x=zero,
        velocity_y=zero,
        velocity_z=zero,
        gas_pressure=jnp.full(shape, 0.5 * args.beta * FIELD ** 2),
        magnetic_field_x=field_x,
        magnetic_field_y=field_y,
        magnetic_field_z=field_z,
    )
    if not finite_volume:
        fields.update(
            interface_magnetic_field_x=face_x,
            interface_magnetic_field_y=face_y,
            interface_magnetic_field_z=face_z,
        )
    state = construct_primitive_state(**fields)
    config = finalize_config(config, state.shape)
    return config, params, registered_variables, state
# -------------------------------------------------------------
# ================ ↑ Initial condition ↑ ======================
# -------------------------------------------------------------


def install_powell_source():
    """
    Add the Godunov-Powell source -B_cc div B_cc (6th-order central divergence
    of the cell-centred field each stage sees; for an ideal gas also
    -(v . B_cc) div B_cc in the energy) to every SSPRK4 stage of the FD MHD
    integrator, by wrapping its per-stage physics-source call in this process.
    """
    ssprk_module = importlib.import_module("astronomix._finite_difference._time_integrators._ssprk")
    original_sources = ssprk_module.explicit_sources
    weights = {1: 45.0 / 60.0, 2: -9.0 / 60.0, 3: 1.0 / 60.0}

    def derivative(field, axis, spacing):
        return sum(
            weight * (jnp.roll(field, -shift, axis) - jnp.roll(field, shift, axis))
            for shift, weight in weights.items()
        ) / spacing

    def sources_with_powell(conserved_state, dt, gamma, config, params, helper_data, registered_variables,
                            **stage_quantities):
        sources = original_sources(conserved_state, dt, gamma, config, params, helper_data, registered_variables,
                                   **stage_quantities)
        magnetic_index = registered_variables.magnetic_index
        momentum_index = registered_variables.velocity_index
        magnetic = [conserved_state[i] for i in (magnetic_index.x, magnetic_index.y, magnetic_index.z)]
        momentum_indices = (momentum_index.x, momentum_index.y, momentum_index.z)
        divergence = sum(derivative(magnetic[axis], axis, config.grid_spacing) for axis in range(3))
        for index, component in zip(momentum_indices, magnetic):
            sources = sources.at[index].add(-dt * component * divergence)
        density = conserved_state[registered_variables.density_index]
        v_dot_b = sum(conserved_state[m] * b for m, b in zip(momentum_indices, magnetic)) / density
        sources = sources.at[registered_variables.energy_index].add(-dt * v_dot_b * divergence)
        return sources

    ssprk_module.explicit_sources = sources_with_powell


def build_diagnostics(config, params, registered_variables, N, normal_vector):
    """The jitted per-snapshot reduction (see the module docstring)."""
    rv = registered_variables
    dx = 1.0 / N
    normal = jnp.asarray(normal_vector)

    def central(field, axis):
        return (jnp.roll(field, -1, axis) - jnp.roll(field, 1, axis)) / (2.0 * dx)

    @jax.jit
    def diagnostics(state):
        density = state[rv.density_index]
        velocity = jnp.stack([state[rv.velocity_index.x], state[rv.velocity_index.y], state[rv.velocity_index.z]])
        magnetic = jnp.stack([state[rv.magnetic_index.x], state[rv.magnetic_index.y], state[rv.magnetic_index.z]])
        pressure = state[rv.pressure_index]
        field_magnitude = jnp.sqrt(jnp.sum(magnetic ** 2, axis=0))
        b_hat = magnetic / jnp.maximum(field_magnitude, 1e-30)
        positive_density = jnp.maximum(density, 1e-30)
        sound2 = GAMMA * jnp.maximum(pressure, 0.0) / positive_density
        total2 = sound2 + field_magnitude ** 2 / positive_density
        signal_sum = 0.0
        for axis in range(3):
            alfven2 = magnetic[axis] ** 2 / positive_density
            fast = jnp.sqrt(0.5 * (total2 + jnp.sqrt(jnp.maximum(total2 ** 2 - 4.0 * sound2 * alfven2, 0.0))))
            signal_sum = signal_sum + jnp.max(jnp.abs(velocity[axis]) + fast)
        divergence = sum(central(magnetic[axis], axis) for axis in range(3))
        return jnp.stack([
            jnp.max(jnp.abs(jnp.sum(velocity * b_hat, axis=0))),
            jnp.max(jnp.abs(jnp.tensordot(normal, velocity, axes=1))),
            jnp.max(jnp.sqrt(jnp.sum(velocity ** 2, axis=0))),
            jnp.min(density),
            jnp.min(field_magnitude),
            params.C_cfl * dx / signal_sum,
            jnp.max(dx * jnp.abs(divergence) / jnp.maximum(field_magnitude, 1e-30)),
            jnp.any(~jnp.isfinite(state)).astype(density.dtype),
        ])

    return diagnostics


NAMES = ("v_par", "v_normal", "v_max", "rho_min", "b_min", "dt", "divb", "nonfinite")


def run(args):
    """Run one case in chunks, stopping at a time-step collapse, NaN or the budget."""
    if args.scheme == "fd_powell":
        install_powell_source()
    config, params, rv, state = build_case(args)
    normal_vector, _, _ = layer_directions(tuple(int(k) for k in args.normal.split(",")))
    diagnostics = build_diagnostics(config, params, rv, args.N, normal_vector)
    tag = f"{args.scheme}_beta{args.beta:g}_w{args.width:g}_N{args.N}_T{args.alfven_times:g}"
    if args.normal != "2,1":
        tag += "_n" + args.normal.replace(",", "")
    end_time = args.alfven_times
    times = [0.0]
    rows = [np.asarray(diagnostics(state), dtype=np.float64)]
    print(f"[{tag}] start: " + " ".join(f"{n}={v:.3e}" for n, v in zip(NAMES, rows[0])), flush=True)
    start = walltime.time()
    status = "passed"
    for chunk in range(args.chunks):
        t0 = end_time * chunk / args.chunks
        t1 = end_time * (chunk + 1) / args.chunks
        state = time_integration(state, config, params._replace(t_start=t0, t_end=t1), rv)
        row = np.asarray(diagnostics(state), dtype=np.float64)
        times.append(t1)
        rows.append(row)
        print(f"[{tag}] t={t1:.3f} " + " ".join(f"{n}={v:.3e}" for n, v in zip(NAMES, row)), flush=True)
        if row[NAMES.index("nonfinite")] > 0:
            status = "nonfinite"
            break
        if row[NAMES.index("dt")] < COLLAPSE_FRACTION * rows[0][NAMES.index("dt")]:
            status = "dt_collapse"
            break
        if walltime.time() - start > 60.0 * args.budget_min:
            status = "budget"
            break
    wall = walltime.time() - start
    rows = np.asarray(rows)
    DATA_DIR.mkdir(exist_ok=True)
    np.savez(
        DATA_DIR / f"{tag}.npz",
        times=np.asarray(times),
        status=status,
        wall=wall,
        scheme=args.scheme,
        beta=args.beta,
        width=args.width,
        N=args.N,
        **{name: rows[:, i] for i, name in enumerate(NAMES)},
    )
    print(f"[{tag}] RESULT {status} at t={times[-1]:.3f}: max|v.b|={rows[:, 0].max():.3e} "
          f"max|v.n|={rows[:, 1].max():.3e} rho_min={rows[:, 3].min():.3e} "
          f"dt {rows[0, 5]:.2e} -> {rows[-1, 5]:.2e}, wall {wall:.0f} s", flush=True)


#: The figure's panels: title and (scheme, beta, width, N, run length, normal, label) per line.
PANELS = (
    ("sharp oblique layer, fixes, $\\beta = 10^{-6}$", (
        ("fd", 1e-6, 0.5, 256, 5, "2,1", "FD (WENO-Z + PP)"),
        ("fd_powell", 1e-6, 0.5, 256, 5, "2,1", "FD + Powell source"),
        ("fd_floor", 1e-6, 0.5, 256, 5, "2,1", "FD + density floor $10^{-3}$"),
        ("fv", 1e-6, 0.5, 256, 5, "2,1", "FV (VL2 + HLLD + GLM)"),
    )),
    ("sharp layer, orientation, $\\beta = 10^{-6}$", (
        ("fd", 1e-6, 0.5, 256, 20, "2,1", "FD, $n \\propto (2, 1)$"),
        ("fd", 1e-6, 0.5, 256, 10, "1,1", "FD, $n \\propto (1, 1)$: div $B_{cc} = 0$"),
        ("fd", 1e-6, 0.5, 256, 10, "1,0", "FD, $n = \\hat{x}$ (1D)"),
        ("fv", 1e-6, 0.5, 256, 20, "2,1", "FV, $n \\propto (2, 1)$"),
    )),
    ("smooth oblique layer, fixed width, $\\beta = 10^{-6}$", (
        ("fd", 1e-6, 2, 128, 5, "2,1", "FD, $N = 128$"),
        ("fd", 1e-6, 4, 256, 5, "2,1", "FD, $N = 256$"),
        ("fd", 1e-6, 8, 512, 5, "2,1", "FD, $N = 512$"),
        ("fd_powell", 1e-6, 4, 256, 5, "2,1", "FD + Powell, $N = 256$"),
        ("fv", 1e-6, 4, 256, 5, "2,1", "FV, $N = 256$"),
    )),
)


def plot():
    """Field-aligned speed against time: sharp and smooth layers, and the FD resolution sweep."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(PANELS), figsize=(5.2 * len(PANELS), 4.0))
    for axis, (title, lines) in zip(axes, PANELS):
        for scheme, beta, width, cells, alfven_times, normal, label in lines:
            suffix = "" if normal == "2,1" else "_n" + normal.replace(",", "")
            path = DATA_DIR / f"{scheme}_beta{beta:g}_w{width:g}_N{cells}_T{alfven_times:g}{suffix}.npz"
            if not path.exists():
                continue
            run = np.load(path)
            axis.semilogy(run["times"], np.maximum(run["v_par"], 1e-8), label=label)
        axis.set_xlabel(r"$t\, v_A / L$")
        axis.set_ylabel(r"max $|\mathbf{v}\cdot\hat{\mathbf{b}}|\, /\, v_A$")
        axis.set_title(title, fontsize=10)
        axis.set_ylim(1e-6, 5.0)
        axis.legend(fontsize=8, frameon=False)
        axis.grid(True, which="both", alpha=0.15)
    figure.tight_layout()
    FIGURE_DIR.mkdir(exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(FIGURE_DIR / f"force_free_layer{suffix}", dpi=200)
    print("wrote", FIGURE_DIR / "force_free_layer.png")


def main():
    """Run one case, or make the figure."""
    args = parse_arguments()
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    if args.plot:
        plot()
    else:
        run(args)


if __name__ == "__main__":
    main()
