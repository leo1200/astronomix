"""
Thermal conduction: damping of a temperature perturbation (finite-difference).

Sets up a periodic 2D box with a small kx=1 temperature sinusoid and integrates
it with and without thermal conduction. Conduction should damp the mode; the
script reports the surviving amplitude, the (adjoint) gradient of that amplitude
with respect to the conductivity, and a plot of the final x-profiles in
``figures/``.
"""

# general
from pathlib import Path

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    BACKWARDS,
    PERIODIC_BOUNDARY,
)
from astronomix.option_classes.simulation_config import IDEAL_GAS

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    DiffusionConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    time_integration,
    get_registered_variables,
)


# Figures are written to the local figures/ directory next to this test.
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

N = 32
PERIODIC2D = BoundarySettings(
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)


def make_config(conduction, diff_mode=None):
    """Build the simulation configuration for the conduction box.

    Args:
        conduction: Whether thermal conduction is active.
        diff_mode: Optional differentiation mode. When given, it is set on the
            configuration together with a checkpointed adjoint (for the gradient
            computation below).

    Returns:
        The simulation configuration.
    """
    config = SimulationConfig(
        equation_of_state=IDEAL_GAS,
        dimensionality=2,
        num_cells=N,
        box_size=1.0,
        mhd=False,
        boundary_settings=PERIODIC2D,
        diffusion_config=DiffusionConfig(thermal_conduction=conduction),
    )
    if diff_mode is not None:
        config = config._replace(differentiation_mode=diff_mode, num_checkpoints=64)
    return config


def initial(config, registered_variables):
    """Construct the initial state: a kx=1 temperature sinusoid at rest.

    The temperature T = p / rho carries the sinusoid via the pressure while the
    density stays uniform, so the perturbation is purely thermal.

    Args:
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The initial primitive state.
    """
    x = jnp.linspace(0, 1, N, endpoint=False)[:, None] * jnp.ones((N, N))
    density = jnp.ones((N, N))
    pressure = 1.0 + 0.05 * jnp.sin(2 * jnp.pi * x)
    zero = jnp.zeros((N, N))
    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zero,
        velocity_y=zero,
        gas_pressure=pressure,
    )


def final_pressure(kappa, conduction, t_end=0.02, diff_mode=None):
    """Integrate the box and return the final pressure field.

    Args:
        kappa: The thermal conductivity.
        conduction: Whether thermal conduction is active.
        t_end: The end time of the integration.
        diff_mode: Optional differentiation mode passed to the configuration.

    Returns:
        The final pressure field.
    """
    config = make_config(conduction, diff_mode)
    registered_variables = get_registered_variables(config)
    initial_state = initial(config, registered_variables)
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=t_end, thermal_conductivity=kappa)
    final_state = time_integration(initial_state, config, params, registered_variables)
    return final_state[registered_variables.pressure_index]


def amplitude(pressure):
    """Return the amplitude of the kx=1 mode of the x-profile.

    The x-profile is the y-average of the pressure; its first (kx=1) Fourier
    coefficient is the conduction-damped sinusoid we track.

    Args:
        pressure: A pressure field.

    Returns:
        The magnitude of the kx=1 Fourier mode of the y-averaged pressure.
    """
    return jnp.abs(jnp.fft.rfft(jnp.mean(pressure, axis=1))[1])


# -------------------------------------------------------------
# =============== ↓ Run: hydro only vs hydro + conduction ↓ ===
# -------------------------------------------------------------

pressure_off = final_pressure(0.0, conduction=False)
pressure_on = final_pressure(0.1, conduction=True)
amplitude_off = float(amplitude(pressure_off))
amplitude_on = float(amplitude(pressure_on))
print(f"kx=1 pressure-mode amplitude: OFF={amplitude_off:.5f}  ON(kappa=0.1)={amplitude_on:.5f}")

# Gradient of the surviving amplitude with respect to the conductivity: exercises
# the checkpointed adjoint and should be negative (more conduction, more damping).
gradient = float(
    jax.grad(lambda k: amplitude(final_pressure(k, conduction=True, diff_mode=BACKWARDS)))(0.1)
)
print(f"d(amplitude)/d(kappa) at kappa=0.1: {gradient:.4e}  (expected < 0)")

# -------------------------------------------------------------
# =============== ↑ Run: hydro only vs hydro + conduction ↑ ===
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Plot the final x-profiles ↓ ===============
# -------------------------------------------------------------

x = jnp.linspace(0, 1, N, endpoint=False)
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(x, jnp.mean(pressure_off, axis=1), "o-", label=f"conduction OFF (amp={amplitude_off:.4f})")
ax.plot(x, jnp.mean(pressure_on, axis=1), "s-", label=f"conduction ON, $\\kappa=0.1$ (amp={amplitude_on:.4f})")
ax.set_xlabel("x")
ax.set_ylabel(r"$\langle p \rangle_y$")
ax.set_title("Thermal conduction damps the temperature perturbation")
ax.legend()
fig.tight_layout()
fig.savefig(figures_dir / "conduction_pressure_profile.png", dpi=200)

# -------------------------------------------------------------
# =============== ↑ Plot the final x-profiles ↑ ===============
# -------------------------------------------------------------
