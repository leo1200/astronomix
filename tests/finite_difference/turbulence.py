"""
Minimal driven turbulence test setup in astronomix.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# import jax
# double precision
# jax.config.update("jax_enable_x64", True)

# numerics
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from matplotlib.colors import LogNorm

# astronomix
from astronomix._fluid_equations._equations import get_absolute_velocity
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig, TurbulentForcingParams
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    ISOTHERMAL,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    SnapshotSettings,
    finalize_config
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

def run_turbulence_tests():
    for num_cells in [32, 64, 128, 256, 512]:
        print(f"Running turbulence test with {num_cells}^3 cells...")
        # Configure the simulation
        config = SimulationConfig(
            solver_mode = FINITE_DIFFERENCE,
            equation_of_state = ISOTHERMAL, # IDEAL_GAS / ISOTHERMAL
            progress_bar = True,
            dimensionality = 3,
            num_cells = num_cells,
            box_size = 1.0,
            mhd = True,
            boundary_settings=BoundarySettings(
                BoundarySettings1D(
                    left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
                ),
                BoundarySettings1D(
                    left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
                ),
                BoundarySettings1D(
                    left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
                ),
            ),
            turbulent_forcing_config = TurbulentForcingConfig(
                turbulent_forcing = True,
            ),
            return_snapshots = True,
            num_snapshots = 50,
            snapshot_settings = SnapshotSettings(
                return_final_state=True,
                return_states = False,
                return_kinetic_energy_spectrum = True,
                return_magnetic_energy_spectrum = True,
                return_helicity_spectrum = True,
            )
        )

        # Set the simulation parameters
        params = SimulationParams(
            C_cfl = 1.5,
            isothermal_sound_speed = 1.0, # only active when equation_of_state is isothermal
            gamma = 5.0 / 3.0, # only active when equation_of_state is ideal_gas
            t_end = 4.0,
            turbulent_forcing_params = TurbulentForcingParams(
                energy_injection_rate = 1.65, # 1.65
            ),
        )

        # Initialize the registered variables
        registered_variables = get_registered_variables(config)

        # Initialize the simulation state
        density = jnp.ones((config.num_cells, config.num_cells, config.num_cells), dtype=jnp.float32)
        velocity_x = jnp.zeros_like(density)
        velocity_y = jnp.zeros_like(density)
        velocity_z = jnp.zeros_like(density)
        # only important for ideal gas
        pressure = jnp.ones_like(density)
        magnetic_field_x = jnp.zeros_like(density)
        magnetic_field_y = jnp.zeros_like(density)
        magnetic_field_z = jnp.ones_like(density) * 0.2
        bxb, byb, bzb = initialize_interface_fields(magnetic_field_x, magnetic_field_y, magnetic_field_z)
        initial_state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=density,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            velocity_z=velocity_z,
            gas_pressure=pressure,
            magnetic_field_x=magnetic_field_x,
            magnetic_field_y=magnetic_field_y,
            magnetic_field_z=magnetic_field_z,
            interface_magnetic_field_x=bxb,
            interface_magnetic_field_y=byb,
            interface_magnetic_field_z=bzb,
        )

        # Finalize the config
        config = finalize_config(config, initial_state.shape)

        # Run the simulation
        result = time_integration(initial_state, config, params, registered_variables)

        if config.snapshot_settings.return_states:
            final_state = result.states[-1]
        else:
            final_state = result.final_state

        # Calculate the rms velocity and the turbulent crossing time
        v = get_absolute_velocity(final_state, config, registered_variables)
        v_rms = jnp.sqrt(jnp.mean(v**2))
        t_cross = config.box_size.x / v_rms
        print(f"RMS velocity: {v_rms:.3f}")
        print(f"Turbulent crossing time: {t_cross:.3f}")

        # Calculate the final Mach number
        final_density = final_state[registered_variables.density_index]
        if config.equation_of_state == IDEAL_GAS:
            final_pressure = final_state[registered_variables.pressure_index]
            final_sound_speed = jnp.sqrt(params.gamma * final_pressure / final_density)
            print(f"Final sound speed: {jnp.mean(final_sound_speed):.3f}")
            final_mach_number = v_rms / jnp.mean(final_sound_speed)
        else:
            final_sound_speed = params.isothermal_sound_speed
            final_mach_number = v_rms / final_sound_speed
        print(f"Final Mach number: {final_mach_number:.3f}")

        # Calculate the final Alfvén Mach number
        if config.mhd:
            final_magnetic_field_x = final_state[registered_variables.magnetic_index.x]
            final_magnetic_field_y = final_state[registered_variables.magnetic_index.y]
            final_magnetic_field_z = final_state[registered_variables.magnetic_index.z]
            final_magnetic_energy_density = 0.5 * (final_magnetic_field_x**2 + final_magnetic_field_y**2 + final_magnetic_field_z**2)
            final_alfven_speed = jnp.sqrt(2 * final_magnetic_energy_density / final_density)
            final_alfven_mach_number = v_rms / jnp.mean(final_alfven_speed)
            print(f"Final Alfvén Mach number: {final_alfven_mach_number:.3f}")

        # Make animations
        print("Creating animations...")
        output_dir = Path("figures")
        output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        kinetic_spectra = result.kinetic_energy_spectrum
        magnetic_spectra = result.magnetic_energy_spectrum
        helicity_spectra = result.helicity_spectrum
        k_energy = result.k_spectra
        k_helicity = result.k_spectra
        time_points = result.time_points

        jnp.savez(
            data_dir / f"turbulence_N{config.num_cells.x}.npz",
            kinetic_spectra=kinetic_spectra,
            magnetic_spectra=magnetic_spectra,
            helicity_spectra=helicity_spectra,
            k_energy=k_energy,
            k_helicity=k_helicity,
            final_state=final_state,
            time_points=time_points,
        )

        if config.snapshot_settings.return_states:
            z_slice = config.num_cells.z // 2
            density_states = result.states[:, registered_variables.density_index, :, :, z_slice]

            density_fig, density_ax = plt.subplots(figsize=(6, 5))
            density_image = density_ax.imshow(
                density_states[0],
                origin="lower",
                extent=(0, config.box_size.x, 0, config.box_size.y),
                norm=LogNorm(vmin=jnp.min(density_states), vmax=jnp.max(density_states)),
                cmap="viridis",
            )
            density_colorbar = make_axes_locatable(density_ax).append_axes("right", size="5%", pad=0.1)
            density_fig.colorbar(density_image, cax=density_colorbar, label="density")
            density_ax.set_xlabel("x")
            density_ax.set_ylabel("y")

            def update_density(frame_idx):
                density_image.set_data(density_states[frame_idx])
                density_ax.set_title(f"Density (frame {frame_idx + 1}/{len(density_states)})")
                return (density_image,)

            density_animation = FuncAnimation(
                density_fig,
                update_density,
                frames=len(density_states),
                interval=10,
                blit=False,
            )
            density_animation.save(output_dir / f"turbulence_test_density_N{config.num_cells.x}.gif", writer=PillowWriter(fps=12))
            plt.close(density_fig)

        spectra_fig, (ax_energy, ax_helicity) = plt.subplots(1, 2, figsize=(12, 5))

        print("shape:", k_energy.shape)

        line_ek, = ax_energy.plot(k_energy, kinetic_spectra[0], label="kinetic")
        line_em, = ax_energy.plot(k_energy, magnetic_spectra[0], label="magnetic")
        ax_energy.set_xscale("log")
        ax_energy.set_yscale("log")
        ax_energy.set_xlabel("k")
        ax_energy.set_ylabel("E(k)")
        k_min = 30 # skip the first noisy bins
        k_max = k_energy[-1]
        ax_energy.set_xlim(k_min, k_max)
        ax_energy.legend(loc="upper right")

        line_hm, = ax_helicity.plot(k_helicity, jnp.abs(helicity_spectra[0]), label="abs(magnetic helicity)")
        # line_hc, = ax_helicity.plot(
        #     k_helicity, cross_helicity_spectra[0], label="cross helicity"
        # )
        ax_helicity.set_xscale("log")
        ax_helicity.set_yscale("log")
        ax_helicity.set_xlabel("k")
        ax_helicity.set_ylabel("H(k)")
        ax_helicity.set_xlim(k_min, k_max)
        ax_helicity.set_ylim(1e-12, 1e-3)
        ax_helicity.legend(loc="upper right")

        plt.tight_layout()

        def update_spectra(frame_idx):
            line_ek.set_ydata(kinetic_spectra[frame_idx])
            line_em.set_ydata(magnetic_spectra[frame_idx])
            line_hm.set_ydata(jnp.abs(helicity_spectra[frame_idx]))
            # line_hc.set_ydata(cross_helicity_spectra[frame_idx])

            current_energy = jnp.concatenate(
                [kinetic_spectra[frame_idx], magnetic_spectra[frame_idx]]
            )
            positive_energy = current_energy[current_energy > 0]
            if positive_energy.size > 0:
                emin = float(jnp.min(positive_energy))
                emax = float(jnp.max(positive_energy))
                ax_energy.set_ylim(max(emin * 0.5, 1e-16), emax * 2.0)

            ax_energy.set_title("Kinetic and Magnetic Energy Spectrum")
            ax_helicity.set_title("Helicity Spectrum")

            return line_ek, line_em, line_hm # , line_hc

        spectra_animation = FuncAnimation(
            spectra_fig,
            update_spectra,
            frames=len(kinetic_spectra),
            interval=10,
            blit=False,
        )
        spectra_animation.save(output_dir / f"turbulence_test_spectra_N{config.num_cells.x}.gif", writer=PillowWriter(fps=12))
        plt.close(spectra_fig)

def compare_spectra():

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # based on the data saved, create a comparison plot of the kinetic and magnetic energy spectra
    fig_comp, axs_comp = plt.subplots(1, 2, figsize=(12, 5))

    num_cell_list = [32, 64, 128, 256]

    for num_cells in num_cell_list:
        data = jnp.load(data_dir / f"turbulence_N{num_cells}.npz")
        k_energy = data["k_energy"]
        kinetic_spectra = data["kinetic_spectra"]
        magnetic_spectra = data["magnetic_spectra"]

        # Plot the kinetic energy spectrum
        axs_comp[0].plot(k_energy, kinetic_spectra[-1], label=f"N={num_cells}")
        axs_comp[0].set_xscale("log")
        axs_comp[0].set_yscale("log")
        axs_comp[0].set_xlabel("k")
        axs_comp[0].set_ylabel("E_kinetic(k)")
        axs_comp[0].set_title("Kinetic Energy Spectrum")
        axs_comp[0].set_xlim(30, 2e3)
        axs_comp[0].set_ylim(1e-10, 1e-2)
        axs_comp[0].legend()

        # Plot the magnetic energy spectrum
        axs_comp[1].plot(k_energy, magnetic_spectra[-1], label=f"N={num_cells}")
        axs_comp[1].set_xscale("log")
        axs_comp[1].set_yscale("log")
        axs_comp[1].set_xlabel("k")
        axs_comp[1].set_ylabel("E_magnetic(k)")
        axs_comp[1].set_title("Magnetic Energy Spectrum")
        axs_comp[1].set_xlim(30, 2e3)
        axs_comp[1].set_ylim(1e-10, 1e-2)
        axs_comp[1].legend()

    fig_comp.tight_layout()
    fig_comp.savefig(output_dir / "turbulence_spectra_comparison.png")

def plot_final_density_slices():

    num_cell_list = [32, 64, 128, 256]

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # also plot the final density slice for all resolutions
    for num_cells in num_cell_list:
        data = jnp.load(data_dir / f"turbulence_N{num_cells}.npz")
        final_state = data["final_state"]
        density = final_state[0]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            density[:, :, num_cells // 2],
            origin="lower",
            extent=(0, 1, 0, 1),
            norm=LogNorm(vmin=jnp.min(density), vmax=jnp.max(density)),
            cmap="viridis",
        )
        cbar = fig.colorbar(im, ax=ax, label="density")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Final Density Slice (N={num_cells})")
        fig.savefig(output_dir / f"turbulence_final_density_N{num_cells}.png")
        plt.close(fig)

if __name__ == "__main__":
    # run_turbulence_tests()
    compare_spectra()
    plot_final_density_slices()