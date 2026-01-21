from typing import Optional, Tuple
import random

from jf1uids import SimulationConfig
from jf1uids import get_helper_data
from jf1uids import SimulationParams
from jf1uids import time_integration
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.data_classes.simulation_snapshot_data import SnapshotData
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.fluid_equations.fluid import construct_primitive_state
from jf1uids import get_registered_variables
from jf1uids.option_classes.simulation_config import SnapshotSettings, finalize_config
from jf1uids.initial_condition_generation.turb import create_turb_field

# jf1uids constants
from jf1uids.option_classes.simulation_config import (
    BACKWARDS,
    FORWARDS,
    HLL,
    HLLC,
    MINMOD,
    OSHER,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

# units
from jf1uids import CodeUnits
from astropy import units as u
import astropy.constants as c
from corrector_src.utils.downaverage import downaverage

import corrector_src.data.blast_creation as blast

from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CorrectorParams,
    CorrectorConfig,
)
from omegaconf import OmegaConf
import numpy as np
import jax.numpy as jnp
import jax.random
import time
from dataclasses import dataclass


@dataclass
class SimulationBundle:
    initial_state: np.ndarray
    config: SimulationConfig
    params: SimulationParams
    helper: HelperData
    reg_vars: RegisteredVariables
    seed: int

    def override_solver_in_the_loop(
        self,
        corrector_config: Optional[CorrectorConfig] = None,
        corrector_params: Optional[CorrectorParams] = None,
    ):
        if corrector_config:
            self.config = self.config._replace(corrector_config=corrector_config)
        if corrector_params:
            self.params = self.params._replace(corrector_params=corrector_params)
        return self

    def override_config(self, strict=False, **overrides):
        valid_fields = self.config._fields
        valid, invalid = {}, []

        for k, v in overrides.items():
            if k in valid_fields:
                valid[k] = v
            else:
                invalid.append(k)

        if invalid:
            msg = f"Invalid config keys: {invalid}"
            if strict:
                raise KeyError(msg)
            else:
                print(f"[override_config] Warning: {msg}")

        self.config = self.config._replace(**valid)
        return self

    def unpack_integrate(self):
        kwargs = {
            "primitive_state": self.initial_state,
            "config": self.config,
            "params": self.params,
            "helper_data": self.helper,
            "registered_variables": self.reg_vars,
        }
        return kwargs

    def copy(self):
        """Create a shallow copy of the bundle."""
        return SimulationBundle(
            self.initial_state.copy(),
            self.config,
            self.params,
            self.helper,
            self.reg_vars,
            self.seed,
        )

    def convert_to_lr(self, downscale_factor):
        initial_state_lr = downaverage(
            state=self.initial_state, downscale_factor=downscale_factor
        )
        config_lr = finalize_config(
            self.config._replace(num_cells=self.config.num_cells // downscale_factor),
            initial_state_lr.shape,
        )
        helper_lr = get_helper_data(config_lr)
        reg_vars_lr = get_registered_variables(config_lr)
        params_lr = self.params  # params independent of config
        return SimulationBundle(
            initial_state_lr,
            config_lr,
            params_lr,
            helper_lr,
            reg_vars_lr,
            self.seed,
        )


class dataset:
    def __init__(self, scenarios_to_use: list[int], cfg_data: OmegaConf):
        self.num_scenarios = len(scenarios_to_use)
        self.complete_scenario_list = [
            "mhd_blast",
            "turbulence",
            "turbulence_mhd",
        ]
        self.scenario_list = [self.complete_scenario_list[i] for i in scenarios_to_use]
        self.default_resolution = cfg_data.hr_res
        self.downscale_factor = cfg_data.downscaling_factor
        self.cfg_data = cfg_data

        # 🧭 Scenario dispatch map
        self._scenario_dispatch = {
            "mhd_blast": self._init_mhd_blast,
            "turbulence": self._init_turbulence,
            "turbulence_mhd": self._init_turbulence_mhd,
        }

        if (
            self.cfg_data.use_specific_snapshot_timepoints
            and self.cfg_data.snapshot_timepoints is not None
        ):
            self.cfg_data.num_snapshots = len(self.cfg_data.snapshot_timepoints)
            # print(
            #     f"Returning snapshots with specific snapshots {self.cfg_data.snapshot_timepoints}"
            # )
        # elif self.cfg_data.return_snapshots:
        #     print(f"Returning {self.cfg_data.num_snapshots} snapshots")

    def _init_mhd_blast(self, resolution: int, **overrides):
        state_tuple = blast.randomized_initial_blast_state(
            resolution, cfg_data=self.cfg_data, rng_seed=overrides.get("rng_seed")
        )
        return SimulationBundle(*state_tuple)

    def _init_turbulence(self, resolution: int, **overrides):
        state_tuple = self.randomized_turbulent_initial_state(
            resolution, mhd=False, rng_seed=overrides.get("rng_seed")
        )
        return SimulationBundle(*state_tuple)

    def _init_turbulence_mhd(self, resolution: int, **overrides):
        state_tuple = self.randomized_turbulent_initial_state(
            resolution, mhd=True, rng_seed=overrides.get("rng_seed")
        )
        return SimulationBundle(*state_tuple)

    def initialize_scenario(self, scenario, resolution, **overrides):
        """Dynamically initialize any scenario by name."""
        if scenario not in self._scenario_dispatch:
            raise ValueError(f"Unknown scenario: {scenario}")
        return self._scenario_dispatch[scenario](resolution, **overrides)

    def sim_initializator(
        self,
        resolution: Optional[int] = None,
        scenario: Optional[str | int] = None,
        rng_seed: Optional[str | int | float] = None,
        corrector_config: Optional[CorrectorConfig] = None,
        corrector_params: Optional[CorrectorParams] = None,
        config_overrides: Optional[dict] = None,
        **overrides,
    ) -> SimulationBundle:
        """returns initial_state, config, params, helper_data, registered_variables, seed"""

        if isinstance(scenario, str) and scenario not in self.scenario_list:
            raise NameError(f"scenario should be in list {self.scenario_list}")
        if scenario is None:
            scenario = random.choice(self.scenario_list)

        resolution = resolution or self.default_resolution

        # Update overrides
        # overrides.update(
        #     {
        #         k: v
        #         for k, v in {
        #             "rng_seed": rng_seed,
        #         }.items()
        #         if v is not None
        #     }
        # )
        if rng_seed is not None:
            overrides["rng_seed"] = rng_seed

        sim_bundle = self._scenario_dispatch[scenario](
            resolution=resolution, **overrides
        )

        if config_overrides is not None:
            sim_bundle.override_config(**config_overrides)
        if corrector_config is not None and corrector_params is not None:
            sim_bundle.override_solver_in_the_loop(
                corrector_config=corrector_config, corrector_params=corrector_params
            )

        return sim_bundle

    def initialize_integrate(
        self,
        resolution: Optional[int] = None,
        scenario: Optional[str | int] = None,
        rng_seed: Optional[int] = None,
        downscale_factor: Optional[int] = None,
        corrector_config: Optional[CorrectorConfig] = None,
        corrector_params: Optional[CorrectorParams] = None,
        config_overrides: Optional[dict] = None,
    ) -> Tuple[
        np.ndarray,
        SimulationBundle,
    ]:
        """
        integrates a simulation and returns the states with the config, resolution, and seed given"""
        sim_bundle = self.sim_initializator(
            resolution,
            scenario,
            rng_seed,
            corrector_config,
            corrector_params,
            config_overrides,
        )

        states = time_integration(**sim_bundle.unpack_integrate()).states

        if downscale_factor is not None:
            states = downaverage(states, downscale_factor)

        return states, sim_bundle

    def hr_lr_initializator(
        self,
        resolution: Optional[int] = None,
        downscale_factor: Optional[int] = None,
        scenario: Optional[str | int] = None,
        rng_seed: Optional[str | int | float] = None,
        corrector_config: Optional[CorrectorConfig] = None,
        corrector_params: Optional[CorrectorParams] = None,
        config_overrides: Optional[dict] = None,
        config_overrides_lr: Optional[dict] = None,
        **overrides,
    ) -> Tuple[
        SimulationBundle, SimulationBundle
        # Tuple[
        #     np.ndarray,
        #     SimulationConfig,
        #     SimulationParams,
        #     HelperData,
        #     RegisteredVariables,
        # ],
        # Tuple[
        #     np.ndarray,
        #     SimulationConfig,
        #     SimulationParams,
        #     HelperData,
        #     RegisteredVariables,
        # ],
    ]:
        """returns hr-lr pair of (state, config, params, helper_data, registered_vars)"""

        sim_bundle_hr = self.sim_initializator(
            resolution=resolution or self.default_resolution,
            scenario=scenario,
            rng_seed=rng_seed,
            corrector_config=corrector_config,
            corrector_params=corrector_params,
            config_overrides=config_overrides,
        )

        sim_bundle_lr = sim_bundle_hr.convert_to_lr(
            downscale_factor=downscale_factor or self.downscale_factor
        )

        # Apply CNN overrides ONLY to LR state
        # if corrector_config is not None:
        #     config_lr = config_lr._replace(corrector_config=corrector_config)
        # if corrector_params is not None:
        #     params_lr = params_lr._replace(corrector_params=corrector_params)
        if config_overrides_lr is not None:
            sim_bundle_lr.override_config(**config_overrides_lr)
        return (sim_bundle_hr, sim_bundle_lr)

    def train_initializator(
        self,
        resolution: Optional[int] = None,
        downscale_factor: Optional[int] = None,
        scenario: Optional[str | int] = None,
        corrector_config: Optional[CorrectorConfig] = None,
        corrector_params: Optional[CorrectorParams] = None,
        rng_seed: Optional[int] = None,
        verbose: bool = False,
        **overrides,
    ) -> Tuple[
        np.ndarray,
        SimulationBundle,
    ]:
        config_overrides_lr = {
            "return_snapshots": False,
            "use_specific_snapshot_timepoints": False,
            "active_nan_checker": True,
        }
        config_overrides_hr = {
            "return_snapshots": True,
            "use_specific_snapshot_timepoints": True,
            "active_nan_checker": True,
        }
        config_overrides_lr_ml = {
            "active_nan_checker": True,
        }
        is_nan_data = True
        while is_nan_data:
            (sim_bundle_hr, sim_bundle_lr) = self.hr_lr_initializator(
                resolution=resolution or self.default_resolution,
                downscale=downscale_factor or self.downscale_factor,
                scenario=scenario,
                config_overrides=config_overrides_hr,
                config_overrides_lr=config_overrides_lr,
                rng_seed=rng_seed,
            )
            is_nan_data, hr_snapshot_data = time_integration(
                **sim_bundle_hr.unpack_integrate()
            )
            if is_nan_data:
                print("Nan found during time integration wo ML enhancing")
                continue

            is_nan_data, _ = time_integration(**sim_bundle_lr.unpack_integrate())
            if is_nan_data:
                print("Nan found during time integration wo ML enhancing")
                continue

        sim_bundle_lr.override_config(**config_overrides_lr_ml)
        sim_bundle_lr.override_solver_in_the_loop(
            corrector_config=corrector_config, corrector_params=corrector_params
        )
        hr_downscaled_states = downaverage(
            hr_snapshot_data.states, downscale_factor or self.downscale_factor
        )
        print(f"rng seed {sim_bundle_hr.seed}")
        return (hr_downscaled_states, sim_bundle_lr)

    def dataset_validation_initializator(
        self,
        resolution: Optional[int] = None,
        downscale_factor: Optional[int] = None,
        scenario: Optional[str | int] = None,
        **overrides,
    ):
        config_overrides_hr = {
            "active_nan_checker": True,
            "snapshot_settings": SnapshotSettings(
                return_states=True, return_total_energy=True, return_total_mass=True
            ),
        }
        config_overrides_lr = {
            "return_snapshots": False,
            "use_specific_snapshot_timepoints": False,
            "active_nan_checker": True,
        }
        is_nan_data = True
        while is_nan_data:
            (sim_bundle_hr, sim_bundle_lr) = self.hr_lr_initializator(
                resolution=resolution or self.default_resolution,
                downscale=downscale_factor or self.downscale_factor,
                scenario=scenario,
                config_overrides=config_overrides_hr,
                config_overrides_lr=config_overrides_lr,
            )

            is_nan_data, hr_snapshot_data = time_integration(
                **sim_bundle_hr.unpack_integrate()
            )
            if is_nan_data:
                print("Nan found during time integration wo ML enhancing")
                continue

            is_nan_data, _ = time_integration(**sim_bundle_lr.unpack_integrate())
            if is_nan_data:
                print("Nan found during time integration wo ML enhancing")
                continue

        hr_downscaled_states = downaverage(
            hr_snapshot_data.states, downscale_factor or self.downscale_factor
        )
        return (
            hr_downscaled_states,
            sim_bundle_lr.initial_state,
            hr_snapshot_data.total_energy[0],
            hr_snapshot_data.total_mass[0],
            sim_bundle_hr.seed,
        )

    def hr_lr_ml_states_integration(
        self,
        resolution: Optional[int] = None,
        downscale_factor: Optional[int] = None,
        scenario: Optional[str | int] = None,
        corrector_config: Optional[CorrectorConfig] = None,
        corrector_params: Optional[CorrectorParams] = None,
        rng_seed: Optional[int] = None,
        verbose: bool = False,
        num_snapshots: int = 50,
        **overrides,
    ) -> Tuple[
        SimulationBundle, SimulationBundle, SnapshotData, SnapshotData, SnapshotData
    ]:
        snapshot_timepoints = jnp.linspace(0.0, self.cfg_data.t_end, num_snapshots)
        # print(snapshot_timepoints)
        config_overrides_lr = {
            "return_snapshots": True,
            "use_specific_snapshot_timepoints": True,
            "active_nan_checker": True,
            "num_snapshots": num_snapshots,
        }
        config_overrides_hr = {
            "return_snapshots": True,
            "use_specific_snapshot_timepoints": True,
            "active_nan_checker": True,
            "num_snapshots": num_snapshots,
            "progress_bar": True,
        }
        config_overrides_lr_ml = {
            "active_nan_checker": True,
            "num_snapshots": num_snapshots,
        }
        is_nan_data = True
        while is_nan_data:
            (sim_bundle_hr, sim_bundle_lr) = self.hr_lr_initializator(
                resolution=resolution or self.default_resolution,
                downscale=downscale_factor or self.downscale_factor,
                scenario=scenario,
                config_overrides=config_overrides_hr,
                config_overrides_lr=config_overrides_lr,
                rng_seed=rng_seed,
            )
            sim_bundle_hr.params = sim_bundle_hr.params._replace(
                snapshot_timepoints=snapshot_timepoints
            )
            sim_bundle_lr.params = sim_bundle_lr.params._replace(
                snapshot_timepoints=snapshot_timepoints
            )
            print("Integrating HR")
            time_hr = time.time()
            is_nan_data, hr_snapshot_data = time_integration(
                **sim_bundle_hr.unpack_integrate()
            )
            print("Time taken for HR ", time.time() - time_hr)
            if is_nan_data:
                print("Nan found during time integration wo ML enhancing")
                continue
            print("Integrating LR")
            time_lr = time.time()
            is_nan_data, lr_snapshot_data = time_integration(
                **sim_bundle_lr.unpack_integrate()
            )
            print("Time taken for LR ", time.time() - time_lr)
            if is_nan_data:
                print("Nan found during time integration wo ML enhancing")
                continue

            sim_bundle_lr.override_config(**config_overrides_lr_ml)
            sim_bundle_lr.override_solver_in_the_loop(
                corrector_config=corrector_config, corrector_params=corrector_params
            )
            print("Integrating ML w SOL")
            time_lr_sol = time.time()
            is_nan_data, lr_ml_snapshot_data = time_integration(
                **sim_bundle_lr.unpack_integrate()
            )
            print("Time taken for LR SOL", time.time() - time_lr_sol)
            if is_nan_data:
                print("Nan found during time integration w ML enhancing")
                continue
            is_nan_data = False

            print(f"rng seed {sim_bundle_hr.seed}")
        return (
            sim_bundle_hr,
            sim_bundle_lr,
            hr_snapshot_data,
            lr_snapshot_data,
            lr_ml_snapshot_data,
        )

    def randomized_turbulent_initial_state(
        self,
        num_cells: int,
        mhd: bool,
        rng_seed: Optional[int] = None,
    ) -> Tuple[
        np.ndarray,
        SimulationConfig,
        SimulationParams,
        HelperData,
        RegisteredVariables,
        int,
    ]:
        "Creates a turbulent initial state with mhd"
        adiabatic_index = 5 / 3
        box_size = 1.0
        dt_max = 0.1

        # setup simulation config
        config = SimulationConfig(
            runtime_debugging=self.cfg_data.debug,
            first_order_fallback=False,
            progress_bar=False,
            dimensionality=3,
            num_ghost_cells=2,
            box_size=box_size,
            num_cells=num_cells,
            mhd=mhd,
            fixed_timestep=self.cfg_data.fixed_timestep,
            differentiation_mode=self.cfg_data.differentiation_mode,
            riemann_solver=HLL,
            limiter=0,
            return_snapshots=self.cfg_data.return_snapshots,
            num_snapshots=self.cfg_data.num_snapshots,
            use_specific_snapshot_timepoints=self.cfg_data.use_specific_snapshot_timepoints,
            boundary_settings=BoundarySettings(),
            num_checkpoints=self.cfg_data.num_checkpoints,
            # boundary_settings=BoundarySettings(
            #    x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            #    y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            #    z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            # ),
        )

        helper_data = get_helper_data(config)
        registered_variables = get_registered_variables(config)

        # setup the unit system
        code_length = 3 * u.parsec
        code_mass = 1 * u.M_sun
        code_velocity = 100 * u.km / u.s
        code_units = CodeUnits(code_length, code_mass, code_velocity)

        # time domain
        C_CFL = 0.4  # Courant-Friedrichs-Lewy number
        t_final = 1.0 * 1e4 * u.yr
        # t_end = t_final.to(code_units.code_time).value
        t_end = self.cfg_data.t_end
        # turbulence
        wanted_rms = 50 * u.km / u.s
        dt_max = 0.1

        # set the simulation parameters
        params = SimulationParams(
            C_cfl=C_CFL,
            dt_max=dt_max,
            gamma=adiabatic_index,
            t_end=t_end,
            snapshot_timepoints=jnp.array(self.cfg_data.snapshot_timepoints),
        )

        # homogeneous initial state
        rho_0 = 2 * c.m_p / u.cm**3
        p_0 = 3e4 * u.K / u.cm**3 * c.k_B

        density = (
            jnp.ones((num_cells, num_cells, num_cells))
            * rho_0.to(code_units.code_density).value
        )

        # turbulence parameters
        turbulence_slope = -2
        kmin = 2
        kmax = 64

        p = (
            jnp.ones((num_cells, num_cells, num_cells))
            * p_0.to(code_units.code_pressure).value
        )
        if rng_seed is None:
            rng_seed = int(time.time() * 1e6) % (2**32 - 1)

        key = jax.random.key(rng_seed)

        keys = jax.random.split(key, 3)
        u_x = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[0]
        )
        u_y = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[1]
        )
        u_z = create_turb_field(
            config.num_cells, 1, turbulence_slope, kmin, kmax, key=keys[2]
        )

        # scale the turbulence to the desired rms velocity
        rms_vel = jnp.sqrt(jnp.mean(u_x**2 + u_y**2 + u_z**2))

        u_x = u_x / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_y = u_y / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_z = u_z / rms_vel * wanted_rms.to(code_units.code_velocity).value
        # construct primitive state

        if mhd:
            grid_spacing = config.box_size / config.num_cells
            x = jnp.linspace(
                grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
            )
            y = jnp.linspace(
                grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
            )
            z = jnp.linspace(
                grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells
            )

            X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

            B_0 = 1 / np.sqrt(2)
            B_x = jnp.zeros_like(X)
            B_y = jnp.zeros_like(X)
            B_z = B_0 * jnp.ones_like(X)

            initial_state = construct_primitive_state(
                config=config,
                registered_variables=registered_variables,
                density=density,
                velocity_x=u_x,
                velocity_y=u_y,
                velocity_z=u_z,
                gas_pressure=p,
                magnetic_field_x=B_x,
                magnetic_field_y=B_y,
                magnetic_field_z=B_z,
            )
        else:
            initial_state = construct_primitive_state(
                config=config,
                registered_variables=registered_variables,
                density=density,
                velocity_x=u_x,
                velocity_y=u_y,
                velocity_z=u_z,
                gas_pressure=p,
            )
        config = finalize_config(config, initial_state.shape)
        return (
            initial_state,
            config,
            params,
            helper_data,
            registered_variables,
            rng_seed,
        )
