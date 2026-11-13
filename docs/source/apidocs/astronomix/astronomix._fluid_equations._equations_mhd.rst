:orphan:

:py:mod:`astronomix._fluid_equations._equations_mhd`
====================================================

.. py:module:: astronomix._fluid_equations._equations_mhd

.. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`thermal_pressure_from_energy_mhd <astronomix._fluid_equations._equations_mhd.thermal_pressure_from_energy_mhd>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.thermal_pressure_from_energy_mhd
          :summary:
   * - :py:obj:`dual_switched_pressure_mhd <astronomix._fluid_equations._equations_mhd.dual_switched_pressure_mhd>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.dual_switched_pressure_mhd
          :summary:
   * - :py:obj:`total_energy_from_primitives_mhd <astronomix._fluid_equations._equations_mhd.total_energy_from_primitives_mhd>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.total_energy_from_primitives_mhd
          :summary:
   * - :py:obj:`conserved_state_from_primitive_mhd <astronomix._fluid_equations._equations_mhd.conserved_state_from_primitive_mhd>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.conserved_state_from_primitive_mhd
          :summary:
   * - :py:obj:`primitive_state_from_conserved_mhd <astronomix._fluid_equations._equations_mhd.primitive_state_from_conserved_mhd>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.primitive_state_from_conserved_mhd
          :summary:
   * - :py:obj:`primitive_state_from_conserved_isothermal <astronomix._fluid_equations._equations_mhd.primitive_state_from_conserved_isothermal>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.primitive_state_from_conserved_isothermal
          :summary:
   * - :py:obj:`conserved_state_from_primitive_isothermal <astronomix._fluid_equations._equations_mhd.conserved_state_from_primitive_isothermal>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.conserved_state_from_primitive_isothermal
          :summary:

API
~~~

.. py:function:: thermal_pressure_from_energy_mhd(E, rho, u_squared, b_squared, gamma)
   :canonical: astronomix._fluid_equations._equations_mhd.thermal_pressure_from_energy_mhd

   .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.thermal_pressure_from_energy_mhd

.. py:function:: dual_switched_pressure_mhd(E, rho, u_squared, b_squared, gamma, internal_energy_density, eta)
   :canonical: astronomix._fluid_equations._equations_mhd.dual_switched_pressure_mhd

   .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.dual_switched_pressure_mhd

.. py:function:: total_energy_from_primitives_mhd(rho, u_squared, p, b_squared, gamma)
   :canonical: astronomix._fluid_equations._equations_mhd.total_energy_from_primitives_mhd

   .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.total_energy_from_primitives_mhd

.. py:function:: conserved_state_from_primitive_mhd(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, gamma: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables) -> astronomix.option_classes.simulation_config.STATE_TYPE
   :canonical: astronomix._fluid_equations._equations_mhd.conserved_state_from_primitive_mhd

   .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.conserved_state_from_primitive_mhd

.. py:function:: primitive_state_from_conserved_mhd(conserved_state: astronomix.option_classes.simulation_config.STATE_TYPE, rhomin: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], pgmin: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], gamma: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables, internal_energy_density=None, clamp: bool = True) -> astronomix.option_classes.simulation_config.STATE_TYPE
   :canonical: astronomix._fluid_equations._equations_mhd.primitive_state_from_conserved_mhd

   .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.primitive_state_from_conserved_mhd

.. py:function:: primitive_state_from_conserved_isothermal(conserved_state: astronomix.option_classes.simulation_config.STATE_TYPE, minimum_density: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables, clamp: bool = True) -> astronomix.option_classes.simulation_config.STATE_TYPE
   :canonical: astronomix._fluid_equations._equations_mhd.primitive_state_from_conserved_isothermal

   .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.primitive_state_from_conserved_isothermal

.. py:function:: conserved_state_from_primitive_isothermal(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables) -> astronomix.option_classes.simulation_config.STATE_TYPE
   :canonical: astronomix._fluid_equations._equations_mhd.conserved_state_from_primitive_isothermal

   .. autodoc2-docstring:: astronomix._fluid_equations._equations_mhd.conserved_state_from_primitive_isothermal
