:orphan:

:py:mod:`astronomix._fluid_equations._equations`
================================================

.. py:module:: astronomix._fluid_equations._equations

.. autodoc2-docstring:: astronomix._fluid_equations._equations
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`primitive_state_from_conserved <astronomix._fluid_equations._equations.primitive_state_from_conserved>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations.primitive_state_from_conserved
          :summary:
   * - :py:obj:`conserved_state_from_primitive <astronomix._fluid_equations._equations.conserved_state_from_primitive>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations.conserved_state_from_primitive
          :summary:
   * - :py:obj:`get_absolute_velocity <astronomix._fluid_equations._equations.get_absolute_velocity>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations.get_absolute_velocity
          :summary:
   * - :py:obj:`pressure_from_internal_energy <astronomix._fluid_equations._equations.pressure_from_internal_energy>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations.pressure_from_internal_energy
          :summary:
   * - :py:obj:`internal_energy_from_energy <astronomix._fluid_equations._equations.internal_energy_from_energy>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations.internal_energy_from_energy
          :summary:
   * - :py:obj:`pressure_from_energy <astronomix._fluid_equations._equations.pressure_from_energy>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations.pressure_from_energy
          :summary:
   * - :py:obj:`total_energy_from_primitives <astronomix._fluid_equations._equations.total_energy_from_primitives>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations.total_energy_from_primitives
          :summary:
   * - :py:obj:`speed_of_sound <astronomix._fluid_equations._equations.speed_of_sound>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equations.speed_of_sound
          :summary:

API
~~~

.. py:function:: primitive_state_from_conserved(conserved_state: astronomix.option_classes.simulation_config.STATE_TYPE, gamma: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables) -> astronomix.option_classes.simulation_config.STATE_TYPE
   :canonical: astronomix._fluid_equations._equations.primitive_state_from_conserved

   .. autodoc2-docstring:: astronomix._fluid_equations._equations.primitive_state_from_conserved

.. py:function:: conserved_state_from_primitive(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, gamma: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables) -> astronomix.option_classes.simulation_config.STATE_TYPE
   :canonical: astronomix._fluid_equations._equations.conserved_state_from_primitive

   .. autodoc2-docstring:: astronomix._fluid_equations._equations.conserved_state_from_primitive

.. py:function:: get_absolute_velocity(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables) -> typing.Union[jaxtyping.Float[jaxtyping.Array, num_cells], jaxtyping.Float[jaxtyping.Array, num_cells_x num_cells_y], jaxtyping.Float[jaxtyping.Array, num_cells_x num_cells_y num_cells_z]]
   :canonical: astronomix._fluid_equations._equations.get_absolute_velocity

   .. autodoc2-docstring:: astronomix._fluid_equations._equations.get_absolute_velocity

.. py:function:: pressure_from_internal_energy(e, rho, gamma)
   :canonical: astronomix._fluid_equations._equations.pressure_from_internal_energy

   .. autodoc2-docstring:: astronomix._fluid_equations._equations.pressure_from_internal_energy

.. py:function:: internal_energy_from_energy(E, rho, u)
   :canonical: astronomix._fluid_equations._equations.internal_energy_from_energy

   .. autodoc2-docstring:: astronomix._fluid_equations._equations.internal_energy_from_energy

.. py:function:: pressure_from_energy(E, rho, u, gamma)
   :canonical: astronomix._fluid_equations._equations.pressure_from_energy

   .. autodoc2-docstring:: astronomix._fluid_equations._equations.pressure_from_energy

.. py:function:: total_energy_from_primitives(rho, u, p, gamma)
   :canonical: astronomix._fluid_equations._equations.total_energy_from_primitives

   .. autodoc2-docstring:: astronomix._fluid_equations._equations.total_energy_from_primitives

.. py:function:: speed_of_sound(rho, p, gamma)
   :canonical: astronomix._fluid_equations._equations.speed_of_sound

   .. autodoc2-docstring:: astronomix._fluid_equations._equations.speed_of_sound
