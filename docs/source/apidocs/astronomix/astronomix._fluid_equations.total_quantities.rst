:py:mod:`astronomix._fluid_equations.total_quantities`
======================================================

.. py:module:: astronomix._fluid_equations.total_quantities

.. autodoc2-docstring:: astronomix._fluid_equations.total_quantities
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`calculate_internal_energy <astronomix._fluid_equations.total_quantities.calculate_internal_energy>`
     - .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_internal_energy
          :summary:
   * - :py:obj:`calculate_radial_momentum <astronomix._fluid_equations.total_quantities.calculate_radial_momentum>`
     - .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_radial_momentum
          :summary:
   * - :py:obj:`calculate_kinetic_energy <astronomix._fluid_equations.total_quantities.calculate_kinetic_energy>`
     - .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_kinetic_energy
          :summary:
   * - :py:obj:`calculate_gravitational_energy <astronomix._fluid_equations.total_quantities.calculate_gravitational_energy>`
     - .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_gravitational_energy
          :summary:
   * - :py:obj:`calculate_total_energy <astronomix._fluid_equations.total_quantities.calculate_total_energy>`
     - .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_total_energy
          :summary:
   * - :py:obj:`calculate_total_mass <astronomix._fluid_equations.total_quantities.calculate_total_mass>`
     - .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_total_mass
          :summary:

API
~~~

.. py:function:: calculate_internal_energy(state, helper_data, gamma, config, registered_variables)
   :canonical: astronomix._fluid_equations.total_quantities.calculate_internal_energy

   .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_internal_energy

.. py:function:: calculate_radial_momentum(state, helper_data, config, registered_variables)
   :canonical: astronomix._fluid_equations.total_quantities.calculate_radial_momentum

   .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_radial_momentum

.. py:function:: calculate_kinetic_energy(state, helper_data, config, registered_variables)
   :canonical: astronomix._fluid_equations.total_quantities.calculate_kinetic_energy

   .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_kinetic_energy

.. py:function:: calculate_gravitational_energy(state, helper_data, gravitational_constant, config, registered_variables)
   :canonical: astronomix._fluid_equations.total_quantities.calculate_gravitational_energy

   .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_gravitational_energy

.. py:function:: calculate_total_energy(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, helper_data: astronomix.data_classes.simulation_helper_data.HelperData, gamma: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], gravitational_constant: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables) -> jaxtyping.Float[jaxtyping.Array, ]
   :canonical: astronomix._fluid_equations.total_quantities.calculate_total_energy

   .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_total_energy

.. py:function:: calculate_total_mass(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, helper_data: astronomix.data_classes.simulation_helper_data.HelperData, config: astronomix.option_classes.simulation_config.SimulationConfig) -> jaxtyping.Float[jaxtyping.Array, ]
   :canonical: astronomix._fluid_equations.total_quantities.calculate_total_mass

   .. autodoc2-docstring:: astronomix._fluid_equations.total_quantities.calculate_total_mass
