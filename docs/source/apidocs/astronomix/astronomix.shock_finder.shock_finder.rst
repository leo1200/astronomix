:py:mod:`astronomix.shock_finder.shock_finder`
==============================================

.. py:module:: astronomix.shock_finder.shock_finder

.. autodoc2-docstring:: astronomix.shock_finder.shock_finder
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`shock_sensor <astronomix.shock_finder.shock_finder.shock_sensor>`
     - .. autodoc2-docstring:: astronomix.shock_finder.shock_finder.shock_sensor
          :summary:
   * - :py:obj:`shock_criteria <astronomix.shock_finder.shock_finder.shock_criteria>`
     - .. autodoc2-docstring:: astronomix.shock_finder.shock_finder.shock_criteria
          :summary:
   * - :py:obj:`find_shock_zone <astronomix.shock_finder.shock_finder.find_shock_zone>`
     - .. autodoc2-docstring:: astronomix.shock_finder.shock_finder.find_shock_zone
          :summary:

API
~~~

.. py:function:: shock_sensor(pressure: astronomix.option_classes.simulation_config.FIELD_TYPE) -> astronomix.option_classes.simulation_config.FIELD_TYPE
   :canonical: astronomix.shock_finder.shock_finder.shock_sensor

   .. autodoc2-docstring:: astronomix.shock_finder.shock_finder.shock_sensor

.. py:function:: shock_criteria(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables, helper_data: astronomix.data_classes.simulation_helper_data.HelperData) -> jax.numpy.ndarray
   :canonical: astronomix.shock_finder.shock_finder.shock_criteria

   .. autodoc2-docstring:: astronomix.shock_finder.shock_finder.shock_criteria

.. py:function:: find_shock_zone(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables, helper_data: astronomix.data_classes.simulation_helper_data.HelperData) -> typing.Tuple[typing.Union[int, jaxtyping.Int[jaxtyping.Array, ]], typing.Union[int, jaxtyping.Int[jaxtyping.Array, ]], typing.Union[int, jaxtyping.Int[jaxtyping.Array, ]]]
   :canonical: astronomix.shock_finder.shock_finder.find_shock_zone

   .. autodoc2-docstring:: astronomix.shock_finder.shock_finder.find_shock_zone
