:py:mod:`astronomix.initial_condition_generation.construct_primitive_state`
===========================================================================

.. py:module:: astronomix.initial_condition_generation.construct_primitive_state

.. autodoc2-docstring:: astronomix.initial_condition_generation.construct_primitive_state
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`construct_primitive_state <astronomix.initial_condition_generation.construct_primitive_state.construct_primitive_state>`
     - .. autodoc2-docstring:: astronomix.initial_condition_generation.construct_primitive_state.construct_primitive_state
          :summary:

API
~~~

.. py:function:: construct_primitive_state(config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables, density: astronomix.option_classes.simulation_config.FIELD_TYPE, velocity_x: typing.Union[astronomix.option_classes.simulation_config.FIELD_TYPE, types.NoneType] = None, velocity_y: typing.Union[astronomix.option_classes.simulation_config.FIELD_TYPE, types.NoneType] = None, velocity_z: typing.Union[astronomix.option_classes.simulation_config.FIELD_TYPE, types.NoneType] = None, magnetic_field_x: typing.Union[astronomix.option_classes.simulation_config.FIELD_TYPE, types.NoneType] = None, magnetic_field_y: typing.Union[astronomix.option_classes.simulation_config.FIELD_TYPE, types.NoneType] = None, magnetic_field_z: typing.Union[astronomix.option_classes.simulation_config.FIELD_TYPE, types.NoneType] = None, interface_magnetic_field_x: typing.Union[astronomix.option_classes.simulation_config.FIELD_TYPE, types.NoneType] = None, interface_magnetic_field_y: typing.Union[astronomix.option_classes.simulation_config.FIELD_TYPE, types.NoneType] = None, interface_magnetic_field_z: typing.Union[astronomix.option_classes.simulation_config.FIELD_TYPE, types.NoneType] = None, gas_pressure: typing.Union[astronomix.option_classes.simulation_config.FIELD_TYPE, types.NoneType] = None, cosmic_ray_pressure: typing.Union[astronomix.option_classes.simulation_config.FIELD_TYPE, types.NoneType] = None, sharding=None) -> astronomix.option_classes.simulation_config.STATE_TYPE
   :canonical: astronomix.initial_condition_generation.construct_primitive_state.construct_primitive_state

   .. autodoc2-docstring:: astronomix.initial_condition_generation.construct_primitive_state.construct_primitive_state
