:py:mod:`astronomix.time_stepping.time_integration`
===================================================

.. py:module:: astronomix.time_stepping.time_integration

.. autodoc2-docstring:: astronomix.time_stepping.time_integration
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`time_integration <astronomix.time_stepping.time_integration.time_integration>`
     - .. autodoc2-docstring:: astronomix.time_stepping.time_integration.time_integration
          :summary:

API
~~~

.. py:function:: time_integration(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, config: astronomix.option_classes.simulation_config.SimulationConfig, params: astronomix.option_classes.simulation_params.SimulationParams, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables, snapshot_callable=None, sharding: typing.Union[types.NoneType, jax.NamedSharding] = None) -> typing.Union[astronomix.option_classes.simulation_config.STATE_TYPE, astronomix.data_classes.simulation_snapshot_data.SnapshotData]
   :canonical: astronomix.time_stepping.time_integration.time_integration

   .. autodoc2-docstring:: astronomix.time_stepping.time_integration.time_integration
