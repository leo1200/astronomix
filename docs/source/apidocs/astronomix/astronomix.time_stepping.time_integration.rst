:py:mod:`astronomix.time_stepping.time_integration`
===================================================

.. py:module:: astronomix.time_stepping.time_integration

.. autodoc2-docstring:: astronomix.time_stepping.time_integration
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LoopState <astronomix.time_stepping.time_integration.LoopState>`
     - .. autodoc2-docstring:: astronomix.time_stepping.time_integration.LoopState
          :summary:

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

.. py:class:: LoopState
   :canonical: astronomix.time_stepping.time_integration.LoopState

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.time_stepping.time_integration.LoopState

   .. py:attribute:: primitive_state
      :canonical: astronomix.time_stepping.time_integration.LoopState.primitive_state
      :type: typing.Any
      :value: None

      .. autodoc2-docstring:: astronomix.time_stepping.time_integration.LoopState.primitive_state

   .. py:attribute:: driving_state
      :canonical: astronomix.time_stepping.time_integration.LoopState.driving_state
      :type: typing.Any
      :value: None

      .. autodoc2-docstring:: astronomix.time_stepping.time_integration.LoopState.driving_state

.. py:function:: time_integration(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, config: astronomix.option_classes.simulation_config.SimulationConfig, params: astronomix.option_classes.simulation_params.SimulationParams, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables, snapshot_callable=None, sharding: typing.Union[types.NoneType, jax.NamedSharding] = None, restart_state: typing.Union[types.NoneType, astronomix.time_stepping.time_integration.LoopState] = None) -> typing.Union[astronomix.option_classes.simulation_config.STATE_TYPE, astronomix.data_classes.simulation_snapshot_data.SnapshotData]
   :canonical: astronomix.time_stepping.time_integration.time_integration

   .. autodoc2-docstring:: astronomix.time_stepping.time_integration.time_integration
