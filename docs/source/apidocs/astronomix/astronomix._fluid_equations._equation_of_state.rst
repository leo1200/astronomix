:orphan:

:py:mod:`astronomix._fluid_equations._equation_of_state`
========================================================

.. py:module:: astronomix._fluid_equations._equation_of_state

.. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PolytropicLaw <astronomix._fluid_equations._equation_of_state.PolytropicLaw>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.PolytropicLaw
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`is_barotropic <astronomix._fluid_equations._equation_of_state.is_barotropic>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.is_barotropic
          :summary:
   * - :py:obj:`barotropic_pressure <astronomix._fluid_equations._equation_of_state.barotropic_pressure>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.barotropic_pressure
          :summary:
   * - :py:obj:`sound_speed_squared <astronomix._fluid_equations._equation_of_state.sound_speed_squared>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.sound_speed_squared
          :summary:
   * - :py:obj:`pressure_of <astronomix._fluid_equations._equation_of_state.pressure_of>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.pressure_of
          :summary:
   * - :py:obj:`barotropic_law <astronomix._fluid_equations._equation_of_state.barotropic_law>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.barotropic_law
          :summary:
   * - :py:obj:`law_pressure <astronomix._fluid_equations._equation_of_state.law_pressure>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.law_pressure
          :summary:
   * - :py:obj:`law_sound_speed <astronomix._fluid_equations._equation_of_state.law_sound_speed>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.law_sound_speed
          :summary:

API
~~~

.. py:function:: is_barotropic(config: astronomix.option_classes.simulation_config.SimulationConfig) -> bool
   :canonical: astronomix._fluid_equations._equation_of_state.is_barotropic

   .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.is_barotropic

.. py:function:: barotropic_pressure(density, params: astronomix.option_classes.simulation_params.SimulationParams, config: astronomix.option_classes.simulation_config.SimulationConfig)
   :canonical: astronomix._fluid_equations._equation_of_state.barotropic_pressure

   .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.barotropic_pressure

.. py:function:: sound_speed_squared(density, pressure, params: astronomix.option_classes.simulation_params.SimulationParams, config: astronomix.option_classes.simulation_config.SimulationConfig)
   :canonical: astronomix._fluid_equations._equation_of_state.sound_speed_squared

   .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.sound_speed_squared

.. py:function:: pressure_of(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, params: astronomix.option_classes.simulation_params.SimulationParams, config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables)
   :canonical: astronomix._fluid_equations._equation_of_state.pressure_of

   .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.pressure_of

.. py:class:: PolytropicLaw
   :canonical: astronomix._fluid_equations._equation_of_state.PolytropicLaw

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.PolytropicLaw

   .. py:attribute:: polytropic_constant
      :canonical: astronomix._fluid_equations._equation_of_state.PolytropicLaw.polytropic_constant
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.PolytropicLaw.polytropic_constant

   .. py:attribute:: polytropic_exponent
      :canonical: astronomix._fluid_equations._equation_of_state.PolytropicLaw.polytropic_exponent
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.PolytropicLaw.polytropic_exponent

.. py:function:: barotropic_law(params: astronomix.option_classes.simulation_params.SimulationParams, config: astronomix.option_classes.simulation_config.SimulationConfig)
   :canonical: astronomix._fluid_equations._equation_of_state.barotropic_law

   .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.barotropic_law

.. py:function:: law_pressure(law, density, minimum_density)
   :canonical: astronomix._fluid_equations._equation_of_state.law_pressure

   .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.law_pressure

.. py:function:: law_sound_speed(law, density)
   :canonical: astronomix._fluid_equations._equation_of_state.law_sound_speed

   .. autodoc2-docstring:: astronomix._fluid_equations._equation_of_state.law_sound_speed
