:orphan:

:py:mod:`astronomix._fluid_equations._passive_scalars`
======================================================

.. py:module:: astronomix._fluid_equations._passive_scalars

.. autodoc2-docstring:: astronomix._fluid_equations._passive_scalars
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`advect_passive_scalars <astronomix._fluid_equations._passive_scalars.advect_passive_scalars>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._passive_scalars.advect_passive_scalars
          :summary:
   * - :py:obj:`split_off_passive_scalars <astronomix._fluid_equations._passive_scalars.split_off_passive_scalars>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._passive_scalars.split_off_passive_scalars
          :summary:
   * - :py:obj:`reattach_passive_scalars <astronomix._fluid_equations._passive_scalars.reattach_passive_scalars>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._passive_scalars.reattach_passive_scalars
          :summary:

API
~~~

.. py:function:: advect_passive_scalars(scalars: jaxtyping.Float[jaxtyping.Array, ...], primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, dt: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], grid_spacing: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables)
   :canonical: astronomix._fluid_equations._passive_scalars.advect_passive_scalars

   .. autodoc2-docstring:: astronomix._fluid_equations._passive_scalars.advect_passive_scalars

.. py:function:: split_off_passive_scalars(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, dt: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables)
   :canonical: astronomix._fluid_equations._passive_scalars.split_off_passive_scalars

   .. autodoc2-docstring:: astronomix._fluid_equations._passive_scalars.split_off_passive_scalars

.. py:function:: reattach_passive_scalars(primitive_state: astronomix.option_classes.simulation_config.STATE_TYPE, passive_scalars, config: astronomix.option_classes.simulation_config.SimulationConfig) -> astronomix.option_classes.simulation_config.STATE_TYPE
   :canonical: astronomix._fluid_equations._passive_scalars.reattach_passive_scalars

   .. autodoc2-docstring:: astronomix._fluid_equations._passive_scalars.reattach_passive_scalars
