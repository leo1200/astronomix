:orphan:

:py:mod:`astronomix._fluid_equations._dual_energy`
==================================================

.. py:module:: astronomix._fluid_equations._dual_energy

.. autodoc2-docstring:: astronomix._fluid_equations._dual_energy
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`advect_internal_energy <astronomix._fluid_equations._dual_energy.advect_internal_energy>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._dual_energy.advect_internal_energy
          :summary:

API
~~~

.. py:function:: advect_internal_energy(internal_energy_density: jaxtyping.Float[jaxtyping.Array, ...], conserved_state: astronomix.option_classes.simulation_config.STATE_TYPE, pressure: jaxtyping.Float[jaxtyping.Array, ...], dt: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], grid_spacing: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: astronomix.option_classes.simulation_config.SimulationConfig, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables)
   :canonical: astronomix._fluid_equations._dual_energy.advect_internal_energy

   .. autodoc2-docstring:: astronomix._fluid_equations._dual_energy.advect_internal_energy
