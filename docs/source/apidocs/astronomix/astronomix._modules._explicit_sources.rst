:orphan:

:py:mod:`astronomix._modules._explicit_sources`
===============================================

.. py:module:: astronomix._modules._explicit_sources

.. autodoc2-docstring:: astronomix._modules._explicit_sources
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`wind_source_active <astronomix._modules._explicit_sources.wind_source_active>`
     - .. autodoc2-docstring:: astronomix._modules._explicit_sources.wind_source_active
          :summary:
   * - :py:obj:`gravity_source_active <astronomix._modules._explicit_sources.gravity_source_active>`
     - .. autodoc2-docstring:: astronomix._modules._explicit_sources.gravity_source_active
          :summary:
   * - :py:obj:`has_explicit_sources <astronomix._modules._explicit_sources.has_explicit_sources>`
     - .. autodoc2-docstring:: astronomix._modules._explicit_sources.has_explicit_sources
          :summary:
   * - :py:obj:`explicit_sources <astronomix._modules._explicit_sources.explicit_sources>`
     - .. autodoc2-docstring:: astronomix._modules._explicit_sources.explicit_sources
          :summary:

API
~~~

.. py:function:: wind_source_active(config: astronomix.option_classes.simulation_config.SimulationConfig) -> bool
   :canonical: astronomix._modules._explicit_sources.wind_source_active

   .. autodoc2-docstring:: astronomix._modules._explicit_sources.wind_source_active

.. py:function:: gravity_source_active(config: astronomix.option_classes.simulation_config.SimulationConfig) -> bool
   :canonical: astronomix._modules._explicit_sources.gravity_source_active

   .. autodoc2-docstring:: astronomix._modules._explicit_sources.gravity_source_active

.. py:function:: has_explicit_sources(config: astronomix.option_classes.simulation_config.SimulationConfig) -> bool
   :canonical: astronomix._modules._explicit_sources.has_explicit_sources

   .. autodoc2-docstring:: astronomix._modules._explicit_sources.has_explicit_sources

.. py:function:: explicit_sources(conserved_state: astronomix.option_classes.simulation_config.STATE_TYPE, dt: jaxtyping.Float[jaxtyping.Array, ], gamma: typing.Union[float, jaxtyping.Float[jaxtyping.Array, ]], config: astronomix.option_classes.simulation_config.SimulationConfig, params: astronomix.option_classes.simulation_params.SimulationParams, helper_data: astronomix.data_classes.simulation_helper_data.HelperData, registered_variables: astronomix.variable_registry.registered_variables.RegisteredVariables, density_fluxes=None, density_increment=None, primitive_state=None) -> astronomix.option_classes.simulation_config.STATE_TYPE
   :canonical: astronomix._modules._explicit_sources.explicit_sources

   .. autodoc2-docstring:: astronomix._modules._explicit_sources.explicit_sources
