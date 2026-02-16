:py:mod:`astronomix.data_classes.simulation_helper_data`
========================================================

.. py:module:: astronomix.data_classes.simulation_helper_data

.. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HelperData <astronomix.data_classes.simulation_helper_data.HelperData>`
     - .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_helper_data <astronomix.data_classes.simulation_helper_data.get_helper_data>`
     - .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.get_helper_data
          :summary:

API
~~~

.. py:class:: HelperData
   :canonical: astronomix.data_classes.simulation_helper_data.HelperData

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData

   .. py:attribute:: geometric_centers
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.geometric_centers
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.geometric_centers

   .. py:attribute:: volumetric_centers
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.volumetric_centers
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.volumetric_centers

   .. py:attribute:: r
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.r
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.r

   .. py:attribute:: r_hat_alpha
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.r_hat_alpha
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.r_hat_alpha

   .. py:attribute:: cell_volumes
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.cell_volumes
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.cell_volumes

   .. py:attribute:: inner_cell_boundaries
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.inner_cell_boundaries
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.inner_cell_boundaries

   .. py:attribute:: outer_cell_boundaries
      :canonical: astronomix.data_classes.simulation_helper_data.HelperData.outer_cell_boundaries
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.HelperData.outer_cell_boundaries

.. py:function:: get_helper_data(config: astronomix.option_classes.simulation_config.SimulationConfig, sharding: typing.Union[types.NoneType, jax.NamedSharding] = None, padded: bool = False, production: bool = False) -> astronomix.data_classes.simulation_helper_data.HelperData
   :canonical: astronomix.data_classes.simulation_helper_data.get_helper_data

   .. autodoc2-docstring:: astronomix.data_classes.simulation_helper_data.get_helper_data
