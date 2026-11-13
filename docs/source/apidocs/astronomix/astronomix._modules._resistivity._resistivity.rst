:orphan:

:py:mod:`astronomix._modules._resistivity._resistivity`
=======================================================

.. py:module:: astronomix._modules._resistivity._resistivity

.. autodoc2-docstring:: astronomix._modules._resistivity._resistivity
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`resistive_face_increment <astronomix._modules._resistivity._resistivity.resistive_face_increment>`
     - .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.resistive_face_increment
          :summary:
   * - :py:obj:`resistive_source_active <astronomix._modules._resistivity._resistivity.resistive_source_active>`
     - .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.resistive_source_active
          :summary:
   * - :py:obj:`resistive_source <astronomix._modules._resistivity._resistivity.resistive_source>`
     - .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.resistive_source
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`XAXIS <astronomix._modules._resistivity._resistivity.XAXIS>`
     - .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.XAXIS
          :summary:
   * - :py:obj:`YAXIS <astronomix._modules._resistivity._resistivity.YAXIS>`
     - .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.YAXIS
          :summary:
   * - :py:obj:`ZAXIS <astronomix._modules._resistivity._resistivity.ZAXIS>`
     - .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.ZAXIS
          :summary:

API
~~~

.. py:data:: XAXIS
   :canonical: astronomix._modules._resistivity._resistivity.XAXIS
   :value: 0

   .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.XAXIS

.. py:data:: YAXIS
   :canonical: astronomix._modules._resistivity._resistivity.YAXIS
   :value: 1

   .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.YAXIS

.. py:data:: ZAXIS
   :canonical: astronomix._modules._resistivity._resistivity.ZAXIS
   :value: 2

   .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.ZAXIS

.. py:function:: resistive_face_increment(bx_interface: jax.numpy.ndarray, by_interface: jax.numpy.ndarray, bz_interface: jax.numpy.ndarray, eta, dt_tilde, grid_spacing: float, dimensionality: int)
   :canonical: astronomix._modules._resistivity._resistivity.resistive_face_increment

   .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.resistive_face_increment

.. py:function:: resistive_source_active(config: astronomix.option_classes.simulation_config.SimulationConfig) -> bool
   :canonical: astronomix._modules._resistivity._resistivity.resistive_source_active

   .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.resistive_source_active

.. py:function:: resistive_source(primitive_state, params, config, registered_variables)
   :canonical: astronomix._modules._resistivity._resistivity.resistive_source

   .. autodoc2-docstring:: astronomix._modules._resistivity._resistivity.resistive_source
