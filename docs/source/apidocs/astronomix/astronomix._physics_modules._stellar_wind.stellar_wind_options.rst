:py:mod:`astronomix._physics_modules._stellar_wind.stellar_wind_options`
========================================================================

.. py:module:: astronomix._physics_modules._stellar_wind.stellar_wind_options

.. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`WindConfig <astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig>`
     - .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig
          :summary:
   * - :py:obj:`WindParams <astronomix._physics_modules._stellar_wind.stellar_wind_options.WindParams>`
     - .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindParams
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`MEO <astronomix._physics_modules._stellar_wind.stellar_wind_options.MEO>`
     - .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.MEO
          :summary:
   * - :py:obj:`EI <astronomix._physics_modules._stellar_wind.stellar_wind_options.EI>`
     - .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.EI
          :summary:
   * - :py:obj:`MEI <astronomix._physics_modules._stellar_wind.stellar_wind_options.MEI>`
     - .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.MEI
          :summary:

API
~~~

.. py:data:: MEO
   :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.MEO
   :value: 0

   .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.MEO

.. py:data:: EI
   :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.EI
   :value: 1

   .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.EI

.. py:data:: MEI
   :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.MEI
   :value: 2

   .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.MEI

.. py:class:: WindConfig
   :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig

   .. py:attribute:: stellar_wind
      :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig.stellar_wind
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig.stellar_wind

   .. py:attribute:: num_injection_cells
      :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig.num_injection_cells
      :type: int
      :value: 10

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig.num_injection_cells

   .. py:attribute:: wind_injection_scheme
      :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig.wind_injection_scheme
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig.wind_injection_scheme

   .. py:attribute:: trace_wind_density
      :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig.trace_wind_density
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindConfig.trace_wind_density

.. py:class:: WindParams
   :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindParams

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindParams

   .. py:attribute:: wind_mass_loss_rate
      :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindParams.wind_mass_loss_rate
      :type: float
      :value: 0.0

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindParams.wind_mass_loss_rate

   .. py:attribute:: wind_final_velocity
      :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindParams.wind_final_velocity
      :type: float
      :value: 0.0

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindParams.wind_final_velocity

   .. py:attribute:: pressure_floor
      :canonical: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindParams.pressure_floor
      :type: float
      :value: 100000.0

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.stellar_wind_options.WindParams.pressure_floor
