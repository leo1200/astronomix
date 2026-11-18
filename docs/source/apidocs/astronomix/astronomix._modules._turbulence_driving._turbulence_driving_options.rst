:orphan:

:py:mod:`astronomix._modules._turbulence_driving._turbulence_driving_options`
=============================================================================

.. py:module:: astronomix._modules._turbulence_driving._turbulence_driving_options

.. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`TurbulenceDrivingConfig <astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig
          :summary:
   * - :py:obj:`TurbulenceDrivingParams <astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams
          :summary:
   * - :py:obj:`TurbulenceDrivingState <astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`BAND <astronomix._modules._turbulence_driving._turbulence_driving_options.BAND>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.BAND
          :summary:
   * - :py:obj:`PARABOLA <astronomix._modules._turbulence_driving._turbulence_driving_options.PARABOLA>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.PARABOLA
          :summary:
   * - :py:obj:`POWER_LAW <astronomix._modules._turbulence_driving._turbulence_driving_options.POWER_LAW>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.POWER_LAW
          :summary:

API
~~~

.. py:data:: BAND
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.BAND
   :value: 0

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.BAND

.. py:data:: PARABOLA
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.PARABOLA
   :value: 1

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.PARABOLA

.. py:data:: POWER_LAW
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.POWER_LAW
   :value: 2

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.POWER_LAW

.. py:class:: TurbulenceDrivingConfig
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig

   .. py:attribute:: turbulence_driving
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.turbulence_driving
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.turbulence_driving

   .. py:attribute:: dimensionality
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.dimensionality
      :type: int
      :value: 3

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.dimensionality

   .. py:attribute:: box_size
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.box_size
      :type: typing.Tuple[float, ...]
      :value: (1.0, 1.0, 1.0)

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.box_size

   .. py:attribute:: spectral_form
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.spectral_form
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.spectral_form

   .. py:attribute:: k_min
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.k_min
      :type: float
      :value: 1.0

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.k_min

   .. py:attribute:: k_max
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.k_max
      :type: float
      :value: 3.0

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.k_max

   .. py:attribute:: power_law_exponent
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.power_law_exponent
      :type: float
      :value: None

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.power_law_exponent

   .. py:attribute:: angles_exponent
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.angles_exponent
      :type: float
      :value: 1.0

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.angles_exponent

   .. py:attribute:: num_steps_per_turnover
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.num_steps_per_turnover
      :type: int
      :value: 10

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.num_steps_per_turnover

   .. py:attribute:: random_seed
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.random_seed
      :type: int
      :value: 42

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.random_seed

   .. py:attribute:: amplitude_auto_adjust
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.amplitude_auto_adjust
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig.amplitude_auto_adjust

.. py:class:: TurbulenceDrivingParams
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams

   .. py:attribute:: velocity
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams.velocity
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: 1.0

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams.velocity

   .. py:attribute:: k_driving
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams.k_driving
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: 2.0

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams.k_driving

   .. py:attribute:: solenoidal_weight
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams.solenoidal_weight
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: 0.5

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams.solenoidal_weight

   .. py:attribute:: amplitude_factor
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams.amplitude_factor
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: 1.0

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams.amplitude_factor

.. py:class:: TurbulenceDrivingState
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState

   .. py:attribute:: ou_phases
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState.ou_phases
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState.ou_phases

   .. py:attribute:: step
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState.step
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState.step

   .. py:attribute:: key
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState.key
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState.key

   .. py:attribute:: amplitude_adjustment
      :canonical: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState.amplitude_adjustment
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState.amplitude_adjustment
