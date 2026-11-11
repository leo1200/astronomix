:py:mod:`astronomix.option_classes.simulation_params`
=====================================================

.. py:module:: astronomix.option_classes.simulation_params

.. autodoc2-docstring:: astronomix.option_classes.simulation_params
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`FixedBoundaryState1D <astronomix.option_classes.simulation_params.FixedBoundaryState1D>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_params.FixedBoundaryState1D
          :summary:
   * - :py:obj:`FixedBoundaryState <astronomix.option_classes.simulation_params.FixedBoundaryState>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_params.FixedBoundaryState
          :summary:
   * - :py:obj:`SimulationParams <astronomix.option_classes.simulation_params.SimulationParams>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams
          :summary:

API
~~~

.. py:class:: FixedBoundaryState1D
   :canonical: astronomix.option_classes.simulation_params.FixedBoundaryState1D

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_params.FixedBoundaryState1D

   .. py:attribute:: left_state
      :canonical: astronomix.option_classes.simulation_params.FixedBoundaryState1D.left_state
      :type: jax.numpy.ndarray
      :value: 'array(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.FixedBoundaryState1D.left_state

   .. py:attribute:: right_state
      :canonical: astronomix.option_classes.simulation_params.FixedBoundaryState1D.right_state
      :type: jax.numpy.ndarray
      :value: 'array(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.FixedBoundaryState1D.right_state

.. py:class:: FixedBoundaryState
   :canonical: astronomix.option_classes.simulation_params.FixedBoundaryState

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_params.FixedBoundaryState

   .. py:attribute:: x
      :canonical: astronomix.option_classes.simulation_params.FixedBoundaryState.x
      :type: astronomix.option_classes.simulation_params.FixedBoundaryState1D
      :value: 'FixedBoundaryState1D(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.FixedBoundaryState.x

   .. py:attribute:: y
      :canonical: astronomix.option_classes.simulation_params.FixedBoundaryState.y
      :type: astronomix.option_classes.simulation_params.FixedBoundaryState1D
      :value: 'FixedBoundaryState1D(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.FixedBoundaryState.y

   .. py:attribute:: z
      :canonical: astronomix.option_classes.simulation_params.FixedBoundaryState.z
      :type: astronomix.option_classes.simulation_params.FixedBoundaryState1D
      :value: 'FixedBoundaryState1D(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.FixedBoundaryState.z

.. py:class:: SimulationParams
   :canonical: astronomix.option_classes.simulation_params.SimulationParams

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams

   .. py:attribute:: C_cfl
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.C_cfl
      :type: float
      :value: 0.4

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.C_cfl

   .. py:attribute:: gravitational_constant
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.gravitational_constant
      :type: float
      :value: 1.0

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.gravitational_constant

   .. py:attribute:: gravitational_potential
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.gravitational_potential
      :type: jax.numpy.array
      :value: 'array(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.gravitational_potential

   .. py:attribute:: viscosity
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.viscosity
      :type: float
      :value: 0.0

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.viscosity

   .. py:attribute:: thermal_conductivity
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.thermal_conductivity
      :type: float
      :value: 0.0

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.thermal_conductivity

   .. py:attribute:: isothermal_sound_speed
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.isothermal_sound_speed
      :type: float
      :value: 1.0

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.isothermal_sound_speed

   .. py:attribute:: gamma
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.gamma
      :type: float
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.gamma

   .. py:attribute:: minimum_density
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.minimum_density
      :type: float
      :value: 1e-14

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.minimum_density

   .. py:attribute:: minimum_pressure
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.minimum_pressure
      :type: float
      :value: 1e-14

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.minimum_pressure

   .. py:attribute:: dt_max
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.dt_max
      :type: float
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.dt_max

   .. py:attribute:: t_start
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.t_start
      :type: float
      :value: 0.0

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.t_start

   .. py:attribute:: t_end
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.t_end
      :type: float
      :value: 0.2

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.t_end

   .. py:attribute:: snapshot_timepoints
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.snapshot_timepoints
      :type: jax.numpy.array
      :value: 'array(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.snapshot_timepoints

   .. py:attribute:: fixed_boundary_state
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.fixed_boundary_state
      :type: astronomix.option_classes.simulation_params.FixedBoundaryState
      :value: 'FixedBoundaryState(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.fixed_boundary_state

   .. py:attribute:: turbulent_forcing_params
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.turbulent_forcing_params
      :type: astronomix._modules._turbulent_forcing._turbulent_forcing_options.TurbulentForcingParams
      :value: 'TurbulentForcingParams(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.turbulent_forcing_params

   .. py:attribute:: wind_params
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.wind_params
      :type: astronomix._modules._stellar_wind.stellar_wind_options.WindParams
      :value: 'WindParams(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.wind_params

   .. py:attribute:: cooling_params
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.cooling_params
      :type: astronomix._modules._cooling.cooling_options.CoolingParams
      :value: 'CoolingParams(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.cooling_params

   .. py:attribute:: neural_net_force_params
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.neural_net_force_params
      :type: astronomix._modules._neural_net_force._neural_net_force_options.NeuralNetForceParams
      :value: 'NeuralNetForceParams(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.neural_net_force_params

   .. py:attribute:: cnn_mhd_corrector_params
      :canonical: astronomix.option_classes.simulation_params.SimulationParams.cnn_mhd_corrector_params
      :type: astronomix._modules._cnn_mhd_corrector._cnn_mhd_corrector_options.CNNMHDconfig
      :value: 'CNNMHDconfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_params.SimulationParams.cnn_mhd_corrector_params
