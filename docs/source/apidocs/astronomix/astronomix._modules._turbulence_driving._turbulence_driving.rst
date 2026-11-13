:orphan:

:py:mod:`astronomix._modules._turbulence_driving._turbulence_driving`
=====================================================================

.. py:module:: astronomix._modules._turbulence_driving._turbulence_driving

.. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`turnover_time <astronomix._modules._turbulence_driving._turbulence_driving.turnover_time>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.turnover_time
          :summary:
   * - :py:obj:`update_interval <astronomix._modules._turbulence_driving._turbulence_driving.update_interval>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.update_interval
          :summary:
   * - :py:obj:`ou_standard_deviation <astronomix._modules._turbulence_driving._turbulence_driving.ou_standard_deviation>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.ou_standard_deviation
          :summary:
   * - :py:obj:`solenoidal_weight_normalisation <astronomix._modules._turbulence_driving._turbulence_driving.solenoidal_weight_normalisation>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.solenoidal_weight_normalisation
          :summary:
   * - :py:obj:`init_turbulence_driving <astronomix._modules._turbulence_driving._turbulence_driving.init_turbulence_driving>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.init_turbulence_driving
          :summary:
   * - :py:obj:`ornstein_uhlenbeck_update <astronomix._modules._turbulence_driving._turbulence_driving.ornstein_uhlenbeck_update>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.ornstein_uhlenbeck_update
          :summary:
   * - :py:obj:`adjust_amplitude <astronomix._modules._turbulence_driving._turbulence_driving.adjust_amplitude>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.adjust_amplitude
          :summary:
   * - :py:obj:`advance_turbulence_driving <astronomix._modules._turbulence_driving._turbulence_driving.advance_turbulence_driving>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.advance_turbulence_driving
          :summary:
   * - :py:obj:`project_modes <astronomix._modules._turbulence_driving._turbulence_driving.project_modes>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.project_modes
          :summary:
   * - :py:obj:`mode_coefficients <astronomix._modules._turbulence_driving._turbulence_driving.mode_coefficients>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.mode_coefficients
          :summary:
   * - :py:obj:`cell_center_coordinates <astronomix._modules._turbulence_driving._turbulence_driving.cell_center_coordinates>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.cell_center_coordinates
          :summary:
   * - :py:obj:`driving_acceleration <astronomix._modules._turbulence_driving._turbulence_driving.driving_acceleration>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.driving_acceleration
          :summary:
   * - :py:obj:`driven_velocity_indices <astronomix._modules._turbulence_driving._turbulence_driving.driven_velocity_indices>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.driven_velocity_indices
          :summary:
   * - :py:obj:`add_velocity_kick <astronomix._modules._turbulence_driving._turbulence_driving.add_velocity_kick>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.add_velocity_kick
          :summary:
   * - :py:obj:`physical_cells <astronomix._modules._turbulence_driving._turbulence_driving.physical_cells>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.physical_cells
          :summary:
   * - :py:obj:`physical_cell_centers <astronomix._modules._turbulence_driving._turbulence_driving.physical_cell_centers>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.physical_cell_centers
          :summary:
   * - :py:obj:`velocity_dispersion <astronomix._modules._turbulence_driving._turbulence_driving.velocity_dispersion>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.velocity_dispersion
          :summary:
   * - :py:obj:`apply_turbulence_driving <astronomix._modules._turbulence_driving._turbulence_driving.apply_turbulence_driving>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.apply_turbulence_driving
          :summary:

API
~~~

.. py:function:: turnover_time(config: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig, params: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.turnover_time

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.turnover_time

.. py:function:: update_interval(config: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig, params: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.update_interval

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.update_interval

.. py:function:: ou_standard_deviation(config: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig, params: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.ou_standard_deviation

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.ou_standard_deviation

.. py:function:: solenoidal_weight_normalisation(solenoidal_weight, num_components: int)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.solenoidal_weight_normalisation

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.solenoidal_weight_normalisation

.. py:function:: init_turbulence_driving(config: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig, dtype=None) -> astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.init_turbulence_driving

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.init_turbulence_driving

.. py:function:: ornstein_uhlenbeck_update(ou_phases, deviates, damping_factor)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.ornstein_uhlenbeck_update

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.ornstein_uhlenbeck_update

.. py:function:: adjust_amplitude(amplitude_adjustment, velocity_dispersion, config, params)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.adjust_amplitude

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.adjust_amplitude

.. py:function:: advance_turbulence_driving(driving_state: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState, time, config: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig, params: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams, velocity_dispersion=None) -> astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.advance_turbulence_driving

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.advance_turbulence_driving

.. py:function:: project_modes(ou_phases, wave_vectors, solenoidal_weight)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.project_modes

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.project_modes

.. py:function:: mode_coefficients(driving_state: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState, config, params)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.mode_coefficients

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.mode_coefficients

.. py:function:: cell_center_coordinates(num_cells: int, cell_width, index_offset: int = 0, dtype=None)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.cell_center_coordinates

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.cell_center_coordinates

.. py:function:: driving_acceleration(driving_state: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState, config: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig, params: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingParams, cell_centers: typing.Sequence[jax.numpy.ndarray]) -> jax.numpy.ndarray
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.driving_acceleration

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.driving_acceleration

.. py:function:: driven_velocity_indices(velocity_index: typing.Union[int, typing.Sequence[int]], num_components: int)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.driven_velocity_indices

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.driven_velocity_indices

.. py:function:: add_velocity_kick(primitive_state, kick, velocity_index: typing.Union[int, typing.Sequence[int]], cells=())
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.add_velocity_kick

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.add_velocity_kick

.. py:function:: physical_cells(primitive_state, config)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.physical_cells

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.physical_cells

.. py:function:: physical_cell_centers(config, dtype)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.physical_cell_centers

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.physical_cell_centers

.. py:function:: velocity_dispersion(primitive_state, config, registered_variables)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.velocity_dispersion

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.velocity_dispersion

.. py:function:: apply_turbulence_driving(primitive_state, driving_state: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingState, dt, config, params, registered_variables)
   :canonical: astronomix._modules._turbulence_driving._turbulence_driving.apply_turbulence_driving

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._turbulence_driving.apply_turbulence_driving
