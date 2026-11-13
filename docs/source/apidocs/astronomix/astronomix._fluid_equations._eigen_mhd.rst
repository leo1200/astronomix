:orphan:

:py:mod:`astronomix._fluid_equations._eigen_mhd`
================================================

.. py:module:: astronomix._fluid_equations._eigen_mhd

.. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`EigenvectorBlocks <astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`diff_safe_sqrt <astronomix._fluid_equations._eigen_mhd.diff_safe_sqrt>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.diff_safe_sqrt
          :summary:

API
~~~

.. py:function:: diff_safe_sqrt(x)
   :canonical: astronomix._fluid_equations._eigen_mhd.diff_safe_sqrt

   .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.diff_safe_sqrt

.. py:class:: EigenvectorBlocks
   :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks

   .. py:attribute:: rho_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.rho_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.rho_interface

   .. py:attribute:: sqrt_rho
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sqrt_rho
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sqrt_rho

   .. py:attribute:: velocity_x_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.velocity_x_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.velocity_x_interface

   .. py:attribute:: velocity_y_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.velocity_y_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.velocity_y_interface

   .. py:attribute:: velocity_z_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.velocity_z_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.velocity_z_interface

   .. py:attribute:: velocity_sq_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.velocity_sq_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.velocity_sq_interface

   .. py:attribute:: magnetic_x_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.magnetic_x_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.magnetic_x_interface

   .. py:attribute:: magnetic_y_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.magnetic_y_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.magnetic_y_interface

   .. py:attribute:: magnetic_z_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.magnetic_z_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.magnetic_z_interface

   .. py:attribute:: bt_normalized_y
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.bt_normalized_y
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.bt_normalized_y

   .. py:attribute:: bt_normalized_z
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.bt_normalized_z
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.bt_normalized_z

   .. py:attribute:: sgn_bx
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sgn_bx
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sgn_bx

   .. py:attribute:: sgn_bt
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sgn_bt
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sgn_bt

   .. py:attribute:: sound_speed_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sound_speed_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sound_speed_interface

   .. py:attribute:: sound_speed_sq_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sound_speed_sq_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sound_speed_sq_interface

   .. py:attribute:: sound_speed_sq_inverse
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sound_speed_sq_inverse
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sound_speed_sq_inverse

   .. py:attribute:: sound_speed_greater_alfven_speed
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sound_speed_greater_alfven_speed
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.sound_speed_greater_alfven_speed

   .. py:attribute:: fast_magnetosonic_velocity_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.fast_magnetosonic_velocity_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.fast_magnetosonic_velocity_interface

   .. py:attribute:: alfven_velocity_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.alfven_velocity_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.alfven_velocity_interface

   .. py:attribute:: slow_magnetosonic_velocity_interface
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.slow_magnetosonic_velocity_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.slow_magnetosonic_velocity_interface

   .. py:attribute:: fast_mode_weighting
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.fast_mode_weighting
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.fast_mode_weighting

   .. py:attribute:: slow_mode_weighting
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.slow_mode_weighting
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.slow_mode_weighting

   .. py:attribute:: gam0
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.gam0
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.gam0

   .. py:attribute:: gam1
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.gam1
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.gam1

   .. py:attribute:: gam2
      :canonical: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.gam2
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd.EigenvectorBlocks.gam2
