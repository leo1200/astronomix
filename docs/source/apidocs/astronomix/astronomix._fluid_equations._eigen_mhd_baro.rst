:orphan:

:py:mod:`astronomix._fluid_equations._eigen_mhd_baro`
=====================================================

.. py:module:: astronomix._fluid_equations._eigen_mhd_baro

.. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`EigenvectorBlocksIso <astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`diff_safe_sqrt <astronomix._fluid_equations._eigen_mhd_baro.diff_safe_sqrt>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.diff_safe_sqrt
          :summary:
   * - :py:obj:`zero_safe_sqrt <astronomix._fluid_equations._eigen_mhd_baro.zero_safe_sqrt>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.zero_safe_sqrt
          :summary:

API
~~~

.. py:function:: diff_safe_sqrt(x)
   :canonical: astronomix._fluid_equations._eigen_mhd_baro.diff_safe_sqrt

   .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.diff_safe_sqrt

.. py:function:: zero_safe_sqrt(x)
   :canonical: astronomix._fluid_equations._eigen_mhd_baro.zero_safe_sqrt

   .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.zero_safe_sqrt

.. py:class:: EigenvectorBlocksIso
   :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso

   .. py:attribute:: rho_interface
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.rho_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.rho_interface

   .. py:attribute:: sqrt_rho
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.sqrt_rho
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.sqrt_rho

   .. py:attribute:: velocity_x_interface
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.velocity_x_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.velocity_x_interface

   .. py:attribute:: velocity_y_interface
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.velocity_y_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.velocity_y_interface

   .. py:attribute:: velocity_z_interface
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.velocity_z_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.velocity_z_interface

   .. py:attribute:: magnetic_x_interface
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.magnetic_x_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.magnetic_x_interface

   .. py:attribute:: magnetic_y_interface
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.magnetic_y_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.magnetic_y_interface

   .. py:attribute:: magnetic_z_interface
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.magnetic_z_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.magnetic_z_interface

   .. py:attribute:: bt_normalized_y
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.bt_normalized_y
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.bt_normalized_y

   .. py:attribute:: bt_normalized_z
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.bt_normalized_z
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.bt_normalized_z

   .. py:attribute:: sgn_bx
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.sgn_bx
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.sgn_bx

   .. py:attribute:: sgn_bt
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.sgn_bt
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.sgn_bt

   .. py:attribute:: sound_speed
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.sound_speed
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.sound_speed

   .. py:attribute:: cs2
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.cs2
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.cs2

   .. py:attribute:: cs2_inverse
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.cs2_inverse
      :type: typing.Union[float, jax.numpy.ndarray]
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.cs2_inverse

   .. py:attribute:: sound_speed_greater_alfven_speed
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.sound_speed_greater_alfven_speed
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.sound_speed_greater_alfven_speed

   .. py:attribute:: fast_magnetosonic_velocity_interface
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.fast_magnetosonic_velocity_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.fast_magnetosonic_velocity_interface

   .. py:attribute:: alfven_velocity_interface
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.alfven_velocity_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.alfven_velocity_interface

   .. py:attribute:: slow_magnetosonic_velocity_interface
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.slow_magnetosonic_velocity_interface
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.slow_magnetosonic_velocity_interface

   .. py:attribute:: fast_mode_weighting
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.fast_mode_weighting
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.fast_mode_weighting

   .. py:attribute:: slow_mode_weighting
      :canonical: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.slow_mode_weighting
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._fluid_equations._eigen_mhd_baro.EigenvectorBlocksIso.slow_mode_weighting
