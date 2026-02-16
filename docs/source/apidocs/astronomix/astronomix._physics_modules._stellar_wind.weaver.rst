:py:mod:`astronomix._physics_modules._stellar_wind.weaver`
==========================================================

.. py:module:: astronomix._physics_modules._stellar_wind.weaver

.. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Weaver <astronomix._physics_modules._stellar_wind.weaver.Weaver>`
     - .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver
          :summary:

API
~~~

.. py:class:: Weaver(v_inf, M_dot, rho_0, p_0, num_xi=100, gamma=5 / 3)
   :canonical: astronomix._physics_modules._stellar_wind.weaver.Weaver

   .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver

   .. rubric:: Initialization

   .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver.__init__

   .. py:method:: calculate_shell_profiles()
      :canonical: astronomix._physics_modules._stellar_wind.weaver.Weaver.calculate_shell_profiles

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver.calculate_shell_profiles

   .. py:method:: get_inner_shock_radius(t)
      :canonical: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_inner_shock_radius

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_inner_shock_radius

   .. py:method:: get_outer_shock_radius(t)
      :canonical: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_outer_shock_radius

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_outer_shock_radius

   .. py:method:: get_critical_radius(t)
      :canonical: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_critical_radius

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_critical_radius

   .. py:method:: get_radial_range_wind_interior(delta_R, t)
      :canonical: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_radial_range_wind_interior

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_radial_range_wind_interior

   .. py:method:: get_radial_range_free_wind(delta_R, t)
      :canonical: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_radial_range_free_wind

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_radial_range_free_wind

   .. py:method:: get_radial_range_undisturbed_ism(delta_R, R_max, t)
      :canonical: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_radial_range_undisturbed_ism

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_radial_range_undisturbed_ism

   .. py:method:: get_pressure_profile(delta_R, R_max, t)
      :canonical: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_pressure_profile

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_pressure_profile

   .. py:method:: get_velocity_profile(delta_R, R_max, t)
      :canonical: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_velocity_profile

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_velocity_profile

   .. py:method:: get_density_profile(delta_R, R_max, t)
      :canonical: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_density_profile

      .. autodoc2-docstring:: astronomix._physics_modules._stellar_wind.weaver.Weaver.get_density_profile
