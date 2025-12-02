:py:mod:`astronomix.units.unit_helpers`
=======================================

.. py:module:: astronomix.units.unit_helpers

.. autodoc2-docstring:: astronomix.units.unit_helpers
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`CodeUnits <astronomix.units.unit_helpers.CodeUnits>`
     - .. autodoc2-docstring:: astronomix.units.unit_helpers.CodeUnits
          :summary:

API
~~~

.. py:class:: CodeUnits(unit_length, unit_mass, unit_velocity)
   :canonical: astronomix.units.unit_helpers.CodeUnits

   .. autodoc2-docstring:: astronomix.units.unit_helpers.CodeUnits

   .. rubric:: Initialization

   .. autodoc2-docstring:: astronomix.units.unit_helpers.CodeUnits.__init__

   .. py:method:: init_from_unit_params(UnitMass_in_g, UnitVelocity_in_cm_per_s)
      :canonical: astronomix.units.unit_helpers.CodeUnits.init_from_unit_params

      .. autodoc2-docstring:: astronomix.units.unit_helpers.CodeUnits.init_from_unit_params

   .. py:method:: get_temperature_from_internal_energy(internal_energy, gamma=5 / 3, hydrogen_abundance=0.76)
      :canonical: astronomix.units.unit_helpers.CodeUnits.get_temperature_from_internal_energy

      .. autodoc2-docstring:: astronomix.units.unit_helpers.CodeUnits.get_temperature_from_internal_energy

   .. py:method:: print_simulation_parameters(final_time_wanted)
      :canonical: astronomix.units.unit_helpers.CodeUnits.print_simulation_parameters

      .. autodoc2-docstring:: astronomix.units.unit_helpers.CodeUnits.print_simulation_parameters
