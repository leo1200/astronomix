:orphan:

:py:mod:`astronomix._fluid_equations._dual_energy_switch`
=========================================================

.. py:module:: astronomix._fluid_equations._dual_energy_switch

.. autodoc2-docstring:: astronomix._fluid_equations._dual_energy_switch
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`dual_energy_internal_energy <astronomix._fluid_equations._dual_energy_switch.dual_energy_internal_energy>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._dual_energy_switch.dual_energy_internal_energy
          :summary:
   * - :py:obj:`dual_energy_switched_pressure <astronomix._fluid_equations._dual_energy_switch.dual_energy_switched_pressure>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._dual_energy_switch.dual_energy_switched_pressure
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DUAL_ENERGY_ETA <astronomix._fluid_equations._dual_energy_switch.DUAL_ENERGY_ETA>`
     - .. autodoc2-docstring:: astronomix._fluid_equations._dual_energy_switch.DUAL_ENERGY_ETA
          :summary:

API
~~~

.. py:data:: DUAL_ENERGY_ETA
   :canonical: astronomix._fluid_equations._dual_energy_switch.DUAL_ENERGY_ETA
   :value: 0.001

   .. autodoc2-docstring:: astronomix._fluid_equations._dual_energy_switch.DUAL_ENERGY_ETA

.. py:function:: dual_energy_internal_energy(internal_energy_from_total, total_energy, internal_energy_density, eta)
   :canonical: astronomix._fluid_equations._dual_energy_switch.dual_energy_internal_energy

   .. autodoc2-docstring:: astronomix._fluid_equations._dual_energy_switch.dual_energy_internal_energy

.. py:function:: dual_energy_switched_pressure(gas_pressure, total_energy, gamma, internal_energy_density, eta)
   :canonical: astronomix._fluid_equations._dual_energy_switch.dual_energy_switched_pressure

   .. autodoc2-docstring:: astronomix._fluid_equations._dual_energy_switch.dual_energy_switched_pressure
