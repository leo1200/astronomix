:py:mod:`astronomix.option_classes.simulation_config`
=====================================================

.. py:module:: astronomix.option_classes.simulation_config

.. autodoc2-docstring:: astronomix.option_classes.simulation_config
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`StaticIntVector <astronomix.option_classes.simulation_config.StaticIntVector>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.StaticIntVector
          :summary:
   * - :py:obj:`StaticFloatVector <astronomix.option_classes.simulation_config.StaticFloatVector>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.StaticFloatVector
          :summary:
   * - :py:obj:`SnapshotSettings <astronomix.option_classes.simulation_config.SnapshotSettings>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings
          :summary:
   * - :py:obj:`BoundarySettings1D <astronomix.option_classes.simulation_config.BoundarySettings1D>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BoundarySettings1D
          :summary:
   * - :py:obj:`BoundarySettings <astronomix.option_classes.simulation_config.BoundarySettings>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BoundarySettings
          :summary:
   * - :py:obj:`GravityConfig <astronomix.option_classes.simulation_config.GravityConfig>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GravityConfig
          :summary:
   * - :py:obj:`PositivityConfig <astronomix.option_classes.simulation_config.PositivityConfig>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PositivityConfig
          :summary:
   * - :py:obj:`SimulationConfig <astronomix.option_classes.simulation_config.SimulationConfig>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`gpu_compute_capability_at_least_80 <astronomix.option_classes.simulation_config.gpu_compute_capability_at_least_80>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.gpu_compute_capability_at_least_80
          :summary:
   * - :py:obj:`finalize_config <astronomix.option_classes.simulation_config.finalize_config>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.finalize_config
          :summary:
   * - :py:obj:`riemann_solver_to_string <astronomix.option_classes.simulation_config.riemann_solver_to_string>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.riemann_solver_to_string
          :summary:
   * - :py:obj:`limiter_to_string <astronomix.option_classes.simulation_config.limiter_to_string>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.limiter_to_string
          :summary:
   * - :py:obj:`solver_mode_to_string <astronomix.option_classes.simulation_config.solver_mode_to_string>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.solver_mode_to_string
          :summary:
   * - :py:obj:`config_to_string <astronomix.option_classes.simulation_config.config_to_string>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.config_to_string
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NATIVE_JAX <astronomix.option_classes.simulation_config.NATIVE_JAX>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.NATIVE_JAX
          :summary:
   * - :py:obj:`PALLAS <astronomix.option_classes.simulation_config.PALLAS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PALLAS
          :summary:
   * - :py:obj:`OPTIMAL_BACKEND <astronomix.option_classes.simulation_config.OPTIMAL_BACKEND>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.OPTIMAL_BACKEND
          :summary:
   * - :py:obj:`POSITIVITY_NONE <astronomix.option_classes.simulation_config.POSITIVITY_NONE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.POSITIVITY_NONE
          :summary:
   * - :py:obj:`POSITIVITY_HARD_FLOOR <astronomix.option_classes.simulation_config.POSITIVITY_HARD_FLOOR>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.POSITIVITY_HARD_FLOOR
          :summary:
   * - :py:obj:`FINITE_VOLUME <astronomix.option_classes.simulation_config.FINITE_VOLUME>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FINITE_VOLUME
          :summary:
   * - :py:obj:`FINITE_DIFFERENCE <astronomix.option_classes.simulation_config.FINITE_DIFFERENCE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FINITE_DIFFERENCE
          :summary:
   * - :py:obj:`FORWARDS <astronomix.option_classes.simulation_config.FORWARDS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FORWARDS
          :summary:
   * - :py:obj:`BACKWARDS <astronomix.option_classes.simulation_config.BACKWARDS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BACKWARDS
          :summary:
   * - :py:obj:`MINMOD <astronomix.option_classes.simulation_config.MINMOD>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MINMOD
          :summary:
   * - :py:obj:`OSHER <astronomix.option_classes.simulation_config.OSHER>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.OSHER
          :summary:
   * - :py:obj:`DOUBLE_MINMOD <astronomix.option_classes.simulation_config.DOUBLE_MINMOD>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DOUBLE_MINMOD
          :summary:
   * - :py:obj:`SUPERBEE <astronomix.option_classes.simulation_config.SUPERBEE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SUPERBEE
          :summary:
   * - :py:obj:`VAN_ALBADA <astronomix.option_classes.simulation_config.VAN_ALBADA>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VAN_ALBADA
          :summary:
   * - :py:obj:`VAN_ALBADA_PP <astronomix.option_classes.simulation_config.VAN_ALBADA_PP>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VAN_ALBADA_PP
          :summary:
   * - :py:obj:`UNSPLIT <astronomix.option_classes.simulation_config.UNSPLIT>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.UNSPLIT
          :summary:
   * - :py:obj:`SPLIT <astronomix.option_classes.simulation_config.SPLIT>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SPLIT
          :summary:
   * - :py:obj:`HLL <astronomix.option_classes.simulation_config.HLL>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLL
          :summary:
   * - :py:obj:`HLLC <astronomix.option_classes.simulation_config.HLLC>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLC
          :summary:
   * - :py:obj:`HLLC_LM <astronomix.option_classes.simulation_config.HLLC_LM>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLC_LM
          :summary:
   * - :py:obj:`LAX_FRIEDRICHS <astronomix.option_classes.simulation_config.LAX_FRIEDRICHS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.LAX_FRIEDRICHS
          :summary:
   * - :py:obj:`HYBRID_HLLC <astronomix.option_classes.simulation_config.HYBRID_HLLC>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HYBRID_HLLC
          :summary:
   * - :py:obj:`AM_HLLC <astronomix.option_classes.simulation_config.AM_HLLC>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.AM_HLLC
          :summary:
   * - :py:obj:`RK2_SSP <astronomix.option_classes.simulation_config.RK2_SSP>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RK2_SSP
          :summary:
   * - :py:obj:`MUSCL <astronomix.option_classes.simulation_config.MUSCL>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MUSCL
          :summary:
   * - :py:obj:`RK4_SSP <astronomix.option_classes.simulation_config.RK4_SSP>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RK4_SSP
          :summary:
   * - :py:obj:`RK4_LSRK <astronomix.option_classes.simulation_config.RK4_LSRK>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RK4_LSRK
          :summary:
   * - :py:obj:`OPEN_BOUNDARY <astronomix.option_classes.simulation_config.OPEN_BOUNDARY>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.OPEN_BOUNDARY
          :summary:
   * - :py:obj:`REFLECTIVE_BOUNDARY <astronomix.option_classes.simulation_config.REFLECTIVE_BOUNDARY>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.REFLECTIVE_BOUNDARY
          :summary:
   * - :py:obj:`PERIODIC_BOUNDARY <astronomix.option_classes.simulation_config.PERIODIC_BOUNDARY>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PERIODIC_BOUNDARY
          :summary:
   * - :py:obj:`FIXED_BOUNDARY <astronomix.option_classes.simulation_config.FIXED_BOUNDARY>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FIXED_BOUNDARY
          :summary:
   * - :py:obj:`MHD_JET_BOUNDARY <astronomix.option_classes.simulation_config.MHD_JET_BOUNDARY>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MHD_JET_BOUNDARY
          :summary:
   * - :py:obj:`FIXED_BOUNDARY_OPEN_MOMENTUM <astronomix.option_classes.simulation_config.FIXED_BOUNDARY_OPEN_MOMENTUM>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FIXED_BOUNDARY_OPEN_MOMENTUM
          :summary:
   * - :py:obj:`PRIMITIVE_GAS_STATE <astronomix.option_classes.simulation_config.PRIMITIVE_GAS_STATE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PRIMITIVE_GAS_STATE
          :summary:
   * - :py:obj:`CONSERVATIVE_GAS_STATE <astronomix.option_classes.simulation_config.CONSERVATIVE_GAS_STATE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.CONSERVATIVE_GAS_STATE
          :summary:
   * - :py:obj:`VELOCITY_ONLY <astronomix.option_classes.simulation_config.VELOCITY_ONLY>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VELOCITY_ONLY
          :summary:
   * - :py:obj:`MAGNETIC_FIELD_ONLY <astronomix.option_classes.simulation_config.MAGNETIC_FIELD_ONLY>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MAGNETIC_FIELD_ONLY
          :summary:
   * - :py:obj:`CARTESIAN <astronomix.option_classes.simulation_config.CARTESIAN>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.CARTESIAN
          :summary:
   * - :py:obj:`CYLINDRICAL <astronomix.option_classes.simulation_config.CYLINDRICAL>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.CYLINDRICAL
          :summary:
   * - :py:obj:`SPHERICAL <astronomix.option_classes.simulation_config.SPHERICAL>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SPHERICAL
          :summary:
   * - :py:obj:`VARAXIS <astronomix.option_classes.simulation_config.VARAXIS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VARAXIS
          :summary:
   * - :py:obj:`XAXIS <astronomix.option_classes.simulation_config.XAXIS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.XAXIS
          :summary:
   * - :py:obj:`YAXIS <astronomix.option_classes.simulation_config.YAXIS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.YAXIS
          :summary:
   * - :py:obj:`ZAXIS <astronomix.option_classes.simulation_config.ZAXIS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.ZAXIS
          :summary:
   * - :py:obj:`GHOST_CELLS <astronomix.option_classes.simulation_config.GHOST_CELLS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GHOST_CELLS
          :summary:
   * - :py:obj:`PERIODIC_ROLL <astronomix.option_classes.simulation_config.PERIODIC_ROLL>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PERIODIC_ROLL
          :summary:
   * - :py:obj:`SIMPLE_SOURCE <astronomix.option_classes.simulation_config.SIMPLE_SOURCE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SIMPLE_SOURCE
          :summary:
   * - :py:obj:`SECOND_ORDER_CONSERVATIVE <astronomix.option_classes.simulation_config.SECOND_ORDER_CONSERVATIVE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SECOND_ORDER_CONSERVATIVE
          :summary:
   * - :py:obj:`FOURTH_ORDER_CONSERVATIVE <astronomix.option_classes.simulation_config.FOURTH_ORDER_CONSERVATIVE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FOURTH_ORDER_CONSERVATIVE
          :summary:
   * - :py:obj:`SINGLE_PRECISION <astronomix.option_classes.simulation_config.SINGLE_PRECISION>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SINGLE_PRECISION
          :summary:
   * - :py:obj:`DOUBLE_PRECISION <astronomix.option_classes.simulation_config.DOUBLE_PRECISION>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DOUBLE_PRECISION
          :summary:
   * - :py:obj:`KINEMATIC_VISCOSITY <astronomix.option_classes.simulation_config.KINEMATIC_VISCOSITY>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.KINEMATIC_VISCOSITY
          :summary:
   * - :py:obj:`DYNAMIC_VISCOSITY <astronomix.option_classes.simulation_config.DYNAMIC_VISCOSITY>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DYNAMIC_VISCOSITY
          :summary:
   * - :py:obj:`IDEAL_GAS <astronomix.option_classes.simulation_config.IDEAL_GAS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.IDEAL_GAS
          :summary:
   * - :py:obj:`ISOTHERMAL <astronomix.option_classes.simulation_config.ISOTHERMAL>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.ISOTHERMAL
          :summary:
   * - :py:obj:`ON_DEVICE <astronomix.option_classes.simulation_config.ON_DEVICE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.ON_DEVICE
          :summary:
   * - :py:obj:`TO_DISK <astronomix.option_classes.simulation_config.TO_DISK>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.TO_DISK
          :summary:
   * - :py:obj:`STATE_TYPE <astronomix.option_classes.simulation_config.STATE_TYPE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.STATE_TYPE
          :summary:
   * - :py:obj:`STATE_TYPE_ALTERED <astronomix.option_classes.simulation_config.STATE_TYPE_ALTERED>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.STATE_TYPE_ALTERED
          :summary:
   * - :py:obj:`FIELD_TYPE <astronomix.option_classes.simulation_config.FIELD_TYPE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FIELD_TYPE
          :summary:

API
~~~

.. py:data:: NATIVE_JAX
   :canonical: astronomix.option_classes.simulation_config.NATIVE_JAX
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.NATIVE_JAX

.. py:data:: PALLAS
   :canonical: astronomix.option_classes.simulation_config.PALLAS
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PALLAS

.. py:data:: OPTIMAL_BACKEND
   :canonical: astronomix.option_classes.simulation_config.OPTIMAL_BACKEND
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.OPTIMAL_BACKEND

.. py:data:: POSITIVITY_NONE
   :canonical: astronomix.option_classes.simulation_config.POSITIVITY_NONE
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.POSITIVITY_NONE

.. py:data:: POSITIVITY_HARD_FLOOR
   :canonical: astronomix.option_classes.simulation_config.POSITIVITY_HARD_FLOOR
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.POSITIVITY_HARD_FLOOR

.. py:data:: FINITE_VOLUME
   :canonical: astronomix.option_classes.simulation_config.FINITE_VOLUME
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FINITE_VOLUME

.. py:data:: FINITE_DIFFERENCE
   :canonical: astronomix.option_classes.simulation_config.FINITE_DIFFERENCE
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FINITE_DIFFERENCE

.. py:data:: FORWARDS
   :canonical: astronomix.option_classes.simulation_config.FORWARDS
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FORWARDS

.. py:data:: BACKWARDS
   :canonical: astronomix.option_classes.simulation_config.BACKWARDS
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BACKWARDS

.. py:data:: MINMOD
   :canonical: astronomix.option_classes.simulation_config.MINMOD
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MINMOD

.. py:data:: OSHER
   :canonical: astronomix.option_classes.simulation_config.OSHER
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.OSHER

.. py:data:: DOUBLE_MINMOD
   :canonical: astronomix.option_classes.simulation_config.DOUBLE_MINMOD
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DOUBLE_MINMOD

.. py:data:: SUPERBEE
   :canonical: astronomix.option_classes.simulation_config.SUPERBEE
   :value: 3

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SUPERBEE

.. py:data:: VAN_ALBADA
   :canonical: astronomix.option_classes.simulation_config.VAN_ALBADA
   :value: 4

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VAN_ALBADA

.. py:data:: VAN_ALBADA_PP
   :canonical: astronomix.option_classes.simulation_config.VAN_ALBADA_PP
   :value: 5

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VAN_ALBADA_PP

.. py:data:: UNSPLIT
   :canonical: astronomix.option_classes.simulation_config.UNSPLIT
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.UNSPLIT

.. py:data:: SPLIT
   :canonical: astronomix.option_classes.simulation_config.SPLIT
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SPLIT

.. py:data:: HLL
   :canonical: astronomix.option_classes.simulation_config.HLL
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLL

.. py:data:: HLLC
   :canonical: astronomix.option_classes.simulation_config.HLLC
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLC

.. py:data:: HLLC_LM
   :canonical: astronomix.option_classes.simulation_config.HLLC_LM
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLC_LM

.. py:data:: LAX_FRIEDRICHS
   :canonical: astronomix.option_classes.simulation_config.LAX_FRIEDRICHS
   :value: 3

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.LAX_FRIEDRICHS

.. py:data:: HYBRID_HLLC
   :canonical: astronomix.option_classes.simulation_config.HYBRID_HLLC
   :value: 4

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HYBRID_HLLC

.. py:data:: AM_HLLC
   :canonical: astronomix.option_classes.simulation_config.AM_HLLC
   :value: 5

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.AM_HLLC

.. py:data:: RK2_SSP
   :canonical: astronomix.option_classes.simulation_config.RK2_SSP
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RK2_SSP

.. py:data:: MUSCL
   :canonical: astronomix.option_classes.simulation_config.MUSCL
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MUSCL

.. py:data:: RK4_SSP
   :canonical: astronomix.option_classes.simulation_config.RK4_SSP
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RK4_SSP

.. py:data:: RK4_LSRK
   :canonical: astronomix.option_classes.simulation_config.RK4_LSRK
   :value: 3

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RK4_LSRK

.. py:data:: OPEN_BOUNDARY
   :canonical: astronomix.option_classes.simulation_config.OPEN_BOUNDARY
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.OPEN_BOUNDARY

.. py:data:: REFLECTIVE_BOUNDARY
   :canonical: astronomix.option_classes.simulation_config.REFLECTIVE_BOUNDARY
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.REFLECTIVE_BOUNDARY

.. py:data:: PERIODIC_BOUNDARY
   :canonical: astronomix.option_classes.simulation_config.PERIODIC_BOUNDARY
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PERIODIC_BOUNDARY

.. py:data:: FIXED_BOUNDARY
   :canonical: astronomix.option_classes.simulation_config.FIXED_BOUNDARY
   :value: 3

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FIXED_BOUNDARY

.. py:data:: MHD_JET_BOUNDARY
   :canonical: astronomix.option_classes.simulation_config.MHD_JET_BOUNDARY
   :value: 4

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MHD_JET_BOUNDARY

.. py:data:: FIXED_BOUNDARY_OPEN_MOMENTUM
   :canonical: astronomix.option_classes.simulation_config.FIXED_BOUNDARY_OPEN_MOMENTUM
   :value: 5

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FIXED_BOUNDARY_OPEN_MOMENTUM

.. py:data:: PRIMITIVE_GAS_STATE
   :canonical: astronomix.option_classes.simulation_config.PRIMITIVE_GAS_STATE
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PRIMITIVE_GAS_STATE

.. py:data:: CONSERVATIVE_GAS_STATE
   :canonical: astronomix.option_classes.simulation_config.CONSERVATIVE_GAS_STATE
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.CONSERVATIVE_GAS_STATE

.. py:data:: VELOCITY_ONLY
   :canonical: astronomix.option_classes.simulation_config.VELOCITY_ONLY
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VELOCITY_ONLY

.. py:data:: MAGNETIC_FIELD_ONLY
   :canonical: astronomix.option_classes.simulation_config.MAGNETIC_FIELD_ONLY
   :value: 3

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MAGNETIC_FIELD_ONLY

.. py:data:: CARTESIAN
   :canonical: astronomix.option_classes.simulation_config.CARTESIAN
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.CARTESIAN

.. py:data:: CYLINDRICAL
   :canonical: astronomix.option_classes.simulation_config.CYLINDRICAL
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.CYLINDRICAL

.. py:data:: SPHERICAL
   :canonical: astronomix.option_classes.simulation_config.SPHERICAL
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SPHERICAL

.. py:data:: VARAXIS
   :canonical: astronomix.option_classes.simulation_config.VARAXIS
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VARAXIS

.. py:data:: XAXIS
   :canonical: astronomix.option_classes.simulation_config.XAXIS
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.XAXIS

.. py:data:: YAXIS
   :canonical: astronomix.option_classes.simulation_config.YAXIS
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.YAXIS

.. py:data:: ZAXIS
   :canonical: astronomix.option_classes.simulation_config.ZAXIS
   :value: 3

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.ZAXIS

.. py:data:: GHOST_CELLS
   :canonical: astronomix.option_classes.simulation_config.GHOST_CELLS
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GHOST_CELLS

.. py:data:: PERIODIC_ROLL
   :canonical: astronomix.option_classes.simulation_config.PERIODIC_ROLL
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PERIODIC_ROLL

.. py:data:: SIMPLE_SOURCE
   :canonical: astronomix.option_classes.simulation_config.SIMPLE_SOURCE
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SIMPLE_SOURCE

.. py:data:: SECOND_ORDER_CONSERVATIVE
   :canonical: astronomix.option_classes.simulation_config.SECOND_ORDER_CONSERVATIVE
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SECOND_ORDER_CONSERVATIVE

.. py:data:: FOURTH_ORDER_CONSERVATIVE
   :canonical: astronomix.option_classes.simulation_config.FOURTH_ORDER_CONSERVATIVE
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FOURTH_ORDER_CONSERVATIVE

.. py:data:: SINGLE_PRECISION
   :canonical: astronomix.option_classes.simulation_config.SINGLE_PRECISION
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SINGLE_PRECISION

.. py:data:: DOUBLE_PRECISION
   :canonical: astronomix.option_classes.simulation_config.DOUBLE_PRECISION
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DOUBLE_PRECISION

.. py:data:: KINEMATIC_VISCOSITY
   :canonical: astronomix.option_classes.simulation_config.KINEMATIC_VISCOSITY
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.KINEMATIC_VISCOSITY

.. py:data:: DYNAMIC_VISCOSITY
   :canonical: astronomix.option_classes.simulation_config.DYNAMIC_VISCOSITY
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DYNAMIC_VISCOSITY

.. py:data:: IDEAL_GAS
   :canonical: astronomix.option_classes.simulation_config.IDEAL_GAS
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.IDEAL_GAS

.. py:data:: ISOTHERMAL
   :canonical: astronomix.option_classes.simulation_config.ISOTHERMAL
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.ISOTHERMAL

.. py:data:: ON_DEVICE
   :canonical: astronomix.option_classes.simulation_config.ON_DEVICE
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.ON_DEVICE

.. py:data:: TO_DISK
   :canonical: astronomix.option_classes.simulation_config.TO_DISK
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.TO_DISK

.. py:class:: StaticIntVector
   :canonical: astronomix.option_classes.simulation_config.StaticIntVector

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.StaticIntVector

   .. py:attribute:: x
      :canonical: astronomix.option_classes.simulation_config.StaticIntVector.x
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.StaticIntVector.x

   .. py:attribute:: y
      :canonical: astronomix.option_classes.simulation_config.StaticIntVector.y
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.StaticIntVector.y

   .. py:attribute:: z
      :canonical: astronomix.option_classes.simulation_config.StaticIntVector.z
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.StaticIntVector.z

.. py:class:: StaticFloatVector
   :canonical: astronomix.option_classes.simulation_config.StaticFloatVector

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.StaticFloatVector

   .. py:attribute:: x
      :canonical: astronomix.option_classes.simulation_config.StaticFloatVector.x
      :type: float
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.StaticFloatVector.x

   .. py:attribute:: y
      :canonical: astronomix.option_classes.simulation_config.StaticFloatVector.y
      :type: float
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.StaticFloatVector.y

   .. py:attribute:: z
      :canonical: astronomix.option_classes.simulation_config.StaticFloatVector.z
      :type: float
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.StaticFloatVector.z

   .. py:method:: __truediv__(other: astronomix.option_classes.simulation_config.StaticIntVector) -> astronomix.option_classes.simulation_config.StaticFloatVector
      :canonical: astronomix.option_classes.simulation_config.StaticFloatVector.__truediv__

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.StaticFloatVector.__truediv__

.. py:data:: STATE_TYPE
   :canonical: astronomix.option_classes.simulation_config.STATE_TYPE
   :value: None

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.STATE_TYPE

.. py:data:: STATE_TYPE_ALTERED
   :canonical: astronomix.option_classes.simulation_config.STATE_TYPE_ALTERED
   :value: None

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.STATE_TYPE_ALTERED

.. py:data:: FIELD_TYPE
   :canonical: astronomix.option_classes.simulation_config.FIELD_TYPE
   :value: None

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FIELD_TYPE

.. py:class:: SnapshotSettings
   :canonical: astronomix.option_classes.simulation_config.SnapshotSettings

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings

   .. py:attribute:: return_states
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_states
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_states

   .. py:attribute:: return_final_state
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_final_state
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_final_state

   .. py:attribute:: return_total_mass
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_total_mass
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_total_mass

   .. py:attribute:: return_total_energy
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_total_energy
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_total_energy

   .. py:attribute:: return_internal_energy
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_internal_energy
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_internal_energy

   .. py:attribute:: return_kinetic_energy
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_kinetic_energy
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_kinetic_energy

   .. py:attribute:: return_gravitational_energy
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_gravitational_energy
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_gravitational_energy

   .. py:attribute:: return_radial_momentum
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_radial_momentum
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_radial_momentum

   .. py:attribute:: return_kinetic_energy_spectrum
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_kinetic_energy_spectrum
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_kinetic_energy_spectrum

   .. py:attribute:: return_magnetic_energy_spectrum
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_magnetic_energy_spectrum
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_magnetic_energy_spectrum

   .. py:attribute:: return_helicity_spectrum
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_helicity_spectrum
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_helicity_spectrum

   .. py:attribute:: return_magnetic_divergence
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_magnetic_divergence
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_magnetic_divergence

   .. py:attribute:: return_temperature_pdf
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.return_temperature_pdf
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.return_temperature_pdf

   .. py:attribute:: num_temperature_bins
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.num_temperature_bins
      :type: int
      :value: 100

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.num_temperature_bins

   .. py:attribute:: temperature_pdf_min
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.temperature_pdf_min
      :type: float
      :value: 1e-10

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.temperature_pdf_min

   .. py:attribute:: temperature_pdf_max
      :canonical: astronomix.option_classes.simulation_config.SnapshotSettings.temperature_pdf_max
      :type: float
      :value: 10000000000.0

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SnapshotSettings.temperature_pdf_max

.. py:class:: BoundarySettings1D
   :canonical: astronomix.option_classes.simulation_config.BoundarySettings1D

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BoundarySettings1D

   .. py:attribute:: left_boundary
      :canonical: astronomix.option_classes.simulation_config.BoundarySettings1D.left_boundary
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BoundarySettings1D.left_boundary

   .. py:attribute:: right_boundary
      :canonical: astronomix.option_classes.simulation_config.BoundarySettings1D.right_boundary
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BoundarySettings1D.right_boundary

.. py:class:: BoundarySettings
   :canonical: astronomix.option_classes.simulation_config.BoundarySettings

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BoundarySettings

   .. py:attribute:: x
      :canonical: astronomix.option_classes.simulation_config.BoundarySettings.x
      :type: astronomix.option_classes.simulation_config.BoundarySettings1D
      :value: 'BoundarySettings1D(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BoundarySettings.x

   .. py:attribute:: y
      :canonical: astronomix.option_classes.simulation_config.BoundarySettings.y
      :type: astronomix.option_classes.simulation_config.BoundarySettings1D
      :value: 'BoundarySettings1D(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BoundarySettings.y

   .. py:attribute:: z
      :canonical: astronomix.option_classes.simulation_config.BoundarySettings.z
      :type: astronomix.option_classes.simulation_config.BoundarySettings1D
      :value: 'BoundarySettings1D(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BoundarySettings.z

.. py:class:: GravityConfig
   :canonical: astronomix.option_classes.simulation_config.GravityConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GravityConfig

   .. py:attribute:: self_gravity
      :canonical: astronomix.option_classes.simulation_config.GravityConfig.self_gravity
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GravityConfig.self_gravity

   .. py:attribute:: self_gravity_version
      :canonical: astronomix.option_classes.simulation_config.GravityConfig.self_gravity_version
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GravityConfig.self_gravity_version

   .. py:attribute:: external_potential
      :canonical: astronomix.option_classes.simulation_config.GravityConfig.external_potential
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GravityConfig.external_potential

   .. py:attribute:: poisson_manual_open_boundaries
      :canonical: astronomix.option_classes.simulation_config.GravityConfig.poisson_manual_open_boundaries
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GravityConfig.poisson_manual_open_boundaries

   .. py:attribute:: work_flux_correction
      :canonical: astronomix.option_classes.simulation_config.GravityConfig.work_flux_correction
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GravityConfig.work_flux_correction

   .. py:attribute:: gravity
      :canonical: astronomix.option_classes.simulation_config.GravityConfig.gravity
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GravityConfig.gravity

.. py:class:: PositivityConfig
   :canonical: astronomix.option_classes.simulation_config.PositivityConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PositivityConfig

   .. py:attribute:: per_step_mode
      :canonical: astronomix.option_classes.simulation_config.PositivityConfig.per_step_mode
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PositivityConfig.per_step_mode

   .. py:attribute:: per_step_specific_floor
      :canonical: astronomix.option_classes.simulation_config.PositivityConfig.per_step_specific_floor
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PositivityConfig.per_step_specific_floor

   .. py:attribute:: clamp_in_estimates
      :canonical: astronomix.option_classes.simulation_config.PositivityConfig.clamp_in_estimates
      :type: bool
      :value: True

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PositivityConfig.clamp_in_estimates

   .. py:attribute:: coldcrush_blend
      :canonical: astronomix.option_classes.simulation_config.PositivityConfig.coldcrush_blend
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PositivityConfig.coldcrush_blend

   .. py:attribute:: coldcrush_blend_factor
      :canonical: astronomix.option_classes.simulation_config.PositivityConfig.coldcrush_blend_factor
      :type: float
      :value: 8.0

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PositivityConfig.coldcrush_blend_factor

.. py:class:: SimulationConfig
   :canonical: astronomix.option_classes.simulation_config.SimulationConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig

   .. py:attribute:: backend
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.backend
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.backend

   .. py:attribute:: pallas_block_shape
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.pallas_block_shape
      :type: typing.Tuple[int, int, int]
      :value: (4, 4, 8)

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.pallas_block_shape

   .. py:attribute:: pallas_use_triton
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.pallas_use_triton
      :type: bool
      :value: True

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.pallas_use_triton

   .. py:attribute:: pallas_interpret
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.pallas_interpret
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.pallas_interpret

   .. py:attribute:: pallas_num_warps
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.pallas_num_warps
      :type: int
      :value: 4

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.pallas_num_warps

   .. py:attribute:: pallas_ct
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.pallas_ct
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.pallas_ct

   .. py:attribute:: solver_mode
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.solver_mode
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.solver_mode

   .. py:attribute:: numerical_precision
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.numerical_precision
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.numerical_precision

   .. py:attribute:: runtime_debugging
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.runtime_debugging
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.runtime_debugging

   .. py:attribute:: donate_state
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.donate_state
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.donate_state

   .. py:attribute:: memory_analysis
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.memory_analysis
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.memory_analysis

   .. py:attribute:: host_helper_data
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.host_helper_data
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.host_helper_data

   .. py:attribute:: print_elapsed_time
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.print_elapsed_time
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.print_elapsed_time

   .. py:attribute:: progress_bar
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.progress_bar
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.progress_bar

   .. py:attribute:: dimensionality
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.dimensionality
      :type: int
      :value: 1

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.dimensionality

   .. py:attribute:: state_struct
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.state_struct
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.state_struct

   .. py:attribute:: geometry
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.geometry
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.geometry

   .. py:attribute:: random_seed
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.random_seed
      :type: int
      :value: 42

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.random_seed

   .. py:attribute:: equation_of_state
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.equation_of_state
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.equation_of_state

   .. py:attribute:: mhd
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.mhd
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.mhd

   .. py:attribute:: positivity_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.positivity_config
      :type: astronomix.option_classes.simulation_config.PositivityConfig
      :value: 'PositivityConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.positivity_config

   .. py:attribute:: gravity_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.gravity_config
      :type: astronomix.option_classes.simulation_config.GravityConfig
      :value: 'GravityConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.gravity_config

   .. py:attribute:: diffusion
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.diffusion
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.diffusion

   .. py:attribute:: viscosity_type
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.viscosity_type
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.viscosity_type

   .. py:attribute:: thermal_conduction
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.thermal_conduction
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.thermal_conduction

   .. py:attribute:: box_size
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.box_size
      :type: typing.Union[float, astronomix.option_classes.simulation_config.StaticFloatVector]
      :value: 1.0

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.box_size

   .. py:attribute:: num_cells
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.num_cells
      :type: typing.Union[int, astronomix.option_classes.simulation_config.StaticIntVector]
      :value: 400

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.num_cells

   .. py:attribute:: reconstruction_order
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.reconstruction_order
      :type: int
      :value: 1

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.reconstruction_order

   .. py:attribute:: limiter
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.limiter
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.limiter

   .. py:attribute:: riemann_solver
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.riemann_solver
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.riemann_solver

   .. py:attribute:: split
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.split
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.split

   .. py:attribute:: time_integrator
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.time_integrator
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.time_integrator

   .. py:attribute:: num_ghost_cells
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.num_ghost_cells
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.num_ghost_cells

   .. py:attribute:: grid_spacing
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.grid_spacing
      :type: float
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.grid_spacing

   .. py:attribute:: boundary_handling
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.boundary_handling
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.boundary_handling

   .. py:attribute:: boundary_settings
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.boundary_settings
      :type: typing.Union[types.NoneType, astronomix.option_classes.simulation_config.BoundarySettings1D, astronomix.option_classes.simulation_config.BoundarySettings]
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.boundary_settings

   .. py:attribute:: fixed_timestep
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.fixed_timestep
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.fixed_timestep

   .. py:attribute:: exact_end_time
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.exact_end_time
      :type: bool
      :value: True

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.exact_end_time

   .. py:attribute:: source_term_aware_timestep
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.source_term_aware_timestep
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.source_term_aware_timestep

   .. py:attribute:: num_timesteps
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.num_timesteps
      :type: int
      :value: 1000

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.num_timesteps

   .. py:attribute:: use_max_adaptive_timestep
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.use_max_adaptive_timestep
      :type: bool
      :value: True

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.use_max_adaptive_timestep

   .. py:attribute:: differentiation_mode
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.differentiation_mode
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.differentiation_mode

   .. py:attribute:: num_checkpoints
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.num_checkpoints
      :type: int
      :value: 100

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.num_checkpoints

   .. py:attribute:: return_snapshots
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.return_snapshots
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.return_snapshots

   .. py:attribute:: snapshot_settings
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.snapshot_settings
      :type: astronomix.option_classes.simulation_config.SnapshotSettings
      :value: 'SnapshotSettings(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.snapshot_settings

   .. py:attribute:: snapshot_storage_mode
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.snapshot_storage_mode
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.snapshot_storage_mode

   .. py:attribute:: snapshot_storage_path
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.snapshot_storage_path
      :type: typing.Union[str, types.NoneType]
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.snapshot_storage_path

   .. py:attribute:: activate_snapshot_callback
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.activate_snapshot_callback
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.activate_snapshot_callback

   .. py:attribute:: use_specific_snapshot_timepoints
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.use_specific_snapshot_timepoints
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.use_specific_snapshot_timepoints

   .. py:attribute:: num_snapshots
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.num_snapshots
      :type: int
      :value: 10

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.num_snapshots

   .. py:attribute:: first_order_fallback
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.first_order_fallback
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.first_order_fallback

   .. py:attribute:: weno_positivity_preserving
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.weno_positivity_preserving
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.weno_positivity_preserving

   .. py:attribute:: turbulent_forcing_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.turbulent_forcing_config
      :type: astronomix._modules._turbulent_forcing._turbulent_forcing_options.TurbulentForcingConfig
      :value: 'TurbulentForcingConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.turbulent_forcing_config

   .. py:attribute:: wind_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.wind_config
      :type: astronomix._modules._stellar_wind.stellar_wind_options.WindConfig
      :value: 'WindConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.wind_config

   .. py:attribute:: cooling_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.cooling_config
      :type: astronomix._modules._cooling.cooling_options.CoolingConfig
      :value: 'CoolingConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.cooling_config

   .. py:attribute:: frame_tracking
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.frame_tracking
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.frame_tracking

   .. py:attribute:: neural_net_force_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.neural_net_force_config
      :type: astronomix._modules._neural_net_force._neural_net_force_options.NeuralNetForceConfig
      :value: 'NeuralNetForceConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.neural_net_force_config

   .. py:attribute:: cnn_mhd_corrector_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.cnn_mhd_corrector_config
      :type: astronomix._modules._cnn_mhd_corrector._cnn_mhd_corrector_options.CNNMHDconfig
      :value: 'CNNMHDconfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.cnn_mhd_corrector_config

.. py:function:: gpu_compute_capability_at_least_80() -> bool
   :canonical: astronomix.option_classes.simulation_config.gpu_compute_capability_at_least_80

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.gpu_compute_capability_at_least_80

.. py:function:: finalize_config(config: astronomix.option_classes.simulation_config.SimulationConfig, state_shape) -> astronomix.option_classes.simulation_config.SimulationConfig
   :canonical: astronomix.option_classes.simulation_config.finalize_config

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.finalize_config

.. py:function:: riemann_solver_to_string(riemann_solver: int) -> str
   :canonical: astronomix.option_classes.simulation_config.riemann_solver_to_string

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.riemann_solver_to_string

.. py:function:: limiter_to_string(limiter: int) -> str
   :canonical: astronomix.option_classes.simulation_config.limiter_to_string

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.limiter_to_string

.. py:function:: solver_mode_to_string(solver_mode: int) -> str
   :canonical: astronomix.option_classes.simulation_config.solver_mode_to_string

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.solver_mode_to_string

.. py:function:: config_to_string(config: astronomix.option_classes.simulation_config.SimulationConfig) -> str
   :canonical: astronomix.option_classes.simulation_config.config_to_string

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.config_to_string
