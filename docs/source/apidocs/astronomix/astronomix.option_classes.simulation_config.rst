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
   * - :py:obj:`BackendConfig <astronomix.option_classes.simulation_config.BackendConfig>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BackendConfig
          :summary:
   * - :py:obj:`FVSolverConfig <astronomix.option_classes.simulation_config.FVSolverConfig>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FVSolverConfig
          :summary:
   * - :py:obj:`FDSolverConfig <astronomix.option_classes.simulation_config.FDSolverConfig>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig
          :summary:
   * - :py:obj:`PassiveScalarConfig <astronomix.option_classes.simulation_config.PassiveScalarConfig>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PassiveScalarConfig
          :summary:
   * - :py:obj:`DiffusionConfig <astronomix.option_classes.simulation_config.DiffusionConfig>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DiffusionConfig
          :summary:
   * - :py:obj:`SimulationConfig <astronomix.option_classes.simulation_config.SimulationConfig>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_solver_config <astronomix.option_classes.simulation_config.get_solver_config>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.get_solver_config
          :summary:
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
   * - :py:obj:`MAXIMUM_STEP_RETRIES <astronomix.option_classes.simulation_config.MAXIMUM_STEP_RETRIES>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MAXIMUM_STEP_RETRIES
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
   * - :py:obj:`RK2_SSP <astronomix.option_classes.simulation_config.RK2_SSP>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RK2_SSP
          :summary:
   * - :py:obj:`MUSCL_HANCOCK <astronomix.option_classes.simulation_config.MUSCL_HANCOCK>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MUSCL_HANCOCK
          :summary:
   * - :py:obj:`VL2 <astronomix.option_classes.simulation_config.VL2>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VL2
          :summary:
   * - :py:obj:`SSPRK54 <astronomix.option_classes.simulation_config.SSPRK54>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SSPRK54
          :summary:
   * - :py:obj:`SSPRK33 <astronomix.option_classes.simulation_config.SSPRK33>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SSPRK33
          :summary:
   * - :py:obj:`LSRK4 <astronomix.option_classes.simulation_config.LSRK4>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.LSRK4
          :summary:
   * - :py:obj:`WENO_FLUX_SPLITTING <astronomix.option_classes.simulation_config.WENO_FLUX_SPLITTING>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.WENO_FLUX_SPLITTING
          :summary:
   * - :py:obj:`RIEMANN_FLUX <astronomix.option_classes.simulation_config.RIEMANN_FLUX>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RIEMANN_FLUX
          :summary:
   * - :py:obj:`RIEMANN_BY_EOS <astronomix.option_classes.simulation_config.RIEMANN_BY_EOS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RIEMANN_BY_EOS
          :summary:
   * - :py:obj:`HLLEM_T <astronomix.option_classes.simulation_config.HLLEM_T>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLEM_T
          :summary:
   * - :py:obj:`HLLD_BOUNDED <astronomix.option_classes.simulation_config.HLLD_BOUNDED>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLD_BOUNDED
          :summary:
   * - :py:obj:`DONOR_CELL <astronomix.option_classes.simulation_config.DONOR_CELL>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DONOR_CELL
          :summary:
   * - :py:obj:`PIECEWISE_LINEAR <astronomix.option_classes.simulation_config.PIECEWISE_LINEAR>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PIECEWISE_LINEAR
          :summary:
   * - :py:obj:`MINMOD <astronomix.option_classes.simulation_config.MINMOD>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MINMOD
          :summary:
   * - :py:obj:`VAN_LEER <astronomix.option_classes.simulation_config.VAN_LEER>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VAN_LEER
          :summary:
   * - :py:obj:`SUPERBEE <astronomix.option_classes.simulation_config.SUPERBEE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SUPERBEE
          :summary:
   * - :py:obj:`DOUBLE_MINMOD <astronomix.option_classes.simulation_config.DOUBLE_MINMOD>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DOUBLE_MINMOD
          :summary:
   * - :py:obj:`HLL <astronomix.option_classes.simulation_config.HLL>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLL
          :summary:
   * - :py:obj:`HLLC <astronomix.option_classes.simulation_config.HLLC>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLC
          :summary:
   * - :py:obj:`LAX_FRIEDRICHS <astronomix.option_classes.simulation_config.LAX_FRIEDRICHS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.LAX_FRIEDRICHS
          :summary:
   * - :py:obj:`HLLD <astronomix.option_classes.simulation_config.HLLD>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLD
          :summary:
   * - :py:obj:`HLL5R <astronomix.option_classes.simulation_config.HLL5R>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLL5R
          :summary:
   * - :py:obj:`GLM <astronomix.option_classes.simulation_config.GLM>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GLM
          :summary:
   * - :py:obj:`CONSTRAINED_TRANSPORT <astronomix.option_classes.simulation_config.CONSTRAINED_TRANSPORT>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.CONSTRAINED_TRANSPORT
          :summary:
   * - :py:obj:`INDUCTION_SOURCE <astronomix.option_classes.simulation_config.INDUCTION_SOURCE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.INDUCTION_SOURCE
          :summary:
   * - :py:obj:`PPERMUTE <astronomix.option_classes.simulation_config.PPERMUTE>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PPERMUTE
          :summary:
   * - :py:obj:`PAIR_ALL_TO_ALL <astronomix.option_classes.simulation_config.PAIR_ALL_TO_ALL>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PAIR_ALL_TO_ALL
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
   * - :py:obj:`ATHENA_SECOND_ORDER <astronomix.option_classes.simulation_config.ATHENA_SECOND_ORDER>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.ATHENA_SECOND_ORDER
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
   * - :py:obj:`POLYTROPIC <astronomix.option_classes.simulation_config.POLYTROPIC>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.POLYTROPIC
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
   * - :py:obj:`FV_TIME_INTEGRATORS <astronomix.option_classes.simulation_config.FV_TIME_INTEGRATORS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FV_TIME_INTEGRATORS
          :summary:
   * - :py:obj:`FD_TIME_INTEGRATORS <astronomix.option_classes.simulation_config.FD_TIME_INTEGRATORS>`
     - .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FD_TIME_INTEGRATORS
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

.. py:data:: MAXIMUM_STEP_RETRIES
   :canonical: astronomix.option_classes.simulation_config.MAXIMUM_STEP_RETRIES
   :value: 6

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MAXIMUM_STEP_RETRIES

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

.. py:data:: RK2_SSP
   :canonical: astronomix.option_classes.simulation_config.RK2_SSP
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RK2_SSP

.. py:data:: MUSCL_HANCOCK
   :canonical: astronomix.option_classes.simulation_config.MUSCL_HANCOCK
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MUSCL_HANCOCK

.. py:data:: VL2
   :canonical: astronomix.option_classes.simulation_config.VL2
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VL2

.. py:data:: SSPRK54
   :canonical: astronomix.option_classes.simulation_config.SSPRK54
   :value: 10

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SSPRK54

.. py:data:: SSPRK33
   :canonical: astronomix.option_classes.simulation_config.SSPRK33
   :value: 11

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SSPRK33

.. py:data:: LSRK4
   :canonical: astronomix.option_classes.simulation_config.LSRK4
   :value: 12

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.LSRK4

.. py:data:: WENO_FLUX_SPLITTING
   :canonical: astronomix.option_classes.simulation_config.WENO_FLUX_SPLITTING
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.WENO_FLUX_SPLITTING

.. py:data:: RIEMANN_FLUX
   :canonical: astronomix.option_classes.simulation_config.RIEMANN_FLUX
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RIEMANN_FLUX

.. py:data:: RIEMANN_BY_EOS
   :canonical: astronomix.option_classes.simulation_config.RIEMANN_BY_EOS
   :value: None

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.RIEMANN_BY_EOS

.. py:data:: HLLEM_T
   :canonical: astronomix.option_classes.simulation_config.HLLEM_T
   :value: 5

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLEM_T

.. py:data:: HLLD_BOUNDED
   :canonical: astronomix.option_classes.simulation_config.HLLD_BOUNDED
   :value: 6

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLD_BOUNDED

.. py:data:: DONOR_CELL
   :canonical: astronomix.option_classes.simulation_config.DONOR_CELL
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DONOR_CELL

.. py:data:: PIECEWISE_LINEAR
   :canonical: astronomix.option_classes.simulation_config.PIECEWISE_LINEAR
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PIECEWISE_LINEAR

.. py:data:: MINMOD
   :canonical: astronomix.option_classes.simulation_config.MINMOD
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.MINMOD

.. py:data:: VAN_LEER
   :canonical: astronomix.option_classes.simulation_config.VAN_LEER
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.VAN_LEER

.. py:data:: SUPERBEE
   :canonical: astronomix.option_classes.simulation_config.SUPERBEE
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SUPERBEE

.. py:data:: DOUBLE_MINMOD
   :canonical: astronomix.option_classes.simulation_config.DOUBLE_MINMOD
   :value: 3

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DOUBLE_MINMOD

.. py:data:: HLL
   :canonical: astronomix.option_classes.simulation_config.HLL
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLL

.. py:data:: HLLC
   :canonical: astronomix.option_classes.simulation_config.HLLC
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLC

.. py:data:: LAX_FRIEDRICHS
   :canonical: astronomix.option_classes.simulation_config.LAX_FRIEDRICHS
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.LAX_FRIEDRICHS

.. py:data:: HLLD
   :canonical: astronomix.option_classes.simulation_config.HLLD
   :value: 3

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLLD

.. py:data:: HLL5R
   :canonical: astronomix.option_classes.simulation_config.HLL5R
   :value: 4

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.HLL5R

.. py:data:: GLM
   :canonical: astronomix.option_classes.simulation_config.GLM
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GLM

.. py:data:: CONSTRAINED_TRANSPORT
   :canonical: astronomix.option_classes.simulation_config.CONSTRAINED_TRANSPORT
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.CONSTRAINED_TRANSPORT

.. py:data:: INDUCTION_SOURCE
   :canonical: astronomix.option_classes.simulation_config.INDUCTION_SOURCE
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.INDUCTION_SOURCE

.. py:data:: PPERMUTE
   :canonical: astronomix.option_classes.simulation_config.PPERMUTE
   :value: 0

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PPERMUTE

.. py:data:: PAIR_ALL_TO_ALL
   :canonical: astronomix.option_classes.simulation_config.PAIR_ALL_TO_ALL
   :value: 1

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PAIR_ALL_TO_ALL

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

.. py:data:: ATHENA_SECOND_ORDER
   :canonical: astronomix.option_classes.simulation_config.ATHENA_SECOND_ORDER
   :value: 3

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.ATHENA_SECOND_ORDER

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

.. py:data:: POLYTROPIC
   :canonical: astronomix.option_classes.simulation_config.POLYTROPIC
   :value: 2

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.POLYTROPIC

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

   .. py:attribute:: poisson_discrete_laplacian
      :canonical: astronomix.option_classes.simulation_config.GravityConfig.poisson_discrete_laplacian
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GravityConfig.poisson_discrete_laplacian

   .. py:attribute:: time_centred_energy_source
      :canonical: astronomix.option_classes.simulation_config.GravityConfig.time_centred_energy_source
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.GravityConfig.time_centred_energy_source

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

   .. py:attribute:: clamp_in_estimates
      :canonical: astronomix.option_classes.simulation_config.PositivityConfig.clamp_in_estimates
      :type: bool
      :value: True

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PositivityConfig.clamp_in_estimates

   .. py:attribute:: step_retry
      :canonical: astronomix.option_classes.simulation_config.PositivityConfig.step_retry
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PositivityConfig.step_retry

.. py:class:: BackendConfig
   :canonical: astronomix.option_classes.simulation_config.BackendConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BackendConfig

   .. py:attribute:: backend
      :canonical: astronomix.option_classes.simulation_config.BackendConfig.backend
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BackendConfig.backend

   .. py:attribute:: pallas_block_shape
      :canonical: astronomix.option_classes.simulation_config.BackendConfig.pallas_block_shape
      :type: typing.Optional[typing.Tuple[int, int, int]]
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BackendConfig.pallas_block_shape

   .. py:attribute:: pallas_use_triton
      :canonical: astronomix.option_classes.simulation_config.BackendConfig.pallas_use_triton
      :type: bool
      :value: True

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BackendConfig.pallas_use_triton

   .. py:attribute:: pallas_interpret
      :canonical: astronomix.option_classes.simulation_config.BackendConfig.pallas_interpret
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BackendConfig.pallas_interpret

   .. py:attribute:: pallas_num_warps
      :canonical: astronomix.option_classes.simulation_config.BackendConfig.pallas_num_warps
      :type: int
      :value: 4

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BackendConfig.pallas_num_warps

   .. py:attribute:: pallas_ct
      :canonical: astronomix.option_classes.simulation_config.BackendConfig.pallas_ct
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BackendConfig.pallas_ct

   .. py:attribute:: use_approximate_rsqrt
      :canonical: astronomix.option_classes.simulation_config.BackendConfig.use_approximate_rsqrt
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BackendConfig.use_approximate_rsqrt

   .. py:attribute:: halo_exchange
      :canonical: astronomix.option_classes.simulation_config.BackendConfig.halo_exchange
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BackendConfig.halo_exchange

   .. py:attribute:: pallas_adjoint
      :canonical: astronomix.option_classes.simulation_config.BackendConfig.pallas_adjoint
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.BackendConfig.pallas_adjoint

.. py:class:: FVSolverConfig
   :canonical: astronomix.option_classes.simulation_config.FVSolverConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FVSolverConfig

   .. py:attribute:: time_integrator
      :canonical: astronomix.option_classes.simulation_config.FVSolverConfig.time_integrator
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FVSolverConfig.time_integrator

   .. py:attribute:: reconstruction
      :canonical: astronomix.option_classes.simulation_config.FVSolverConfig.reconstruction
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FVSolverConfig.reconstruction

   .. py:attribute:: limiter
      :canonical: astronomix.option_classes.simulation_config.FVSolverConfig.limiter
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FVSolverConfig.limiter

   .. py:attribute:: riemann_solver
      :canonical: astronomix.option_classes.simulation_config.FVSolverConfig.riemann_solver
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FVSolverConfig.riemann_solver

   .. py:attribute:: divergence_control
      :canonical: astronomix.option_classes.simulation_config.FVSolverConfig.divergence_control
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FVSolverConfig.divergence_control

   .. py:attribute:: parabolic_cleaning
      :canonical: astronomix.option_classes.simulation_config.FVSolverConfig.parabolic_cleaning
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FVSolverConfig.parabolic_cleaning

   .. py:attribute:: glm_extended_source
      :canonical: astronomix.option_classes.simulation_config.FVSolverConfig.glm_extended_source
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FVSolverConfig.glm_extended_source

   .. py:attribute:: first_order_flux_correction
      :canonical: astronomix.option_classes.simulation_config.FVSolverConfig.first_order_flux_correction
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FVSolverConfig.first_order_flux_correction

.. py:class:: FDSolverConfig
   :canonical: astronomix.option_classes.simulation_config.FDSolverConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig

   .. py:attribute:: scheme
      :canonical: astronomix.option_classes.simulation_config.FDSolverConfig.scheme
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig.scheme

   .. py:attribute:: time_integrator
      :canonical: astronomix.option_classes.simulation_config.FDSolverConfig.time_integrator
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig.time_integrator

   .. py:attribute:: weno_z
      :canonical: astronomix.option_classes.simulation_config.FDSolverConfig.weno_z
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig.weno_z

   .. py:attribute:: weno_admissible_face_state
      :canonical: astronomix.option_classes.simulation_config.FDSolverConfig.weno_admissible_face_state
      :type: bool
      :value: True

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig.weno_admissible_face_state

   .. py:attribute:: weno_positivity_preserving
      :canonical: astronomix.option_classes.simulation_config.FDSolverConfig.weno_positivity_preserving
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig.weno_positivity_preserving

   .. py:attribute:: weno_ad_frozen_weights
      :canonical: astronomix.option_classes.simulation_config.FDSolverConfig.weno_ad_frozen_weights
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig.weno_ad_frozen_weights

   .. py:attribute:: ct_upwind_dissipation
      :canonical: astronomix.option_classes.simulation_config.FDSolverConfig.ct_upwind_dissipation
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig.ct_upwind_dissipation

   .. py:attribute:: ct_upwind_strength
      :canonical: astronomix.option_classes.simulation_config.FDSolverConfig.ct_upwind_strength
      :type: float
      :value: 0.03

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig.ct_upwind_strength

   .. py:attribute:: dual_energy
      :canonical: astronomix.option_classes.simulation_config.FDSolverConfig.dual_energy
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig.dual_energy

   .. py:attribute:: riemann_solver
      :canonical: astronomix.option_classes.simulation_config.FDSolverConfig.riemann_solver
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FDSolverConfig.riemann_solver

.. py:class:: PassiveScalarConfig
   :canonical: astronomix.option_classes.simulation_config.PassiveScalarConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PassiveScalarConfig

   .. py:attribute:: num_scalars
      :canonical: astronomix.option_classes.simulation_config.PassiveScalarConfig.num_scalars
      :type: int
      :value: 0

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PassiveScalarConfig.num_scalars

   .. py:attribute:: bounds
      :canonical: astronomix.option_classes.simulation_config.PassiveScalarConfig.bounds
      :type: typing.Tuple[typing.Optional[typing.Tuple[float, float]], ...]
      :value: ()

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PassiveScalarConfig.bounds

   .. py:attribute:: cfl
      :canonical: astronomix.option_classes.simulation_config.PassiveScalarConfig.cfl
      :type: float
      :value: 0.4

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PassiveScalarConfig.cfl

   .. py:attribute:: max_substeps
      :canonical: astronomix.option_classes.simulation_config.PassiveScalarConfig.max_substeps
      :type: int
      :value: 8

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.PassiveScalarConfig.max_substeps

.. py:class:: DiffusionConfig
   :canonical: astronomix.option_classes.simulation_config.DiffusionConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DiffusionConfig

   .. py:attribute:: viscosity
      :canonical: astronomix.option_classes.simulation_config.DiffusionConfig.viscosity
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DiffusionConfig.viscosity

   .. py:attribute:: viscosity_type
      :canonical: astronomix.option_classes.simulation_config.DiffusionConfig.viscosity_type
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DiffusionConfig.viscosity_type

   .. py:attribute:: resistivity
      :canonical: astronomix.option_classes.simulation_config.DiffusionConfig.resistivity
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DiffusionConfig.resistivity

   .. py:attribute:: thermal_conduction
      :canonical: astronomix.option_classes.simulation_config.DiffusionConfig.thermal_conduction
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DiffusionConfig.thermal_conduction

   .. py:attribute:: conduction_density_weighted
      :canonical: astronomix.option_classes.simulation_config.DiffusionConfig.conduction_density_weighted
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DiffusionConfig.conduction_density_weighted

   .. py:attribute:: conduction_order
      :canonical: astronomix.option_classes.simulation_config.DiffusionConfig.conduction_order
      :type: int
      :value: 2

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.DiffusionConfig.conduction_order

.. py:class:: SimulationConfig
   :canonical: astronomix.option_classes.simulation_config.SimulationConfig

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig

   .. py:attribute:: dimensionality
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.dimensionality
      :type: int
      :value: 1

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.dimensionality

   .. py:attribute:: geometry
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.geometry
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.geometry

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

   .. py:attribute:: boundary_settings
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.boundary_settings
      :type: typing.Union[types.NoneType, astronomix.option_classes.simulation_config.BoundarySettings1D, astronomix.option_classes.simulation_config.BoundarySettings]
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.boundary_settings

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

   .. py:attribute:: solver_mode
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.solver_mode
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.solver_mode

   .. py:attribute:: solver_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.solver_config
      :type: typing.Union[types.NoneType, astronomix.option_classes.simulation_config.FVSolverConfig, astronomix.option_classes.simulation_config.FDSolverConfig]
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.solver_config

   .. py:attribute:: positivity_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.positivity_config
      :type: astronomix.option_classes.simulation_config.PositivityConfig
      :value: 'PositivityConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.positivity_config

   .. py:attribute:: backend_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.backend_config
      :type: astronomix.option_classes.simulation_config.BackendConfig
      :value: 'BackendConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.backend_config

   .. py:attribute:: gravity_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.gravity_config
      :type: astronomix.option_classes.simulation_config.GravityConfig
      :value: 'GravityConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.gravity_config

   .. py:attribute:: diffusion_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.diffusion_config
      :type: astronomix.option_classes.simulation_config.DiffusionConfig
      :value: 'DiffusionConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.diffusion_config

   .. py:attribute:: passive_scalar_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.passive_scalar_config
      :type: astronomix.option_classes.simulation_config.PassiveScalarConfig
      :value: 'PassiveScalarConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.passive_scalar_config

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

   .. py:attribute:: turbulence_driving_config
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.turbulence_driving_config
      :type: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig
      :value: 'TurbulenceDrivingConfig(...)'

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.turbulence_driving_config

   .. py:attribute:: per_step_update
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.per_step_update
      :type: typing.Callable | None
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.per_step_update

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

   .. py:attribute:: num_timesteps
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.num_timesteps
      :type: int
      :value: 1000

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.num_timesteps

   .. py:attribute:: differentiation_mode
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.differentiation_mode
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.differentiation_mode

   .. py:attribute:: rematerialize_stages
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.rematerialize_stages
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.rematerialize_stages

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

   .. py:attribute:: memory_analysis
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.memory_analysis
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.memory_analysis

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

   .. py:attribute:: num_ghost_cells
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.num_ghost_cells
      :type: int
      :value: 2

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

   .. py:attribute:: source_term_aware_timestep
      :canonical: astronomix.option_classes.simulation_config.SimulationConfig.source_term_aware_timestep
      :type: bool
      :value: False

      .. autodoc2-docstring:: astronomix.option_classes.simulation_config.SimulationConfig.source_term_aware_timestep

.. py:data:: FV_TIME_INTEGRATORS
   :canonical: astronomix.option_classes.simulation_config.FV_TIME_INTEGRATORS
   :value: ()

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FV_TIME_INTEGRATORS

.. py:data:: FD_TIME_INTEGRATORS
   :canonical: astronomix.option_classes.simulation_config.FD_TIME_INTEGRATORS
   :value: ()

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.FD_TIME_INTEGRATORS

.. py:function:: get_solver_config(config: astronomix.option_classes.simulation_config.SimulationConfig) -> typing.Union[astronomix.option_classes.simulation_config.FVSolverConfig, astronomix.option_classes.simulation_config.FDSolverConfig]
   :canonical: astronomix.option_classes.simulation_config.get_solver_config

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.get_solver_config

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

.. py:function:: config_to_string(config: astronomix.option_classes.simulation_config.SimulationConfig) -> str
   :canonical: astronomix.option_classes.simulation_config.config_to_string

   .. autodoc2-docstring:: astronomix.option_classes.simulation_config.config_to_string
