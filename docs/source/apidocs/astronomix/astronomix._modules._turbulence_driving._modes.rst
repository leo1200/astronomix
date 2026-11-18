:orphan:

:py:mod:`astronomix._modules._turbulence_driving._modes`
========================================================

.. py:module:: astronomix._modules._turbulence_driving._modes

.. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DrivingModes <astronomix._modules._turbulence_driving._modes.DrivingModes>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.DrivingModes
          :summary:
   * - :py:obj:`SeparableLattice <astronomix._modules._turbulence_driving._modes.SeparableLattice>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.SeparableLattice
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`driving_wavenumber_range <astronomix._modules._turbulence_driving._modes.driving_wavenumber_range>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.driving_wavenumber_range
          :summary:
   * - :py:obj:`build_driving_modes <astronomix._modules._turbulence_driving._modes.build_driving_modes>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.build_driving_modes
          :summary:
   * - :py:obj:`separable_lattice <astronomix._modules._turbulence_driving._modes.separable_lattice>`
     - .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.separable_lattice
          :summary:

API
~~~

.. py:class:: DrivingModes
   :canonical: astronomix._modules._turbulence_driving._modes.DrivingModes

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.DrivingModes

   .. py:attribute:: wave_vectors
      :canonical: astronomix._modules._turbulence_driving._modes.DrivingModes.wave_vectors
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.DrivingModes.wave_vectors

   .. py:attribute:: amplitudes
      :canonical: astronomix._modules._turbulence_driving._modes.DrivingModes.amplitudes
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.DrivingModes.amplitudes

.. py:class:: SeparableLattice
   :canonical: astronomix._modules._turbulence_driving._modes.SeparableLattice

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.SeparableLattice

   .. py:attribute:: wave_numbers
      :canonical: astronomix._modules._turbulence_driving._modes.SeparableLattice.wave_numbers
      :type: typing.Tuple[numpy.ndarray, ...]
      :value: None

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.SeparableLattice.wave_numbers

   .. py:attribute:: index
      :canonical: astronomix._modules._turbulence_driving._modes.SeparableLattice.index
      :type: typing.Tuple[numpy.ndarray, ...]
      :value: None

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.SeparableLattice.index

   .. py:attribute:: mirrored
      :canonical: astronomix._modules._turbulence_driving._modes.SeparableLattice.mirrored
      :type: numpy.ndarray
      :value: None

      .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.SeparableLattice.mirrored

.. py:function:: driving_wavenumber_range(config: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig) -> typing.Tuple[float, float]
   :canonical: astronomix._modules._turbulence_driving._modes.driving_wavenumber_range

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.driving_wavenumber_range

.. py:function:: build_driving_modes(config: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig) -> astronomix._modules._turbulence_driving._modes.DrivingModes
   :canonical: astronomix._modules._turbulence_driving._modes.build_driving_modes

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.build_driving_modes

.. py:function:: separable_lattice(config: astronomix._modules._turbulence_driving._turbulence_driving_options.TurbulenceDrivingConfig) -> astronomix._modules._turbulence_driving._modes.SeparableLattice
   :canonical: astronomix._modules._turbulence_driving._modes.separable_lattice

   .. autodoc2-docstring:: astronomix._modules._turbulence_driving._modes.separable_lattice
