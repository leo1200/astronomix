:py:mod:`astronomix.parallel.distributed`
=========================================

.. py:module:: astronomix.parallel.distributed

.. autodoc2-docstring:: astronomix.parallel.distributed
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DistInfo <astronomix.parallel.distributed.DistInfo>`
     - .. autodoc2-docstring:: astronomix.parallel.distributed.DistInfo
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`init_distributed <astronomix.parallel.distributed.init_distributed>`
     - .. autodoc2-docstring:: astronomix.parallel.distributed.init_distributed
          :summary:

API
~~~

.. py:class:: DistInfo
   :canonical: astronomix.parallel.distributed.DistInfo

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix.parallel.distributed.DistInfo

   .. py:attribute:: process_count
      :canonical: astronomix.parallel.distributed.DistInfo.process_count
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.parallel.distributed.DistInfo.process_count

   .. py:attribute:: process_index
      :canonical: astronomix.parallel.distributed.DistInfo.process_index
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.parallel.distributed.DistInfo.process_index

   .. py:attribute:: local_device_count
      :canonical: astronomix.parallel.distributed.DistInfo.local_device_count
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.parallel.distributed.DistInfo.local_device_count

   .. py:attribute:: global_device_count
      :canonical: astronomix.parallel.distributed.DistInfo.global_device_count
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix.parallel.distributed.DistInfo.global_device_count

   .. py:attribute:: is_distributed
      :canonical: astronomix.parallel.distributed.DistInfo.is_distributed
      :type: bool
      :value: None

      .. autodoc2-docstring:: astronomix.parallel.distributed.DistInfo.is_distributed

.. py:function:: init_distributed(verbose: bool = True) -> astronomix.parallel.distributed.DistInfo
   :canonical: astronomix.parallel.distributed.init_distributed

   .. autodoc2-docstring:: astronomix.parallel.distributed.init_distributed
