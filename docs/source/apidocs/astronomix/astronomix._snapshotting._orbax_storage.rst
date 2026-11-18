:orphan:

:py:mod:`astronomix._snapshotting._orbax_storage`
=================================================

.. py:module:: astronomix._snapshotting._orbax_storage

.. autodoc2-docstring:: astronomix._snapshotting._orbax_storage
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LoopCheckpoint <astronomix._snapshotting._orbax_storage.LoopCheckpoint>`
     - .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.LoopCheckpoint
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`loop_checkpointer <astronomix._snapshotting._orbax_storage.loop_checkpointer>`
     - .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.loop_checkpointer
          :summary:
   * - :py:obj:`save_loop_checkpoint <astronomix._snapshotting._orbax_storage.save_loop_checkpoint>`
     - .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.save_loop_checkpoint
          :summary:
   * - :py:obj:`latest_step <astronomix._snapshotting._orbax_storage.latest_step>`
     - .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.latest_step
          :summary:
   * - :py:obj:`load_loop_checkpoint <astronomix._snapshotting._orbax_storage.load_loop_checkpoint>`
     - .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.load_loop_checkpoint
          :summary:

API
~~~

.. py:class:: LoopCheckpoint
   :canonical: astronomix._snapshotting._orbax_storage.LoopCheckpoint

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.LoopCheckpoint

   .. py:attribute:: time
      :canonical: astronomix._snapshotting._orbax_storage.LoopCheckpoint.time
      :type: float
      :value: None

      .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.LoopCheckpoint.time

   .. py:attribute:: primitive_state
      :canonical: astronomix._snapshotting._orbax_storage.LoopCheckpoint.primitive_state
      :type: typing.Any
      :value: None

      .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.LoopCheckpoint.primitive_state

   .. py:attribute:: driving_state
      :canonical: astronomix._snapshotting._orbax_storage.LoopCheckpoint.driving_state
      :type: typing.Any
      :value: None

      .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.LoopCheckpoint.driving_state

   .. py:attribute:: num_iterations
      :canonical: astronomix._snapshotting._orbax_storage.LoopCheckpoint.num_iterations
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.LoopCheckpoint.num_iterations

   .. py:attribute:: step
      :canonical: astronomix._snapshotting._orbax_storage.LoopCheckpoint.step
      :type: int
      :value: None

      .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.LoopCheckpoint.step

.. py:function:: loop_checkpointer(directory)
   :canonical: astronomix._snapshotting._orbax_storage.loop_checkpointer

   .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.loop_checkpointer

.. py:function:: save_loop_checkpoint(writer: astronomix._snapshotting._orbax_storage._LoopCheckpointWriter, step: int, *, time, primitive_state, driving_state, num_iterations) -> None
   :canonical: astronomix._snapshotting._orbax_storage.save_loop_checkpoint

   .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.save_loop_checkpoint

.. py:function:: latest_step(directory) -> typing.Optional[int]
   :canonical: astronomix._snapshotting._orbax_storage.latest_step

   .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.latest_step

.. py:function:: load_loop_checkpoint(directory, step: typing.Optional[int] = None, *, sharding: typing.Optional[jax.sharding.Sharding] = None) -> astronomix._snapshotting._orbax_storage.LoopCheckpoint
   :canonical: astronomix._snapshotting._orbax_storage.load_loop_checkpoint

   .. autodoc2-docstring:: astronomix._snapshotting._orbax_storage.load_loop_checkpoint
