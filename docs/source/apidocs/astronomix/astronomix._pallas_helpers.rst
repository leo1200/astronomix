:orphan:

:py:mod:`astronomix._pallas_helpers`
====================================

.. py:module:: astronomix._pallas_helpers

.. autodoc2-docstring:: astronomix._pallas_helpers
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`pallas_mesh_context <astronomix._pallas_helpers.pallas_mesh_context>`
     - .. autodoc2-docstring:: astronomix._pallas_helpers.pallas_mesh_context
          :summary:
   * - :py:obj:`sharded_roll <astronomix._pallas_helpers.sharded_roll>`
     - .. autodoc2-docstring:: astronomix._pallas_helpers.sharded_roll
          :summary:
   * - :py:obj:`diffable_pallas_call <astronomix._pallas_helpers.diffable_pallas_call>`
     - .. autodoc2-docstring:: astronomix._pallas_helpers.diffable_pallas_call
          :summary:
   * - :py:obj:`diffable_pallas_call_n <astronomix._pallas_helpers.diffable_pallas_call_n>`
     - .. autodoc2-docstring:: astronomix._pallas_helpers.diffable_pallas_call_n
          :summary:
   * - :py:obj:`pallas_vjp_call <astronomix._pallas_helpers.pallas_vjp_call>`
     - .. autodoc2-docstring:: astronomix._pallas_helpers.pallas_vjp_call
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DEFAULT_BLOCK_CELLS <astronomix._pallas_helpers.DEFAULT_BLOCK_CELLS>`
     - .. autodoc2-docstring:: astronomix._pallas_helpers.DEFAULT_BLOCK_CELLS
          :summary:

API
~~~

.. py:data:: DEFAULT_BLOCK_CELLS
   :canonical: astronomix._pallas_helpers.DEFAULT_BLOCK_CELLS
   :value: 128

   .. autodoc2-docstring:: astronomix._pallas_helpers.DEFAULT_BLOCK_CELLS

.. py:function:: pallas_mesh_context(mesh, spec=None, halo_exchange=PAIR_ALL_TO_ALL)
   :canonical: astronomix._pallas_helpers.pallas_mesh_context

   .. autodoc2-docstring:: astronomix._pallas_helpers.pallas_mesh_context

.. py:function:: sharded_roll(x, shift: int, axis: int)
   :canonical: astronomix._pallas_helpers.sharded_roll

   .. autodoc2-docstring:: astronomix._pallas_helpers.sharded_roll

.. py:function:: diffable_pallas_call(state, params, *, pallas_branch, native_branch)
   :canonical: astronomix._pallas_helpers.diffable_pallas_call

   .. autodoc2-docstring:: astronomix._pallas_helpers.diffable_pallas_call

.. py:function:: diffable_pallas_call_n(primals, *, pallas_branch, native_branch)
   :canonical: astronomix._pallas_helpers.diffable_pallas_call_n

   .. autodoc2-docstring:: astronomix._pallas_helpers.diffable_pallas_call_n

.. py:function:: pallas_vjp_call(state, aux, *, pallas_forward, pallas_backward)
   :canonical: astronomix._pallas_helpers.pallas_vjp_call

   .. autodoc2-docstring:: astronomix._pallas_helpers.pallas_vjp_call
