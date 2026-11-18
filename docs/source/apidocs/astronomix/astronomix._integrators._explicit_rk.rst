:orphan:

:py:mod:`astronomix._integrators._explicit_rk`
==============================================

.. py:module:: astronomix._integrators._explicit_rk

.. autodoc2-docstring:: astronomix._integrators._explicit_rk
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ssprk4 <astronomix._integrators._explicit_rk.ssprk4>`
     - .. autodoc2-docstring:: astronomix._integrators._explicit_rk.ssprk4
          :summary:
   * - :py:obj:`ssprk33 <astronomix._integrators._explicit_rk.ssprk33>`
     - .. autodoc2-docstring:: astronomix._integrators._explicit_rk.ssprk33
          :summary:
   * - :py:obj:`lsrk4 <astronomix._integrators._explicit_rk.lsrk4>`
     - .. autodoc2-docstring:: astronomix._integrators._explicit_rk.lsrk4
          :summary:

API
~~~

.. py:function:: ssprk4(u0, dt, *, rhs, pre_stage=_identity, post_stage=_identity, finalize=_identity)
   :canonical: astronomix._integrators._explicit_rk.ssprk4

   .. autodoc2-docstring:: astronomix._integrators._explicit_rk.ssprk4

.. py:function:: ssprk33(u0, dt, *, rhs, pre_stage=_identity, post_stage=_identity, finalize=_identity)
   :canonical: astronomix._integrators._explicit_rk.ssprk33

   .. autodoc2-docstring:: astronomix._integrators._explicit_rk.ssprk33

.. py:function:: lsrk4(u0, dt, *, pre_stage=_identity, finalize=_identity, rhs=None, lsrk_increment=None)
   :canonical: astronomix._integrators._explicit_rk.lsrk4

   .. autodoc2-docstring:: astronomix._integrators._explicit_rk.lsrk4
