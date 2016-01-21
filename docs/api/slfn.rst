SLFN Solvers
------------

Background solvers for Single Layer Feed-forward Network (SLFN) that do all heavy-lifting
computations, a separate solver accelerates computations for separate hardware (GPU, etc.).
Interface is defined by ``SLFN`` class.

Use different solvers by passing optional parameter ``accelerator`` to ``ELM`` or ``HPELM``.


.. automodule:: hpelm.nnets.slfn
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hpelm.nnets.slfn_python
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: hpelm.nnets.slfn_skcuda
    :members:
    :undoc-members:
    :show-inheritance:
