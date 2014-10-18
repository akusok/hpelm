High Performance ELM
--------

Extreme Learning Machine (ELM) with regularization for very large models.
 
Uses MPI for distributed computing, iterative computation for limiting memory 
consumption, and fast system solvers.


Example usage::

    >>> from hpelm import ELM
    >>> error = ELM(X,Y)