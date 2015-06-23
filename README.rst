High Performance ELM
--------

Extreme Learning Machine (ELM) with model selection and regularizations.

In-memory ELM works, check hpelm/tests folder.
MAGMA acceleration works, check hpelm/acc/setup_gpu.py.


Example usage::

    >>> from hpelm import ELM
    >>> elm = ELM(X.shape[1], T.shape[1])
    >>> elm.add_neurons(20, "sigm")
    >>> elm.add_neurons(10, "rbf_l2")
    >>> elm.train(X, T, "LOO")
    >>> Y = elm.predict(X)

If you use the toolbox, cite our paper that will be published in IEEE Access.