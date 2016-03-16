High Performance toolbox for Extreme Learning Machines.
--------

Extreme learning machines (ELM) are a particular kind of Artificial Neural Networks,
which solve classification and regression problems. Their performance is comparable
to a classical Multilayer Perceptron trained with Error Back-Propagation algorithm,
but the training time is up to 6 orders of magnitude smaller. (yes, a million times!)

ELMs are suitable for processing huge datasets and dealing with Big Data,
and this toolbox is created as their fastest and most scalable implementation.

Documentation is available here: http://hpelm.readthedocs.org, 
it uses Numpydocs.

NEW: Parallel HP-ELM tutorial! See the documentation: http://hpelm.readthedocs.org

Highlights:
    - Efficient matrix math implementation without bottlenecks
    - Efficient data storage (HDF5 file format)
    - Data size not limited by the available memory
    - GPU accelerated computations (if you have one)
    - Regularization and model selection (for in-memory models)

Main classes:
    - hpelm.ELM for in-memory computations (dataset fits into RAM)
    - hpelm.HPELM for out-of-memory computations (dataset on disk in HDF5 format)

Example usage::
    >>> from hpelm import ELM
    >>> elm = ELM(X.shape[1], T.shape[1])
    >>> elm.add_neurons(20, "sigm")
    >>> elm.add_neurons(10, "rbf_l2")
    >>> elm.train(X, T, "LOO")
    >>> Y = elm.predict(X)

If you use the toolbox, cite our open access paper "High Performance Extreme Learning Machines: A Complete Toolbox for Big Data Applications" in IEEE Access.
http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7140733&newsearch=true&queryText=High%20Performance%20Extreme%20Learning%20Machines

@ARTICLE{7140733,
author={Akusok, A. and Bj\"{o}rk, K.-M. and Miche, Y. and Lendasse, A.},
journal={Access, IEEE},
title={High-Performance Extreme Learning Machines: A Complete Toolbox for Big Data Applications},
year={2015},
volume={3},
pages={1011-1025},
doi={10.1109/ACCESS.2015.2450498},
ISSN={2169-3536},
month={},}