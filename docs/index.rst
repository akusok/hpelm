.. hpelm documentation master file, created by
   sphinx-quickstart on Sun Nov  1 20:13:17 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HP-ELM documentation
=================================

HP-ELM is a high performance Neural Network toolbox for solving large problems, especially
in Big Data. It supports datasets of any size and GPU acceleration, both with modest memory
consumption and fast non-iterative optimization. Train a neural network with 32000 neurons on MNIST
dataset in 1 minute on your desktop! [*]_

.. [*] Neural Network with 32000 sigmoid neurons trained on MNIST training set with 60000 samples,
       at Intel i7-4790K(4.6GHz), 16GB RAM, GTX Titan Black --- 62.8 seconds.


Extreme Learning Machines (ELM)
-------------------------------

Extreme Learning Machine is a training algorithm for Single hidden Layer Feed-forward Neural
networks (SLFN). It's distinctive feature is random selection of input weights, after which
the output weights are computed in one step. The one-step solution provides a huge speedup
(> x1000) compared to iterative training algorithms for SLFN like error back-propagation (BP),
also known as Multilayer Perceptron (MLP). ELM accuracy with default settings is comparable
to MLP, so it is a ready-available fast replacement for MLP for any practical purposes.

ELM follows an opposite "philosophy" to a popular Deep Learning methodology, that aims for
the best accuracy in the world in complex tasks at a cost of a very long training time,
easily x1,000,000 compared to ELM. Deep Learning also incurs great development costs
because it requires highly skilled scientific personnel for model tuning and many man-years
of work. While Deep Learning fits best for global challenges like machine translation and
self-driving vehicles, ELM is the best model for anything else: prototyping, any low-cost or
short-term projects, and for obtaining results on Big Data in reasonable time.





.. sidebar:: References for ELM

    * `Main website <http://www.ntu.edu.sg/home/egbhuang/>`_
    * Recent `surway <https://scholar.google.fi/scholar?cluster=10231859643125368934&hl=en&as_sdt=0,5>`_
    * `HPELM toolbox paper <http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7140733&punumber%3D6287639>`_

.. toctree::
    api/index




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

