High Performance toolbox for Extreme Learning Machines
--------

Extreme Learning Machine (ELM) is a machine learning model universally suitable for classification and regression 
problems. It includes one or several types of hidden neurons concatenated together into the hidden neuron layer.
Each neuron type has its own connection to input layer (dense, sparse or pairwise function based), and an element-wise
transformation function applied on hidden layer output that is usually non-linear and bound. ELM model also includes
a linear solver for the output weights, with several options and multiple parameters available: batch solvers,
L2 and L1 regularization, iterative addition and removal ("forgetting") or training data samples, Lanczos finite
iterative solvers, GPU-accelerated solvers, and distributed solvers.

ELM toolbox supports export of trained models into Scikit-Learn compatible format for inference, 
and Scikit-Learn compatible models for training with limited solver options 
(and reduced performance at very large tasks).

The main feature of ELM are randomly selected and fixed parameters of hidden neurons that never change. 
This provides an explicit solution for output weights, unlike in traditional neural networks that have to be solved 
iteratively as optimal weights of input and output layers depend on each other. 
ELM performance is comparable to a classical Multilayer Perceptron trained with error back-propagation algorithm, 
but explicit solution reduces training time by up to 6 orders of magnitude. (*yes, a million times!*)

ELMs are suitable for processing huge datasets and dealing with Big Data,
and this toolbox is created as their fastest and most scalable implementation.

Documentation is available here: http://hpelm.readthedocs.org, it uses Numpydocs.

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
```python
    from hpelm import ELM
    elm = ELM(X.shape[1], T.shape[1])
    elm.add_neurons(20, "sigm")
    elm.add_neurons(10, "rbf_l2")
    elm.train(X, T, "LOO")
    Y = elm.predict(X)
```

If you use the toolbox, cite our open access paper
["High Performance Extreme Learning Machines: A Complete Toolbox for Big Data Applications"](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7140733&newsearch=true&queryText=High%20Performance%20Extreme%20Learning%20Machines)
 in IEEE Access.


```text
@ARTICLE{7140733,
  author={Akusok, A. and Bj\"{o}rk, K.-M. and Miche, Y. and Lendasse, A.},
  journal={Access, IEEE},
  title={High-Performance Extreme Learning Machines: A Complete Toolbox for Big Data Applications},
  year={2015},
  volume={3},
  pages={1011-1025},
  doi={10.1109/ACCESS.2015.2450498},
  ISSN={2169-3536},
  month={},
}
```
