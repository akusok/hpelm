.. _parallel:

Running HP-ELM in parallel
==========================


An ELM model is very easy to run in parallel. Its solution has two main steps: compute helper matrices :math:`HH`
and :math:`HT` (99% runtime for large dataset and many hidden neurons) and solve output matrix :math:`B` from :math:`HH` 
and :math:`HT` (1% runtime). Partial matrices :math:`HH^p` and :math:`HT^p` are computed from different parts of input data
independently, and then simply summed together: :math:`HH = HH^1 + HH^2 + ... + HH^n`, :math:`HT = HT^1 + HT^2 + ... + HT^n`.
The final solution of :math:`B` cannot be easily split across multiple computers, but it is very fast anyway.

.. note::
    On a single computer HP-ELM already uses all the cores. Parallel HP-ELM takes advantage of distributing work across
    multiple machines, for instance on a computer cluster.


An example of running HP-ELM in parallel is given below. Separate code blocks are in different files.

1. Put data on a disk in HDF5 format. For example:

    .. code:: python

        X = np.random.rand(1000, 10)
        T = np.random.rand(1000, 3)
        hX = modules.make_hdf5(X, "dataX.hdf5")
        hT = modules.make_hdf5(T, "dataT.hdf5")

2. Create an HP-ELM model with neurons, and save it to a file.

    .. code:: python

        model0 = HPELM(10, 3)
        model0.add_neurons(10, 'lin')
        model0.add_neurons(5, 'tanh')
        model0.add_neurons(15, 'sigm')
        model0.save("fmodel.h5")

3. Compute partial matrices :math:`HH^p, HT^p` on different machines in parallel by running different Python scripts. 
   All scripts can read data from the same data files (then you need to set parameters `istart` and `icount` that
   specify where to start reading data and how many rows to read). Scripts can also read data from separate
   files which you have prepared and distributed, or even from the given Numpy matrices (not sure about that :).
   
   All scripts write their partial matrices :math:`HH^p, HT^p` to the same files on disk, incrementing existing data
   in these files. Writes are multiprocess-safe using file locks (from `fasteners` library). HP-ELM will create starting 
   files with zero matrices :math:`HH, HT` for you if they don't exist yet.

   .. note::
        The folder where :math:`HH, HT` files are located must be writable by all parallel scripts, because they use
        auxiliary files as write locks.

   .. code:: python

        model1 = HPELM(10, 3)
        model1.load("model.pkl")
        model1.add_data("dataX.hdf5", "dataT.hdf5", istart=0, icount=100, fHH="HH.hdf5", fHT="HT.hdf5")

   .. code:: python

        model2 = HPELM(10, 3)
        model2.load("model.pkl")
        model2.add_data("dataX.hdf5", "dataT.hdf5", istart=100, icount=100, fHH="HH.hdf5", fHT="HT.hdf5")

   .. code:: python

        model3 = HPELM(10, 3)
        model3.load("model.pkl")
        model3.add_data("dataX.hdf5", "dataT.hdf5", istart=200, icount=800, fHH="HH.hdf5", fHT="HT.hdf5")

4. Run final solution step on one machine, and save the trained model. You can then get your predictions.

   .. code:: python

        model4 = HPELM(10, 3)
        model4.load("model.pkl")
        model4.solve_corr("HH.hdf5", "HT.hdf5")
        model4.save("model.pkl")

   .. code:: python

        model5 = HPELM(10, 3)
        model5.load("model.pkl")
        model5.predict("dataX.hdf5", "predictedY.hdf5")
        err_train = model5.error("dataX.hdf5", "predictedY.hdf5")
        print "Training error is", err_train
