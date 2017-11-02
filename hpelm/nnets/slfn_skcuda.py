# -*- coding: utf-8 -*-
"""Nvidia GPU-accelerated solver based on Scikit-CUDA, works without compiling anything.

GPU computations run in asyncronous mode: GPU is processing one batch of data while CPU prepares
the next batch. Loads GPU for 100% without waiting times, very fast and efficient. The requied
Scikit-CUDA is a single-line install in Linux: ``pip install scikit-cuda``. Tested on CUDA 7.

Created on Sat Sep 12 13:10:23 2015
@author: akusok
"""

from slfn import SLFN
import numpy as np
from scipy.linalg import lapack
from scipy.spatial.distance import cdist

from pycuda import gpuarray, cumath, autoinit
from pycuda.compiler import SourceModule
from skcuda import linalg, misc, cublas



class SLFNSkCUDA(SLFN):
    """Single Layer Feed-forward Network (SLFN) implementation on GPU with pyCUDA.

    To choose a specific GPU, use environmental variable ``CUDA_DEVICE``, for exampe
    ``CUDA_DEVICE=0 python myscript1.py & CUDA_DEVICE=1 python myscript2.py``.

    In single precision, only upper triangular part of HH matrix is computed to speedup the method.
    """

    def __init__(self, inputs, outputs, norm=None, precision=np.float64):
        super(SLFNSkCUDA, self).__init__(inputs, outputs, norm, precision)

        # startup GPU
        #self.ctx = misc.init_context(misc.init_device(nDevice))  # NO NO NO, crashes and does not release memory
        # use CUDA_DEVICE=0 python my-script.py
        try:
            linalg.init()
        except OSError as e:
            pass  # no 'cusolver' library which is paid and not needed
            # print "error initializing scikit-cuda: %s" % e
            # print "ignore if toolbox works"

        # precision-dependent stuff
        if precision is np.float64:
            self.posv = lapack.dposv
        else:
            self.posv = lapack.sposv
            self.handle = cublas.cublasCreate()

        # prepare GPU function kernels
        kernel = """
            __global__ void dev_sigm(%s *a) {
                unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
                a[idx] = 1.0 / ( exp(a[idx]) + 1 );
            }
            """
        kernel = kernel % "double" if self.precision is np.float64 else kernel % "float"
        self.dev_sigm = SourceModule(kernel).get_function("dev_sigm")
        self.dev_sigm.prepare("P")

        # GPU transformation functions
        self.func["lin"] = self._dev_lin
        self.func["sigm"] = self._dev_sigm
        self.func["tanh"] = self._dev_tanh
        self.func["rbf_l1"] = self._dev_rbfl1
        self.func["rbf_l2"] = self._dev_rbfl2
        self.func["rbf_linf"] = self._dev_rbflinf

    def _dev_lin(self, devX, devW, devB):
        """Linear function on GPU.

        Returns:
            devH (gpuarray): GPU matrix with the result.
        """
        devH = misc.add_matvec(linalg.dot(devX, devW), devB, axis=1)
        return devH

    def _dev_sigm(self, devX, devW, devB):
        """Compute Sigmoid on GPU for a given array and return array."""

#        def sigm(a):
#            block = a._block
#            grid = (int(np.ceil(1.0 * np.prod(a.shape) / block[0])), 1)
#            dev_sigm.prepared_call(grid, block, a.gpudata)
#            return a

        devH = misc.add_matvec(linalg.dot(devX, devW), devB, axis=1)
        block = devH._block
        grid = (int(np.ceil(1.0 * np.prod(devH.shape) / block[0])), 1)
        self.dev_sigm.prepared_call(grid, block, devH.gpudata)
        return devH

    def _dev_tanh(self, devX, devW, devB):
        """Hyperbolic tangent function on GPU.

        Returns:
            devH (gpuarray): GPU matrix with the result.
        """
        devH = misc.add_matvec(linalg.dot(devX, devW), devB, axis=1)
        cumath.tanh(devH, out=devH)
        return devH

    def _dev_rbfl1(self, devX, devW, devB):
        # TODO: make proper GPU implementation of RBF_L1
        X = devX.get()
        W = devW.get()
        B = devB.get()
        devH = gpuarray.to_gpu(np.exp(-cdist(X, W.T, "cityblock")**2 / B))
        return devH

    def _dev_rbfl2(self, devX, devW, devB):
        # TODO: make proper GPU implementation of RBF_L2
        X = devX.get()
        W = devW.get()
        B = devB.get()
        devH = gpuarray.to_gpu(np.exp(-cdist(X, W.T, "euclidean")**2 / B))
        return devH

    def _dev_rbflinf(self, devX, devW, devB):
        # TODO: make proper GPU implementation of RBF_Linf
        X = devX.get()
        W = devW.get()
        B = devB.get()
        devH = gpuarray.to_gpu(np.exp(-cdist(X, W.T, "chebyshev")**2 / B))
        return devH

    def add_neurons(self, number, func, W, B):
        """Add prepared neurons to the SLFN, merge with existing ones.

        Adds a number of specific neurons to SLFN network. Weights and biases
        must be provided for that function.

        If neurons of such type already exist, they are merged together.

        Args:
            number (int): the number of new neurons to add
            func (str): transformation function of hidden layer. Linear function creates a linear model.
            W (matrix): a 2-D matrix of neuron weights, size (`inputs` * `number`)
            B (vector): a 1-D vector of neuron biases, size (`number` * 1)
        """
        ntypes = [nr[1] for nr in self.neurons]  # existing types of neurons
        if func in ntypes:
            # add to an existing neuron type
            i = ntypes.index(func)
            nn0, _, devW, devB = self.neurons[i]
            number = nn0 + number
            devW = gpuarray.to_gpu(np.hstack((devW.get(), W)))
            devB = gpuarray.to_gpu(np.hstack((devB.get(), B)))
            self.neurons[i] = (number, func, devW, devB)
        else:
            # create a new neuron type
            devW = gpuarray.to_gpu(W)
            devB = gpuarray.to_gpu(B)
            self.neurons.append((number, func, devW, devB))
        self.reset()
        self.B = None

    def reset(self):
        """ Resets intermediate training results, releases memory that they use.

        Keeps solution of ELM, so a trained ELM remains operational.
        Can be called to free memory after an ELM is trained.
        """
        self.L = sum([n[0] for n in self.neurons])  # get number of neurons
        self.HH = None
        self.HT = None

    def _project(self, X, dev=False):
        """Projects X to H, an auxiliary function that implements a particular projection.

        For actual projection, use `ELM.project()` instead.

        Args:
            X (matrix): an input data matrix, size (N * `inputs`)
            dev (bool, optional): whether leave result in the GPU memory

        Returns:
            H (matrix): an SLFN hidden layer representation, size (N * `L`) where 'L' is number of neurons
        """
        assert self.neurons is not None, "ELM has no neurons"
        X = np.array(X, order="C", dtype=self.precision)
        devX = gpuarray.to_gpu(X)
        devH = gpuarray.empty((X.shape[0], self.L), dtype=self.precision)
        i = 0
        for nn, ftype, devW, devB in self.neurons:
            devH[:, i:i+nn] = self.func[ftype](devX, devW, devB)
            i += nn

        H = devH if dev else devH.get()
        return H

    def _predict(self, X, dev=False):
        """Predict a batch of data. Auxiliary function that implements a particular prediction.

        For prediction, use `ELM.predict()` instead.

        Args:
            X (matrix): input data size (N * `inputs`)
            dev (bool, optional): whether leave result in the GPU memory

        Returns:
            Y (matrix): predicted outputs size (N * `outputs`), always in float/double format.
        """
        assert self.B is not None, "Solve the task before predicting"
        devH = self._project(X, dev=True)
        devY = linalg.dot(devH, self.B)
        Y = devY if dev else devY.get()
        return Y

    def add_batch(self, X, T, wc=None):
        """Add a batch of training data to an iterative solution, weighted if neeed.

        The batch is processed as a whole, the training data is splitted in `ELM.add_data()` method.
        With parameters HH_out, HT_out, the output will be put into these matrices instead of model.

        Args:
            X (matrix): input data matrix size (N * `inputs`)
            T (matrix): output data matrix size (N * `outputs`)
            wc (vector): vector of weights for data samples, one weight per sample, size (N * 1)
            HH_out, HT_out (matrix, optional): output matrices to add batch result into, always given together
        """
        devH = self._project(X, dev=True)
        T = np.array(T, order="C", dtype=self.precision)
        devT = gpuarray.to_gpu(T)
        if wc is not None:  # apply weights if given
            w = np.array(wc**0.5, dtype=self.precision)[:, None]  # re-shape to column matrix
            devWC = gpuarray.to_gpu(w)
            misc.mult_matvec(devH, devWC, axis=0, out=devH)
            misc.mult_matvec(devT, devWC, axis=0, out=devT)

        if self.HH is None:  # initialize space for self.HH, self.HT
            self.HT = misc.zeros((self.L, self.outputs), dtype=self.precision)
            self.HH = linalg.eye(self.L, dtype=self.precision)
            self.HH *= self.norm

        linalg.add_dot(devH, devT, self.HT, transa='T')
        if self.precision is np.float64:
            linalg.add_dot(devH, devH, self.HH, transa='T')
        else:
            cublas.cublasSsyrk(self.handle, 'L', 'N', self.L, X.shape[0], 1, devH.ptr, self.L, 1, self.HH.ptr, self.L)
#        self.ctx.synchronize()  # GPU runs asyncronously without that

    def solve(self):
        """Compute output weights B, with fix for unstable solution.
        """
        HH = self.HH.get()
        HT = self.HT.get()
        B = self.solve_corr(HH, HT)
        self.B = gpuarray.to_gpu(B)

    def solve_corr(self, HH, HT):
        """Compute output weights B for given HH and HT.

        Simple but inefficient version, see a better one in solver_python.

        Args:
            HH (matrix): covariance matrix of hidden layer represenation H, size (`L` * `L`)
            HT (matrix): correlation matrix between H and outputs T, size (`L` * `outputs`)
        """
        _, B, info = self.posv(HH, HT)
        if info > 0:
            print("ELM covariance matrix is not full rank; solving with SVD (slow)")
            print("This happened because you have duplicated or too many neurons")
            HH = np.triu(HH) + np.triu(HH, k=1).T
            B = np.linalg.lstsq(HH, HT)[0]
        B = np.array(B, order='C', dtype=self.precision)
        return B

    def _prune(self, idx):
        """Leave only neurons with the given indexes.
        """
        idx = list(idx)
        neurons = []
        for k, func, devW, devB in self.neurons:
            ix1 = [i for i in idx if i < k]  # index for current neuron type
            idx = [i-k for i in idx if i >= k]
            number = len(ix1)
            W = devW.get()
            W = np.array(W[:, ix1], order='C')
            devW = gpuarray.to_gpu(W)
            B = devB.get()
            B = np.array(B[ix1], order='C')
            devB = gpuarray.to_gpu(B)
            neurons.append((number, func, devW, devB))
        self.neurons = neurons
        # reset invalid parameters
        self.reset()
        self.B = None

    def get_B(self):
        """Return B as a numpy array.
        """
        if self.B is None:
            B = None
        else:
            B = self.B.get()
        return B

    def set_B(self, B):
        """Set B as a numpy array.

        Args:
            B (matrix): output layer weights matrix, size (`L` * `outputs`)
        """
        assert B.shape[0] == self.L, "Incorrect first dimension: %d expected, %d found" % (self.L, B.shape[0])
        assert B.shape[1] == self.outputs, "Incorrect output dimension: %d expected, %d found" % (self.outputs, B.shape[1])
        self.B = gpuarray.to_gpu(B.astype(self.precision))

    def get_corr(self):
        """Return current correlation matrices.
        """
        if self.HH is None:
            HH = None
            HT = None
        else:
            HH = self.HH.get()
            HT = self.HT.get()
            HH = np.triu(HH) + np.triu(HH, k=1).T
        return HH, HT

    def set_corr(self, HH, HT):
        """Set pre-computed correlation matrices.

        Args:
            HH (matrix): covariance matrix of hidden layer represenation H, size (`L` * `L`)
            HT (matrix): correlation matrix between H and outputs T, size (`L` * `outputs`)
        """
        assert self.neurons is not None, "Add or load neurons before using ELM"
        assert HH.shape[0] == HH.shape[1], "HH must be a square matrix"
        msg = "Wrong HH dimension: (%d, %d) expected, %s found" % (self.L, self.L, HH.shape)
        assert HH.shape[0] == self.L, msg
        assert HH.shape[0] == HT.shape[0], "HH and HT must have the same number of rows (%d)" % self.L
        assert HT.shape[1] == self.outputs, "Number of columns in HT must equal number of outputs (%d)" % self.outputs
        self.HH = gpuarray.to_gpu(HH.astype(self.precision))
        self.HT = gpuarray.to_gpu(HT.astype(self.precision))

    def get_neurons(self):
        """Return current neurons.

        Returns:
            neurons (list of tuples (number/int, func/string, W/matrix, B/vector)): current neurons in the model
        """
        neurons = []
        for number, func, devW, devB in self.neurons:
            neurons.append((number, func, devW.get(), devB.get()))
        return neurons
















