# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

import numpy as np
import multiprocessing as mp
from time import time
from hpelm.modules import make_hdf5, ireader, iwriter, _prepare_fHH, _write_fHH
from tables import open_file
from elm import ELM
# TODO: fix double storage of self.(nnet).outputs and self.(nnet).inputs problem


class HPELM(ELM):
    """Interface for training High-Performance Extreme Learning Machines (HP-ELM).

    Args:
        inputs (int): dimensionality of input data, or number of data features
        outputs (int): dimensionality of output data, or number of classes
        batch (int, optional): batch size for data processing in ELM, reduces memory requirements. Does not work
            for model structure selection (validation, cross-validation, Leave-One-Out). Can be changed later directly
            as a class attribute.
        accelerator (string, optional): type of accelerated ELM to use: None, 'GPU', ...
        precision (optional): data precision to use, supports signle ('single', '32' or numpy.float32) or double
            ('double', '64' or numpy.float64). Single precision is faster but may cause numerical errors. Majority
            of GPUs work in single precision. Default: **double**.
        norm (double, optinal): L2-normalization parameter, **None** gives the default value.
        tprint (int, optional): ELM reports its progess every `tprint` seconds or after every batch,
            whatever takes longer.

    Class attributes; attributes that simply store initialization or `train()` parameters are omitted.

    Attributes:
        nnet (object): Implementation of neural network with computational methods, but without
            complex logic. Different implementations are given by different classes: for Python, for GPU, etc.
            See ``hpelm.nnets`` folder for particular files. You can implement your own computational algorithm
            by inheriting from ``hpelm.nnets.SLFN`` and overwriting some methods.
        flist (list of strings): Awailable types of neurons, use them when adding new neurons.

    Note:
        The 'hdf5' type denotes a name of HDF5 file type with a single 2-dimensional array inside. HPELM uses PyTables
        interface to HDF5: http://www.pytables.org/. For HDF5 array examples, see
        http://www.pytables.org/usersguide/libref/homogenous_storage.html. Array name is irrelevant,
        but there must be **only one array per HDF5 file**.

        A 2-dimensional Numpy.ndarray can also be used.
    """

    def train(self, fX, fT, *args, **kwargs):
        """Universal training interface for HP-ELM model.

        Always trains a basic ELM model without model structure selection.
        L2-regularization is available as `norm` parameter at HPELM initialization.
        Number of neurons selection with validation set for trained HPELM is available in `train_hpv()` method.

        Args:
            fX (hdf5): input data on disk, size (N * `inputs`)
            fT (hdf5): outputs data on disk, size (N * `outputs`)
            'c'/'wc'/'mc' (string, choose one): train HPELM for classification ('c'), classification with weighted
                classes ('wc') or multi-label classification ('mc') with several correct classes per data sample.
                In classification, number of `outputs` is the number of classes; correct class(es) for each sample
                has value 1 and incorrect classes have 0.

        Keyword Args:
            istart (int, optional): index of first data sample to use from `fX`, `istart` < N. If not given,
                all data from `fX` is used. Sample with index `istart` is used for training, indexing is 0-based.
            icount (int, optional): number of data samples to use from `fX`, starting from `istart`, automatically
                adjusted to `istart` + `icount` <= N. If not given, all data starting from `start` is used.
                The last sample used for training is `istart`+`icount`-1, so you can index data as:
                istart_1=0, icount_1=1000; istart_2=1000, icount_2=1000; istart_3=2000, icount_3=1000, ...
            batch (int, optional): batch size for ELM, overwrites batch size from the initialization
        """
        # TODO: add start/stop indexes for training on a part of HDF5 files.
        # TODO: move to h5py completely with async io (mpio), because I don't need pyTables features
        # TODO: explain why I don't support parallel processing (huge amount of data to transfer, or fast enough)
        # TODO: support appending HH, HT into the same given files, waiting for others
        X, T = self._checkdata(fX, fT)
        self._train_parse_args(args, kwargs)

        istart = 0
        icount = np.inf
        if "istart" in kwargs.keys():
            istart = max(0, int(kwargs["istart"]))
        if "icount" in kwargs.keys():
            icount = kwargs["icount"]
        self.add_data(X, T, istart=istart, icount=icount)
        self.nnet.solve()

    def add_data(self, fX, fT, istart=0, icount=np.inf, fHH=None, fHT=None):
        """Feed new training data (X,T) to HP-ELM model in batches: does not solve ELM itself.

        This method prepares an intermediate solution data, that takes the most time. After that, obtaining
        the solution is fast.

        The intermediate solution consists of two matrices: `HH` and `HT`. They can be in memory for a model computed
        at once, or stored on disk for a model computed in parts or in parallel.

        For iterative solution, provide file names for on-disk matrices in the input parameters `fHH` and `fHT`.
        They will be created if they don't exist, or new results will be merged with the existing ones. This method is
        multiprocess-safe for parallel writing into files `fHH` and `fHT`, that allows you to easily compute ELM
        in parallel. The multiprocess-safeness uses Python module 'fasteners' and a lock file, which is named
        fHH+'.lock' and fHT+'.lock'.

        Args:
            fX (hdf5): (part of) input training data size (N * `inputs`)
            fT (hdf5) (part of) output training data size (N * `outputs`)
            istart (int, optional): index of first data sample to use from `fX`, `istart` < N. If not given,
                all data from `fX` is used. Sample with index `istart` is used for training, indexing is 0-based.
            icount (int, optional): number of data samples to use from `fX`, starting from `istart`, automatically
                adjusted to `istart` + `icount` <= N. If not given, all data starting from `start` is used.
                The last sample used for training is `istart`+`icount`-1, so you can index data as:
                istart_1=0, icount_1=1000; istart_2=1000, icount_2=1000; istart_3=2000, icount_3=1000, ...
            fHH, fHT (string, optional): file names for storing HH and HT matrices. Files are created if they don't
                exist, or new result is added to the existing files if they exist. Parallel writing to the same
                `fHH`, `fHT` files is multiprocess-safe, made specially for parallel training of HP-ELM. Another use
                is to split a very long training of huge ELM into smaller parts, so the training can be interrupted
                and resumed later.

        """
        # initialize
        assert len(self.nnet.neurons) > 0, "Add neurons to ELM before using it"
        X, T = self._checkdata(fX, fT)
        N = X.shape[0]
        HH, HT = _prepare_fHH(fHH, fHT, self.nnet.L, self.outputs, self.precision, self.nnet.norm)
        # custom range adjustments
        icount = min(istart + icount, N)
        nb = int(np.ceil(float(icount) / self.batch))  # number of batches

        # weighted classification initialization
        if self.classification == "wc" and self.wc is None:
            ns = np.zeros((self.outputs,))
            for b in xrange(nb):  # batch sum is much faster
                start = b*self.batch + istart
                stop = min((b+1)*self.batch + istart, icount + istart)
                ns += T[start:stop].sum(axis=0)
            ns = ns.astype(self.precision)
            self.wc = ns.sum() / ns  # class weights normalized to number of samples

        # main loop over all the data
        t = time()
        t0 = time()
        eta = 0
        wc_vector = None
        for b in xrange(nb):
            start = b*self.batch + istart
            stop = min((b+1)*self.batch + istart, icount + istart)
            Xb = X[start:stop]
            Tb = T[start:stop]
            if self.classification == "wc":
                wc_vector = self.wc[np.where(Tb == 1)[1]]  # weights for samples in the batch

            self.nnet.add_batch(Xb, Tb, wc_vector, HH_out=HH, HT_out=HT)

            # report time
            eta = int(((time()-t0) / (b+1)) * (nb-b-1))
            if time() - t > self.tprint:
                print "processing batch %d/%d, eta %d:%02d:%02d" % (b+1, nb, eta/3600, (eta % 3600)/60, eta % 60)
                t = time()

        # if storing output to disk
        if HH is not None and HT is not None:
            _write_fHH(fHH, fHT, HH, HT)

    def solve_corr(self, fHH, fHT):
        """Solves an ELM model with the given (covariance) fHH and (correlation) fHT HDF5 files.

        Args:
            fHH (hdf5): an hdf5 file with intermediate solution data
            fHT (hdf5): an hdf5 file with intermediate solution data
        """
        try:
            h5 = open_file(fHH, "r")
        except:
            raise IOError("Cannot read HDF5 file at %s" % fHH)
        node = None
        for node in h5.walk_nodes():
            pass  # find a node with whatever name
        if node:
            HH = node[:]
        else:
            raise IOError("Empty HDF5 file at %s" % fHH)
        h5.close()

        try:
            h5 = open_file(fHT, "r")
        except:
            raise IOError("Cannot read HDF5 file at %s" % fHT)
        node = None
        for node in h5.walk_nodes():
            pass  # find a node with whatever name
        if node:
            HT = node[:]
        else:
            raise IOError("Empty HDF5 file at %s" % fHT)
        h5.close()

        L = self.nnet.L
        c = self.nnet.outputs
        assert len(self.nnet.neurons) > 0, "Cannot solve ELM without neurons"
        assert HH.shape[0] == L and HH.shape[1] == L, "HH has wrong shape: (%d,%d) expected, (%d,%d) found" \
                                                      % (L, L, HH.shape[0], HH.shape[1])
        assert HT.shape[0] == L and HT.shape[1] == c, "HT has wrong shape: (%d,%d) expected, (%d,%d) found" \
                                                      % (L, c, HH.shape[0], HH.shape[1])
        B = self.nnet.solve_corr(HH, HT)
        self.nnet.set_B(B)

    def predict(self, fX, fY, istart=0, icount=np.inf):
        """Iterative predict outputs and save them to HDF5, can use custom range.

        Args:
            fX (hdf5): hdf5 filename with input data from which outputs are predicted
            fY (hdf5): hdf5 filename to store output data into
            istart (int, optional): index of first data sample to use from `fX`, `istart` < N. If not given,
                all data from `fX` is used. Sample with index `istart` is used for training, indexing is 0-based.
            icount (int, optional): number of data samples to use from `fX`, starting from `istart`, automatically
                adjusted to `istart` + `icount` <= N. If not given, all data starting from `start` is used.
                The last sample used for training is `istart`+`icount`-1, so you can index data as:
                istart_1=0, icount_1=1000; istart_2=1000, icount_2=1000; istart_3=2000, icount_3=1000, ...
        """
        assert len(self.nnet.neurons) > 0, "Add neurons to ELM and train it before using"
        assert self.nnet.B is not None, "Train ELM before predicting"
        X, _ = self._checkdata(fX, None)
        N = X.shape[0]
        # custom range adjustments
        icount = min(istart + icount, N)
        nb = int(np.ceil(float(icount) / self.batch))  # number of batches
        # make file to store results
        make_hdf5((icount, self.outputs), fY, dtype=self.precision)
        h5 = open_file(fY, "a")
        for Y in h5.walk_nodes():
            pass  # find a node with whatever name

        t = time()
        t0 = time()
        eta = 0
        for b in xrange(0, nb):
            start = b*self.batch + istart
            stop = min((b+1)*self.batch + istart, icount + istart)

            # get data
            Xb = X[start:stop]
            # process data
            Yb = self.nnet._predict(Xb)
            # write data
            Y[start-start:stop-istart] = Yb

            # report time
            eta = int(((time()-t0) / (b+1)) * (nb-b-1))
            if time() - t > self.tprint:
                print "processing batch %d/%d, eta %d:%02d:%02d" % (b+1, nb, eta/3600, (eta % 3600)/60, eta % 60)
                t = time()

        h5.flush()
        h5.close()

    def project(self, fX, fH, istart=0, icount=np.inf):
        """Iteratively project input data from HDF5 into HPELM hidden layer, and save in another HDF5.

        Args:
            fX (hdf5): hdf5 filename with input data from which outputs are predicted
            fH (hdf5): hdf5 filename to store output data into
            istart (int, optional): index of first data sample to use from `fX`, `istart` < N. If not given,
                all data from `fX` is used. Sample with index `istart` is used for training, indexing is 0-based.
            icount (int, optional): number of data samples to use from `fX`, starting from `istart`, automatically
                adjusted to `istart` + `icount` <= N. If not given, all data starting from `start` is used.
                The last sample used for training is `istart`+`icount`-1, so you can index data as:
                istart_1=0, icount_1=1000; istart_2=1000, icount_2=1000; istart_3=2000, icount_3=1000, ...
        """
        assert len(self.nnet.neurons) > 0, "Add neurons to ELM before using it"
        X, _ = self._checkdata(fX, None)
        N = X.shape[0]
        # custom range adjustments
        print N
        print icount
        print istart
        icount = min(istart + icount, N)
        nb = int(np.ceil(float(icount) / self.batch))  # number of batches
        # make file to store results
        make_hdf5((icount, self.nnet.L), fH, dtype=self.precision)
        h5 = open_file(fH, "a")
        for H in h5.walk_nodes():
            pass  # find a node with whatever name

        t = time()
        t0 = time()
        eta = 0
        for b in xrange(0, nb):
            start = b*self.batch + istart
            stop = min((b+1)*self.batch + istart, icount + istart)

            # get data
            Xb = X[start:stop]
            # process data
            Hb = self.nnet._project(Xb)
            # write data
            H[start-start:stop-istart] = Hb

            # report time
            eta = int(((time()-t0) / (b+1)) * (nb-b-1))
            if time() - t > self.tprint:
                print "processing batch %d/%d, eta %d:%02d:%02d" % (b+1, nb, eta/3600, (eta % 3600)/60, eta % 60)
                t = time()

        h5.flush()
        h5.close()

    def _error(self, Y1, T1, H1=None, Beta=None, rank=None):
        """Do projection and calculate error in batch mode.

        HPELM-specific iterative error for all usage cases.
        Can be _error(Y, T) or _error(None, T, H, Beta, rank)

        :param T: - true targets for error calculation
        :param H: - projected data for error calculation
        :param Beta: - current projection matrix
        :param rank: - selected neurons (= columns of H)
        """
        if Y1 is None:
            H, T = self._checkdata(H1, T1)
            assert rank.shape[0] == Beta.shape[0], "Wrong dimension of Beta for the given ranking"
            assert T.shape[1] == Beta.shape[1], "Wrong dimension of Beta for the given outputs"
            nn = rank.shape[0]
        else:
            _, Y = self._checkdata(None, Y1)
            _, T = self._checkdata(None, T1)
            nn = np.sum([n1[1] for n1 in self.neurons])
        N = T.shape[0]
        batch = max(self.batch, nn)
        nb = N / batch  # number of batches
        if N > batch * nb:
            nb += 1

        if self.classification == "c":
            err = 0
            for b in xrange(nb):
                start = b*batch
                stop = min((b+1)*batch, N)
                Tb = np.array(T[start:stop])
                if Y1 is None:
                    Hb = H[start:stop, rank]
                    Yb = np.dot(Hb, Beta)
                else:
                    Yb = np.array(Y[start:stop])
                errb = np.mean(Yb.argmax(1) != Tb.argmax(1))
                err += errb * float(stop-start)/N

        elif self.classification == "wc":  # weighted classification
            c = T.shape[1]
            errc = np.zeros(c)
            for b in xrange(nb):
                start = b*batch
                stop = min((b+1)*batch, N)
                Tb = np.array(T[start:stop])
                if Y1 is None:
                    Hb = H[start:stop, rank]
                    Yb = np.dot(Hb, Beta)
                else:
                    Yb = np.array(Y[start:stop])
                for i in xrange(c):  # per-class MSE
                    idxc = Tb[:, i] == 1
                    errb = np.mean(Yb[idxc].argmax(1) != i)
                    errc[i] += errb * float(stop-start)/N
            err = np.mean(errc * self.wc)

        elif self.classification == "mc":
            err = 0
            for b in xrange(nb):
                start = b*batch
                stop = min((b+1)*batch, N)
                Tb = np.array(T[start:stop])
                if Y1 is None:
                    Hb = H[start:stop, rank]
                    Yb = np.dot(Hb, Beta)
                else:
                    Yb = np.array(Y[start:stop])
                errb = np.mean((Yb > 0.5) != (Tb > 0.5))
                err += errb * float(stop-start)/N

        else:  # MSE error
            err = 0
            for b in xrange(nb):
                start = b*batch
                stop = min((b+1)*batch, N)
                Tb = T[start:stop]
                if Y1 is None:
                    Hb = H[start:stop, rank]
                    Yb = np.dot(Hb, Beta)
                else:
                    Yb = Y[start:stop]
                errb = np.mean((Tb - Yb)**2)
                err += errb * float(stop-start)/N

        return err

    def train_hpv(self, HH, HT, Xv, Tv, steps=10):
        X, T = self._checkdata(Xv, Tv)
        N = X.shape[0]
        nn = HH.shape[0]

        nns = np.logspace(np.log(3), np.log(nn), steps, base=np.e, endpoint=True)
        nns = np.ceil(nns).astype(np.int)
        nns = np.unique(nns)  # numbers of neurons to check
        print nns
        k = nns.shape[0]
        err = np.zeros((k,))  # errors for these numbers of neurons
        nb = int(np.ceil(float(N) / self.batch))

        Betas = []  # keep all betas in memory
        for l in nns:
            Betas.append(self.nnet.solve_corr(HH[:l, :l], HT[:l, :]))

        t = time()
        t0 = time()
        eta = 0
        for b in xrange(nb):
            eta = int(((time()-t0) / (b+0.0000001)) * (nb-b))
            if time() - t > self.tprint:
                print "processing batch %d/%d, eta %d:%02d:%02d" % (b+1, nb, eta/3600, (eta % 3600)/60, eta % 60)
                t = time()
            start = b*self.batch
            stop = min((b+1)*self.batch, N)
            alpha = float(stop-start)/N
            Tb = np.array(T[start:stop])
            Xb = np.array(X[start:stop])
            Hb = self.nnet._project(Xb)
            for i in xrange(k):
                hb1 = Hb[:, :nns[i]]
                Yb = np.dot(hb1, Betas[i])
                err[i] += self._error(Yb, Tb) * alpha

        k_opt = np.argmin(err)
        best_nn = nns[k_opt]
        self._prune(np.arange(best_nn))
        self.nnet.B = Betas[k_opt]
        del Betas
        print "%d of %d neurons selected with a validation set" % (best_nn, nn)
        if best_nn > nn*0.9:
            print "Hint: try re-training with more hidden neurons"
        return nns, err

    def train_myhpv(self, HH, HT, Xv, Tv, steps=10):
        X, T = self._checkdata(Xv, Tv)
        N = X.shape[0]
        nn = HH.shape[0]

        nns = np.logspace(np.log(3), np.log(nn), steps, base=np.e, endpoint=True)
        nns = np.ceil(nns).astype(np.int)
        nns = np.unique(nns)  # numbers of neurons to check
        k = nns.shape[0]
        err = np.zeros((k, 2, 2))  # errors for these numbers of neurons
        nb = int(np.ceil(float(N) / self.batch))

        Betas = []  # keep all betas in memory
        for l in nns:
            Betas.append(self.nnet.solve_corr(HH[:l, :l], HT[:l, :]))

        t = time()
        t0 = time()
        eta = 0
        for b in xrange(nb):
            eta = int(((time()-t0) / (b+0.0000001)) * (nb-b))
            if time() - t > self.tprint:
                print "processing batch %d/%d, eta %d:%02d:%02d" % (b+1, nb, eta/3600, (eta % 3600)/60, eta % 60)
                t = time()
            start = b*self.batch
            stop = min((b+1)*self.batch, N)
            Tb = np.array(T[start:stop])
            Xb = np.array(X[start:stop])
            Hb = self.nnet._project(Xb)
            Tc = np.argmax(Tb, axis=1)
            for i in xrange(k):
                hb1 = Hb[:, :nns[i]]
                Yb = np.dot(hb1, Betas[i])
                Yc = np.argmax(Yb, axis=1)
                err[i, 0, 0] += np.sum((Tc == 0)*(Yc == 0))
                err[i, 0, 1] += np.sum((Tc == 0)*(Yc == 1))
                err[i, 1, 0] += np.sum((Tc == 1)*(Yc == 0))
                err[i, 1, 1] += np.sum((Tc == 1)*(Yc == 1))

        return nns, err, N

    # async-IO versions of methods

    def train_async(self, fX, fT, *args, **kwargs):
        """Training HPELM with asyncronous I/O, good for network drives, etc. See `train()` for reference.

        Spawns new processes using Python's `multiprocessing` module.
        """
        X, T = self._checkdata(fX, fT)
        self._train_parse_args(args, kwargs)

        istart = 0
        icount = np.inf
        if "istart" in kwargs.keys():
            istart = max(0, int(kwargs["istart"]))
        if "icount" in kwargs.keys():
            icount = kwargs["icount"]
        self.add_data_async(fX, fT, istart=istart, icount=icount)
        self.nnet.solve()

    def add_data_async(self, fX, fT, istart=0, icount=np.inf, fHH=None, fHT=None):
        """Version of `add_data()` with asyncronous I/O. See `add_data()` for reference.

        Spawns new processes using Python's `multiprocessing` module, and requires more memory than non-async version.
        """
        # initialize
        assert len(self.nnet.neurons) > 0, "Add neurons to ELM before using it"
        X, T = self._checkdata(fX, fT)
        N = X.shape[0]
        HH, HT = _prepare_fHH(fHH, fHT, self.nnet.L, self.outputs, self.precision, self.nnet.norm)
        # custom range adjustments
        icount = min(istart + icount, N)
        nb = int(np.ceil(float(icount) / self.batch))

        # weighted classification initialization
        if self.classification == "wc" and self.wc is None:
            ns = np.zeros((self.outputs,))
            for b in xrange(nb):  # batch sum is much faster
                start = b*self.batch + istart
                stop = min((b+1)*self.batch + istart, icount + istart)
                ns += T[start:stop].sum(axis=0)
            ns = ns.astype(self.precision)
            self.wc = ns.sum() / ns  # class weights normalized to number of samples

        # close X and T files opened by _checkdata()
        h5 = self.opened_hdf5.pop()
        h5.close()
        h5 = self.opened_hdf5.pop()
        h5.close()

        # start async reader and writer for HDF5 files
        qX_in = mp.Queue()
        qX_out = mp.Queue(1)
        readerX = mp.Process(target=ireader, args=(fX, qX_in, qX_out))
        readerX.daemon = True
        readerX.start()
        qT_in = mp.Queue()
        qT_out = mp.Queue(1)
        readerT = mp.Process(target=ireader, args=(fT, qT_in, qT_out))
        readerT.daemon = True
        readerT.start()

        # main loop over all the data
        t = time()
        t0 = time()
        eta = 0
        wc_vector = None
        for b in xrange(0, nb+1):
            start_next = b*self.batch + istart
            stop_next = min((b+1)*self.batch + istart, icount + istart)
            # prefetch data
            qX_in.put((start_next, stop_next))  # asyncronous reading of next data batch
            qT_in.put((start_next, stop_next))

            if b > 0:  # first iteration only prefetches data
                Xb = qX_out.get()
                Tb = qT_out.get()
                if self.classification == "wc":
                    wc_vector = self.wc[np.where(Tb == 1)[1]]  # weights for samples in the batch

                self.nnet.add_batch(Xb, Tb, wc_vector, HH_out=HH, HT_out=HT)

            # report time
            eta = int(((time()-t0) / (b+1)) * (nb-b-1))
            if time() - t > self.tprint:
                print "processing batch %d/%d, eta %d:%02d:%02d" % (b+1, nb, eta/3600, (eta % 3600)/60, eta % 60)
                t = time()

        # close async reader and writer
        readerX.join()
        readerT.join()

        # if storing output to disk
        if HH is not None and HT is not None:
            _write_fHH(fHH, fHT, HH, HT)

    def predict_async(self, fX, fY, istart=0, icount=np.inf):
        """Version of `predict()` with asyncronous I/O. See `predict()` for reference.

        Spawns new processes using Python's `multiprocessing` module, and requires more memory than non-async version.
        """
        assert len(self.nnet.neurons) > 0, "Add neurons to ELM and train it before using"
        assert self.nnet.B is not None, "Train ELM before predicting"
        X, _ = self._checkdata(fX, None)
        N = X.shape[0]
        # custom range adjustments
        icount = min(istart + icount, N)
        nb = int(np.ceil(float(icount) / self.batch))  # number of batches
        # make file to store results
        make_hdf5((icount, self.outputs), fY)

        # start async reader and writer for HDF5 files
        qr_in = mp.Queue()
        qr_out = mp.Queue(1)
        reader = mp.Process(target=ireader, args=(fX, qr_in, qr_out))
        reader.daemon = True
        reader.start()
        qw_in = mp.Queue(1)
        writer = mp.Process(target=iwriter, args=(fY, qw_in))
        writer.daemon = True
        writer.start()

        t = time()
        t0 = time()
        eta = 0
        for b in xrange(0, nb+1):
            start_next = b*self.batch + istart
            stop_next = min((b+1)*self.batch + istart, icount + istart)
            # prefetch data
            qr_in.put((start_next, stop_next))  # asyncronous reading of next data batch

            if b > 0:  # first iteration only prefetches data
                # get data
                Xb = qr_out.get()
                # process data
                Yb = self.nnet._predict(Xb)
                # save data
                qw_in.put((Yb, start-istart, stop-istart))

            start = start_next
            stop = stop_next
            # report time
            eta = int(((time()-t0) / (b+1)) * (nb-b-1))
            if time() - t > self.tprint:
                print "processing batch %d/%d, eta %d:%02d:%02d" % (b+1, nb, eta/3600, (eta % 3600)/60, eta % 60)
                t = time()

        qw_in.put(None)
        reader.join()
        writer.join()

















