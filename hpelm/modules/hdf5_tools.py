# -*- coding: utf-8 -*-
"""Different tools to work with datasets in HDF5 file format.

Created on Thu Apr  2 21:12:46 2015

@author: akusok
"""

import numpy as np
from six import string_types
from six.moves import xrange
import csv
from tables import open_file, Atom, Filters
import os
import fasteners  # inter-process file lock


def _prepare_fHH(fHH, fHT, nnet, precision):
    """Prepares files for fHH, fHT if they are needed.

    Args:
        fHH (string): hdf5 filename to store HH, None for ignore disk storage
        fHT (string): hdf5 filename to store HT, None for ignore disk storage
        nent (nnets Object): neural network implementation from HPELM
        precision (np.float32/64): precision
    """
    if (fHH is not None) and (fHT is not None):
        # reset accumulated data in ELM
        nnet.reset()
        L = nnet.L
        outputs = nnet.outputs
        norm = nnet.norm

        # process fHH
        if os.path.isfile(fHH):
            h5 = open_file(fHH, 'r')
            node = None
            for node in h5.walk_nodes():
                pass  # find a node with whatever name
            try:
                assert node is not None, "Matrix in %d does not exist" % fHH
                assert node is not None and node.shape[0] == L and node.shape[1] == L, \
                       "Matrix in %d has a wrong shape: (%d, %d) expected, (%d, %d) found" % \
                       (fHH, L, L, node.shape[0], node.shape[1])
            except AssertionError as e:
                raise  # re-raise same error
            finally:
                h5.close()
        else:
            make_hdf5(np.eye(L, L, dtype=precision)*norm, fHH, precision)

        # process fHT
        if os.path.isfile(fHT):
            h5 = open_file(fHT, 'r')
            node = None
            for node in h5.walk_nodes():
                pass  # find a node with whatever name
            try:
                assert node is not None, "Matrix in %d does not exist" % fHT
                assert node is not None and node.shape[0] == L and node.shape[1] == outputs, \
                       "Matrix in %d has a wrong shape: (%d, %d) expected, (%d, %d) found" % \
                       (fHT, L, outputs, node.shape[0], node.shape[1])
            except AssertionError as e:
                raise  # re-raise same error
            finally:
                h5.close()
        else:
            make_hdf5((L, outputs), fHT, precision)

def _write_fHH(fHH, fHT, HH, HT):
    """Writes HH,HT data into fHH,fHT files, multi-process safe with lock file.

    Lock file has the same name as fHH,fHT, but with '.lock' extension.
    """
    fHH_lock = fHH + ".lock"
    with fasteners.InterProcessLock(fHH_lock):
        h5 = open_file(fHH, "a")
        for node in h5.walk_nodes():
            pass  # find a node with whatever name
        node[:] += HH
        h5.flush()
        h5.close()

    fHT_lock = fHT + ".lock"
    with fasteners.InterProcessLock(fHT_lock):
        h5 = open_file(fHT, "a")
        for node in h5.walk_nodes():
            pass  # find a node with whatever name
        node[:] += HT
        h5.flush()
        h5.close()


def normalize_hdf5(h5file, mean=None, std=None, batch=None):
    """Calculates and applies normalization to data in HDF5 file.

    :param mean: - known vector of mean values
    :param std: - known vector of standard deviations
    :param batch: - number of rows to read at once, default is a native batch size
    """

    h5 = open_file(h5file, "a")
    for node in h5.walk_nodes():
        pass  # find a node with whatever name
    dt = node.dtype
    N, d = node.shape  # HDF5 files are transposed, for Matlab compatibility
    if batch is None:
        batch = node.chunkshape[0]
    nb = N/batch
    if N > nb*batch:
        nb += 1  # add last incomplete step

    if mean is None or std is None:
        if node.attrs.mean is None:  # data was not normalized before
            print("calculating mean and standard deviation of data")
            E_x = np.zeros((d,), dtype=np.float64)
            E_x2 = np.zeros((d,), dtype=np.float64)
            for b in xrange(nb):
                start = b*batch
                step = min(batch, N-start)
                X1 = node[start: start+step, :].astype(np.float64)
                E_x += np.mean(X1, 0) * (1.0*step/N)
                E_x2 += np.mean(X1**2, 0) * (1.0*step/N)
            mean = E_x
            E2_x = E_x**2
            std = (E_x2 - E2_x)**0.5
            node.attrs.mean = mean
            node.attrs.std = std
            return mean, std
        else:  # data is already normalized
            print("data was already normalized, returning 'mean', 'std' parameters")
            print("if you want to run normalization anyway, call the function with 'mean' and 'std' params")
            return node.attrs.mean, node.attrs.std
    else:
        assert len(mean) == d, "Incorrect lenght of a vector of means: %d expected, %d found" % (d, len(mean))
        assert len(std) == d, "Incorrect lenght of a vector of standard deviations: %d expected, %d found" % (d, len(std))
        node.attrs.mean = mean
        node.attrs.std = std
    std[std == 0] = 1  # prevent division by zero for std=0

    print("applying normalization")
    for b in xrange(nb):
        start = b*batch
        step = min(batch, N-start)
        X = node[start: start+step].astype(np.float64)
        X = (X - mean) / std
        node[start: start+step] = X.astype(dt)

    h5.close()  # closing file
    return mean, std


#def oversample(data, targets, classes):
#    pass


def make_hdf5(data, h5file, dtype=np.float64, delimiter=" ", skiprows=0, comp_level=0):
    """Makes an HDF5 file from whatever given data.

    :param data: - input data in Numpy.ndarray or filename, or a shape tuple
    :param h5file: - name (and path) of the output HDF5 file
    :param delimiter: - data delimiter for text, csv files
    :param comp_level: - compression level of the HDF5 file
    """
    assert comp_level < 10, "Compression level must be 0-9 (0 for no compression)"
    fill = ""

    # open data file
    if isinstance(data, np.ndarray):
        X = data
    elif isinstance(data, string_types) and data[-3:] in ['npy']:
        X = np.load(data)
    elif isinstance(data, string_types) and data[-3:] in ['.gz', 'bz2']:
        X = np.loadtxt(data, dtype=dtype, delimiter=delimiter, skiprows=skiprows)
    elif isinstance(data, string_types) and data[-3:] in ['txt', 'csv']:
        # iterative out-of-memory loader for huge .csv/.txt files
        fill = "iter"
        # check data dimensionality
        with open(data, "rU") as f:
            for _ in xrange(skiprows):
                f.readline()
            reader = csv.reader(f, delimiter=delimiter)
            for line in reader:
                X = np.fromiter(line, dtype=dtype)
                break
    elif isinstance(data, tuple) and len(data) == 2:
        X = np.empty((1, 1))
        fill = "empty"
    else:
        assert False, "Input data must be Numpy ndarray, .npy file, or .txt/.csv text file (compressed .gz/.bz2)"

    # process data
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    assert len(X.shape) == 2, "Data in Numpy ndarray must have 2 dimensions"
    # create hdf5 file
    if comp_level > 0:
        flt = Filters(complevel=comp_level, shuffle=True)
    else:
        flt = Filters(complevel=0)
    h5 = open_file(h5file, "w")
    a = Atom.from_dtype(np.dtype(dtype), dflt=0)
    # write data to hdf5 file
    if fill == "iter":  # iteratively fill the data
        h5data = h5.create_earray(h5.root, "data", a, (0, X.shape[0]), filters=flt)
        with open(data, "rU") as f:
            for _ in xrange(skiprows):
                f.readline()
            reader = csv.reader(f, delimiter=delimiter)
            for line in reader:
                row = np.fromiter(line, dtype=dtype)
                h5data.append(row[np.newaxis, :])
    elif fill == "empty":  # no fill at all
        h5data = h5.create_carray(h5.root, "data", a, data, filters=flt)
    else:  # write whole data matrix
        h5data = h5.create_carray(h5.root, "data", a, X.shape, filters=flt)
        h5data[:] = X
    # close the file
    h5data.attrs.mean = None
    h5data.attrs.std = None
    h5.flush()
    h5.close()


def _ireader(fX, q_in, q_out):
    """Asyncronous reader for an HDF5 file.

    q_in - a (start, stop) tuple of read indexes; if start >= stop then reader terminates
    q_out - a queue for chunks red from a disk
    """
    assert isinstance(fX, string_types), "Asyncronous I/O only supported with HDF5 data files"
    hX = open_file(fX, "r")
    for X in hX.walk_nodes():
        pass  # find a node with whatever name

    while True:  # returning data chunks on demand
        start, stop = q_in.get()
        if start >= stop:
            break
        q_out.put(X[start:stop])
    hX.close()


def _iwriter(fX, q_in):
    """Asyncronous writer for an HDF5 file.

    q_in - a (Xbatch, start, stop) tuple of data to write indexes; if q_in is None then writer terminates
    """
    assert isinstance(fX, string_types), "Asyncronous I/O only supported with HDF5 data files"
    hX = open_file(fX, "a")
    for X in hX.walk_nodes():
        pass  # find a node with whatever name

    while True:  # returning data chunks on demand
        data = q_in.get()
        if data is None:
            break
        Xb, start, stop = data
        X[start:stop] = Xb
    X.flush()
    hX.close()


if __name__ == "__main__":
    # def make_hdf5(data, h5file, dtype=np.float64, delimiter=" ", skiprows=0, comp_level=0):
    # make_hdf5("textfile.txt", "text.h5")
    # make_hdf5("textfile.txt", "textz.h5", comp_level=3)
    normalize_hdf5("text.h5")
    print("Done!")
