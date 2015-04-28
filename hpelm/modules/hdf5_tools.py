# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:12:46 2015

@author: akusok
"""

import numpy as np
import csv
from tables import open_file, Atom, Filters


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
            print "calculating mean and standard deviation of data"
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
            print "data was already normalized, returning 'mean', 'std' parameters"
            print "if you want to run normalization anyway, call the function with 'mean' and 'std' params"
            return node.attrs.mean, node.attrs.std
    else:
        assert len(mean) == d, "Incorrect lenght of a vector of means: %d expected, %d found" % (d, len(mean))
        assert len(std) == d, "Incorrect lenght of a vector of standard deviations: %d expected, %d found" % (d, len(std))
        node.attrs.mean = mean
        node.attrs.std = std
    std[std == 0] = 1  # prevent division by zero for std=0

    print "applying normalization"
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
    elif isinstance(data, basestring) and data[-3:] in ['npy']:
        X = np.load(data)
    elif isinstance(data, basestring) and data[-3:] in ['.gz', 'bz2']:
        X = np.loadtxt(data, dtype=dtype, delimiter=delimiter, skiprows=skiprows)
    elif isinstance(data, basestring) and data[-3:] in ['txt', 'csv']:
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
        X = np.empty(data)
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
        h5data = h5.create_carray(h5.root, "data", a, X.shape, filters=flt)
    else:  # write whole data matrix
        h5data = h5.create_carray(h5.root, "data", a, X.shape, filters=flt)
        h5data[:] = X
    # close the file
    h5data.attrs.mean = None
    h5data.attrs.std = None
    h5.flush()
    h5.close()


if __name__ == "__main__":
    # def make_hdf5(data, h5file, dtype=np.float64, delimiter=" ", skiprows=0, comp_level=0):
    # make_hdf5("textfile.txt", "text.h5")
    # make_hdf5("textfile.txt", "textz.h5", comp_level=3)
    normalize_hdf5("text.h5")
    print "Done!"
