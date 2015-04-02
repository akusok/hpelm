# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:28:02 2014

@author: akusok
"""

import numpy as np
from tables import openFile, Atom


def h5write(filename, varname, data):
    """Writes one data matrix to HDF5 file.
    
    Similar to Matlab function.
    """
    assert isinstance(filename, basestring), "file name must be a string"
    assert isinstance(varname, basestring), "variable name must be a string"
    assert isinstance(data, np.ndarray), "data must be a Numpy array"
    if len(data.shape) == 1:
        data = data.reshape(-1,1)
    
    # remove leading "/" from variable name
    if varname[0] == "/":
        varname = varname[1:]

    try:
        h5 = openFile(filename, "w")
        a = Atom.from_dtype(data.dtype)
        h5.create_array(h5.root, varname, data.T, atom=a)  # transpose for Matlab compatibility
        h5.flush()
    finally:
        h5.close()
    
    
def h5read(filename):
    """Reads one data matrix from HDF5 file, variable name does not matter.
    
    Similar to Matlab function.
    """
    h5 = openFile(filename)
    for node in h5.walk_nodes():  # find the last node with whatever name
        pass
    M = node[:].T  # transpose for Matlab compatibility
    h5.close()
    return M
        