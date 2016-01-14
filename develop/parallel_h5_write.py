import numpy as np
from tables import open_file
import sys
import fasteners


def make(fname):
    h5 = open_file(fname, "w")
    h5.create_array(h5.root, "Data", obj=np.zeros((1000, 1000), dtype=np.float64))
    h5.flush()
    h5.close()


def run(fname):
    flock = fname[:-2]+"lock"
    with fasteners.InterProcessLock(flock):
        h5 = open_file(fname, "a")
        d = np.ones((1000, 1000))
        print "adding"
        h5.root.Data[:] += d
        h5.flush()
        h5.close()

if __name__ == "__main__":
    fname = sys.argv[1]
    if len(sys.argv) > 2:
        make(fname)
    else:
        run(fname)