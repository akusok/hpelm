#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext_module = Extension(
    "f_apply",
    ["f_apply.pyx", "mp_func.c"],
    include_dirs=[numpy.get_include()], 
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    name = 'f_apply',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)
