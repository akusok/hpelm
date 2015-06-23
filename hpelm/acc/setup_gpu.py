# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 17:21:12 2014

@author: akusok
"""

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import commands
import numpy
import sys



def pkgconfig(*packages, **kw):
    """Returns nicely organized stuff from PKGCONFIG.

    Found on the internet, returns a dictionary with
    libraries, library dirs, include dirs, extra arguments

    To test, run in terminal: "pkg-config --libs --cflags magma"

    To add "magma" to pkg-config:
    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/magma/lib/pkgconfig
    use your own path to installed magma + lib/pkgconfig
    """
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    for token in commands.getoutput("pkg-config --libs --cflags %s" % ' '.join(packages)).split():
        if token[:2] in flag_map:
            kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
        else:
            kw.setdefault('extra_compile_args', []).append(token)
    return kw


setup(cmdclass={'build_ext': build_ext},
      ext_modules=[
          Extension(
              "gpu_solver",
              sources=["gpu_solver.pyx",
                       "gpu_code.cpp"],
              language="c++",
              extra_compile_args=pkgconfig("magma")["extra_compile_args"],
              include_dirs=[numpy.get_include()] + pkgconfig("magma")["include_dirs"],
              libraries=pkgconfig("magma")["libraries"],
              library_dirs=pkgconfig("magma")["library_dirs"])
          ]
      )
