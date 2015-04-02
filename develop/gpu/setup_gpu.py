#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import commands
import numpy


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
              "magma_solver",
              sources=["magma_solver.pyx",
                       "gpu_solver.cpp"],
              language="c++",
              extra_compile_args=pkgconfig("magma")["extra_compile_args"],
              include_dirs=[numpy.get_include()] + pkgconfig("magma")["include_dirs"],
              libraries=pkgconfig("magma")["libraries"],
              library_dirs=pkgconfig("magma")["library_dirs"])
          ]
      )



"""
setup(cmdclass = {'build_ext': build_ext},
	ext_modules = [
	    Extension("magma_solver",
            sources = ["magma_solver.pyx", "gpu_solver.cpp"],
            language="c++",
            extra_compile_args=["-DADD_  -DHAVE_CUBLAS"],
            include_dirs = [numpy.get_include(),
                            "/usr/local/magma/include",
                            "/usr/local/cuda-6.5/include",
                            "/opt/intel/composerxe/mkl/include"],
            libraries = ["magma", 
                         "mkl_intel_lp64",
                         "mkl_intel_thread",
                         "mkl_core",
                         "iomp5",
                         "pthread",
                         "cublas",
                         "cudart",
                         "stdc++",
                         "gfortran",
                         "m"],
            library_dirs = ["/usr/local/magma/lib",
                            "/opt/intel/composerxe/mkl/lib/intel64",
                            "/usr/local/cuda-6.5/lib64",
                            "/opt/intel/composerxe/lib/intel64"]
            )
    ]
)
"""