# -*- coding: utf-8 -*-

import os

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    ireqs = ['numpy']
else:
    ireqs = [
          'numpy',
          'scipy>=0.12',
          'tables',
          'fasteners',
          'six',
          'nose'
      ]


############################
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

# sphinx-apidoc -f -o docs hpelm; cd docs; make html; cd ../

setup(name='hpelm',
      version='1.0.5',
      description='High-Performance implementation of an Extreme Learning Machine',
      long_description=readme(),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      keywords='ELM HPC regression classification ANN',
      url='https://www.researchgate.net/profile/Anton_Akusok',
      author='Anton Akusok',
      author_email='akusok.a@gmail.com',
      license='BSD (3-clause)',
      packages=['hpelm',
                'hpelm.modules',
                'hpelm.tests',
                'hpelm.nnets'],
      install_requires=ireqs,  # ReadTheDocs fix
      scripts=['bin/elm_naive.py'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
