# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 17:21:12 2014

@author: akusok
"""

from setuptools import setup
#from distutils.core import setup

def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='hpelm',
      version='0.6.6',
      description='High-Performance implementation of an\
                   Extreme Learning Machine',
      long_description=readme(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.7',
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
                'hpelm.acc'],
      install_requires=[
          'numpy',
          'numexpr',
          'scipy>=0.12',
          'tables',
          'cython'
      ],
      scripts=['bin/elm_naive.py'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
