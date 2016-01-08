#!/bin/bash

rm cuda_solver.cpp
python setup_cuda.py build_ext --inplace
python -c "import cuda_solver as cs; print dir(cs)"
python try_cuda.py
echo "Done"
