#!/bin/bash
#$ -q AL
#$ -pe smp 1
OMP_NUM_THREADS=1 python /Users/akusok/Dropbox/Documents/X-ELM/hpelm/datasets/benchmark.py $1
