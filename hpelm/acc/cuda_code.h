#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

class CudaSolver {
    public:
        CudaSolver( int, int, double* HH, double* HT );
        void get_corr( double* HH, double* HT );
        void add_data( int N, double* H, double* T );
        void finalize();
};
