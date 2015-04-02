#include "magma.h"

class GpuSolver {
    public:
        GpuSolver( int, int );
        ~GpuSolver();
        void add_data( int m, double* X, double* T );
        void solve( double* X );
};
