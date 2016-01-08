#include "magma.h"

class GpuSolver {
    public:
        GpuSolver( int, int, double* A, double* B );
        void add_data( int m, double* X, double* T );
        void get_corr( double* XX, double* XT );
        void solve( double* X );
        void finalize();
};

void solve_corr( int n, int nrhs, double* A, double* B, double* X );
