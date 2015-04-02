#include "magma.h"

class GpuSolver {
    public:
        GpuSolver( int, int, double* A, double* B );
        ~GpuSolver();
        void add_data( int m, double* X, double* T );
        void get_corr( double* XX, double* XT );
        void solve( double* X );
};

void solve_corr( int n, int nrhs, double* A, double* B, double* X );
