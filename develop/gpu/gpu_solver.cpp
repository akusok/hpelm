#include <stdio.h>
#include <stdlib.h>
#include "magma.h"

class GpuSolver {
        int n, nrhs;
        magmaDouble_ptr dA=NULL, dB=NULL;
    public:
        GpuSolver( int, int );
        ~GpuSolver(){
            magma_free( dA );
            magma_free( dB );
            magma_finalize();
        };
        void add_data( magma_int_t m, magmaDouble_ptr X, magmaDouble_ptr T );
        void solve( magmaDouble_ptr X );
};



// init a zero matrix on GPU to store X'*X
GpuSolver::GpuSolver ( int nn, int outs ) {

    magma_init();

    n = nn;
    nrhs = outs;

    magma_int_t i, j;
    magmaDouble_ptr A=NULL, B=NULL;

    magma_dmalloc_cpu( &A, n*n );
    magma_dmalloc_cpu( &B, n*nrhs );
    magma_dmalloc( &dA, n*n );
    magma_dmalloc( &dB, n*nrhs );
    if ( dA == NULL || dB == NULL || A == NULL || B == NULL ) {
        fprintf( stderr, "malloc failed - not enough GPU or system memory?\n" );
        goto cleanup;
    }

    for( j=0; j < n; ++j ) {
        for( i=0; i < n; ++i ) { A[i + j*n] = 0; }
    }
    for( j=0; j < nrhs; ++j ) {
        for( i=0; i < n; ++i ) { B[i + j*n] = 0; }
    }

    magma_dsetmatrix( n, n, A, n, dA, n );
    magma_dsetmatrix( n, nrhs, B, n, dB, n );

  cleanup:
    magma_free_cpu( A );
    magma_free_cpu( B );
};



// init a zero matrix on GPU to store X'*X
void GpuSolver::add_data ( magma_int_t m, magmaDouble_ptr X, magmaDouble_ptr T ) {

    real_Double_t   time;
    magmaDouble_ptr dX=NULL, dT=NULL;

    magma_dmalloc( &dX, m*n );
    magma_dmalloc( &dT, m*nrhs );
    if ( dX == NULL || dT == NULL ) {
        fprintf( stderr, "malloc failed - not enough GPU or system memory?\n" );
        goto cleanup;
    }

    magma_dsetmatrix( m, n, X, m, dX, m );
    magma_dsetmatrix( m, nrhs, T, m, dT, m );

    time = magma_sync_wtime( NULL );
    magma_dgemm( MagmaTrans, MagmaNoTrans, n, nrhs, m,
                 1, dX, m,
                    dT, m,
                 1, dB, n );
    magma_dgemm( MagmaTrans, MagmaNoTrans, n, n, m,
                 1, dX, m,
                    dX, m,
                 1, dA, n );
    time = magma_sync_wtime( NULL ) - time;
    fprintf( stdout, "added data in %f sec\n", time );


  cleanup:
    magma_free( dX );
    magma_free( dT );
};



// Solve dA * dX = dB, where dA and dX are stored in GPU device memory.
// Internally, MAGMA uses a hybrid CPU + GPU algorithm.
void GpuSolver::solve( magmaDouble_ptr X )
{

    real_Double_t   gpu_time;
    magmaDouble_ptr dX=NULL, dWORKD=NULL, Z=NULL;
    float *dWORKS=NULL;
    magma_int_t qrsv_iters;
    magma_int_t info = 0;
    
    magma_dmalloc( &dX, n*nrhs );
    magma_dmalloc( &dWORKD, n*nrhs );
    magma_smalloc( &dWORKS, n*(n+nrhs) );
    if ( dX == NULL || dWORKD == NULL || dWORKS == NULL ) {
        fprintf( stderr, "malloc failed - not enough GPU memory?\n" );
        goto cleanup;
    }    
    
    gpu_time = magma_wtime();
    magma_dsposv_gpu( MagmaUpper, n, nrhs,
                      dA, n, dB, n, dX, n, 
                      dWORKD, dWORKS, &qrsv_iters, &info );
    gpu_time = magma_wtime() - gpu_time;
    fprintf( stdout, "DSPOSV GPU solution time = %fs\n", gpu_time);
    if ( qrsv_iters == -3 ) {fprintf( stderr, "cannot factor input matrix in single precision, bad initialization?\n"); }
    if ( info != 0 ) { fprintf( stderr, "magma_dsposv_gpu failed with info=%d\n", info ); }

    magma_dgetmatrix( n, nrhs, dX, n, X, n );
    
cleanup:
    magma_free( dX );
}
    


// ------------------------------------------------------------


void dfill_matrix( magma_int_t m, magma_int_t n, magmaDouble_ptr A, magma_int_t lda )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    
    magma_int_t i, j;
    for( j=0; j < n; ++j ) {
        for( i=0; i < m; ++i ) {
            A(i,j) = rand() / ((double) RAND_MAX); 
        }
    }
}


int main( int argc, char** argv )
{
    
    magma_int_t n = 15000;
    magma_int_t lda  = n;
    magma_int_t ldx  = lda;
    magma_int_t nrhs = 1000;
    
    printf( "using MAGMA GPU interface\n" );

    GpuSolver g = GpuSolver(n, nrhs);

    // fill some data
    magmaDouble_ptr A=NULL, B=NULL;
    magma_dmalloc_cpu( &A, n*n );
    magma_dmalloc_cpu( &B, n*nrhs );
    for (int i = 0; i < 2; i++) {
        dfill_matrix( n, n, A, n );
        dfill_matrix( n, nrhs, B, n );    
        g.add_data( n, A, B );
    };


    magmaDouble_ptr X=NULL;
    magma_dmalloc_cpu( &X, n*nrhs );

    g.solve( X );

    magma_free_cpu( A );
    magma_free_cpu( B );
    magma_free_cpu( X );

    return 0;
}















































