#include <stdio.h>
#include <stdlib.h>
#include "magma.h"

class GpuSolver {
        int n, nrhs;
        magmaDouble_ptr dA=NULL, dB=NULL;
        magma_queue_t queue;
    public:
        GpuSolver( int, int, magmaDouble_ptr A, magmaDouble_ptr B );
        void add_data( magma_int_t m, magmaDouble_ptr X, magmaDouble_ptr T );
        void get_corr( magmaDouble_ptr XX, magmaDouble_ptr XT );
        void solve( magmaDouble_ptr X );
        void finalize();
};



// init a zero matrix on GPU to store X'*X, add normalization
GpuSolver::GpuSolver ( int nn, int outs, magmaDouble_ptr A, magmaDouble_ptr B ) {

    magma_int_t err;
    magma_int_t num = 0;
    magma_device_t device;
    magma_init();

    err = magma_get_devices( &device, 1, &num );
    if ( err != MAGMA_SUCCESS or num < 1 ) {
        fprintf( stderr, "magma_get_devices failed: %d\n", (int) err );
        exit(-1);
    }
    err = magma_queue_create( device, &queue );
    if ( err != MAGMA_SUCCESS ) {
        fprintf( stderr, "magma_queue_create failed: %d\n", (int) err );
        exit(-1);
    }

    n = nn;
    nrhs = outs;

    magma_dmalloc( &dA, n*n );
    magma_dmalloc( &dB, n*nrhs );
    if ( dA == NULL || dB == NULL ) { fprintf( stderr, "malloc failed - not enough GPU memory?\n" ); }

    magma_dsetmatrix( n, n, A, 0, n, dA, 0, n, queue );
    magma_dsetmatrix( n, nrhs, B, 0, n, dB, 0, n, queue );
};


// update covariance matrices with new data
void GpuSolver::add_data ( magma_int_t m, magmaDouble_ptr X, magmaDouble_ptr T ) {

    real_Double_t   time;
    magmaDouble_ptr dX=NULL, dT=NULL;

    magma_dmalloc( &dX, m*n );
    magma_dmalloc( &dT, m*nrhs );
    if ( dX == NULL || dT == NULL ) {
        fprintf( stderr, "malloc failed - not enough GPU memory?\n" );
        goto cleanup;
    }

    magma_dsetmatrix( m, n, X, 0, m, dX, 0, m, queue );
    magma_dsetmatrix( m, nrhs, T, 0, m, dT, 0, m, queue );

    // time = magma_sync_wtime( NULL );
    magma_dgemm( MagmaTrans, MagmaNoTrans, n, nrhs, m,
                 1, dX, 0, m,
                    dT, 0, m,
                 1, dB, 0, n, queue );
    magma_dgemm( MagmaTrans, MagmaNoTrans, n, n, m,
                 1, dX, 0, m,
                    dX, 0, m,
                 1, dA, 0, n, queue );
    // time = magma_sync_wtime( NULL ) - time;
    // fprintf( stdout, "added data in %f sec\n", time );

  cleanup:
    magma_free( dX );
    magma_free( dT );
};



// return current covariance matrices
void GpuSolver::get_corr ( magmaDouble_ptr XX, magmaDouble_ptr XT ) {

    magma_dgetmatrix( n, n, dA, 0, n, XX, 0, n, queue );
    magma_dgetmatrix( n, nrhs, dB, 0, n, XT, 0, n, queue );
};



// free memory
void GpuSolver::finalize( ) {
    magma_free( dA );
    magma_free( dB );
    magma_finalize();
}


// Solve dA * dX = dB, where dA and dX are stored in GPU device memory.
// Internally, MAGMA uses a hybrid CPU + GPU algorithm.
void GpuSolver::solve( magmaDouble_ptr X )
{

    //real_Double_t   gpu_time;
    //magmaDouble_ptr dX=NULL, dWORKD=NULL;
    //float *dWORKS=NULL;
    magma_int_t qrsv_iters;
    magma_int_t info = 0;
    
    //magma_dmalloc( &dX, n*nrhs );
    //magma_dmalloc( &dWORKD, n*nrhs );
    //magma_smalloc( &dWORKS, n*(n+nrhs) );
    //if ( dX == NULL || dWORKD == NULL || dWORKS == NULL ) {
    //    fprintf( stderr, "malloc failed - not enough GPU memory?\n" );
    //    goto cleanup;
    //}    
    
    // gpu_time = magma_wtime();
    magma_dsposv_mic( MagmaUpper, n, nrhs,
                      dA, 0, n, dB, 0, n, 
                      &info, queue );
    // gpu_time = magma_wtime() - gpu_time;
    // fprintf( stdout, "DSPOSV GPU solution time = %fs\n", gpu_time);
    if ( qrsv_iters == -3 ) {fprintf( stderr, "cannot factor input matrix in single precision, bad initialization?\n"); }
    if ( info != 0 ) { fprintf( stderr, "magma_dsposv_gpu failed with info=%d\n", info ); }

    magma_dgetmatrix( n, nrhs, dB, 0, n, X, 0, n, queue );
    
//cleanup:
    //magma_free( dX );
    //magma_free( dWORKD );
    //magma_free( dWORKS );
}
    



// ------------------------------------------------------------
// ------------------------------------------------------------


// Independent solver for dA * dX = dB, where dA and dX are stored in GPU device memory.
// Internally, MAGMA uses a hybrid CPU + GPU algorithm.
//void solve_corr( magma_int_t n, magma_int_t nrhs, magmaDouble_ptr A, magmaDouble_ptr B, magmaDouble_ptr X )
//{
//    magma_init();
//
//    real_Double_t   gpu_time;
//    magmaDouble_ptr dA=NULL, dB=NULL, dX=NULL, dWORKD=NULL;
//    float *dWORKS=NULL;
//    magma_int_t qrsv_iters;
//    magma_int_t info = 0;
//    
//    magma_dmalloc( &dA, n*n );
//    magma_dmalloc( &dB, n*nrhs );
//    magma_dmalloc( &dX, n*nrhs );
//    magma_dmalloc( &dWORKD, n*nrhs );
//    magma_smalloc( &dWORKS, n*(n+nrhs) );
//    if ( dA == NULL || dB == NULL || dX == NULL || dWORKD == NULL || dWORKS == NULL ) {
//        fprintf( stderr, "malloc failed - not enough GPU memory?\n" );
//        goto cleanup;
//    }
//    
//    // send data to GPU (round n to ldda)
//    magma_dsetmatrix( n, n, A, n, dA, n );
//    magma_dsetmatrix( n, nrhs, B, n, dB, n );
//
//    gpu_time = magma_wtime();
//    magma_dsposv_gpu( MagmaUpper, n, nrhs,
//                      dA, n, dB, n, dX, n, 
//                      dWORKD, dWORKS, &qrsv_iters, &info );
//    gpu_time = magma_wtime() - gpu_time;
//    fprintf( stdout, "DSPOSV GPU solution time = %fs\n", gpu_time);
//    if ( qrsv_iters == -3 ) {fprintf( stderr, "cannot factor input matrix in single precision, bad initialization?\n"); }
//    if ( info != 0 ) { fprintf( stderr, "magma_dsposv_gpu failed with info=%d\n", info ); }
//
//    magma_dgetmatrix( n, nrhs, dX, n, X, n );
//    
//cleanup:
//    magma_free( dA );
//    magma_free( dB );
//    magma_free( dX );
//    magma_free( dWORKD );
//    magma_free( dWORKS );
//    magma_finalize();
//}
















































