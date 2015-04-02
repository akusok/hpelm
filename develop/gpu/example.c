// This is a simple standalone example. See README.txt

#include <stdio.h>
#include <stdlib.h>

//#include "cublas_v2.h"     // if you need CUBLAS, include before magma.h
#include "magma.h"
//#include "magma_lapack.h"  // if you need BLAS & LAPACK


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


void dprint( magma_int_t m, magma_int_t n, magmaDouble_ptr A, magma_int_t lda )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    
    magma_int_t i, j;
    for( j=0; j < n; ++j ) {
        for( i=0; i < m; ++i ) {
            fprintf(stdout, "%.03f ", A(i,j)); 
        }
        //fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}

void dprint2( magma_int_t m, magma_int_t n, magmaDouble_ptr A, magma_int_t lda )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    
    magma_int_t i, j;
    for( j=0; j < n; ++j ) {
        for( i=0; i < m; ++i ) {
            fprintf(stdout, "%.03f ", A(i,j)); 
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}

// ------------------------------------------------------------
// Solve dA * dX = dB, where dA and dX are stored in GPU device memory.
// Internally, MAGMA uses a hybrid CPU + GPU algorithm.
void gpu_solve( magma_int_t n, magma_int_t nrhs, magmaDouble_ptr A, magmaDouble_ptr B, magmaDouble_ptr X )
{
    magma_init();

    real_Double_t   gpu_time;
    magmaDouble_ptr dA=NULL, dB=NULL, dX=NULL, dWORKD=NULL;
    float *dWORKS=NULL;
    magma_int_t qrsv_iters;
    magma_int_t info = 0;
    
    magma_dmalloc( &dA, n*n );
    magma_dmalloc( &dB, n*nrhs );
    magma_dmalloc( &dX, n*nrhs );
    magma_dmalloc( &dWORKD, n*nrhs );
    magma_smalloc( &dWORKS, n*(n+nrhs) );
    if ( dA == NULL || dB == NULL || dX == NULL || dWORKD == NULL || dWORKS == NULL ) {
        fprintf( stderr, "malloc failed - not enough GPU memory?\n" );
        goto cleanup;
    }
    
    // send data to GPU (round n to ldda)
    magma_dsetmatrix( n, n, A, n, dA, n );
    magma_dsetmatrix( n, nrhs, B, n, dB, n );

    //gpu_time = magma_wtime();
    //magma_dsgesv_gpu( MagmaNoTrans, n, nrhs,
    //                  dA, ldda, ipiv, d_ipiv,
    //                  dB, lddb, dX, lddx, 
    //                  dWORKD, dWORKS, &qrsv_iters, &info );
    //gpu_time = magma_wtime() - gpu_time;
    //fprintf( stdout, "DSGESV GPU time = %fs with %d iterations (info %d)\n", gpu_time, qrsv_iters, info);

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
    magma_free( dA );
    magma_free( dB );
    magma_free( dX );
    magma_finalize();
}




// ------------------------------------------------------------
int main( int argc, char** argv )
{
    magma_init();
    
    magma_int_t n = 20000;
    magma_int_t lda  = n;
    magma_int_t ldx  = lda;
    magma_int_t nrhs = 1000;
    
    printf( "using MAGMA GPU interface\n" );

    magmaDouble_ptr A=NULL, B=NULL, X=NULL;
    magma_dmalloc_cpu( &A, lda*n );
    magma_dmalloc_cpu( &B, ldx*nrhs );
    magma_dmalloc_cpu( &X, n*nrhs );

    dfill_matrix( n, n, A, n );
    dfill_matrix( n, nrhs, B, n );    
    gpu_solve( n, nrhs, A, B, X );

    magma_free_cpu( A );
    magma_free_cpu( B );
    magma_free_cpu( X );

    magma_finalize();
    return 0;
}















































