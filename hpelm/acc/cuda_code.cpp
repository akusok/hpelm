#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// H_(N,L), T_(N,C), HH(L,L), HT(L,C)

class CudaSolver {
        int L, C;
        double* devHH;
        double* devHT;
        double* one;
        double* ones;
        cublasHandle_t handle;
    public:
        CudaSolver( int, int, double*, double* );
        void get_corr( double* HH, double* HT );
        void add_data( int N, double* H, double* T );
        void finalize();
};


// init a zero matrix on GPU to store X'*X, add normalization
CudaSolver::CudaSolver ( int nL, int nC, double* HH, double* HT ) {
    cudaError_t cudaStat;
    cublasStatus_t stat;

    L = nL;
    C = nC;


    cudaStat = cudaMalloc((void**)&devHH, L*L*sizeof(*HH));
    if (cudaStat != cudaSuccess) { printf ("devHH device memory allocation failed"); }
    fprintf(stdout, "Allocating devHH success\n");

    cudaStat = cudaMalloc((void**)&devHT, L*C*sizeof(*HT));
    if (cudaStat != cudaSuccess) { printf ("devHT device memory allocation failed"); }
    fprintf(stdout, "Allocating devHT success\n");

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed\n"); }
    fprintf(stdout, "CUBLAS initialization success\n");

    stat = cublasSetMatrix (L, L, sizeof(*HH), HH, L, devHH, L);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("devHH upload failed"); }
    fprintf(stdout, "devHH upload success\n");

    stat = cublasSetMatrix (C, L, sizeof(*HT), HT, C, devHT, C);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("devHT upload failed"); }
    fprintf(stdout, "devHT upload success\n");
};


// update covariance matrices with new data
void CudaSolver::add_data ( int N, double* H, double* T ) {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    double* devH;
    double* devT;
    double one = 1.0;

    cudaStat = cudaMalloc((void**)&devH, N*L*sizeof(*H));
    if (cudaStat != cudaSuccess) { printf ("devH device memory allocation failed"); goto cleanup1; }
//    fprintf(stdout, "Allocating devH success\n");

    cudaStat = cudaMalloc((void**)&devT, N*C*sizeof(*T));
    if (cudaStat != cudaSuccess) { printf ("devT device memory allocation failed"); goto cleanup2; }
//    fprintf(stdout, "Allocating devT success\n");

    stat = cublasSetMatrix (N, L, sizeof(*H), H, N, devH, N);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("devH upload failed"); }
//    fprintf(stdout, "devH upload success\n");

    stat = cublasSetMatrix (N, C, sizeof(*T), T, N, devT, N);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("devT upload failed"); }
//    fprintf(stdout, "devT upload success\n");


    stat = cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                       L, N, &one, devH, N, &one, devHH, L);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("H'H update failed\n"); }
//    fprintf(stdout, "H'H update success\n");

    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                       L, C, N,
                       &one, devH,  N,
                            devT,  N,
                       &one, devHT, L);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("H'T update failed\n"); }
//    fprintf(stdout, "H'T update success\n");


   cleanup2:
    cudaFree (devT);
//    fprintf(stdout, "devT freed success\n");
   cleanup1:
    cudaFree (devH);
//    fprintf(stdout, "devH freed success\n");

};


// return current covariance matrices
void CudaSolver::get_corr ( double* HH, double* HT ) {
    cublasStatus_t stat;

    stat = cublasGetMatrix (L, L, sizeof(*HH), devHH, L, HH, L);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("devHH download failed"); }   

    stat = cublasGetMatrix (L, C, sizeof(*HT), devHT, L, HT, L);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("devHT download failed"); }   
};


// free memory
void CudaSolver::finalize( ) {
    fprintf(stdout, "Solver finalized\n");
    cudaFree (devHH);
    cudaFree (devHT);
    cublasDestroy(handle);
}

















