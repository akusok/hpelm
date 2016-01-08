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
    double* ones;

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

    stat = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("Setting pointer mode failed"); }
    fprintf(stdout, "Setting pointer mode success\n");

    stat = cublasSetMatrix (L, L, sizeof(*HH), HH, L, devHH, L);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("devHH upload failed"); }
    fprintf(stdout, "devHH upload success\n");

    stat = cublasSetMatrix (C, L, sizeof(*HT), HT, C, devHT, C);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("devHT upload failed"); }
    fprintf(stdout, "devHT upload success\n");


    ones = (double*) malloc (1*sizeof(double));
    ones[0] = 1.0;
    cudaMalloc((void**)&one, 1*sizeof(double));
    cublasSetVector(1, sizeof(*ones), ones, 1, one, 1);
    fprintf(stdout, "here\n");

};


// update covariance matrices with new data
void CudaSolver::add_data ( int N, double* H, double* T ) {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    double* devH;
    double* devT;

    cudaStat = cudaMalloc((void**)&devH, N*L*sizeof(*H));
    if (cudaStat != cudaSuccess) { printf ("devH device memory allocation failed"); goto cleanup; }
    fprintf(stdout, "Allocating devH success\n");

    cudaStat = cudaMalloc((void**)&devT, N*C*sizeof(*T));
    if (cudaStat != cudaSuccess) { printf ("devT device memory allocation failed"); goto cleanup; }
    fprintf(stdout, "Allocating devT success\n");

    stat = cublasSetMatrix (N, L, sizeof(*H), H, N, devH, N);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("devH upload failed"); }
    fprintf(stdout, "devH upload success\n");

    stat = cublasSetMatrix (N, C, sizeof(*T), T, N, devT, N);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("devT upload failed"); }
    fprintf(stdout, "devT upload success\n");


    stat = cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                       L, N, one, devH, L, one, devHH, L);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("H'H update failed"); }
    fprintf(stdout, "H'H update success\n");

//
//    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
//                       L, C, N,
//                       one, devH,  L,//L
//                            devT,  C,//C
//                       one, devHT, L);

    if (stat != CUBLAS_STATUS_SUCCESS) { printf ("H'T update failed\n"); }
    fprintf(stdout, "H'T update success\n");


   cleanup:
    cudaFree (devH);
    fprintf(stdout, "devH freed success\n");
    cudaFree (devT);
    fprintf(stdout, "devT freed success\n");


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
    cudaFree (one);
    cublasDestroy(handle);
}

















