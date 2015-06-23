#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>  // replace with proper Lapack headers

void load_file(double *X, int n, int d)
{
	FILE *myfile;

	myfile = fopen("X.bin","r");
	fread(X, sizeof(double), n*d, myfile);
	fclose(myfile);
	printf("X: \n");
	/*for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++) {
			printf("%f ", U[i + j*M]);
		}
		printf("\n");
	}*/
}

void my_lls()
{
	// M - number of samples
	// N - number of columns

	int M = 10000;
	int N = 5;
	int d = sizeof(double);

	double *A = (double*) malloc(M*N*d);
	double *s = (double*) malloc(N*d);
	double *U = (double*) malloc(M*N*d);
	double *VT = (double*) malloc(N*N*d);

	FILE *myfile;
	myfile = fopen("A.bin","r");
	fread(A, d, M*N, myfile);
	fclose(myfile);

	char S = 'S';
	int lwork = M + 3*N;
	lwork = (lwork < 5*N) ? 5*N : lwork;
	double *work = (double*) malloc(lwork*d);
	int info;

	// correct SVD
	dgesvd_(&S, &S, &M, &N, A, &M, s, U, &M, VT, &N, work, &lwork, &info);


	//*********************************************************************


	myfile = fopen("A.bin","r");
	fread(A, d, M*N, myfile);
	fclose(myfile);

	int K = 2;
	double *B = (double*) malloc(M*K*d);
	for (int i=0; i<M; i++) {
		B[i] = A[i];
		B[M+i] = A[M+i];
	}

	double *C = (double*) malloc(N*K*d);

	// correct dgemm
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, N, K, M, 1.0, A, M, B, M, 0.0, C, N);


	//*********************************************************************


	/*
	for (int i=0; i<M; i++) {
		for (int j=0; j<N; j++) {
			printf("%f ", U[i + j*M]);
		}
		printf("\n");
	}
	*/
}

void my_new()
{
	FILE *myfile;
	int M = 10000;
	int N = 5;
	int K = 2;
	int d = sizeof(double);

	double* H = (double*) malloc(M*N*d);
	double* Y = (double*) malloc(M*K*d);
	double* s = (double*) malloc(N*d);
	double rcond = -1.0;
	int rank;
	int info;

	myfile = fopen("H.bin","r");
	fread(H, d, M*N, myfile);
	fclose(myfile);

	myfile = fopen("Y.bin","r");
	fread(Y, d, M*N, myfile);
	fclose(myfile);

	int lwork = -1;
	double *work = (double*) malloc(2*d);
	dgelss_(&M, &N, &K, H, &M, Y, &M, s, &rcond, &rank, work, &lwork, &info); // lwork query

	lwork = (int) work[0];
	work = (double*) malloc(lwork*d);

	// correct Least Squares Solution
	dgelss_(&M, &N, &K, H, &M, Y, &M, s, &rcond, &rank, work, &lwork, &info);

	// return projection matrix
	double* W = (double*) malloc(N*K*d);
	for (int i=0; i<N; i++) {
		for (int j=0; j<K; j++) {
			W[i + j*N] = Y[i + j*M];
		}
	}

	myfile = fopen("W.bin","w");
	fwrite(W, d, N*K, myfile);
	fclose(myfile);


	printf("%d\n", info);
}


int main()
{
	my_new();
}
