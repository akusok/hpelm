#include <math.h>

void mp_func(double *H, const int *f, int n, int k)
{
    int i, j, ofc;
    #pragma omp parallel for private(i, j, ofc)
    for (i=0; i<n; i++) {
        for (j=0; j<k; j++) {
            if (f[j] == 1) {
                ofc = i*k + j;
                H[ofc] = tanh(H[ofc]);
            }
        }    
    }
}