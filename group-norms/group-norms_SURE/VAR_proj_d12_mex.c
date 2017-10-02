/*==================================================================
 * V = VAR_proj_d12_mex(X, G, Rb, A, La);
 *
 * Parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include "mex.h"
#include "../src/SURE.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const int I = mxGetM(prhs[0]);
    const int J = mxGetN(prhs[0]);
    const int N = mxGetNumberOfElements(prhs[1]);
    const int L = mxGetM(prhs[3]);
    const int ***G = mxMalloc(sizeof(int**)*N); 
    int n, g, nG;

    if (mxIsDouble(prhs[0])){
        const double **Rb = mxMalloc(sizeof(double*)*N);
        for (n=0; n<N; n++){
            Rb[n] = (double*) mxGetData(mxGetCell(prhs[2], n));
            nG = mxGetN(mxGetCell(prhs[2], n));
            G[n] = mxMalloc(sizeof(int*)*(nG+1));
            for (g=0; g<=nG; g++){
                G[n][g] = (int*) mxGetData(mxGetCell(mxGetCell(prhs[1], n), g));
            }
        }
        const double *La = (double*) mxGetData(prhs[3]);
        plhs[0] = mxCreateNumericMatrix(L, I, mxDOUBLE_CLASS, mxREAL);
        double* V = (double*) mxGetData(plhs[0]);
        if (mxIsComplex(prhs[0])){
            const double *Xr = (double*) mxGetData(prhs[0]);
            const double *Xi = (double*) mxGetImagData(prhs[0]);
            VAR_proj_d12_cplx_double(Xr, Xi, G, Rb, La, V, I, J, N, L);
        }else{
            const double *X = (double*) mxGetData(prhs[0]);
            VAR_proj_d12_real_double(X, G, Rb, La, V, I, J, N, L);
        }
    }else{
        const float **Rb = mxMalloc(sizeof(float*)*N);
        for (n=0; n<N; n++){
            Rb[n] = (float*) mxGetData(mxGetCell(prhs[2], n));
            nG = mxGetN(mxGetCell(prhs[2], n));
            G[n] = mxMalloc(sizeof(int*)*(nG+1));
            for (g=0; g<=nG; g++){
                G[n][g] = (int*) mxGetData(mxGetCell(mxGetCell(prhs[1], n), g));
            }
        }
        const float *La = (float*) mxGetData(prhs[3]);
        plhs[0] = mxCreateNumericMatrix(L, I, mxSINGLE_CLASS, mxREAL);
        float* V = (float*) mxGetData(plhs[0]);
        if (mxIsComplex(prhs[0])){
            const float *Xr = (float*) mxGetData(prhs[0]);
            const float *Xi = (float*) mxGetImagData(prhs[0]);
            VAR_proj_d12_cplx_single(Xr, Xi, G, Rb, La, V, I, J, N, L);
        }else{
            const float *X = (float*) mxGetData(prhs[0]);
            VAR_proj_d12_real_single(X, G, Rb, La, V, I, J, N, L);
        }
    }
}
