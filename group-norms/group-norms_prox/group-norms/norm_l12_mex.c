/*==================================================================
 * norm = norm_l12_mex(X, G, La);
 *
 * Parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include "mex.h"
#include "../src/group_norms.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const int I = mxGetM(prhs[0]);
    const int N = mxGetN(prhs[2]);
    const int **G = mxMalloc(sizeof(int*)*(N+1));
    int g;
    for (g=0; g<=N; g++){
        G[g] = (int*) mxGetData(mxGetCell(prhs[1], g));
    }
    
    if (mxIsDouble(prhs[0])){
        const double *La = (double*) mxGetData(prhs[2]);
        plhs[0] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
        double *norm = (double*) mxGetData(plhs[0]);
        if (mxIsComplex(prhs[0])){
            const double *Xr = (double*) mxGetData(prhs[0]);
            const double *Xi = (double*) mxGetImagData(prhs[0]);
            *norm = norm_l12_cplx_double(Xr, Xi, G, La, I);
        }else{
            const double *X = (double*) mxGetData(prhs[0]);
            *norm = norm_l12_real_double(X, G, La, I);
        }
    }else{
        const float *La = (float*) mxGetData(prhs[2]);
        plhs[0] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        float *norm = (float*) mxGetData(plhs[0]);
        if (mxIsComplex(prhs[0])){
            const float *Xr = (float*) mxGetData(prhs[0]);
            const float *Xi = (float*) mxGetImagData(prhs[0]);
            *norm = norm_l12_cplx_single(Xr, Xi, G, La, I);
        }else{
            const float *X = (float*) mxGetData(prhs[0]);
            *norm = norm_l12_real_single(X, G, La, I);
        }
    }
    mxFree(G);
}
