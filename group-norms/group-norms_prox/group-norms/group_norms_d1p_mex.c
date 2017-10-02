/*==================================================================
 * Nb = group_norms_d1p_mex(X, G, p, W);
 *
 * parallel implementation with OpenMP API
 *
 * Hugo Raguet 2015
 *==================================================================*/

#include "mex.h"
#include "../src/group_norms.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const int I = mxGetM(prhs[0]);
    const int N = (int) mxGetScalar(mxGetCell(prhs[1], 0));
    const int **G = mxMalloc(sizeof(int*)*(N+1));
    int g;
    for (g=0; g<=N; g++){
        G[g] = (int*) mxGetData(mxGetCell(prhs[1], g));
    }

    if (mxIsDouble(prhs[0])){
        const double p = nrhs > 2  ?  (double) mxGetScalar(prhs[2])  :  2.;
        const double *W = nrhs > 3  ?  (double*) mxGetData(prhs[3])  :  NULL;
        plhs[0] = mxCreateNumericMatrix(I, N, mxDOUBLE_CLASS, mxREAL);
        double *Nb = (double*) mxGetData(plhs[0]);
        if (mxIsComplex(prhs[0])){
            const double *Xr = (double*) mxGetData(prhs[0]);
            const double *Xi = (double*) mxGetImagData(prhs[0]);
            group_norms_d1p_cplx_double(Nb, Xr, Xi, G, I, p, W);
        }else{
            const double *X = (double*) mxGetData(prhs[0]);
            group_norms_d1p_real_double(Nb, X, G, I, p, W);
        }
    }else{
        const float p = nrhs > 2  ?  (float) mxGetScalar(prhs[2])  :  2.;
        const float *W = nrhs > 3  ?  (float*) mxGetData(prhs[3])  :  NULL;
        plhs[0] = mxCreateNumericMatrix(I, N, mxSINGLE_CLASS, mxREAL);
        float *Nb = (float*) mxGetData(plhs[0]);
        if (mxIsComplex(prhs[0])){
            const float *Xr = (float*) mxGetData(prhs[0]);
            const float *Xi = (float*) mxGetImagData(prhs[0]);
            group_norms_d1p_cplx_single(Nb, Xr, Xi, G, I, p, W);
        }else{
            const float *X = (float*) mxGetData(prhs[0]);
            group_norms_d1p_real_single(Nb, X, G, I, p, W);
        }
    }
    mxFree(G);
}
