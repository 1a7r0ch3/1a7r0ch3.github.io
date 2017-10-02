/*==================================================================
 * Y = proj_d12_mex(X, G, Bnd);
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
        const double *Bnd = (double*) mxGetData(prhs[2]);
        plhs[0] = mxDuplicateArray(prhs[0]);
        if (mxIsComplex(plhs[0])){
            double *Yr = (double*) mxGetData(plhs[0]);
            double *Yi = (double*) mxGetImagData(plhs[0]);
            proj_d12_cplx_double(Yr, Yi, G, Bnd, I);
        }else{
            double *Y = (double*) mxGetData(plhs[0]);
            proj_d12_real_double(Y, G, Bnd, I);
        }
    }else{
        const float *Bnd = (float*) mxGetData(prhs[2]);
        plhs[0] = mxDuplicateArray(prhs[0]);
        if (mxIsComplex(plhs[0])){
            float *Yr = (float*) mxGetData(plhs[0]);
            float *Yi = (float*) mxGetImagData(plhs[0]);
            proj_d12_cplx_single(Yr, Yi, G, Bnd, I);
        }else{
            float *Y = (float*) mxGetData(plhs[0]);
            proj_d12_real_single(Y, G, Bnd, I);
        }
    }
    mxFree(G);
}
