/*==================================================================
 * SURE = SURE_prox_rw12_mex(SURE, La, Yb, Sb2, S2_Yb, Mu);
 *
 * Parallel implementation with OpenMP API
 *
 * Hugo Raguet 2014
 *================================================================*/

#include "mex.h"
#include "../src/SURE.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    plhs[0] = mxDuplicateArray(prhs[0]);
    const int K = mxGetN(prhs[0]);
    const int L = mxGetM(prhs[0]);
    const int M = mxGetM(prhs[3]);
    if (mxIsDouble(prhs[0])){
        double *SURE = (double*) mxGetData(plhs[0]);
        const double *La = (double*) mxGetData(prhs[1]);
        const int *Idx = (int*) mxGetData(prhs[2]);
        const double *Yb = (double*) mxGetData(prhs[3]);
        const double *Sb2 = (double*) mxGetData(prhs[4]);
        const double *S2_Yb = (double*) mxGetData(prhs[5]);
        const double *Mu = (double*) mxGetData(prhs[6]);
        SURE_prox_rw12_double(SURE, La, Idx, Yb, Sb2, S2_Yb, Mu, K, L, M);
    }else{
        float *SURE = (float*) mxGetData(plhs[0]);
        const float *La = (float*) mxGetData(prhs[1]);
        const int *Idx = (int*) mxGetData(prhs[2]);
        const float *Yb = (float*) mxGetData(prhs[3]);
        const float *Sb2 = (float*) mxGetData(prhs[4]);
        const float *S2_Yb = (float*) mxGetData(prhs[5]);
        const float *Mu = (float*) mxGetData(prhs[6]);
        SURE_prox_rw12_single(SURE, La, Idx, Yb, Sb2, S2_Yb, Mu, K, L, M);
    }
}
