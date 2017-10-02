/*==================================================================
 * Hugo Raguet 2016
 *================================================================*/

#include "mex.h"
#include "../include/SURE.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const int V = mxGetNumberOfElements(prhs[0]);
    const int E = mxGetNumberOfElements(prhs[2]);
    const int L = mxGetNumberOfElements(prhs[3]);
    const int *Eu = (int*) mxGetData(prhs[4]);
    const int *Ev = (int*) mxGetData(prhs[5]);
    const int verbose = (int) mxGetScalar(prhs[6]);
    if (mxIsDouble(prhs[0])){
        plhs[0] = mxCreateNumericMatrix(L, 1, mxDOUBLE_CLASS, mxREAL);
        plhs[1] = mxCreateNumericMatrix(L, 1, mxDOUBLE_CLASS, mxREAL);
        plhs[2] = mxCreateNumericMatrix(V, 1, mxDOUBLE_CLASS, mxREAL);
        double *SURE = (double*) mxGetData(plhs[0]);
        double *VAR = (double*) mxGetData(plhs[1]);
        double *W = (double*) mxGetData(plhs[2]);
        const double *Y = (double*) mxGetData(prhs[0]);
        const double *S2 = (double*) mxGetData(prhs[1]);
        const double *Mu = (double*) mxGetData(prhs[2]);
        const double *La = (double*) mxGetData(prhs[3]);
        SURE_VAR_prox_graph_d1<double>(SURE, VAR, W, Y, S2, Mu, La, L, V, E, Eu, Ev, verbose);
    }else{
        plhs[0] = mxCreateNumericMatrix(L, 1, mxSINGLE_CLASS, mxREAL);
        plhs[1] = mxCreateNumericMatrix(L, 1, mxSINGLE_CLASS, mxREAL);
        plhs[2] = mxCreateNumericMatrix(V, 1, mxSINGLE_CLASS, mxREAL);
        float *SURE = (float*) mxGetData(plhs[0]);
        float *VAR = (float*) mxGetData(plhs[1]);
        float *W = (float*) mxGetData(plhs[2]);
        const float *Y = (float*) mxGetData(prhs[0]);
        const float *S2 = (float*) mxGetData(prhs[1]);
        const float *Mu = (float*) mxGetData(prhs[2]);
        const float *La = (float*) mxGetData(prhs[3]);
        SURE_VAR_prox_graph_d1<float>(SURE, VAR, W, Y, S2, Mu, La, L, V, E, Eu, Ev, verbose);
    }
}
