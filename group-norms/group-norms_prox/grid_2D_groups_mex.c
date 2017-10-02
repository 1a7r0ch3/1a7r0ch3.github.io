/*==================================================================
 * G = grid_2D_groups_mex([ROI|Xsz], si, sj, shi, shj);
 *
 * Create list of indices constituing a regular nonoverlaping two dimensional
 * block structure.
 *
 * Hugo Raguet 2015
 *================================================================*/

#include "mex.h"
#include "../src/group_norms.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* retrieve inputs */
    const int N = mxGetNumberOfElements(prhs[0]);
    const bool *ROI = N > 2  ?  (bool *) mxGetLogicals(prhs[0])  :  NULL;
    int I, J;
    if (N > 2){
        I = mxGetM(prhs[0]);
        J = mxGetN(prhs[0]);
    }else{
        const double *s = (double*) mxGetData(prhs[0]);
        I = (int) s[0];
        J = N > 1  ?  (int) s[1]  :  I;
    }
    const int si = (int) mxGetScalar(prhs[1]);
    const int sj = (int) mxGetScalar(prhs[2]);
    const int shi = (int) mxGetScalar(prhs[3]);
    const int shj = (int) mxGetScalar(prhs[4]);
    /***  compute indices  ***/
    int **G = grid_2D_groups(ROI, I, J, si, sj, shi, shj);
    /* warning: stuff in G are allocated using malloc
     * risk of memory leak if program returns or crash before it is freed */
    /***  create and fill output  ***/
    const int sz[] = {1, G[0][0] + 1};
    plhs[0] = mxCreateCellArray(2, sz);
    mxArray *cell;
    int *cptr;
    /* group structure meta-info */
    cell = mxCreateNumericMatrix(1, 2, mxINT32_CLASS, mxREAL);
    cptr = mxGetData(cell);
    /* copy */
    cptr[0] = G[0][0];
    cptr[1] = G[0][1];
    mxSetCell(plhs[0], 0, cell);
    /* free */
    free(G[0]);
    /* groups indices */
    int g, i;
    for (g=1; g<sz[1]; g++){
        cell = mxCreateNumericMatrix(1, G[g][0] + 1, mxINT32_CLASS, mxREAL);
        cptr = mxGetData(cell);
        /* copy */
        for (i=0; i<=G[g][0]; i++){ cptr[i] = G[g][i]; }
        mxSetCell(plhs[0], g, cell);
        /* free */
        free(G[g]);
    }
    free(G);
}
