/*==================================================================
 * parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include <omp.h>
#include <math.h>

void proj_l1inf_real_double(double *X, const int **G, const double *Bnd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	double bnd; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, bnd)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            bnd = Bnd[idg];
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                X[idx] = bnd < X[idx]  ?  bnd  :  X[idx];
				X[idx] = -bnd > X[idx]  ?  -bnd  :  X[idx];
			}
		}
	}
}

void proj_l1inf_cplx_double(double *Xr, double *Xi, const int **G, const double *Bnd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	double bnd, mod2, shrink; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, bnd, mod2, shrink)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            bnd = Bnd[idg];
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
				mod2 = Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx];
			    if (mod2 > bnd*bnd){
				    shrink = bnd/sqrt(mod2);
                    Xr[idx] *= shrink;
                    Xi[idx] *= shrink;
                }
            }
		}
	}
}

void proj_l1inf_real_single(float *X, const int **G, const float *Bnd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	float bnd; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, bnd)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            bnd = Bnd[idg];
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                X[idx] = bnd < X[idx]  ?  bnd  :  X[idx];
				X[idx] = -bnd > X[idx]  ?  -bnd  :  X[idx];
			}
		}
	}
}

void proj_l1inf_cplx_single(float *Xr, float *Xi, const int **G, const float *Bnd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	float bnd, mod2, shrink; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, bnd, mod2, shrink)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            bnd = Bnd[idg];
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
				mod2 = Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx];
			    if (mod2 > bnd*bnd){
				    shrink = bnd/sqrtf(mod2);
                    Xr[idx] *= shrink;
                    Xi[idx] *= shrink;
                }
            }
		}
	}
}
