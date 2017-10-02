/*==================================================================
 * parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include <omp.h>
#include <math.h>

void proj_l12_real_double(double *X, const int **G, const double *Bnd, const int I)
{
	int i, j, g, idg;
	const int *gidx;
	double x, bnd, ng2, shrink; 
    #pragma omp parallel for private(i, j, g, idg, gidx, x, bnd, ng2, shrink)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            bnd = Bnd[idg];
			ng2 = 0.;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                x = X[i+I*gidx[j]];
				ng2 += x*x;
			}
			if (ng2 > bnd*bnd){
				shrink = bnd/sqrt(ng2);
			    for (j=1; j<=gidx[0]; j++){
                    X[i+I*gidx[j]] *= shrink;
                }
            }
		}
	}
}

void proj_l12_cplx_double(double *Xr, double *Xi, const int **G, const double *Bnd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	double bnd, ng2, shrink; 
    #pragma omp parallel for private(i, j, g, idg, idx, gidx, bnd, ng2, shrink)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            bnd = Bnd[idg];
			ng2 = 0.;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
				ng2 += Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx];
			}
			if (ng2 > bnd*bnd){
				shrink = bnd/sqrt(ng2);
			    for (j=1; j<=gidx[0]; j++){
                    idx = i + I*gidx[j];
                    Xr[idx] *= shrink;
                    Xi[idx] *= shrink;
                }
            }
		}
	}
}

void proj_l12_real_single(float *X, const int **G, const float *Bnd, const int I)
{
	int i, j, g, idg;
	const int *gidx;
	float x, bnd, ng2, shrink; 
    #pragma omp parallel for private(i, j, g, idg, gidx, x, bnd, ng2, shrink)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            bnd = Bnd[idg];
			ng2 = 0.f;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                x = X[i+I*gidx[j]];
				ng2 += x*x;
			}
			if (ng2 > bnd*bnd){
				shrink = bnd/sqrtf(ng2);
			    for (j=1; j<=gidx[0]; j++){
                    X[i+I*gidx[j]] *= shrink;
                }
            }
		}
	}
}

void proj_l12_cplx_single(float *Xr, float *Xi, const int **G, const float *Bnd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	float bnd, ng2, shrink; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, bnd, ng2, shrink)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            bnd = Bnd[idg];
			ng2 = 0.f;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
				ng2 += Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx];
			}
			if (ng2 > bnd*bnd){
				shrink = bnd/sqrtf(ng2);
			    for (j=1; j<=gidx[0]; j++){
                    idx = i + I*gidx[j];
                    Xr[idx] *= shrink;
                    Xi[idx] *= shrink;
                }
            }
		}
	}
}
