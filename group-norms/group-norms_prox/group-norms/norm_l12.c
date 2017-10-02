/*==================================================================
 * parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include <omp.h>
#include <math.h>

double norm_l12_real_double(const double *X, const int **G, const double *La, const int I)
{
	int i, j, g, idg;
    const int *gidx;
	double x, ng2, norm = 0.;
    #pragma omp parallel for private(i, j, g, idg, gidx, x, ng2) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			ng2 = 0.;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                x = X[i+I*gidx[j]];
				ng2 += x*x;
			}
            norm += La[idg]*sqrt(ng2);
		}
	}
    return norm;
}

double norm_l12_cplx_double(const double *Xr, const double *Xi, const int **G, const double *La, const int I)
{
	int i, j, g, idx, idg;
    const int *gidx;
	double ng2, norm = 0.; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, ng2) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			ng2 = 0.;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
				ng2 += Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx];
			}
            norm += La[idg]*sqrt(ng2);
		}
	}
    return norm;
}

float norm_l12_real_single(const float *X, const int **G, const float *La, const int I)
{
	int i, j, g, idg;
    const int *gidx;
	float x, ng2, norm = 0.f;
    #pragma omp parallel for private(i, j, g, idg, gidx, x, ng2) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			ng2 = 0.;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                x = X[i+I*gidx[j]];
				ng2 += x*x;
			}
            norm += La[idg]*sqrtf(ng2);
		}
	}
    return norm;
}

float norm_l12_cplx_single(const float *Xr, const float *Xi, const int **G, const float *La, const int I)
{
	int i, j, g, idx, idg;
    const int *gidx;
	float ng2, norm = 0.f; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, ng2) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			ng2 = 0.f;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
				ng2 += Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx];
			}
            norm += La[idg]*sqrtf(ng2);
		}
	}
    return norm;
}
