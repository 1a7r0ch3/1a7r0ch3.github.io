/*==================================================================
 * parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include <omp.h>
#include <math.h>

double norm_l1inf_real_double(const double *X, const int **G, const double *La, const int I)
{
	int i, j, g, idg;
	const int *gidx;
	double absx, nb, norm = 0.;
    #pragma omp parallel for private(i, j, g, idg, gidx, absx, nb) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			nb = 0.;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                absx = X[i+I*gidx[j]];
                absx = absx > 0.  ?  absx  :  -absx;
				nb = nb > absx  ?  nb  :  absx;
			}
            norm += La[idg]*nb;
		}
	}
    return norm;
}

double norm_l1inf_cplx_double(const double *Xr, const double *Xi, const int **G, const double *La, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	double modx, nb2, norm = 0.; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, modx, nb2) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			nb2 = 0.;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                modx = Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx];
                nb2 = nb2 > modx  ?  nb2  :  modx;
			}
            norm += La[idg]*sqrt(nb2);
		}
	}
    return norm;
}

float norm_l1inf_real_single(const float *X, const int **G, const float *La, const int I)
{
	int i, j, g, idg;
	const int *gidx;
	float absx, nb, norm = 0.f;
    #pragma omp parallel for private(i, j, g, idg, gidx, absx, nb) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			nb = 0.f;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                absx = X[i+I*gidx[j]];
                absx = absx > 0.f  ?  absx  :  -absx;
				nb = nb > absx  ?  nb  :  absx;
			}
            norm += La[idg]*nb;
		}
	}
    return norm;
}

float norm_l1inf_cplx_single(const float *Xr, const float *Xi, const int **G, const float *La, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	float modx, nb2, norm = 0.f; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, modx, nb2) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			nb2 = 0.f;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                modx = Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx];
                nb2 = nb2 > modx  ?  nb2  :  modx;
			}
            norm += La[idg]*sqrtf(nb2);
		}
	}
    return norm;
}
