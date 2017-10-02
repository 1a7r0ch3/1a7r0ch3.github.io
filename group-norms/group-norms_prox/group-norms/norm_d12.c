/*==================================================================
 * parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include <omp.h>
#include <math.h>

double norm_d12_real_double(const double *X, const int **G, const double *La, const int I)
{
	int i, j, g, idg;
    const int *gidx;
	double x, ng2, avg, norm = 0.;
    #pragma omp parallel for private(i, j, g, idg, gidx, x, avg, ng2) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            ng2 = 0.;
            avg = 0.;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                x = X[i+I*gidx[j]];
                avg += x;
				ng2 += x*x;
			}
            ng2 -= avg*avg/gidx[0];
            if (ng2 > 0.){ /* can be negative due to round-off errors */
                norm += La[idg]*sqrt(ng2);
            }
		}
	}
    return norm;
}

double norm_d12_cplx_double(const double *Xr, const double *Xi, const int **G, const double *La, const int I)
{
	int i, j, g, idx, idg;
    const int *gidx;
	double xr, xi, avgr, avgi, ng2, norm = 0.; 

    #pragma omp parallel for private(i, j, g, idx, idg, gidx, xr, xi, avgr, avgi, ng2) reduction(+: norm)
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            avgr = 0.;
            avgi = 0.;
			ng2 = 0.;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i+I*gidx[j];
                xr = Xr[idx];
                xi = Xi[idx];
                avgr += xr;
                avgi += xi;
				ng2  += xr*xr + xi*xi;
			}
            ng2 -= (avgr*avgr + avgi*avgi)/gidx[0];
            if (ng2 > 0.){ /* can be negative due round-off errors */
                norm += La[idg]*sqrt(ng2);
            }
		}
	}
    return norm;
}

float norm_d12_real_single(const float *X, const int **G, const float *La, const int I)
{
	int i, j, g, idg;
    const int *gidx;
	float x, ng2, avg, norm = 0.f;
    #pragma omp parallel for private(i, j, g, idg, gidx, x, avg, ng2) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			ng2 = 0.f;
            avg = 0.f;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                x = X[i+I*gidx[j]];
                avg += x;
				ng2 += x*x;
			}
            ng2 -= avg*avg/gidx[0];
            if (ng2 > 0.f){ /* can be negative due to round-off errors */
                norm += La[idg]*sqrtf(ng2);
            }
		}
	}
    return norm;
}

float norm_d12_cplx_single(const float *Xr, const float *Xi, const int **G, const float *La, const int I)
{
	int i, j, g, idx, idg;
    const int *gidx;
	float xr, xi, avgr, avgi, ng2, norm = 0.f; 

    #pragma omp parallel for private(i, j, g, idx, idg, gidx, xr, xi, avgr, avgi, ng2) reduction(+: norm)
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            avgr = 0.f;
            avgi = 0.f;
			ng2 = 0.f;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i+I*gidx[j];
                xr = Xr[idx];
                xi = Xi[idx];
                avgr += xr;
                avgi += xi;
				ng2  += xr*xr + xi*xi;
			}
            ng2 -= (avgr*avgr + avgi*avgi)/gidx[0];
            if (ng2 > 0.f){ /* can be negative due to round-off errors */
                norm += La[idg]*sqrtf(ng2);
            }
		}
	}
    return norm;
}
