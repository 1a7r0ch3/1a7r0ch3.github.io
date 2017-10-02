/*==================================================================
 * parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include <omp.h>
#include <math.h>

double norm_d1inf_real_double(const double *X, const int **G, const double *La, const int I)
{
	int i, j, g, idg;
    const int *gidx;
	double x, avg, min_d, max_d, norm = 0.;
    #pragma omp parallel for private(i, j, g, idg, gidx, x, avg, min_d, max_d) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            min_d = INFINITY;
            max_d = -INFINITY;
            avg = 0.;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                x = X[i+I*gidx[j]];
                avg += x;
                min_d = min_d < x  ?  min_d  :  x;
                max_d = max_d > x  ?  max_d  :  x;
			}
            avg /= gidx[0];
            min_d = avg - min_d;
            max_d = max_d - avg;
            max_d = max_d > min_d  ?  max_d  :  min_d;
            norm += La[idg]*max_d;
		}
	}
    return norm;
}

double norm_d1inf_cplx_double(const double *Xr, const double *Xi, const int **G, const double *La, const int I)
{
	int i, j, g, idx, idg;
    const int *gidx;
	double xpr, xpi, avgr, avgi, nb2, norm = 0.; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, xpr, xpi, avgr, avgi, nb2) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            avgr = 0.;
            avgi = 0.;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                avgr += Xr[idx];
                avgi += Xi[idx];
			}
            avgr /= gidx[0];
            avgi /= gidx[0];
            nb2 = 0.;
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                xpr = Xr[idx] - avgr;
                xpi = Xi[idx] - avgi;
                xpr = xpr*xpr + xpi*xpi;
                nb2 = nb2 > xpr  ?  nb2  :  xpr;
            }
            norm += La[idg]*sqrt(nb2);
		}
	}
    return norm;
}

float norm_d1inf_real_single(const float *X, const int **G, const float *La, const int I)
{
	int i, j, g, idg;
    const int *gidx;
	float x, avg, min_d, max_d, norm = 0.f;
    #pragma omp parallel for private(i, j, g, idg, gidx, x, avg, min_d, max_d) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            min_d =  INFINITY;
            max_d = -INFINITY;
            avg = 0.f;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                x = X[i+I*gidx[j]];
                avg += x;
                min_d = min_d < x  ?  min_d  :  x;
                max_d = max_d > x  ?  max_d  :  x;
			}
            avg /= gidx[0];
            min_d = avg - min_d;
            max_d = max_d - avg;
            max_d = max_d > min_d  ?  max_d  :  min_d;
            norm += La[idg]*max_d;
		}
	}
    return norm;
}

float norm_d1inf_cplx_single(const float *Xr, const float *Xi, const int **G, const float *La, const int I)
{
	int i, j, g, idx, idg;
    const int *gidx;
	float xpr, xpi, avgr, avgi, nb2, norm = 0.f; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, xpr, xpi, avgr, avgi, nb2) reduction(+: norm)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            avgr = 0.f;
            avgi = 0.f;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                avgr += Xr[idx];
                avgi += Xi[idx];
			}
            avgr /= gidx[0];
            avgi /= gidx[0];
            nb2 = 0.f;
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                xpr = Xr[idx] - avgr;
                xpi = Xi[idx] - avgi;
                xpr = xpr*xpr + xpi*xpi;
                nb2 = nb2 > xpr  ?  nb2  :  xpr;
            }
            norm += La[idg]*sqrtf(nb2);
		}
	}
    return norm;
}
