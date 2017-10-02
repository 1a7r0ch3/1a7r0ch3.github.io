/*==================================================================
 * parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include <omp.h>
#include <math.h>

void prox_l12_d12_real_double(double *X, const int **G, const double *Thrl, const double *Thrd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	double thrl, thrd, ng2, avg, avg2, shrink, shrinkd; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, thrl, thrd, ng2, avg, avg2, shrink, shrinkd)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
        for (idg=I*g, i=0; i<I; i++, idg++){
            thrl = Thrl[idg];
            thrd = Thrd[idg];
            avg = 0.;
            ng2 = 0.;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                avg += X[idx];
                ng2 += X[idx]*X[idx];
            }
            avg /= gidx[0];
            avg2 = avg*avg*gidx[0];
            ng2 -= avg2;
            if (ng2 > thrd*thrd){
                shrinkd = 1. - thrd/sqrt(ng2);
                ng2 = avg2 + shrinkd*shrinkd*ng2;
                if (ng2 > thrl*thrl){
                    shrink = 1. - thrl/sqrt(ng2);
                    avg2 = shrink*avg; /* reuse available variable */
                    shrink *= shrinkd;
                    for (j=1; j<=gidx[0]; j++){
                        idx = i + I*gidx[j];
                        X[idx] = avg2 + shrink*(X[idx] - avg);
                    }
                }else{
                    for (j=1; j<=gidx[0]; j++){
                        X[i+I*gidx[j]] = 0.;
                    }
                }
            }else{
                if (avg2 > thrl*thrl){
                    avg *= 1. - thrl/sqrt(avg2);
                    for (j=1; j<=gidx[0]; j++){
                        X[i+I*gidx[j]] = avg;
                    }
                }else{
                    for (j=1; j<=gidx[0]; j++){
                        X[i+I*gidx[j]] = 0.;
                    }
                }
            }
        }
    }
}

void prox_l12_d12_cplx_double(double *Xr, double *Xi, const int **G, const double *Thrl, const double *Thrd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	double thrl, thrd, ng2, avgr, avgi, avg2, shrink, shrinkd; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, thrl, thrd, ng2, avgr, avgi, avg2, shrink, shrinkd)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
        for (idg=I*g, i=0; i<I; i++, idg++){
            thrl = Thrl[idg];
            thrd = Thrd[idg];
            avgr = 0.;
            avgi = 0.;
            ng2 = 0.;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                avgr += Xr[idx];
                avgi += Xi[idx];
                ng2 += Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx];
            }
            avgr /= gidx[0];
            avgi /= gidx[0];
            avg2 = (avgr*avgr + avgi*avgi)*gidx[0];
            ng2 -= avg2;
            if (ng2 > thrd*thrd){
                shrinkd = 1. - thrd/sqrt(ng2);
                ng2 = avg2 + shrinkd*shrinkd*ng2;
                if (ng2 > thrl*thrl){
                    shrink = 1. - thrl/sqrt(ng2);
                    thrl = shrink*avgr; /* reuse available variable */
                    thrd = shrink*avgi; /* reuse available variable */
                    shrink *= shrinkd;
                    for (j=1; j<=gidx[0]; j++){
                        idx = i + I*gidx[j];
                        Xr[idx] = thrl + shrink*(Xr[idx] - avgr);
                        Xi[idx] = thrd + shrink*(Xi[idx] - avgi);
                    }
                }else{
                    for (j=1; j<=gidx[0]; j++){
                        idx = i + I*gidx[j];
                        Xr[idx] = 0.;
                        Xi[idx] = 0.;
                    }
                }
            }else{
                if (avg2 > thrl*thrl){
                    shrink = 1. - thrl/sqrt(avg2);
                    avgr *= shrink;
                    avgi *= shrink;
                    for (j=1; j<=gidx[0]; j++){
                        idx = i + I*gidx[j];
                        Xr[idx] = avgr;
                        Xi[idx] = avgi;
                    }
                }else{
                    for (j=1; j<=gidx[0]; j++){
                        idx = i + I*gidx[j];
                        Xr[idx] = 0.;
                        Xi[idx] = 0.;
                    }
                }
            }
        }
    }
}

void prox_l12_d12_real_single(float *X, const int **G, const float *Thrl, const float *Thrd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	float thrl, thrd, ng2, avg, avg2, shrink, shrinkd; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, thrl, thrd, ng2, avg, avg2, shrink, shrinkd)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
        for (idg=I*g, i=0; i<I; i++, idg++){
            thrl = Thrl[idg];
            thrd = Thrd[idg];
            avg = 0.f;
            ng2 = 0.f;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                avg += X[idx];
                ng2 += X[idx]*X[idx];
            }
            avg /= gidx[0];
            avg2 = avg*avg*gidx[0];
            ng2 -= avg2;
            if (ng2 > thrd*thrd){
                shrinkd = 1.f - thrd/sqrtf(ng2);
                ng2 = avg2 + shrinkd*shrinkd*ng2;
                if (ng2 > thrl*thrl){
                    shrink = 1.f - thrl/sqrtf(ng2);
                    avg2 = shrink*avg; /* reuse available variable */
                    shrink *= shrinkd;
                    for (j=1; j<=gidx[0]; j++){
                        idx = i + I*gidx[j];
                        X[idx] = avg2 + shrink*(X[idx] - avg);
                    }
                }else{
                    for (j=1; j<=gidx[0]; j++){
                        X[i+I*gidx[j]] = 0.f;
                    }
                }
            }else{
                if (avg2 > thrl*thrl){
                    avg *= 1.f - thrl/sqrtf(avg2);
                    for (j=1; j<=gidx[0]; j++){
                        X[i+I*gidx[j]] = avg;
                    }
                }else{
                    for (j=1; j<=gidx[0]; j++){
                        X[i+I*gidx[j]] = 0.f;
                    }
                }
            }
        }
    }
}

void prox_l12_d12_cplx_single(float *Xr, float *Xi, const int **G, const float *Thrl, const float *Thrd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	float thrl, thrd, ng2, avgr, avgi, avg2, shrink, shrinkd; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, thrl, thrd, ng2, avgr, avgi, avg2, shrink, shrinkd)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
        for (idg=I*g, i=0; i<I; i++, idg++){
            thrl = Thrl[idg];
            thrd = Thrd[idg];
            avgr = 0.f;
            avgi = 0.f;
            ng2 = 0.f;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                avgr += Xr[idx];
                avgi += Xi[idx];
                ng2 += Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx];
            }
            avgr /= gidx[0];
            avgi /= gidx[0];
            avg2 = (avgr*avgr + avgi*avgi)*gidx[0];
            ng2 -= avg2;
            if (ng2 > thrd*thrd){
                shrinkd = 1.f - thrd/sqrtf(ng2);
                ng2 = avg2 + shrinkd*shrinkd*ng2;
                if (ng2 > thrl*thrl){
                    shrink = 1.f - thrl/sqrtf(ng2);
                    thrl = shrink*avgr; /* reuse available variable */
                    thrd = shrink*avgi; /* reuse available variable */
                    shrink *= shrinkd;
                    for (j=1; j<=gidx[0]; j++){
                        idx = i + I*gidx[j];
                        Xr[idx] = thrl + shrink*(Xr[idx] - avgr);
                        Xi[idx] = thrd + shrink*(Xi[idx] - avgi);
                    }
                }else{
                    for (j=1; j<=gidx[0]; j++){
                        idx = i + I*gidx[j];
                        Xr[idx] = 0.f;
                        Xi[idx] = 0.f;
                    }
                }
            }else{
                if (avg2 > thrl*thrl){
                    shrink = 1.f - thrl/sqrtf(avg2);
                    avgr *= shrink;
                    avgi *= shrink;
                    for (j=1; j<=gidx[0]; j++){
                        idx = i + I*gidx[j];
                        Xr[idx] = avgr;
                        Xi[idx] = avgi;
                    }
                }else{
                    for (j=1; j<=gidx[0]; j++){
                        idx = i + I*gidx[j];
                        Xr[idx] = 0.f;
                        Xi[idx] = 0.f;
                    }
                }
            }
        }
    }
}
