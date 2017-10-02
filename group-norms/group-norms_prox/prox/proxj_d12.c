/*==================================================================
 * parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include <omp.h>
#include <math.h>

void proxj_d12_real_double(double *X, const int **G, const double *Thr, const double *Bnd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	double thr, bnd, ng2, avg, shrink; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, thr, bnd, ng2, avg, shrink)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            thr = Thr[idg];
            bnd = Bnd[idg];
            avg = 0.;
			ng2 = 0.;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                avg += X[idx];
				ng2 += X[idx]*X[idx];
			}
            avg /= gidx[0];
            ng2 -= avg*avg*gidx[0];
			if (ng2 > thr*thr){
                ng2 = sqrt(ng2);
				shrink = (1. - thr/ng2) < bnd/ng2  ?  1. - thr/ng2  :  bnd/ng2;
			    for (j=1; j<=gidx[0]; j++){
                    idx = i + I*gidx[j];
                    X[idx] = avg + shrink*(X[idx] - avg);
                }
            }else{
                for (j=1; j<=gidx[0]; j++){
                    X[i+I*gidx[j]] = avg;
                }
            }
		}
	}
}

void proxj_d12_cplx_double(double *Xr, double *Xi, const int **G, const double *Thr, const double *Bnd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	double thr, bnd, ng2, avgr, avgi, shrink; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, thr, bnd, ng2, avgr, avgi, shrink)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            thr = Thr[idg];
            bnd = Bnd[idg];
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
            ng2 -= (avgr*avgr + avgi*avgi)*gidx[0];
			if (ng2 > thr*thr){
                ng2 = sqrt(ng2);
                shrink = (1. - thr/ng2) < bnd/ng2  ?  1. - thr/ng2  :  bnd/ng2;
			    for (j=1; j<=gidx[0]; j++){
                    idx = i + I*gidx[j];
                    Xr[idx] = avgr + shrink*(Xr[idx] - avgr);
                    Xi[idx] = avgi + shrink*(Xi[idx] - avgi);
                }
            }else{
                for (j=1; j<=gidx[0]; j++){
                    idx = i + I*gidx[j];
                    Xr[idx] = avgr;
                    Xi[idx] = avgi;
                }
            }
		}
	}
}

void proxj_d12_real_single(float *X, const int **G, const float *Thr, const float *Bnd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	float thr, bnd, ng2, avg, shrink; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, thr, bnd, ng2, avg, shrink)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            thr = Thr[idg];
            bnd = Bnd[idg];
            avg = 0.f;
			ng2 = 0.f;
            /* iterate over all coefficients in the group */
			for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                avg += X[idx];
				ng2 += X[idx]*X[idx];
			}
            avg /= gidx[0];
            ng2 -= avg*avg*gidx[0];
			if (ng2 > thr*thr){
                ng2 = sqrtf(ng2);
				shrink = (1.f - thr/ng2) < bnd/ng2  ?  1.f - thr/ng2  :  bnd/ng2;
			    for (j=1; j<=gidx[0]; j++){
                    idx = i + I*gidx[j];
                    X[idx] = avg + shrink*(X[idx] - avg);
                }
            }else{
                for (j=1; j<=gidx[0]; j++){
                    X[i+I*gidx[j]] = avg;
                }
            }
		}
	}
}

void proxj_d12_cplx_single(float *Xr, float *Xi, const int **G, const float *Thr, const float *Bnd, const int I)
{
	int i, j, g, idx, idg;
	const int *gidx;
	float thr, bnd, ng2, avgr, avgi, shrink; 
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, thr, bnd, ng2, avgr, avgi, shrink)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            thr = Thr[idg];
            bnd = Bnd[idg];
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
            ng2 -= (avgr*avgr + avgi*avgi)*gidx[0];
			if (ng2 > thr*thr){
                ng2 = sqrtf(ng2);
                shrink = (1.f - thr/ng2) < bnd/ng2  ?  1.f - thr/ng2  :  bnd/ng2;
			    for (j=1; j<=gidx[0]; j++){
                    idx = i + I*gidx[j];
                    Xr[idx] = avgr + shrink*(Xr[idx] - avgr);
                    Xi[idx] = avgi + shrink*(Xi[idx] - avgi);
                }
            }else{
                for (j=1; j<=gidx[0]; j++){
                    idx = i + I*gidx[j];
                    Xr[idx] = avgr;
                    Xi[idx] = avgi;
                }
            }
		}
	}
}
