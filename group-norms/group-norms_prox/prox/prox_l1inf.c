/*==================================================================
 * parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include <alloca.h>
#include <omp.h>
#include <math.h>

void prox_l1inf_real_double(double *X, const int **G, const double *Thr, const int I)
{
	int i, j, g, idx, idg, l, m, n;
	const int *gidx;
	double *sort_coef, *csum_coef, coef, max_coef, thr;
    #pragma omp parallel private(i, j, g, idx, idg, gidx, l, m, n, sort_coef, csum_coef, coef, max_coef, thr)
    {
	sort_coef = alloca(sizeof(double)*(G[0][1]+1));
	csum_coef = alloca(sizeof(double)*G[0][1]); 
    #pragma omp for
    /* iterate over all groups */
	for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            thr = Thr[idg];
            for (j=0; j<gidx[0]; j++){ sort_coef[j] = 0.; }
            n = 0;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                coef = X[i+I*gidx[j]];
                coef = (coef > 0.) ? coef : -coef;
                /* insertion sort */
                m = 0;
                while (sort_coef[m] > coef){ m++; }
                for (l=n++; l>=m; l--){
                    sort_coef[l+1] = sort_coef[l];
                }
                sort_coef[m] = coef;
            }
            /* cumulative sum */
            csum_coef[0] = sort_coef[0];
            for (l=1; l<n; l++){
                csum_coef[l] = sort_coef[l]+csum_coef[l-1];
            }
            /* compute derivatives */
            for (l=1; l<n; l++){
                sort_coef[l] = (l+1)*sort_coef[l]-csum_coef[l]+thr;
            }
            /* find change of sign */
            l = 1;
            while (0.<=sort_coef[l] && l<n){ l++; }
            /* compute optimal maximal coefficient */
            max_coef = (csum_coef[l-1] - thr)/l;
            /* threshold coefficients */
			if (max_coef > 0.){
                for (j=1; j<=gidx[0]; j++){
                    idx = i + I*gidx[j];
                    coef = X[idx];
                    if (max_coef < coef){
                        X[idx] = max_coef;
                    }else if (-max_coef > coef){
                        X[idx] = -max_coef;
                    }
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

void prox_l1inf_cplx_double(double *Xr, double *Xi, const int **G, const double *Thr, const int I)
{
	int i, j, g, idx, idg, l, m, n;
	const int *gidx;
	double *sort_coef, *csum_coef, *coef, max_coef, thr;
    #pragma omp parallel private(i, j, g, idx, idg, gidx, l, m, n, sort_coef, csum_coef, coef, max_coef, thr)
    {
	sort_coef = alloca(sizeof(double)*(G[0][1]+1));
	csum_coef = alloca(sizeof(double)*G[0][1]); 
    coef = alloca(sizeof(double)*G[0][1]);
    #pragma omp for
    /* iterate over all groups */
	for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            thr = Thr[idg];
            for (j=0; j<gidx[0]; j++){ sort_coef[j] = 0.; }
            n = 0;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                /* insertion sort */
                idx = i + I*gidx[j];
                coef[j-1] = sqrt(Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx]);
                m = 0;
                while (sort_coef[m] > coef[j-1]){ m++; }
                for (l=n++; l>=m; l--){
                    sort_coef[l+1] = sort_coef[l];
                }
                sort_coef[m] = coef[j-1];
            }
            /* cumulative sum */
            csum_coef[0] = sort_coef[0];
            for (l=1; l<n; l++){
                csum_coef[l] = sort_coef[l]+csum_coef[l-1];
            }
            /* compute derivatives */
            for (l=1; l<n; l++){
                sort_coef[l] = (l+1)*sort_coef[l]-csum_coef[l]+thr;
            }
            /* find change of sign */
            l = 1;
            while (0.<=sort_coef[l] && l<n){ l++; }
            /* compute optimal maximal coefficient */
            max_coef = (csum_coef[l-1] - thr)/l;
            /* threshold coefficients */
			if (max_coef > 0.){
                for (j=1; j<=gidx[0]; j++){
                if (coef[j-1] > max_coef){
                    idx = i + I*gidx[j];
                    thr = max_coef/coef[j-1];
                    Xr[idx] *= thr;
                    Xi[idx] *= thr;
                }}
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

void prox_l1inf_real_single(float *X, const int **G, const float *Thr, const int I)
{
	int i, j, g, idx, idg, l, m, n;
	const int *gidx;
	float *sort_coef, *csum_coef, coef, max_coef, thr;
    #pragma omp parallel private(i, j, g, idx, idg, gidx, l, m, n, sort_coef, csum_coef, coef, max_coef, thr)
    {
	sort_coef = alloca(sizeof(float)*(G[0][1]+1));
	csum_coef = alloca(sizeof(float)*G[0][1]); 
    #pragma omp for
    /* iterate over all groups */
	for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            thr = Thr[idg];
            for (j=0; j<gidx[0]; j++){ sort_coef[j] = 0.f; }
            n = 0;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                coef = X[i+I*gidx[j]];
                coef = (coef > 0.f) ? coef : -coef;
                /* insertion sort */
                m = 0;
                while (sort_coef[m] > coef){ m++; }
                for (l=n++; l>=m; l--){
                    sort_coef[l+1] = sort_coef[l];
                }
                sort_coef[m] = coef;
            }
            /* cumulative sum */
            csum_coef[0] = sort_coef[0];
            for (l=1; l<n; l++){
                csum_coef[l] = sort_coef[l]+csum_coef[l-1];
            }
            /* compute derivatives */
            for (l=1; l<n; l++){
                sort_coef[l] = (l+1)*sort_coef[l]-csum_coef[l]+thr;
            }
            /* find change of sign */
            l = 1;
            while (0.f<=sort_coef[l] && l<n){ l++; }
            /* compute optimal maximal coefficient */
            max_coef = (csum_coef[l-1] - thr)/l;
            /* threshold coefficients */
			if (max_coef > 0.f){
                for (j=1; j<=gidx[0]; j++){
                    idx = i + I*gidx[j];
                    coef = X[idx];
                    if (max_coef < coef){
                        X[idx] = max_coef;
                    }else if (-max_coef > coef){
                        X[idx] = -max_coef;
                    }
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

void prox_l1inf_cplx_single(float *Xr, float *Xi, const int **G, const float *Thr, const int I)
{
	int i, j, g, idx, idg, l, m, n;
	const int *gidx;
	float *sort_coef, *csum_coef, *coef, max_coef, thr;
    #pragma omp parallel private(i, j, g, idx, idg, gidx, l, m, n, sort_coef, csum_coef, coef, max_coef, thr)
    {
	sort_coef = alloca(sizeof(float)*(G[0][1]+1));
	csum_coef = alloca(sizeof(float)*G[0][1]); 
    coef = alloca(sizeof(float)*G[0][1]);
    #pragma omp for
    /* iterate over all groups */
	for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
            thr = Thr[idg];
            for (j=0; j<gidx[0]; j++){ sort_coef[j] = 0.f; }
            n = 0;
            /* iterate over all coefficients in the group */
            for (j=1; j<=gidx[0]; j++){
                /* insertion sort */
                idx = i + I*gidx[j];
                coef[j-1] = sqrtf(Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx]);
                m = 0;
                while (sort_coef[m] > coef[j-1]){ m++; }
                for (l=n++; l>=m; l--){
                    sort_coef[l+1] = sort_coef[l];
                }
                sort_coef[m] = coef[j-1];
            }
            /* cumulative sum */
            csum_coef[0] = sort_coef[0];
            for (l=1; l<n; l++){
                csum_coef[l] = sort_coef[l]+csum_coef[l-1];
            }
            /* compute derivatives */
            for (l=1; l<n; l++){
                sort_coef[l] = (l+1)*sort_coef[l]-csum_coef[l]+thr;
            }
            /* find change of sign */
            l = 1;
            while (0.f<=sort_coef[l] && l<n){ l++; }
            /* compute optimal maximal coefficient */
            max_coef = (csum_coef[l-1] - thr)/l;
            /* threshold coefficients */
			if (max_coef > 0.f){
                for (j=1; j<=gidx[0]; j++){
                if (coef[j-1] > max_coef){
                    idx = i + I*gidx[j];
                    thr = max_coef/coef[j-1];
                    Xr[idx] *= thr;
                    Xi[idx] *= thr;
                }}
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
