/*==================================================================
 * parallel implementation with OpenMP API
 *
 * Hugo Raguet 2015
 *==================================================================*/

#include <stddef.h>
#include <omp.h>
#include "norms_wl1p.c"

void group_norms_l1p_real_double(double *Nb, const double *X, const int **G, const int I, const double p, const double *W)
{
	int i, j, g, idx, idg;
    const int *gidx;
	double ngp; 
    /* select the convenient auxiliary function */
    void (*add_pow)(double*, double, const double, const double*, const int);
    double (*radical)(const double, const double);
    double invp = 1./p;
    if (p == 1.){ /*  actually compute the average  */
        add_pow = W == NULL  ?  add_pow_l11_real_double  :  add_pow_wl11_real_double;
        radical = radical_1_double;
    }else if (p == 2.){
        add_pow = W == NULL  ?  add_pow_l12_real_double  :  add_pow_wl12_real_double;
        radical = radical_2_double;
    }else if (p == INFINITY){
        add_pow = add_pow_l1inf_real_double;
        radical = radical_1_double;
    }else{
        add_pow = W == NULL  ?  add_pow_l1p_real_double  :  add_pow_wl1p_real_double;
        radical = radical_p_double;
    }
    
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, ngp)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			ngp = 0.;
            /* compute norm */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                add_pow(&ngp, X[idx], p, W, idx);
            }
            Nb[idg] = radical(ngp, invp);
		}
	}
}

void group_norms_l1p_cplx_double(double *Nb, const double *Xr, const double *Xi, const int **G, const int I, const double p, const double *W)
{
	int i, j, g, idx, idg;
    const int *gidx;
	double ngp; 

    /* select the convenient auxiliary function */
    void (*add_pow)(double*, double, const double, const double, const double*, const int);
    double (*radical)(const double, const double);
    double invp = 1./p;
    if (p == 1.){
        add_pow = W == NULL  ?  add_pow_l11_cplx_double  :  add_pow_wl11_cplx_double;
        radical = radical_1_double;
    }else if (p == 2.){
        add_pow = W == NULL  ?  add_pow_l12_cplx_double  :  add_pow_wl12_cplx_double;
        radical = radical_2_double;
    }else if (p == INFINITY){
        add_pow = add_pow_l1inf_cplx_double;
        radical = radical_1_double;
    }else{
        add_pow = W == NULL  ?  add_pow_l1p_cplx_double  :  add_pow_wl1p_cplx_double;
        radical = radical_p_double;
    }

    #pragma omp parallel for private(i, j, g, idx, idg, gidx, ngp)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			ngp = 0.;
            /* compute norm */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                add_pow(&ngp, Xr[idx], Xi[idx], p, W, idx);
            }
            Nb[idg] = radical(ngp, invp);
		}
	}
}

void group_norms_l1p_real_single(float *Nb, const float *X, const int **G, const int I, const float p, const float *W)
{
	int i, j, g, idx, idg;
    const int *gidx;
	float ngp; 

    /* select the convenient auxiliary function */
    void (*add_pow)(float*, const float, const float, const float*, const int);
    float (*radical)(const float, const float);
    float invp = 1.f/p;
    if (p == 1.f){
        add_pow = W == NULL  ?  add_pow_l11_real_single  :  add_pow_wl11_real_single;
        radical = radical_1_single;
    }else if (p == 2.f){
        add_pow = W == NULL  ?  add_pow_l12_real_single  :  add_pow_wl12_real_single;
        radical = radical_2_single;
    }else if (p == INFINITY){
        add_pow = add_pow_l1inf_real_single;
        radical = radical_1_single;
    }else{
        add_pow = W == NULL  ?  add_pow_l1p_real_single  :  add_pow_wl1p_real_single;
        radical = radical_p_single;
    }
    
    #pragma omp parallel for private(i, j, g, idx, idg, gidx, ngp)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			ngp = 0.f;
            /* compute norm */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                add_pow(&ngp, X[idx], p, W, idx);
            }
            Nb[idg] = radical(ngp, invp);
		}
	}
}

void group_norms_l1p_cplx_single(float *Nb, const float *Xr, const float *Xi, const int **G, const int I, const float p, const float *W)
{
	int i, j, g, idx, idg;
    const int *gidx;
	float ngp; 

    /* select the convenient auxiliary function */
    void (*add_pow)(float*, const float, const float, const float, const float*, const int);
    float (*radical)(const float, const float);
    float invp = 1.f/p;
    if (p == 1.f){
        add_pow = W == NULL  ?  add_pow_l11_cplx_single  :  add_pow_wl11_cplx_single;
        radical = radical_1_single;
    }else if (p == 2.f){
        add_pow = W == NULL  ?  add_pow_l12_cplx_single  :  add_pow_wl12_cplx_single;
        radical = radical_2_single;
    }else if (p == INFINITY){
        add_pow = add_pow_l1inf_cplx_single;
        radical = radical_1_single;
    }else{
        add_pow = W == NULL  ?  add_pow_l1p_cplx_single  :  add_pow_wl1p_cplx_single;
        radical = radical_p_single;
    }

    #pragma omp parallel for private(i, j, g, idx, idg, gidx, ngp)
    /* iterate over all groups */
    for (g=0; g<G[0][0]; g++){
        gidx = G[g+1];
        /* iterate over all frames */
	    for (idg=I*g, i=0; i<I; i++, idg++){
			ngp = 0.f;
            /* compute norm */
            for (j=1; j<=gidx[0]; j++){
                idx = i + I*gidx[j];
                add_pow(&ngp, Xr[idx], Xi[idx], p, W, idx);
            }
            Nb[idg] = radical(ngp, invp);
		}
	}
}
