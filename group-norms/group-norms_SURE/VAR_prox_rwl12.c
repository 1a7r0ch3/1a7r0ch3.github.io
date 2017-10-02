/*==================================================================
 * parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include <alloca.h>
#include <omp.h>
#include <math.h>

void VAR_prox_rwl12_real_double(const double *X, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L)
{
	int i, j, l, g, n, idx, idg;
    const int *gidx;
	double la, rb, s, s2, *Xv, *Xa, v; 
    #pragma omp parallel private(i, j, l, g, n, idx, idg, gidx, la, rb, s, s2, Xv, Xa, v)
    {
    Xa = alloca(sizeof(double)*J);
    Xv = alloca(sizeof(double)*J);
    /* iterate over all penalizations */
    #pragma omp for
    for (l=0; l<L; l++){
        /* iterate over all frames */
        for (idg=l, i=0; i<I; i++, idg+=L){
            la = La[idg];
            for (j=0; j<J; j++){
                Xv[j] = 0.;
                Xa[j] = 0.;
            }
            /* iterate over all estimators */
            for (n=0; n<N; n++){
                /* iterate over all groups */
                idx = i;
                for (g=0; g<G[n][0][0]; g++){
                    gidx = G[n][g+1];
                    rb = Rb[n][idx];
                    idx += I;
                    if (rb > la){
                        rb = la/rb;
        				s = (1. - rb + sqrt(- 3.*rb*rb + 2.*rb + 1.))/2.;
                        s2 = s*s;
                        for (j=1; j<=gidx[0]; j++){
                            Xv[gidx[j]] += s2;
                            Xa[gidx[j]] += s;
                        }
                    }
                }
            }
            v = 0.;
            idx = i;
            for (j=0; j<J; j++){
                v += X[idx]*X[idx]*(Xv[j] - Xa[j]*Xa[j]/N)/N;
                idx += I;
            }
            V[idg] = v;
        }
    }
    }
}

void VAR_prox_rwl12_cplx_double(const double *Xr, const double *Xi, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L)
{
	int i, j, l, g, n, idx, idg;
    const int *gidx;
	double la, rb, s, s2, *Xv, *Xa, v; 
    #pragma omp parallel private(i, j, l, g, n, idx, idg, gidx, la, rb, s, s2, Xv, Xa, v)
    {
    Xa = alloca(sizeof(double)*J);
    Xv = alloca(sizeof(double)*J);
    /* iterate over all penalizations */
    #pragma omp for
    for (l=0; l<L; l++){
        /* iterate over all frames */
        for (idg=l, i=0; i<I; i++, idg+=L){
            la = La[idg];
            for (j=0; j<J; j++){
                Xv[j] = 0.;
                Xa[j] = 0.;
            }
            /* iterate over all estimators */
            for (n=0; n<N; n++){
                /* iterate over all groups */
                idx = i;
                for (g=0; g<G[n][0][0]; g++){
                    gidx = G[n][g+1];
                    rb = Rb[n][idx];
                    idx += I;
                    if (rb > la){
                        rb = la/rb;
        				s = (1. - rb + sqrt(- 3.*rb*rb + 2.*rb + 1.))/2.;
                        s2 = s*s;
                        for (j=1; j<=gidx[0]; j++){
                            Xv[gidx[j]] += s2;
                            Xa[gidx[j]] += s;
                        }
                    }/* else: suppress the group ? should be the same for all i... */
                }
            }
            v = 0.;
            idx = i;
            for (j=0; j<J; j++){
                v += (Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx])*(Xv[j] - Xa[j]*Xa[j]/N)/N;
                idx += I;
            }
            V[idg] = v;
        }
    }
    }
}

void VAR_prox_rwl12_real_single(const float *X, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L)
{
	int i, j, l, g, n, idx, idg;
    const int *gidx;
	float la, rb, s, s2, *Xv, *Xa, v; 
    
    #pragma omp parallel private(i, j, l, g, n, idx, idg, gidx, la, rb, s, s2, Xv, Xa, v)
    {
    Xa = alloca(sizeof(float)*J);
    Xv = alloca(sizeof(float)*J);
    /* iterate over all penalizations */
    #pragma omp for
    for (l=0; l<L; l++){
        /* iterate over all frames */
        for (idg=l, i=0; i<I; i++, idg+=L){
            la = La[idg];
            for (j=0; j<J; j++){
                Xv[j] = 0.f;
                Xa[j] = 0.f;
            }
            /* iterate over all estimators */
            for (n=0; n<N; n++){
                /* iterate over all groups */
                idx = i;
                for (g=0; g<G[n][0][0]; g++){
                    gidx = G[n][g+1];
                    rb = Rb[n][idx];
                    idx += I;
                    if (rb > la){
                        rb = la/rb;
        				s = (1.f - rb + sqrt(- 3.f*rb*rb + 2.f*rb + 1.f))/2.f;
                        s2 = s*s;
                        for (j=1; j<=gidx[0]; j++){
                            Xv[gidx[j]] += s2;
                            Xa[gidx[j]] += s;
                        }
                    }/* else: suppress the group ? should be the same for all i... */
                }
            }
            v = 0.f;
            idx = i;
            for (j=0; j<J; j++){
                v += X[idx]*X[idx]*(Xv[j] - Xa[j]*Xa[j]/N)/N;
                idx += I;
            }
            V[idg] = v;
        }
    }
    }
}

void VAR_prox_rwl12_cplx_single(const float *Xr, const float *Xi, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L)
{
	int i, j, l, g, n, idx, idg;
    const int *gidx;
	float la, rb, s, s2, *Xv, *Xa, v; 
    #pragma omp parallel private(i, j, l, g, n, idx, idg, gidx, la, rb, s, s2, Xv, Xa, v)
    {
    Xa = alloca(sizeof(float)*J);
    Xv = alloca(sizeof(float)*J);
    /* iterate over all penalizations */
    #pragma omp for
    for (l=0; l<L; l++){
        /* iterate over all frames */
        for (idg=l, i=0; i<I; i++, idg+=L){
            la = La[idg];
            for (j=0; j<J; j++){
                Xv[j] = 0.f;
                Xa[j] = 0.f;
            }
            /* iterate over all estimators */
            for (n=0; n<N; n++){
                /* iterate over all groups */
                idx = i;
                for (g=0; g<G[n][0][0]; g++){
                    gidx = G[n][g+1];
                    rb = Rb[n][idx];
                    idx += I;
                    if (rb > la){
                        rb = la/rb;
        				s = (1.f - rb + sqrt(- 3.f*rb*rb + 2.f*rb + 1.f))/2.f;
                        s2 = s*s;
                        for (j=1; j<=gidx[0]; j++){
                            Xv[gidx[j]] += s2;
                            Xa[gidx[j]] += s;
                        }
                    }/* else: suppress the group ? should be the same for all i... */
                }
            }
            v = 0.f;
            idx = i;
            for (j=0; j<J; j++){
                v += (Xr[idx]*Xr[idx] + Xi[idx]*Xi[idx])*(Xv[j] - Xa[j]*Xa[j]/N)/N;
                idx += I;
            }
            V[idg] = v;
        }
    }
    }
}
