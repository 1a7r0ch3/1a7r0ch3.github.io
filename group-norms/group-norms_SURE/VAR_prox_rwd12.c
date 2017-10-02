/*==================================================================
 * parallel implementation with OpenMP API
 * 
 * Hugo Raguet 2015
 *================================================================*/

#include <stdlib.h>
#include <alloca.h>
#include <omp.h>
#include <math.h>

void VAR_prox_rwd12_real_double(const double *X, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L)
{
	int i, j, l, g, n, idx, idg;
    const int *gidx;
	double la, rb, s, s2, xb, xb2, xs, *Xvb, *Xvp, *Xvr, *Xab, *Xap, v; 
    /***  compute averages over the groups  ***/
    double **Xb = malloc(sizeof(double*)*N);
    #pragma omp parallel for private (n, g, i, j, idg, gidx, xb)
    for (n=0; n<N; n++){
        Xb[n] = malloc(sizeof(double)*I*G[n][0][0]);
        /* iterate over all groups */
        for (idg=0, g=0; g<G[n][0][0]; g++){
            gidx = G[n][g+1];
            /* iterate over all frames */
            for (i=0; i<I; i++, idg++){
                xb = 0.;
                for (j=1; j<=gidx[0]; j++){ xb += X[i+I*gidx[j]]; }
                Xb[n][idg] = xb/gidx[0];
            }
        }
    }
    /***  compute variances accross group structures  ***/
    #pragma omp parallel private(i, j, l, g, n, idx, idg, gidx, la, rb, s, s2, xb, xb2, xs, Xvb, Xvp, Xvr, Xab, Xap, v)
    { /* beware of stack overflow, malloc would be safer */
    Xvb = alloca(sizeof(double)*J);
    Xvp = alloca(sizeof(double)*J);
    Xvr = alloca(sizeof(double)*J);
    Xab = alloca(sizeof(double)*J);
    Xap = alloca(sizeof(double)*J);
    /* iterate over all penalizations */
    #pragma omp for
    for (l=0; l<L; l++){
        /* iterate over all frames */
        for (idg=l, i=0; i<I; i++, idg+=L){
            la = La[idg];
            for (j=0; j<J; j++){
                Xvb[j] = 0.;
                Xvp[j] = 0.;
                Xvr[j] = 0.;
                Xab[j] = 0.;
                Xap[j] = 0.;
            }
            /* iterate over all estimators */
            for (n=0; n<N; n++){
                /* iterate over all groups */
                idx = i;
                for (g=0; g<G[n][0][0]; g++){
                    gidx = G[n][g+1];
                    rb = Rb[n][idx];
                    if (rb > la){
                        rb = la/rb;
        				s = (1. - rb + sqrt(- 3.*rb*rb + 2.*rb + 1.))/2.;
                        s2 = s*s;
                        xb = (1. - s)*Xb[n][idx];
                        xb2 = xb*xb;
                        xs = s*xb;
                        for (j=1; j<=gidx[0]; j++){
                            Xvb[gidx[j]] += xb2;
                            Xvp[gidx[j]] += s2;
                            Xvr[gidx[j]] += xs;
                            Xab[gidx[j]] += xb;
                            Xap[gidx[j]] += s;
                        }
                    }else{
                        xb = Xb[n][idx];
                        xb2 = xb*xb;
                        for (j=1; j<=gidx[0]; j++){
                            Xvb[gidx[j]] += xb2;
                            Xab[gidx[j]] += xb;
                        }
                    }
                    idx += I;
                }
            }
            v = 0.;
            idx = i;
            for (j=0; j<J; j++){
                Xab[j] += Xap[j]*X[idx];
                Xvb[j] += X[idx]*(X[idx]*Xvp[j] + 2.*Xvr[j]);
                v += (Xvb[j] - Xab[j]*Xab[j]/N)/N;
                idx += I;
            }
            V[idg] = v;
        }
    }
    }
    for (n=0; n<N; n++){ free(Xb[n]); }
    free(Xb);
}

void VAR_prox_rwd12_cplx_double(const double *Xr, const double *Xi, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L)
{
	int i, j, l, g, n, idx, idg;
    const int *gidx;
	double la, rb, s, s2, xbr, xbi, xb2, xsr, xsi, *Xvb, *Xvp, *Xvr, *Xvi, *Xar, *Xai, *Xap, v; 
    /***  compute averages over the groups  ***/
    double **Xbr = malloc(sizeof(double*)*N);
    double **Xbi = malloc(sizeof(double*)*N);
    #pragma omp parallel for private (n, g, i, j, idg, gidx, xbr, xbi)
    for (n=0; n<N; n++){
        Xbr[n] = malloc(sizeof(double)*I*G[n][0][0]);
        Xbi[n] = malloc(sizeof(double)*I*G[n][0][0]);
        /* iterate over all groups */
        for (idg=0, g=0; g<G[n][0][0]; g++){
            gidx = G[n][g+1];
            /* iterate over all frames */
            for (i=0; i<I; i++, idg++){
                xbr = 0.;
                xbi = 0.;
                for (j=1; j<=gidx[0]; j++){
                    xbr += Xr[i+I*gidx[j]];
                    xbi += Xi[i+I*gidx[j]];
                }
                Xbr[n][idg] = xbr/gidx[0];
                Xbi[n][idg] = xbi/gidx[0];
            }
        }
    }
    
    /***  compute variances accross group structures  ***/
    #pragma omp parallel private(i, j, l, g, n, idx, idg, gidx, la, rb, s, s2, xbr, xbi, xb2, xsr, xsi, Xvb, Xvp, Xvr, Xvi, Xar, Xai, Xap, v)
    { /* beware of stack overflow, malloc would be safer */
    Xvb = alloca(sizeof(double)*J);
    Xvp = alloca(sizeof(double)*J);
    Xvr = alloca(sizeof(double)*J);
    Xvi = alloca(sizeof(double)*J);
    Xar = alloca(sizeof(double)*J);
    Xai = alloca(sizeof(double)*J);
    Xap = alloca(sizeof(double)*J);
    /* iterate over all penalizations */
    #pragma omp for
    for (l=0; l<L; l++){
        /* iterate over all frames */
        for (idg=l, i=0; i<I; i++, idg+=L){
            la = La[idg];
            for (j=0; j<J; j++){
                Xvb[j] = 0.;
                Xvp[j] = 0.;
                Xvr[j] = 0.;
                Xvi[j] = 0.;
                Xar[j] = 0.;
                Xai[j] = 0.;
                Xap[j] = 0.;
            }
            /* iterate over all estimators */
            for (n=0; n<N; n++){
                /* iterate over all groups */
                idx = i;
                for (g=0; g<G[n][0][0]; g++){
                    gidx = G[n][g+1];
                    rb = Rb[n][idx];
                    if (rb > la){
                        rb = la/rb;
        				s = (1. - rb + sqrt(- 3.*rb*rb + 2.*rb + 1.))/2.;
                        s2 = s*s;
                        xbr = (1. - s)*Xbr[n][idx];
                        xbi = (1. - s)*Xbi[n][idx];
                        xb2 = xbr*xbr + xbi*xbi;
                        xsr = s*xbr;
                        xsi = s*xbi;
                        for (j=1; j<=gidx[0]; j++){
                            Xvb[gidx[j]] += xb2;
                            Xvp[gidx[j]] += s2;
                            Xvr[gidx[j]] += xsr;
                            Xvi[gidx[j]] += xsi;
                            Xar[gidx[j]] += xbr;
                            Xai[gidx[j]] += xbi;
                            Xap[gidx[j]] += s;
                        }
                    }else{
                        xbr = Xbr[n][idx];
                        xbi = Xbi[n][idx];
                        xb2 = xbr*xbr + xbi*xbi;
                        for (j=1; j<=gidx[0]; j++){
                            Xvb[gidx[j]] += xb2;
                            Xar[gidx[j]] += xbr;
                            Xai[gidx[j]] += xbi;
                        }
                    }
                    idx += I;
                }
            }
            v = 0.;
            idx = i;
            for (j=0; j<J; j++){
                Xar[j] += Xap[j]*Xr[idx];
                Xai[j] += Xap[j]*Xi[idx];
                Xvb[j] += Xr[idx]*(Xr[idx]*Xvp[j] + 2.*Xvr[j]) + \
                          Xi[idx]*(Xi[idx]*Xvp[j] + 2.*Xvi[j]);
                v += (Xvb[j] - (Xar[j]*Xar[j] + Xai[j]*Xai[j])/N)/N;
                idx += I;
            }
            V[idg] = v;
        }
    }
    }
    for (n=0; n<N; n++){
        free(Xbr[n]);
        free(Xbi[n]);
    }
    free(Xbr);
    free(Xbi);
}

void VAR_prox_rwd12_real_single(const float *X, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L)
{
	int i, j, l, g, n, idx, idg;
    const int *gidx;
	float la, rb, s, s2, xb, xb2, xs, *Xvb, *Xvp, *Xvr, *Xab, *Xap, v; 
    /***  compute averages over the groups  ***/
    float **Xb = malloc(sizeof(float*)*N);
    #pragma omp parallel for private (n, g, i, j, idg, gidx, xb)
    for (n=0; n<N; n++){
        Xb[n] = malloc(sizeof(float)*I*G[n][0][0]);
        /* iterate over all groups */
        for (idg=0, g=0; g<G[n][0][0]; g++){
            gidx = G[n][g+1];
            /* iterate over all frames */
            for (i=0; i<I; i++, idg++){
                xb = 0.f;
                for (j=1; j<=gidx[0]; j++){ xb += X[i+I*gidx[j]]; }
                Xb[n][idg] = xb/gidx[0];
            }
        }
    }
    /***  compute variances accross group structures  ***/
    #pragma omp parallel private(i, j, l, g, n, idx, idg, gidx, la, rb, s, s2, xb, xb2, xs, Xvb, Xvp, Xvr, Xab, Xap, v)
    { /* beware of stack overflow, malloc would be safer */
    Xvb = alloca(sizeof(float)*J);
    Xvp = alloca(sizeof(float)*J);
    Xvr = alloca(sizeof(float)*J);
    Xab = alloca(sizeof(float)*J);
    Xap = alloca(sizeof(float)*J);
    /* iterate over all penalizations */
    #pragma omp for
    for (l=0; l<L; l++){
        /* iterate over all frames */
        for (idg=l, i=0; i<I; i++, idg+=L){
            la = La[idg];
            for (j=0; j<J; j++){
                Xvb[j] = 0.f;
                Xvp[j] = 0.f;
                Xvr[j] = 0.f;
                Xab[j] = 0.f;
                Xap[j] = 0.f;
            }
            /* iterate over all estimators */
            for (n=0; n<N; n++){
                /* iterate over all groups */
                idx = i;
                for (g=0; g<G[n][0][0]; g++){
                    gidx = G[n][g+1];
                    rb = Rb[n][idx];
                    if (rb > la){
                        rb = la/rb;
        				s = (1.f - rb + sqrt(- 3.f*rb*rb + 2.f*rb + 1.f))/2.f;
                        s2 = s*s;
                        xb = (1.f - s)*Xb[n][idx];
                        xb2 = xb*xb;
                        xs = s*xb;
                        for (j=1; j<=gidx[0]; j++){
                            Xvb[gidx[j]] += xb2;
                            Xvp[gidx[j]] += s2;
                            Xvr[gidx[j]] += xs;
                            Xab[gidx[j]] += xb;
                            Xap[gidx[j]] += s;
                        }
                    }else{
                        xb = Xb[n][idx];
                        xb2 = xb*xb;
                        for (j=1; j<=gidx[0]; j++){
                            Xvb[gidx[j]] += xb2;
                            Xab[gidx[j]] += xb;
                        }
                    }
                    idx += I;
                }
            }
            v = 0.f;
            idx = i;
            for (j=0; j<J; j++){
                Xab[j] += Xap[j]*X[idx];
                Xvb[j] += X[idx]*(X[idx]*Xvp[j] + 2.f*Xvr[j]);
                v += (Xvb[j] - Xab[j]*Xab[j]/N)/N;
                idx += I;
            }
            V[idg] = v;
        }
    }
    }
    for (n=0; n<N; n++){ free(Xb[n]); }
    free(Xb);
}

void VAR_prox_rwd12_cplx_single(const float *Xr, const float *Xi, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L)
{
	int i, j, l, g, n, idx, idg;
    const int *gidx;
	float la, rb, s, s2, xbr, xbi, xb2, xsr, xsi, *Xvb, *Xvp, *Xvr, *Xvi, *Xar, *Xai, *Xap, v; 
    /***  compute averages over the groups  ***/
    float **Xbr = malloc(sizeof(float*)*N);
    float **Xbi = malloc(sizeof(float*)*N);
    #pragma omp parallel for private (n, g, i, j, idg, gidx, xbr, xbi)
    for (n=0; n<N; n++){
        Xbr[n] = malloc(sizeof(float)*I*G[n][0][0]);
        Xbi[n] = malloc(sizeof(float)*I*G[n][0][0]);
        /* iterate over all groups */
        for (idg=0, g=0; g<G[n][0][0]; g++){
            gidx = G[n][g+1];
            /* iterate over all frames */
            for (i=0; i<I; i++, idg++){
                xbr = 0.f;
                xbi = 0.f;
                for (j=1; j<=gidx[0]; j++){
                    xbr += Xr[i+I*gidx[j]];
                    xbi += Xi[i+I*gidx[j]];
                }
                Xbr[n][idg] = xbr/gidx[0];
                Xbi[n][idg] = xbi/gidx[0];
            }
        }
    }
    /***  compute variances accross group structures  ***/
    #pragma omp parallel private(i, j, l, g, n, idx, idg, gidx, la, rb, s, s2, xbr, xbi, xb2, xsr, xsi, Xvb, Xvp, Xvr, Xvi, Xar, Xai, Xap, v)
    { /* beware of stack overflow, malloc would be safer */
    Xvb = alloca(sizeof(float)*J);
    Xvp = alloca(sizeof(float)*J);
    Xvr = alloca(sizeof(float)*J);
    Xvi = alloca(sizeof(float)*J);
    Xar = alloca(sizeof(float)*J);
    Xai = alloca(sizeof(float)*J);
    Xap = alloca(sizeof(float)*J);
    /* iterate over all penalizations */
    #pragma omp for
    for (l=0; l<L; l++){
        /* iterate over all frames */
        for (idg=l, i=0; i<I; i++, idg+=L){
            la = La[idg];
            for (j=0; j<J; j++){
                Xvb[j] = 0.f;
                Xvp[j] = 0.f;
                Xvr[j] = 0.f;
                Xvi[j] = 0.f;
                Xar[j] = 0.f;
                Xai[j] = 0.f;
                Xap[j] = 0.f;
            }
            /* iterate over all estimators */
            for (n=0; n<N; n++){
                /* iterate over all groups */
                idx = i;
                for (g=0; g<G[n][0][0]; g++){
                    gidx = G[n][g+1];
                    rb = Rb[n][idx];
                    if (rb > la){
                        rb = la/rb;
        				s = (1.f - rb + sqrt(- 3.f*rb*rb + 2.f*rb + 1.f))/2.f;
                        s2 = s*s;
                        xbr = (1.f - s)*Xbr[n][idx];
                        xbi = (1.f - s)*Xbi[n][idx];
                        xb2 = xbr*xbr + xbi*xbi;
                        xsr = s*xbr;
                        xsi = s*xbi;
                        for (j=1; j<=gidx[0]; j++){
                            Xvb[gidx[j]] += xb2;
                            Xvp[gidx[j]] += s2;
                            Xvr[gidx[j]] += xsr;
                            Xvi[gidx[j]] += xsi;
                            Xar[gidx[j]] += xbr;
                            Xai[gidx[j]] += xbi;
                            Xap[gidx[j]] += s;
                        }
                    }else{
                        xbr = Xbr[n][idx];
                        xbi = Xbi[n][idx];
                        xb2 = xbr*xbr + xbi*xbi;
                        for (j=1; j<=gidx[0]; j++){
                            Xvb[gidx[j]] += xb2;
                            Xar[gidx[j]] += xbr;
                            Xai[gidx[j]] += xbi;
                        }
                    }
                    idx += I;
                }
            }
            v = 0.f;
            idx = i;
            for (j=0; j<J; j++){
                Xar[j] += Xap[j]*Xr[idx];
                Xai[j] += Xap[j]*Xi[idx];
                Xvb[j] += Xr[idx]*(Xr[idx]*Xvp[j] + 2.f*Xvr[j]) + \
                          Xi[idx]*(Xi[idx]*Xvp[j] + 2.f*Xvi[j]);
                v += (Xvb[j] - (Xar[j]*Xar[j] + Xai[j]*Xai[j])/N)/N;
                idx += I;
            }
            V[idg] = v;
        }
    }
    }
    for (n=0; n<N; n++){
        free(Xbr[n]);
        free(Xbi[n]);
    }
    free(Xbr);
    free(Xbi);
}
