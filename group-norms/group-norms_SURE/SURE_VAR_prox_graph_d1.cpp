/*==================================================================
 * Hugo Raguet 2016
 *================================================================*/
#include <stdlib.h>
#include <alloca.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#ifdef MEX
    #include "mex.h"
    #define FLUSH mexEvalString("drawnow expose")
#else
    #define FLUSH fflush(stdout)
#endif

/* constants of the correct type */
#define ZERO ((real) 0.) 
#define ONE ((real) 1.)
#define TWO ((real) 2.)
#define HALF ((real) 0.5)

/* minimum problem size each thread should take care of within parallel regions */
#define CHUNKSIZE ((int) 1000)

static inline int compute_num_threads(const int size)
{
#ifdef _OPENMP
    if (size > omp_get_num_procs()*CHUNKSIZE){
        return omp_get_num_procs();
    }else{
        return 1 + (size - CHUNKSIZE)/CHUNKSIZE;
    }
#else
    return 1;
#endif
}

template <typename real>
void SURE_VAR_prox_graph_d1(real *SURE, real *VAR, real *W, const real *Y, \
                            const real *S2, const real *Mu, const real *La, \
                            const int L, const int V, const int E, \
                            const int *Eu, const int *Ev, const int verbose)
/* 13 arguments */
{
    /***  control the number of threads with Open MP  ***/
    const int ntV = compute_num_threads(V);
    const int ntE = compute_num_threads(E);
    const int ntLE = compute_num_threads(L*E);

    /***  initialize general variables  ***/
    if (verbose){ printf("Initialization... "); FLUSH; }
    int u, v, e, l; /* vertices, edges, indices */
    real a, d, la, *AVGl, SUREl, VARl; /* general purpose temporary real scalars */

    /* compute the weights for average */
    for (v = 0; v < V; v++){ W[v] = ZERO; }
    /* this task cannot be parallelized easily */
    for (e = 0; e < E; e++){
        u = Eu[e];
        v = Ev[e];
        W[u] += ONE;
        if (u != v){ W[v] += ONE; }
    }
    #pragma omp parallel for private(v) num_threads(ntV)
    for (v = 0; v < V; v++){ W[v] = ONE/W[v]; }

    /* precompute some quantities */
    real *WS2 = (real*) malloc(sizeof(real)*V);
    real *A = (real*) malloc(sizeof(real)*E);
    real *D = (real*) malloc(sizeof(real)*E);
    #pragma omp parallel for private(v) num_threads(ntV)
    for (v = 0; v < V; v++){ WS2[v] = W[v]*S2[v]; }
    #pragma omp parallel for private(e, u, v) num_threads(ntE)
    for (e = 0; e < E; e++){
        u = Eu[e];
        v = Ev[e];
        A[e] = HALF*(Y[u] + Y[v]);
        D[e] = HALF*(Y[u] - Y[v]);
    }
    if (verbose){ printf("done.\n"); FLUSH; }

    /* process */
    if (verbose){ printf("Compute SURE and variance for graph d1 denoising... "); FLUSH; }
    #pragma omp parallel private(l, la, SUREl, VARl, AVGl, e, u, v, a, d) num_threads(ntLE)
    {
    AVGl = (real*) alloca(sizeof(real)*V);
    #pragma omp for
    for (l = 0; l < L; l++){
        SUREl = ZERO;
        for (v = 0; v < V; v++){ AVGl[v] = ZERO; }
        for (e = 0; e < E; e++){
            u = Eu[e];
            v = Ev[e];
            a = A[e];
            if (u == v){
                SUREl += WS2[u];
                AVGl[u] += a;
                continue;
            }
            d = D[e];
            la = La[l]*Mu[e];
            if (d > la){
                d = d - la;
            }else if (d < -la){
                d = d + la;
            }else{
                SUREl += (W[u] + W[v])*d*d;
                AVGl[u] += a;
                AVGl[v] += a;
                continue;
            }
            SUREl += (W[u] + W[v])*la*la + WS2[u] + WS2[v];
            AVGl[u] += a + d;
            AVGl[v] += a - d;
        }
        SURE[l] = SUREl;
        VARl = ZERO;
        for (v = 0; v < V; v++){ AVGl[v] = W[v]*AVGl[v]; }
        for (e = 0; e < E; e++){
            u = Eu[e];
            v = Ev[e];
            a = A[e];
            if (u == v){
                VARl += W[u]*(a - AVGl[u])*(a - AVGl[u]);
                continue;
            }
            d = D[e];
            la = La[l]*Mu[e];
            if (d > la){
                d = d - la;
            }else if (d < -la){
                d = d + la;
            }else{
                VARl += W[u]*(a - AVGl[u])*(a - AVGl[u]);
                VARl += W[v]*(a - AVGl[v])*(a - AVGl[v]);
                continue;
            }
            VARl += W[u]*(a + d - AVGl[u])*(a + d - AVGl[u]);
            VARl += W[v]*(a - d - AVGl[v])*(a - d - AVGl[v]);
        }
        VAR[l] = VARl;
    }
    }
    if (verbose){ printf("done.\n"); FLUSH; }
    /* free stuff */
    free(WS2);
    free(A);
    free(D);
}

/* instantiate for compilation */
template void SURE_VAR_prox_graph_d1<float>(float*, float*, float*, const float*, \
                                        const float*, const float*, const float*, \
                                        const int, const int, const int, \
                                        const int*, const int*, const int);

template void SURE_VAR_prox_graph_d1<double>(double*, double*, double*, const double*, \
                                        const double*, const double*, const double*, \
                                        const int, const int, const int, \
                                        const int*, const int*, const int);
