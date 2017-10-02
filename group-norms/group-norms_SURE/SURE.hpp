/*==================================================================
 * SURE and variance accross edges for graph d1 denoising estimator
 *
 * parallel implementation with OpenMP API
 *
 * Recall: over one edge, with d1(la,(xu,xv)) = la |xu - xv|,
 *
 * prox_{d1,la}((xu,xv)) = ((xu+xv)/2 + (1 - 2la/|xu - xv|)(xu - xv)/2)   
 *                         ((xu+xv)/2 + (1 - 2la/|xu - xv|)(xv - xu)/2),
 *                                                  if |xu - xv| >  2 la
 *                         ((xu+xv)/2)
 *                         ((xu+xv)/2),             if |xu - xv| <= 2 la
 *
 * SURE(prox_{d1,la}, (xu,xv)) = 2 la^v + su^v + sv^v  if |xu - xv| >  2 la
 *                               1/2 |xu - xv|^v       if |xu - xv| <= 2 la
 *      
 * Reference: H. Raguet, A Signal Processing Approach to Voltage Sensitive Dye,
 * Chapter V: "Risk Estimation for Parameter Selection in Proximal Denoising",
 *  Ph.D. Thesis, 2014.
 * 
 * Hugo Raguet 2016
 *================================================================*/
#ifndef SURE_H
#define SURE_H

template <typename real>
void SURE_VAR_prox_graph_d1(real *SURE, real *VAR, real *W, const real *Y, \
                            const real *S2, const real *Mu, const real *La, \
                            const int L, const int V, const int E, \
                            const int *Eu, const int *Ev, const int verbose);
/* 13 arguments:
 * SURE, VAR - SURE and VAR values for different penalization scaling,
 *             arrays of length L
 * W         - for each node, inverse of the number of edges involving this node,
 *             array of length V (computed by the function)
 * Y         - observations, array of length V
 * S2        - noise variance on observations, array of length V
 * Mu        - individual scaling on each edge, array of length E
 * La        - list of overall scaling to test, array of length L
 * L         - number of overall scaling to test
 * V, E      - number of vertices and of (undirected) edges
 * Eu        - for each edge, index of one vertex, array of length E
 * Ev        - for each edge, index of the other vertex, array of length E
 *             Every vertex should belong to at least one edge. If it is not the
 *             case, a workaround is to add an edge from the vertex to itself
 *             with a nonzero penalization coefficient.
 * verbose   - if nonzero, display information on the progress */

#endif
