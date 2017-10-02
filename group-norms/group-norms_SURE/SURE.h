/*==================================================================
 * some routines helping SURE computations
 * * * parallel implementation with OpenMP API
 * TODO: - update the iterative SURE routine with group structures given as
 *         sets of indices
 * 
 * Hugo Raguet 2015
 *================================================================*/
#ifndef SURE_H
#define SURE_H

/***  distance term for SURE reweighted (l|d)1,2-norm denoising estimator  ***/
void SURE_prox_rw12_double(double *SURE, const double *La, const int *Idx, const double *Yb, const double *Sb2, const double *S2_Yb, const double *Mu, const int K, const int L, const int M);
void SURE_prox_rw12_single(float *SURE, const float *La, const int *Idx, const float *Yb, const float *Sb2, const float *S2_Yb, const float *Mu, const int K, const int L, const int M);

/***  variance accross grids for group-norm based denoising estimators  ***/
void VAR_proj_d12_real_double(const double *X, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_proj_d12_cplx_double(const double *Xr, const double *Xi, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_proj_d12_real_single(const float *X, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);
void VAR_proj_d12_cplx_single(const float *Xr, const float *Xi, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);
void VAR_proj_l12_real_double(const double *X, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_proj_l12_cplx_double(const double *Xr, const double *Xi, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_proj_l12_real_single(const float *X, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);
void VAR_proj_l12_cplx_single(const float *Xr, const float *Xi, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);
void VAR_prox_d12_real_double(const double *X, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_prox_d12_cplx_double(const double *Xr, const double *Xi, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_prox_d12_real_single(const float *X, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);
void VAR_prox_d12_cplx_single(const float *Xr, const float *Xi, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);
void VAR_prox_l12_real_double(const double *X, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_prox_l12_cplx_double(const double *Xr, const double *Xi, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_prox_l12_real_single(const float *X, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);
void VAR_prox_l12_cplx_single(const float *Xr, const float *Xi, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);
void VAR_prox_rwl12_real_double(const double *X, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_prox_rwl12_cplx_double(const double *Xr, const double *Xi, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_prox_rwl12_real_single(const float *X, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);
void VAR_prox_rwl12_cplx_single(const float *Xr, const float *Xi, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);
void VAR_prox_rwd12_real_double(const double *X, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_prox_rwd12_cplx_double(const double *Xr, const double *Xi, const int ***G, const double **Rb, const double* La, double *V, const int I, const int J, const int N, const int L);
void VAR_prox_rwd12_real_single(const float *X, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);
void VAR_prox_rwd12_cplx_single(const float *Xr, const float *Xi, const int ***G, const float **Rb, const float* La, float *V, const int I, const int J, const int N, const int L);

/***  iterative SURE - those must be updated  ***/
void prox_d12_grid_2D_dif_real(double *X, double *Xd, const double *Thr, const int I, const int J, const int K, const int N, const int si, const int sj, const int shi, const int shj, const bool *ROI);
void prox_d12_grid_2D_dif_cplx(double *Xr, double *Xi, double *Xdr, double *Xdi, const double *Thr, const int I, const int J, const int K, const int N, const int si, const int sj, const int shi, const int shj, const bool *ROI);
void prox_l12_grid_2D_dif_real(double *X, double *Xd, const double *Thr, const int I, const int J, const int K, const int N, const int si, const int sj, const int shi, const int shj, const bool *ROI);
void prox_l12_grid_2D_dif_cplx(double *Xr, double *Xi, double *Xdr, double *Xdi, const double *Thr, const int I, const int J, const int K, const int N, const int si, const int sj, const int shi, const int shj, const bool *ROI);

#endif
