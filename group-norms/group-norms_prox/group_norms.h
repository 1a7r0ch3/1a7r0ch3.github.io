/*==================================================================
 * Methods for manipulating group norms.
 *
 * Hugo Raguet 2015
 *================================================================*/
#ifndef GROUP_NORMS_H
#define GROUP_NORMS_H
#include <stdint.h>

/***  create list of indices constituting a regular nonoverlapping two
 ***  dimensional block structure  ***/
int ** grid_2D_groups(const uint8_t *ROI, const int I, const int J, const int si, const int sj, const int shi, const int shj);

/***  group norm of each group  ***/
void group_norms_d1p_real_double(double *Nb, const double *X, const int **G, const int I, const double p, const double *W);
void group_norms_d1p_cplx_double(double *Nb, const double *Xr, const double *Xi, const int **G, const int I, const double p, const double *W);
void group_norms_d1p_real_single(float *Nb, const float *X, const int **G, const int I, const float p, const float *W);
void group_norms_d1p_cplx_single(float *Nb, const float *Xr, const float *Xi, const int **G, const int I, const float p, const float *W);
void group_norms_l1p_real_double(double *Nb, const double *X, const int **G, const int I, const double p, const double *W);
void group_norms_l1p_cplx_double(double *Nb, const double *Xr, const double *Xi, const int **G, const int I, const double p, const double *W);
void group_norms_l1p_real_single(float *Nb, const float *X, const int **G, const int I, const float p, const float *W);
void group_norms_l1p_cplx_single(float *Nb, const float *Xr, const float *Xi, const int **G, const int I, const float p, const float *W);

/***  compute complete group norm  ***/
double norm_d1inf_real_double(const double *X, const int **G, const double *La, const int I);
double norm_d1inf_cplx_double(const double *Xr, const double *Xi, const int **G, const double *La, const int I);
float norm_d1inf_real_single(const float *X, const int **G, const float *La, const int I);
float norm_d1inf_cplx_single(const float *Xr, const float *Xi, const int **G, const float *La, const int I);
double norm_d12_real_double(const double *X, const int **G, const double *La, const int I);
double norm_d12_cplx_double(const double *Xr, const double *Xi, const int **G, const double *La, const int I);
float norm_d12_real_single(const float *X, const int **G, const float *La, const int I);
float norm_d12_cplx_single(const float *Xr, const float *Xi, const int **G, const float *La, const int I);
double norm_l1inf_real_double(const double *X, const int **G, const double *La, const int I);
double norm_l1inf_cplx_double(const double *Xr, const double *Xi, const int **G, const double *La, const int I);
float norm_l1inf_real_single(const float *X, const int **G, const float *La, const int I);
float norm_l1inf_cplx_single(const float *Xr, const float *Xi, const int **G, const float *La, const int I);
double norm_l12_real_double(const double *X, const int **G, const double *La, const int I);
double norm_l12_cplx_double(const double *Xr, const double *Xi, const int **G, const double *La, const int I);
float norm_l12_real_single(const float *X, const int **G, const float *La, const int I);
float norm_l12_cplx_single(const float *Xr, const float *Xi, const int **G, const float *La, const int I);

/***  group norm penalized denoising estimators  ***/
void proj_d12_real_double(double *X, const int **G, const double *Bnd, const int I);
void proj_d12_cplx_double(double *Xr, double *Xi, const int **G, const double *Bnd, const int I);
void proj_d12_real_single(float *X, const int **G, const float *Bnd, const int I);
void proj_d12_cplx_single(float *Xr, float *Xi, const int **G, const float *Bnd, const int I);
void proj_l1inf_real_double(double *X, const int **G, const double *Bnd, const int I);
void proj_l1inf_cplx_double(double *Xr, double *Xi, const int **G, const double *Bnd, const int I);
void proj_l1inf_real_single(float *X, const int **G, const float *Bnd, const int I);
void proj_l1inf_cplx_single(float *Xr, float *Xi, const int **G, const float *Bnd, const int I);
void proj_l12_real_double(double *X, const int **G, const double *Bnd, const int I);
void proj_l12_cplx_double(double *Xr, double *Xi, const int **G, const double *Bnd, const int I);
void proj_l12_real_single(float *X, const int **G, const float *Bnd, const int I);
void proj_l12_cplx_single(float *Xr, float *Xi, const int **G, const float *Bnd, const int I);
void prox_d12_real_double(double *X, const int **G, const double *Thr, const int I);
void prox_d12_cplx_double(double *Xr, double *Xi, const int **G, const double *Thr, const int I);
void prox_d12_real_single(float *X, const int **G, const float *Thr, const int I);
void prox_d12_cplx_single(float *Xr, float *Xi, const int **G, const float *Thr, const int I);
void prox_l1inf_real_double(double *X, const int **G, const double *Thr, const int I);
void prox_l1inf_cplx_double(double *Xr, double *Xi, const int **G, const double *Thr, const int I);
void prox_l1inf_real_single(float *X, const int **G, const float *Thr, const int I);
void prox_l1inf_cplx_single(float *Xr, float *Xi, const int **G, const float *Thr, const int I);
void prox_l12_real_double(double *X, const int **G, const double *Thr, const int I);
void prox_l12_cplx_double(double *Xr, double *Xi, const int **G, const double *Thr, const int I);
void prox_l12_real_single(float *X, const int **G, const float *Thr, const int I);
void prox_l12_cplx_single(float *Xr, float *Xi, const int **G, const float *Thr, const int I);
void prox_l12_d12_real_double(double *X, const int **G, const double *Thrl, const double *Thrd, const int I);
void prox_l12_d12_cplx_double(double *Xr, double *Xi, const int **G, const double *Thrl, const double *Thrd, const int I);
void prox_l12_d12_real_single(float *X, const int **G, const float *Thrl, const float *Thrd, const int I);
void prox_l12_d12_cplx_single(float *Xr, float *Xi, const int **G, const float *Thrl, const float *Thrd, const int I);
void prox_rwl12_real_double(double *X, const int **G, const double *Thr, const int I);
void prox_rwl12_cplx_double(double *Xr, double *Xi, const int **G, const double *Thr, const int I);
void prox_rwl12_real_single(float *X, const int **G, const float *Thr, const int I);
void prox_rwl12_cplx_single(float *Xr, float *Xi, const int **G, const float *Thr, const int I);
void proxj_d12_real_double(double *X, const int **G, const double *Thr, const double *Bnd, const int I);
void proxj_d12_cplx_double(double *Xr, double *Xi, const int **G, const double *Thr, const double *Bnd, const int I);
void proxj_d12_real_single(float *X, const int **G, const float *Thr, const float *Bnd, const int I);
void proxj_d12_cplx_single(float *Xr, float *Xi, const int **G, const float *Thr, const float *Bnd, const int I);
void proxj_l12_real_double(double *X, const int **G, const double *Thr, const double *Bnd, const int I);
void proxj_l12_cplx_double(double *Xr, double *Xi, const int **G, const double *Thr, const double *Bnd, const int I);
void proxj_l12_real_single(float *X, const int **G, const float *Thr, const float *Bnd, const int I);
void proxj_l12_cplx_single(float *Xr, float *Xi, const int **G, const float *Thr, const float *Bnd, const int I);

/* gradient of squared d2,2-norms */
void grad_d22_2_real_double(double *X, const int **G, const double *La, const int I);
void grad_d22_2_cplx_double(double *Xr, double *Xi, const int **G, const double *La, const int I);
void grad_d22_2_real_single(float *X, const int **G, const float *La, const int I);
void grad_d22_2_cplx_single(float *Xr, float *Xi, const int **G, const float *La, const int I);

/***  compute squared d2,2-norm  ***/
double norm_d22_2_real_double(const double *X, const int **G, const double *La, const int I);
double norm_d22_2_cplx_double(const double *Xr, const double *Xi, const int **G, const double *La, const int I);
float norm_d22_2_real_single(const float *X, const int **G, const float *La, const int I);
float norm_d22_2_cplx_single(const float *Xr, const float *Xi, const int **G, const float *La, const int I);
#endif
