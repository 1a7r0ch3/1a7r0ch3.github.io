/*==================================================================
 * Hugo Raguet 2015
 *==================================================================*/

#include <math.h>

/* real inputs, unweighted, double precision */
void add_pow_l11_real_double(double *ngp, double x, const double p, const double *W, const int i)
    { *ngp += x>0. ? x : -x; }
void add_pow_l12_real_double(double *ngp, double x, const double p, const double *W, const int i)
    { *ngp += x*x; }
void add_pow_l1inf_real_double(double *ngp, double x, const double p, const double *W, const int i)
    { x = x>0. ? x : -x; *ngp = *ngp>x ? *ngp : x; }
void add_pow_l1p_real_double(double *ngp, double x, const double p, const double *W, const int i)
    { x = x>0. ? x : -x; *ngp += pow(x, p); }

/* complex inputs, unweighted, double precision */
void add_pow_l11_cplx_double(double *ngp, double xr, const double xi, const double p, const double *W, const int i)
    { *ngp += sqrt(xr*xr + xi*xi); }
void add_pow_l12_cplx_double(double *ngp, double xr, const double xi, const double p, const double *W, const int i)
    { *ngp += xr*xr + xi*xi; }
void add_pow_l1inf_cplx_double(double *ngp, double xr, const double xi, const double p, const double *W, const int i)
    { xr = sqrt(xr*xr + xi*xi); *ngp = *ngp>xr ? *ngp : xr; }
void add_pow_l1p_cplx_double(double *ngp, double xr, const double xi, const double p, const double *W, const int i)
    { xr = sqrt(xr*xr + xi*xi); *ngp += pow(xr, p); }

/* real inputs, weighted, double precision */
void add_pow_wl11_real_double(double *ngp, double x, const double p, const double *W, const int i)
    { *ngp += x>0. ? W[i]*x : W[i]*(-x); }
void add_pow_wl12_real_double(double *ngp, double x, const double p, const double *W, const int i)
    { *ngp += W[i]*x*x; }
void add_pow_wl1inf_real_double(double *ngp, double x, const double p, const double *W, const int i)
    { x = x>0. ? x : -x; *ngp = *ngp>x ? *ngp : W[i]*x; }
void add_pow_wl1p_real_double(double *ngp, double x, const double p, const double *W, const int i)
    { x = x>0. ? x : -x; *ngp += W[i]*pow(x, p); }

/* complex inputs, weighted, double precision */
void add_pow_wl11_cplx_double(double *ngp, double xr, const double xi, const double p, const double *W, const int i)
    { *ngp += W[i]*sqrt(xr*xr + xi*xi); }
void add_pow_wl12_cplx_double(double *ngp, double xr, const double xi, const double p, const double *W, const int i)
    { *ngp += W[i]*(xr*xr + xi*xi); }
void add_pow_wl1p_cplx_double(double *ngp, double xr, const double xi, const double p, const double *W, const int i)
    { xr = sqrt(xr*xr + xi*xi); *ngp += W[i]*pow(xr, p); }

/* radicals */
double radical_1_double(const double ngp, const double invp){ return ngp; }
double radical_2_double(const double ngp, const double invp){ return sqrt(ngp); }
double radical_p_double(const double ngp, const double invp){ return pow(ngp, invp); }

/* real inputs, unweighted, single precision */
void add_pow_l11_real_single(float *ngp, float x, const float p, const float *W, const int i)
    { *ngp += x>0.f ? x : -x; }
void add_pow_l12_real_single(float *ngp, float x, const float p, const float *W, const int i)
    { *ngp += x*x; }
void add_pow_l1inf_real_single(float *ngp, float x, const float p, const float *W, const int i)
    { x = x>0.f ? x : -x; *ngp = *ngp>x ? *ngp : x; }
void add_pow_l1p_real_single(float *ngp, float x, const float p, const float *W, const int i)
    { x = x>0.f ? x : -x; *ngp += powf(x, p); }

/* complex inputs, unweighted, single precision */
void add_pow_l11_cplx_single(float *ngp, float xr, const float xi, const float p, const float *W, const int i)
    { *ngp += sqrtf(xr*xr + xi*xi); }
void add_pow_l12_cplx_single(float *ngp, float xr, const float xi, const float p, const float *W, const int i)
    { *ngp += xr*xr + xi*xi; }
void add_pow_l1inf_cplx_single(float *ngp, float xr, const float xi, const float p, const float *W, const int i)
    { xr = sqrtf(xr*xr + xi*xi); *ngp = *ngp>xr ? *ngp : xr; }
void add_pow_l1p_cplx_single(float *ngp, float xr, const float xi, const float p, const float *W, const int i)
    { xr = sqrtf(xr*xr + xi*xi); *ngp += powf(xr, p); }

/* real inputs, weighted, single precision */
void add_pow_wl11_real_single(float *ngp, float x, const float p, const float *W, const int i)
    { *ngp += x>0.f ? W[i]*x : W[i]*(-x); }
void add_pow_wl12_real_single(float *ngp, float x, const float p, const float *W, const int i)
    { *ngp += W[i]*x*x; }
void add_pow_wl1inf_real_single(float *ngp, float x, const float p, const float *W, const int i)
    { x = x>0.f ? x : -x; *ngp = *ngp>x ? *ngp : W[i]*x; }
void add_pow_wl1p_real_single(float *ngp, float x, const float p, const float *W, const int i)
    { x = x>0.f ? x : -x; *ngp += W[i]*powf(x, p); }

/* complex inputs, weighted, single precision */
void add_pow_wl11_cplx_single(float *ngp, float xr, const float xi, const float p, const float *W, const int i)
    { *ngp += W[i]*sqrtf(xr*xr + xi*xi); }
void add_pow_wl12_cplx_single(float *ngp, float xr, const float xi, const float p, const float *W, const int i)
    { *ngp += W[i]*(xr*xr + xi*xi); }
void add_pow_wl1p_cplx_single(float *ngp, float xr, const float xi, const float p, const float *W, const int i)
    { xr = sqrtf(xr*xr + xi*xi); *ngp += W[i]*powf(xr, p); }

/* radicals, single precision */
float radical_1_single(const float ngp, const float invp){ return ngp; }
float radical_2_single(const float ngp, const float invp){ return sqrtf(ngp); }
float radical_p_single(const float ngp, const float invp){ return powf(ngp, invp); }
