/*==================================================================
 * parallel implementation with omp api
 *
 * Hugo Raguet 2014
 *================================================================*/

#include <omp.h>
#include <math.h>

void SURE_prox_rw12_double(double *SURE, const double *La, const int *Idx, const double *Yb, const double *Sb2, const double *S2_Yb, const double *Mu, const int K, const int L, const int M)
{
    int k, l, m, Lk, Mk;
    double la, la2, ybla, sqrd, diff;
    #pragma omp parallel for private(k, l, m, Lk, Mk, la, la2, ybla, sqrd, diff)
    for (k=0; k<K; k++){
        l = L*k;
        Lk = l + L;
        Mk = M*(k + 1);
        while (l<Lk && (la = La[l])>=0.){
            la2 = la*la;
            m = Mk - M + Idx[l];
            while (++m<Mk && Yb[m]==la){ SURE[l] += la2*Mu[m]*Mu[m]; }
            for (; m<Mk; m++){
                ybla = Yb[m] + la;
                sqrd = sqrt(ybla*ybla - 4.*la2);
                diff = (ybla - sqrd)*Mu[m];
                SURE[l] += diff*diff/4. + (1. + (sqrd - la)/Yb[m])*Sb2[m] + \
                                        (1. + (4.*la - ybla)/sqrd)*la*S2_Yb[m];
            }
            l++;
        }
    }
}

void SURE_prox_rw12_single(float *SURE, const float *La, const int *Idx, const float *Yb, const float *Sb2, const float *S2_Yb, const float *Mu, const int K, const int L, const int M)
{
    int k, l, m, Lk, Mk;
    float la, la2, ybla, sqrd, diff;
    #pragma omp parallel for private(k, l, m, Lk, Mk, la, la2, ybla, sqrd, diff)
    for (k=0; k<K; k++){
        l = L*k;
        Lk = l + L;
        Mk = M*(k + 1);
        while (l<Lk && (la = La[l])>=0.f){
            la2 = la*la;
            m = Mk - M + Idx[l];
            while (++m<Mk && Yb[m]==la){ SURE[l] += la2*Mu[m]*Mu[m]; }
            for (; m<Mk; m++){
                ybla = Yb[m] + la;
                sqrd = sqrtf(ybla*ybla - 4.f*la2);
                diff = (ybla - sqrd)*Mu[m];
                SURE[l] += diff*diff/4.f + (1.f + (sqrd - la)/Yb[m])*Sb2[m] + \
                                      (1.f + (4.f*la - ybla)/sqrd)*la*S2_Yb[m];
            }
            l++;
        }
    }
}
