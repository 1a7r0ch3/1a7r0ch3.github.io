/*==================================================================
 * Create list of indices constituing a regular nonoverlaping two dimensional
 * block structure.
 *
 * Hugo Raguet 2015
 *================================================================*/

#include <stdlib.h>
#include <stdint.h>

int ** grid_2D_groups(const uint8_t *ROI, const int I, const int J, const int si, const int sj, const int shi, const int shj)
{
	int i, j, l, ib, jb, g = 0, max_l = 0, *gidx;
	const int offi = shi > 0  ?  shi - si  :  0;
	const int offj = shj > 0  ?  shj - sj  :  0;
    const int Ib = 1 + (I - offi - 1)/si; /* ceil((I - offi)/si) */
	const int Jb = 1 + (J - offj - 1)/sj; /* ceil((J - offj)/sj) */
    int **G = malloc(sizeof(int*)*(Ib*Jb + 1)); /* size is an upper bound, unused indices will never be allocated */
    /* compute indices within ROI */
    int *IDX = malloc(sizeof(int)*I*J);
    if (ROI != NULL){
        for (i=0, l=0; l<I*J; l++){ IDX[l] = ROI[l] ? i++ : -1; }
    }
    /* iterate over all groups */
    for (ib=0; ib<Ib; ib++){
    for (jb=0; jb<Jb; jb++){
        l = 0;
        /* iterate over all coefficients in the group */
        for (j=offj+jb*sj; j<offj+(jb+1)*sj; j++){
        if (0<=j && j<J){
            for (i=offi+ib*si; i<offi+(ib+1)*si; i++){
            if (0<=i && i<I){
                if (ROI==NULL || ROI[i+I*j]){ l++; }
            }}
        }}
        if (l > 0){ /* group is nonempty: store group length and indices */
            max_l = l > max_l  ?  l  :  max_l;
            gidx = malloc(sizeof(int)*(l + 1));
            gidx[0] = l;
            l = 0;
            for (j=offj+jb*sj; j<offj+(jb+1)*sj; j++){
            if (0<=j && j<J){
                for (i=offi+ib*si; i<offi+(ib+1)*si; i++){
                if (0<=i && i<I){
                    if (ROI == NULL){ 
                        gidx[++l] = i + I*j;
                    }else if (ROI[i+I*j]){
                        gidx[++l] = IDX[i+I*j];
                    }
                }}
            }}
            G[++g] = gidx;
        }
    }}
    /* meta-info of group structure */
    G[0] = malloc(sizeof(int)*2);
    G[0][0] = g;
    G[0][1] = max_l;
    /* free memory and return group structure */
    free(IDX);
    return G;
}
