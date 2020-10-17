#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
double* a;
double* b;
double* ca;
int D;
int M;
int P;

double nrm(int i, int j){
    double diff;
    double dist = 0.;
    int d; int m;
    for (d = 0; d < D; d++){
        for(m = 0; m < M; m++){
            diff = *(a + P * M * d + P * m + (i - 1)) - *(b + P * M * d + P * m + (j - 1));
            dist += diff * diff;
        }
    }
    dist = sqrt(dist);
    //if ( i == j){
    //  printf("%f \n", dist);
    //}
    return dist;
}

double recursive_c(int i, int j){
    double *ca_ij; /* Pointer to `ca(i, j)`, just to simplify notation */
    ca_ij = ca + (i - 1) * P + (j - 1);

    /* This implements the algorithm from [1] */
    if (*ca_ij > -1.0){
        return *ca_ij;
    }
    else if ((i == 1) && (j == 1)){
        *ca_ij = nrm(1, 1);
    }
    else if ((i > 1) && (j == 1)){
        *ca_ij = fmax(recursive_c(i - 1, 1), nrm(i, 1));
    }
    else if ((i == 1) && (j > 1)){
        *ca_ij = fmax(recursive_c(1, j - 1), nrm(1, j));
    }
    else if ((i > 1) && (j > 1)){
        *ca_ij = fmax(
                      fmin(fmin(
                           recursive_c(i - 1, j    ),
                           recursive_c(i - 1, j - 1)),
                           recursive_c(i,     j - 1)),
                      nrm(i, j));
        // printf("%f \n", &ca_ij);
    }
    else{
        *ca_ij = DBL_MAX;
    }

    return *ca_ij;
}

double frechet_distance(int _D, int _M, int _P, double *_a, double *_b){
    D = _D; M = _M; P = _P;
    a = _a; b = _b;
    // printf("%d %d %d \n" ,D, M, P);
    // printf("%f", a[30]);

    int k; /* Index for initialisation of `ca`*/
    int i = P;
    int j = P;

    /* Allocate memory for `ca` */
    ca = (double *) malloc(i * j * sizeof(double));

    /* Initialise it with -1.0 */
    for (k = 0; k < i * j; k++){
        *(ca + k) = -1.0;
    }

    /* Call the recursive computation of the coupling measure */
    double ans = recursive_c(i, j);

    /* Free memory */
    free(ca);
    return ans;
}
