#ifndef SOFA_HELPER_LCPCALC_H
#define SOFA_HELPER_LCPCALC_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

namespace sofa
{

namespace helper
{


#define EPSILON_LCP		0.00000000001	// epsilon pour tests = 0
#define MAX_BOU	50	// nombre maximal de boucles de calcul

int resoudreLCP(int, double *, double **, double *);
int lcp_lexicolemke(int, double *, double **, double *);
// same with pre-allocated matrix A
int lcp_lexicolemke(int, double *, double **, double **, double *);


void afficheSyst(double *q,double **M, int *base, double **mat, int dim);
void afficheLCP(double *q, double **M, int dim);

} // namespace helper

} // namespace sofa

#endif
