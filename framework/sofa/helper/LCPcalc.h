/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_LCPCALC_H
#define SOFA_HELPER_LCPCALC_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>

namespace sofa
{

namespace helper
{

//#define NULL 0

using namespace system::thread;

#define EPSILON_LCP		0.00000000001	// epsilon pour tests = 0
#define EPSILON_CONV	0.001			// for GS convergence
#define MAX_BOU	50	// nombre maximal de boucles de calcul

class SOFA_HELPER_API LCP
{
private:
    int maxConst;
    double* dfree;
    double** W;
    double* f;
    double tol;
    int numItMax;
    bool useInitialF;
    double mu;
    int dim;
    unsigned int nbConst;
public:
    LCP(unsigned int maxConstraint);
    ~LCP();
    void reset(void);
    //LCP& operator=(LCP& lcp);
    inline double** getW(void) {return W;};
    inline double& getMu(void) { return mu;};
    inline double* getDfree(void) {return dfree;};
    inline double getTolerance(void) {return tol;};
    inline double getMaxIter(void) {return numItMax;};
    inline double* getF(void) {return f;};
    inline bool useInitialGuess(void) {return useInitialF;};
    inline unsigned int getNbConst(void) {return nbConst;};
    inline unsigned int setNbConst(unsigned int nbC) {nbConst = nbC; return 0;};
    inline unsigned int getMaxConst(void) {return maxConst;};
};



inline SOFA_HELPER_API void set3Dof(double *vector, int index, double &vx, double &vy, double &vz)
{vector[3*index]=vx; vector[3*index+1]=vy; vector[3*index+2]=vz;}
inline SOFA_HELPER_API void add3Dof(double *vector, int index, double &vx, double &vy, double &vz)
{vector[3*index]+=vx; vector[3*index+1]+=vy; vector[3*index+2]+=vz;}
inline SOFA_HELPER_API double normError(double &f1x, double &f1y, double &f1z, double &f2x, double &f2y, double &f2z)
{
    return sqrt( ((f2x-f1x)*(f2x-f1x) + (f2y-f1y)*(f2y-f1y) + (f2z-f1z)*(f2z-f1z)) /
            (f1x*f1x + f1y*f1y + f1z*f1z) ) ;
}

inline SOFA_HELPER_API double absError(double &f1x, double &f1y, double &f1z, double &f2x, double &f2y, double &f2z)
{return sqrt ((f2x-f1x)*(f2x-f1x) + (f2y-f1y)*(f2y-f1y) + (f2z-f1z)*(f2z-f1z));}


SOFA_HELPER_API int resoudreLCP(int, double *, double **, double *);
SOFA_HELPER_API int lcp_lexicolemke(int, double *, double **, double *);
// same with pre-allocated matrix A
SOFA_HELPER_API int lcp_lexicolemke(int, double *, double **, double **, double *);


SOFA_HELPER_API void afficheSyst(double *q,double **M, int *base, double **mat, int dim);
SOFA_HELPER_API void afficheLCP(double *q, double **M, int dim);
SOFA_HELPER_API void afficheLCP(double *q, double **M, double *f, int dim);

typedef SOFA_HELPER_API double FemClipsReal;
SOFA_HELPER_API void gaussSeidelLCP1(int, FemClipsReal *,FemClipsReal **, FemClipsReal *, double , int );




// inverted SymMatrix 3x3 //
class SOFA_HELPER_API LocalBlock33
{
public:
    LocalBlock33() {computed=false;};
    ~LocalBlock33() {};

    void compute(double &w11, double &w12, double &w13, double &w22, double &w23, double &w33);
    void stickState(double &dn, double &dt, double &ds, double &fn, double &ft, double &fs);
    void slipState(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs);

    // computation of a new state using a simple gauss-seidel loop // pseudo-potential
    void GS_State(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs);

    // computation of a new state using a simple gauss-seidel loop // pseudo-potential (new: dn, dt, ds already take into account current value of fn, ft and fs)
    void New_GS_State(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs);

    // computation of a new state using biPotential approach
    void BiPotential(double &mu, double &dn, double &dt, double &ds, double &fn, double &ft, double &fs);

    void setPreviousForce(double &fn, double &ft, double &fs) {f_1[0]=fn; f_1[1]=ft; f_1[2]=fs;}

    bool computed;
public:

    double w[6];
    double wInv[6];
    double det;
    double f_1[3]; // previous value of force
};

// Gauss-Seidel like algorithm for contacts
SOFA_HELPER_API int nlcp_gaussseidel(int, double *, double**, double *, double, double, int, bool, bool verbose=false);
// Timed Gauss-Seidel like algorithm for contacts
SOFA_HELPER_API int nlcp_gaussseidelTimed(int, double *, double**, double *, double, double, int, bool, double timeout, bool verbose=false);
} // namespace helper

} // namespace sofa

#endif
