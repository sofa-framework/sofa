/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_LCPCALC_H
#define SOFA_HELPER_LCPCALC_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sofa/helper/system/thread/CTime.h>
#include <vector>
#include <ostream>


namespace sofa
{

namespace helper
{

#define EPSILON_LCP		0.00000000001	// epsilon pour tests = 0
#define EPSILON_CONV	0.001			// for GS convergence
#define MAX_BOU	50	// nombre maximal de boucles de calcul

class SOFA_HELPER_API LCP
{
    int maxConst;
    SReal* dfree;
    SReal** W;
    SReal* f, *f_1;
    SReal* d;
    SReal tol;
    int numItMax;
    bool useInitialF;
    SReal mu;
    int dim;  //=3*nbContact !!
    //unsigned int nbConst;


public:
    LCP();
    ~LCP();
    void reset(void);
    void allocate (unsigned int maxConst);
    inline SReal** getW(void) {return W;};
    inline SReal& getMu(void) { return mu;};
    inline SReal* getDfree(void) {return dfree;};
    inline SReal getTolerance(void) {return tol;};
    inline SReal getMaxIter(void) {return numItMax;};
    inline SReal* getF(void) {return f;};
    inline SReal* getF_1(void) {return f_1;};
    inline SReal* getD(void) {return d;};
    inline bool useInitialGuess(void) {return useInitialF;};
    inline unsigned int getDim(void) {return dim;};
    inline unsigned int setDim(unsigned int nbC) {dim = nbC; return 0;};
    inline unsigned int getMaxConst(void) {return maxConst;};
    inline void setNumItMax(int input_numItMax) {numItMax = input_numItMax;};
    inline void setTol(SReal input_tol) {tol = input_tol;};

    void setLCP(unsigned int input_dim, SReal *input_dfree, SReal **input_W, SReal *input_f, SReal &input_mu, SReal &input_tol, int input_numItMax);

    void solveNLCP(bool convergenceTest, std::vector<SReal>* residuals = nullptr, std::vector<SReal>* violations = nullptr);
    int it; // to get the number of iteration that is necessary for convergence
    SReal error; // to get the error at the end of the convergence
};




inline SOFA_HELPER_API void set3Dof(SReal*vector, int index, SReal vx, SReal vy, SReal vz)
{vector[3*index]=vx; vector[3*index+1]=vy; vector[3*index+2]=vz;}
inline SOFA_HELPER_API void add3Dof(SReal*vector, int index, SReal vx, SReal vy, SReal vz)
{vector[3*index]+=vx; vector[3*index+1]+=vy; vector[3*index+2]+=vz;}
inline SOFA_HELPER_API SReal normError(SReal f1x, SReal f1y, SReal f1z, SReal f2x, SReal f2y, SReal f2z)
{
    return sqrt( ((f2x-f1x)*(f2x-f1x) + (f2y-f1y)*(f2y-f1y) + (f2z-f1z)*(f2z-f1z)) /
            (f1x*f1x + f1y*f1y + f1z*f1z) ) ;
}

inline SOFA_HELPER_API SReal absError(SReal f1x, SReal f1y, SReal f1z, SReal f2x, SReal f2y, SReal f2z)
{return sqrt ((f2x-f1x)*(f2x-f1x) + (f2y-f1y)*(f2y-f1y) + (f2z-f1z)*(f2z-f1z));}


SOFA_HELPER_API SOFA_LCPCALC_RESOUDRELCP_DEPRECATED() int resoudreLCP(int, SReal *, SReal **, SReal *);
SOFA_HELPER_API int solveLCP(int, SReal *, SReal **, SReal *);


SOFA_HELPER_API SOFA_LCPCALC_AFFICHESYST_DEPRECATED() void afficheSyst(SReal *q,SReal **M, int *base, SReal **mat, int dim);
SOFA_HELPER_API SOFA_LCPCALC_AFFICHELCP_DEPRECATED() void afficheLCP(SReal *q, SReal **M, int dim);
SOFA_HELPER_API SOFA_LCPCALC_AFFICHELCP_DEPRECATED() void afficheLCP(SReal *q, SReal **M, SReal *f, int dim);
SOFA_HELPER_API void printSyst(SReal* q, SReal** M, int* base, SReal** mat, int dim);
SOFA_HELPER_API void printLCP(SReal* q, SReal** M, int dim);
SOFA_HELPER_API void printLCP(SReal* q, SReal** M, SReal* f, int dim);
SOFA_HELPER_API void resultToString(std::ostream& s, SReal *f, int dim);

typedef SReal FemClipsReal;
SOFA_HELPER_API void gaussSeidelLCP1(int dim, FemClipsReal * q, FemClipsReal ** M, FemClipsReal * res, SReal tol, int numItMax, SReal minW=0.0, SReal maxF=0.0, std::vector<SReal>* residuals = nullptr);



// inverted SymMatrix 3x3 //
class SOFA_HELPER_API LocalBlock33
{
public:
    LocalBlock33() {computed=false;};
    ~LocalBlock33() {};

    void compute(SReal &w11, SReal &w12, SReal &w13, SReal &w22, SReal &w23, SReal &w33);
    void stickState(SReal &dn, SReal &dt, SReal &ds, SReal &fn, SReal &ft, SReal &fs);
    void slipState(SReal &mu, SReal &dn, SReal &dt, SReal &ds, SReal &fn, SReal &ft, SReal &fs);

    // computation of a new state using a simple gauss-seidel loop // pseudo-potential
    void GS_State(SReal &mu, SReal &dn, SReal &dt, SReal &ds, SReal &fn, SReal &ft, SReal &fs);

    // computation of a new state using a simple gauss-seidel loop // pseudo-potential (new: dn, dt, ds already take into account current value of fn, ft and fs)
    void New_GS_State(SReal &mu, SReal &dn, SReal &dt, SReal &ds, SReal &fn, SReal &ft, SReal &fs);

    // computation of a new state using biPotential approach
    void BiPotential(SReal &mu, SReal &dn, SReal &dt, SReal &ds, SReal &fn, SReal &ft, SReal &fs);

    void setPreviousForce(SReal &fn, SReal &ft, SReal &fs) {f_1[0]=fn; f_1[1]=ft; f_1[2]=fs;}

    bool computed;

    SReal w[6];
    SReal wInv[6];
    SReal det;
    SReal f_1[3]; // previous value of force
};

// Multigrid algorithm for contacts
SOFA_HELPER_API int nlcp_multiGrid(int dim, SReal *dfree, SReal**W, SReal *f, SReal mu, SReal tol, int numItMax, bool useInitialF, SReal** W_coarse, std::vector<int> &contact_group, unsigned int num_group,  bool verbose=false);
SOFA_HELPER_API int nlcp_multiGrid_2levels(int dim, SReal *dfree, SReal**W, SReal *f, SReal mu, SReal tol, int numItMax, bool useInitialF,
        std::vector< int> &contact_group, unsigned int num_group, std::vector< int> &constraint_group, std::vector<SReal> &constraint_group_fact, bool verbose, std::vector<SReal>* residuals1 = nullptr, std::vector<SReal>* residuals2 = nullptr);
SOFA_HELPER_API int nlcp_multiGrid_Nlevels(int dim, SReal *dfree, SReal**W, SReal *f, SReal mu, SReal tol, int numItMax, bool useInitialF,
        std::vector< std::vector< int> > &contact_group_hierarchy, std::vector<unsigned int> Tab_num_group, std::vector< std::vector< int> > &constraint_group_hierarchy, std::vector< std::vector< SReal> > &constraint_group_fact_hierarchy, bool verbose, std::vector<SReal> *residualsN = nullptr, std::vector<SReal> *residualLevels = nullptr, std::vector<SReal> *violations = nullptr);

// Gauss-Seidel like algorithm for contacts
SOFA_HELPER_API int nlcp_gaussseidel(int dim, SReal*dfree, SReal**W, SReal*f, SReal mu, SReal tol, int numItMax, bool useInitialF, bool verbose = false, SReal minW=0.0, SReal maxF=0.0, std::vector<SReal>* residuals = nullptr, std::vector<SReal>* violations = nullptr);
// Timed Gauss-Seidel like algorithm for contacts
SOFA_HELPER_API int nlcp_gaussseidelTimed(int, SReal*, SReal**, SReal*, SReal, SReal, int, bool, SReal timeout, bool verbose=false);
} // namespace helper

} // namespace sofa

#endif
